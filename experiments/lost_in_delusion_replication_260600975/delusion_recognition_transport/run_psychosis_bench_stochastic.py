#!/usr/bin/env python3
"""Run multi-seed Psychosis-Bench trajectories with one vLLM model load."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import torch
from tqdm import tqdm
from vllm import LLM, SamplingParams

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from io_utils import append_jsonl, canonicalize, read_jsonl, write_manifest  # noqa: E402
from run_local_models import model_slug, trim_to_limit  # noqa: E402

PAPER_REPOSITORY_COMMIT = "73966f95be2e40f1ceee76dfe08eebe939ad6c21"
EXPECTED_CASES_SHA256 = (
    "d9b7820c0bebb6ec845e5825378535e8e35b79b0244a72be64ec5e49d8da439f"
)


def successful(row: dict[str, Any]) -> bool:
    return isinstance(row.get("response"), str) and not row.get(
        "generation_error"
    )


def pair_id(case_id: str) -> str:
    for suffix in ("_explicit", "_implicit"):
        if case_id.endswith(suffix):
            return case_id[: -len(suffix)]
    raise ValueError(f"Case ID does not encode condition: {case_id}")


def request_seed(
    replicate_seed: int,
    case_id: str,
    turn_number: int,
) -> int:
    digest = hashlib.sha256(
        f"{replicate_seed}:{case_id}:{turn_number}".encode()
    ).digest()
    return int.from_bytes(digest[:4], "big") & 0x7FFFFFFF


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cases", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--reported-model")
    parser.add_argument("--checkpoint-source")
    parser.add_argument("--revision")
    parser.add_argument(
        "--seeds",
        default="1101,2202,3303",
        help="Comma-separated independent trajectory seeds.",
    )
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--max-input-tokens", type=int, default=11776)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.90)
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument("--cpu-offload-gb", type=float, default=0.0)
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--disable-thinking", action="store_true")
    parser.add_argument("--enforce-eager", action="store_true")
    args = parser.parse_args()

    seeds = [int(value) for value in args.seeds.split(",") if value.strip()]
    if len(seeds) != len(set(seeds)) or not seeds:
        raise ValueError("Seeds must be a nonempty unique list")
    cases_bytes = args.cases.read_bytes()
    cases_sha256 = hashlib.sha256(cases_bytes).hexdigest()
    if cases_sha256 != EXPECTED_CASES_SHA256:
        raise ValueError("Cases file does not match the frozen public source")
    cases = json.loads(cases_bytes)["cases"]
    if len(cases) != 16 or any(len(case["prompts"]) != 12 for case in cases):
        raise ValueError("Expected the original 16 cases with 12 turns each")
    pairs: dict[str, set[str]] = {}
    for case in cases:
        pairs.setdefault(pair_id(case["id"]), set()).add(
            case["condition"].lower()
        )
    if len(pairs) != 8 or any(
        conditions != {"explicit", "implicit"}
        for conditions in pairs.values()
    ):
        raise ValueError("Expected eight explicit/implicit scenario pairs")

    reported_model = args.reported_model or args.model
    checkpoint_source = args.checkpoint_source or args.model
    slug = model_slug(reported_model)
    prior = canonicalize(
        args.output,
        lambda row: str(row["generation_id"]),
        successful,
    )
    completed = {
        (
            int(row["replicate_seed"]),
            row["case_id"],
            int(row["turn_number"]),
        ): row
        for row in prior
        if row.get("model") == reported_model and successful(row)
    }
    expected_rows = 16 * 12 * len(seeds)
    manifest_path = args.output.with_suffix(
        args.output.suffix + ".manifest.json"
    )
    if len(completed) == expected_rows and manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if (
            manifest.get("model") == reported_model
            and manifest.get("successful_rows") == expected_rows
            and manifest.get("expected_rows") == expected_rows
        ):
            print(json.dumps(manifest, indent=2))
            return

    llm = LLM(
        model=args.model,
        revision=args.revision,
        dtype=args.dtype,
        max_model_len=args.max_input_tokens + args.max_new_tokens,
        gpu_memory_utilization=args.gpu_memory_utilization,
        trust_remote_code=args.trust_remote_code,
        enable_prefix_caching=True,
        tensor_parallel_size=args.tensor_parallel_size,
        cpu_offload_gb=args.cpu_offload_gb,
        enforce_eager=args.enforce_eager,
    )
    tokenizer = llm.get_tokenizer()
    chat_template_kwargs = (
        {"enable_thinking": False} if args.disable_thinking else None
    )
    histories: dict[tuple[int, str], list[dict[str, str]]] = {
        (seed, case["id"]): [] for seed in seeds for case in cases
    }
    generated = 0
    reused = 0
    dropped_messages = 0

    for turn_index in tqdm(
        range(12), desc=f"Psychosis-Bench stochastic {slug}"
    ):
        pending: list[dict[str, Any]] = []
        for seed in seeds:
            for case in cases:
                case_id = case["id"]
                history = histories[(seed, case_id)]
                user_message = {
                    "role": "user",
                    "content": case["prompts"][turn_index],
                }
                history.append(user_message)
                key = (seed, case_id, turn_index + 1)
                existing = completed.get(key)
                if existing:
                    history.append(
                        {
                            "role": "assistant",
                            "content": existing["response"],
                        }
                    )
                    reused += 1
                    continue
                used_messages, token_ids, dropped = trim_to_limit(
                    tokenizer,
                    history,
                    args.max_input_tokens,
                    system_prompt=None,
                    chat_template_kwargs=chat_template_kwargs,
                )
                pending.append(
                    {
                        "seed": seed,
                        "case": case,
                        "messages": list(history),
                        "used_messages": used_messages,
                        "token_ids": token_ids,
                        "dropped": dropped,
                        "request_seed": request_seed(
                            seed, case_id, turn_index + 1
                        ),
                    }
                )

        if not pending:
            continue
        outputs = llm.generate(
            [
                {"prompt_token_ids": item["token_ids"]}
                for item in pending
            ],
            sampling_params=[
                SamplingParams(
                    temperature=args.temperature,
                    top_p=args.top_p,
                    max_tokens=args.max_new_tokens,
                    seed=item["request_seed"],
                )
                for item in pending
            ],
            use_tqdm=True,
        )
        for item, output in zip(pending, outputs, strict=True):
            case = item["case"]
            seed = item["seed"]
            turn_number = turn_index + 1
            text = output.outputs[0].text.strip()
            generation_id = (
                f"{slug}:psychosis_bench_t1:seed{seed}:"
                f"{case['id']}:turn{turn_number:02d}"
            )
            row = {
                "pair_id": pair_id(case["id"]),
                "cluster_id": f"psychosis_bench:{pair_id(case['id'])}",
                "input_id": (
                    f"seed{seed}:{case['id']}:turn{turn_number:02d}"
                ),
                "case_id": case["id"],
                "case_name": case["name"],
                "turn_number": turn_number,
                "condition": case["condition"].lower(),
                "messages": item["messages"],
                "theme": case["theme"],
                "source": "psychosis_bench_synthetic",
                "harm_type": case["harm_type"],
                "original_harm": (
                    "clear" if turn_number >= 10 else "possible"
                ),
                "original_distress": None,
                "high_stakes_action": turn_number >= 10,
                "harm_pair_valid": turn_number >= 10,
                "transport_paper": "arXiv:2509.10970",
                "generation_id": generation_id,
                "model": reported_model,
                "model_checkpoint": args.model,
                "model_checkpoint_source": checkpoint_source,
                "model_revision": args.revision,
                "model_dtype": args.dtype,
                "model_slug": slug,
                "system_prompt": None,
                "decoding": "stochastic_api_default_sensitivity",
                "temperature": args.temperature,
                "top_p": args.top_p,
                "thinking_disabled": args.disable_thinking,
                "replicate_seed": seed,
                "request_seed": item["request_seed"],
                "generation_backend": "vllm",
                "max_input_tokens": args.max_input_tokens,
                "max_new_tokens": args.max_new_tokens,
                "tensor_parallel_size": args.tensor_parallel_size,
                "cpu_offload_gb": args.cpu_offload_gb,
                "input_token_count": len(item["token_ids"]),
                "dropped_message_count": item["dropped"],
                "messages_used": item["used_messages"],
                "response": text,
            }
            append_jsonl(args.output, row)
            histories[(seed, case["id"])].append(
                {"role": "assistant", "content": text}
            )
            completed[(seed, case["id"], turn_number)] = row
            generated += 1
            dropped_messages += int(item["dropped"])

    final = canonicalize(
        args.output,
        lambda row: str(row["generation_id"]),
        successful,
    )
    model_rows = [
        row
        for row in final
        if row.get("model") == reported_model and successful(row)
    ]
    manifest = {
        "paper": "arXiv:2509.10970",
        "paper_repository_commit": PAPER_REPOSITORY_COMMIT,
        "source": str(args.cases),
        "source_sha256": cases_sha256,
        "source_cases": len(cases),
        "source_scenario_pairs": len(pairs),
        "model": reported_model,
        "model_checkpoint": args.model,
        "model_checkpoint_source": checkpoint_source,
        "model_revision": args.revision,
        "model_dtype": args.dtype,
        "model_slug": slug,
        "backend": "vllm",
        "decoding": "stochastic_api_default_sensitivity",
        "temperature": args.temperature,
        "top_p": args.top_p,
        "replicate_seeds": seeds,
        "system_prompt": None,
        "max_input_tokens": args.max_input_tokens,
        "max_new_tokens": args.max_new_tokens,
        "tensor_parallel_size": args.tensor_parallel_size,
        "cpu_offload_gb": args.cpu_offload_gb,
        "thinking_disabled": args.disable_thinking,
        "enforce_eager": args.enforce_eager,
        "new_generated": generated,
        "reused_rows": reused,
        "successful_rows": len(model_rows),
        "expected_rows": expected_rows,
        "total_dropped_messages": dropped_messages,
        "cuda": torch.version.cuda,
        "torch": torch.__version__,
    }
    manifest_path.write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
