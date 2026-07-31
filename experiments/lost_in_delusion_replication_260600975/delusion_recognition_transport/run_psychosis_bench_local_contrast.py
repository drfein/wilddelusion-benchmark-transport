#!/usr/bin/env python3
"""Run matched local explicit/implicit contrasts on synthetic trajectories."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import defaultdict
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
EXPECTED_SEEDS = (1101, 2202, 3303)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_json(value: Any) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def successful(row: dict[str, Any]) -> bool:
    return isinstance(row.get("response"), str) and not row.get(
        "generation_error"
    )


def base_pair_id(case_id: str) -> str:
    for suffix in ("_explicit", "_implicit"):
        if case_id.endswith(suffix):
            return case_id[: -len(suffix)]
    raise ValueError(f"Case ID lacks explicitness suffix: {case_id}")


def shared_request_seed(
    replicate_seed: int,
    pair_id: str,
    history_anchor: str,
    turn_number: int,
) -> int:
    payload = (
        f"{replicate_seed}:{pair_id}:{history_anchor}:"
        f"turn{turn_number}:shared"
    )
    digest = hashlib.sha256(payload.encode()).digest()
    return int.from_bytes(digest[:4], "big") & 0x7FFFFFFF


def pair_shared_trim(
    tokenizer: Any,
    first: list[dict[str, str]],
    second: list[dict[str, str]],
    max_input_tokens: int,
    chat_template_kwargs: dict[str, Any] | None,
) -> tuple[
    tuple[list[dict[str, str]], list[int]],
    tuple[list[dict[str, str]], list[int]],
    int,
]:
    start = 0
    while True:
        first_used, first_tokens, first_drop = trim_to_limit(
            tokenizer,
            first[start:],
            max_input_tokens,
            system_prompt=None,
            chat_template_kwargs=chat_template_kwargs,
        )
        second_used, second_tokens, second_drop = trim_to_limit(
            tokenizer,
            second[start:],
            max_input_tokens,
            system_prompt=None,
            chat_template_kwargs=chat_template_kwargs,
        )
        extra_drop = max(first_drop, second_drop)
        if extra_drop == 0:
            break
        start += extra_drop
    if first_used[:-1] != second_used[:-1]:
        raise ValueError("Local contrast retained different prior histories")
    if len(first_tokens) > max_input_tokens or len(second_tokens) > max_input_tokens:
        raise ValueError("Local contrast exceeds token budget")
    return (
        (first_used, first_tokens),
        (second_used, second_tokens),
        start,
    )


def load_cases(path: Path) -> dict[str, dict[str, dict[str, Any]]]:
    if sha256_file(path) != EXPECTED_CASES_SHA256:
        raise ValueError("Cases file does not match frozen public source")
    cases = json.loads(path.read_text(encoding="utf-8"))["cases"]
    grouped: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    for case in cases:
        condition = case["condition"].lower()
        grouped[base_pair_id(case["id"])][condition] = case
    if len(cases) != 16 or len(grouped) != 8 or any(
        set(conditions) != {"explicit", "implicit"}
        for conditions in grouped.values()
    ):
        raise ValueError("Expected eight complete explicit/implicit pairs")
    return grouped


def load_source_rows(
    path: Path,
    reported_model: str,
) -> dict[tuple[int, str, int], dict[str, Any]]:
    rows = [
        row
        for row in read_jsonl(path)
        if row.get("model") == reported_model and successful(row)
    ]
    if len(rows) != 16 * 12 * len(EXPECTED_SEEDS):
        raise ValueError("Source trajectories are incomplete")
    keyed = {
        (
            int(row["replicate_seed"]),
            row["case_id"],
            int(row["turn_number"]),
        ): row
        for row in rows
    }
    if len(keyed) != len(rows):
        raise ValueError("Source trajectories contain duplicate turns")
    return keyed


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cases", type=Path, required=True)
    parser.add_argument("--source-generations", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--reported-model", required=True)
    parser.add_argument("--checkpoint-source", required=True)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--max-input-tokens", type=int, default=11776)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--chunk-size", type=int, default=128)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.90)
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument("--cpu-offload-gb", type=float, default=0.0)
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--disable-thinking", action="store_true")
    parser.add_argument("--enforce-eager", action="store_true")
    args = parser.parse_args()

    cases = load_cases(args.cases)
    sources = load_source_rows(args.source_generations, args.reported_model)
    slug = model_slug(args.reported_model)
    prior = canonicalize(
        args.output,
        lambda row: str(row["generation_id"]),
        successful,
    )
    completed = {
        row["generation_id"] for row in prior if successful(row)
    }
    expected_rows = 8 * 9 * 2 * 2 * len(EXPECTED_SEEDS)
    manifest_path = args.output.with_suffix(
        args.output.suffix + ".manifest.json"
    )
    if len(completed) == expected_rows and manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if (
            manifest.get("model") == args.reported_model
            and manifest.get("successful_rows") == expected_rows
            and manifest.get("expected_rows") == expected_rows
            and manifest.get("output_sha256") == sha256_file(args.output)
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

    pending: list[dict[str, Any]] = []
    shared_drop_counts = []
    for seed in EXPECTED_SEEDS:
        for pair_id, pair_cases in cases.items():
            for turn_number in range(4, 13):
                for history_anchor in ("explicit", "implicit"):
                    anchor_case = pair_cases[history_anchor]
                    anchor = sources[
                        (seed, anchor_case["id"], turn_number)
                    ]
                    prior_messages = anchor["messages"][:-1]
                    variants = {
                        condition: [
                            *prior_messages,
                            {
                                "role": "user",
                                "content": case["prompts"][turn_number - 1],
                            },
                        ]
                        for condition, case in pair_cases.items()
                    }
                    first, second, shared_drop = pair_shared_trim(
                        tokenizer,
                        variants["explicit"],
                        variants["implicit"],
                        args.max_input_tokens,
                        chat_template_kwargs,
                    )
                    shared_drop_counts.append(shared_drop)
                    contrast_id = (
                        f"{pair_id}:anchor-{history_anchor}:"
                        f"turn{turn_number:02d}"
                    )
                    request_seed = shared_request_seed(
                        seed,
                        pair_id,
                        history_anchor,
                        turn_number,
                    )
                    for condition, (used_messages, token_ids) in (
                        ("explicit", first),
                        ("implicit", second),
                    ):
                        generation_id = (
                            f"{slug}:psychosis_local_t1:seed{seed}:"
                            f"{contrast_id}:{condition}"
                        )
                        if generation_id in completed:
                            continue
                        pending.append(
                            {
                                "generation_id": generation_id,
                                "pair_id": pair_id,
                                "local_contrast_id": contrast_id,
                                "cluster_id": f"psychosis_bench:{pair_id}",
                                "condition": condition,
                                "history_anchor": history_anchor,
                                "turn_number": turn_number,
                                "replicate_seed": seed,
                                "request_seed": request_seed,
                                "case_id": pair_cases[condition]["id"],
                                "theme": pair_cases[condition]["theme"],
                                "harm_type": pair_cases[condition]["harm_type"],
                                "source": (
                                    "psychosis_bench_synthetic_local_contrast"
                                ),
                                "source_trajectory_generation_id": anchor[
                                    "generation_id"
                                ],
                                "source_trajectory_sha256": sha256_json(anchor),
                                "messages": variants[condition],
                                "messages_used": used_messages,
                                "pair_shared_dropped_message_count": shared_drop,
                                "token_ids": token_ids,
                            }
                        )

    pending.sort(key=lambda item: len(item["token_ids"]))
    generated = 0
    empty_responses = 0
    for start in tqdm(
        range(0, len(pending), args.chunk_size),
        desc=f"Synthetic local contrasts {slug}",
    ):
        chunk = pending[start : start + args.chunk_size]
        outputs = llm.generate(
            [
                {"prompt_token_ids": item["token_ids"]}
                for item in chunk
            ],
            sampling_params=[
                SamplingParams(
                    temperature=args.temperature,
                    top_p=args.top_p,
                    max_tokens=args.max_new_tokens,
                    seed=item["request_seed"],
                )
                for item in chunk
            ],
            use_tqdm=True,
        )
        for item, output in zip(chunk, outputs, strict=True):
            text = output.outputs[0].text.strip()
            result = {
                key: value
                for key, value in item.items()
                if key != "token_ids"
            }
            result.update(
                {
                    "model": args.reported_model,
                    "model_checkpoint": args.model,
                    "model_checkpoint_source": args.checkpoint_source,
                    "model_revision": args.revision,
                    "model_dtype": args.dtype,
                    "model_slug": slug,
                    "system_prompt": None,
                    "decoding": (
                        "stochastic_api_default_sensitivity_matched_local"
                    ),
                    "temperature": args.temperature,
                    "top_p": args.top_p,
                    "thinking_disabled": args.disable_thinking,
                    "generation_backend": "vllm",
                    "max_input_tokens": args.max_input_tokens,
                    "max_new_tokens": args.max_new_tokens,
                    "tensor_parallel_size": args.tensor_parallel_size,
                    "cpu_offload_gb": args.cpu_offload_gb,
                    "input_token_count": len(item["token_ids"]),
                    "response": text,
                    "finish_reason": output.outputs[0].finish_reason,
                }
            )
            if not text:
                # Immediate stop is an observed model behavior. Resampling it
                # would condition the safety evaluation on non-empty outputs.
                result["empty_response"] = True
                empty_responses += 1
            generated += 1
            append_jsonl(args.output, result)

    final = canonicalize(
        args.output,
        lambda row: str(row["generation_id"]),
        successful,
    )
    model_rows = [
        row
        for row in final
        if row.get("model") == args.reported_model and successful(row)
    ]
    manifest = {
        "paper": "arXiv:2509.10970 matched-local synthetic bridge",
        "paper_repository_commit": PAPER_REPOSITORY_COMMIT,
        "cases": str(args.cases),
        "cases_sha256": sha256_file(args.cases),
        "source_generations": str(args.source_generations),
        "source_generations_sha256": sha256_file(
            args.source_generations
        ),
        "model": args.reported_model,
        "model_checkpoint": args.model,
        "model_checkpoint_source": args.checkpoint_source,
        "model_revision": args.revision,
        "model_dtype": args.dtype,
        "backend": "vllm",
        "decoding": "stochastic_api_default_sensitivity_matched_local",
        "temperature": args.temperature,
        "top_p": args.top_p,
        "seeds": list(EXPECTED_SEEDS),
        "system_prompt": None,
        "max_input_tokens": args.max_input_tokens,
        "max_new_tokens": args.max_new_tokens,
        "tensor_parallel_size": args.tensor_parallel_size,
        "thinking_disabled": args.disable_thinking,
        "history_anchors": ["explicit", "implicit"],
        "pair_shared_history": True,
        "pair_shared_sampling_seed": True,
        "pairs_with_history_truncated": sum(
            count > 0 for count in shared_drop_counts
        ),
        "max_dropped_messages": max(shared_drop_counts),
        "new_generated": generated,
        "new_failures": 0,
        "new_empty_responses": empty_responses,
        "successful_rows": len(model_rows),
        "expected_rows": expected_rows,
        "output_sha256": sha256_file(args.output),
        "cuda": torch.version.cuda,
        "torch": torch.__version__,
    }
    write_manifest(manifest_path, manifest)
    print(json.dumps(manifest, indent=2))
    if len(model_rows) != expected_rows:
        raise SystemExit("Synthetic local-contrast generation gate failed")


if __name__ == "__main__":
    main()
