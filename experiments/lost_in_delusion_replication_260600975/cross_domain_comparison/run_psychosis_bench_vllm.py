#!/usr/bin/env python3
"""Run the original Psychosis-Bench conversations with a local vLLM model."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import torch
from tqdm import tqdm
from vllm import LLM, SamplingParams

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from io_utils import append_jsonl, canonicalize, read_jsonl, write_manifest  # noqa: E402
from run_local_models import model_slug, trim_to_limit  # noqa: E402


def successful(row: dict[str, Any]) -> bool:
    return bool(row.get("response")) and not row.get("generation_error")


def pair_id(case_id: str) -> str:
    for suffix in ("_explicit", "_implicit"):
        if case_id.endswith(suffix):
            return case_id[: -len(suffix)]
    raise ValueError(f"Case ID does not encode condition: {case_id}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cases", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument(
        "--reported-model",
        help="Canonical base-model ID to record when loading a quantized checkpoint.",
    )
    parser.add_argument("--max-input-tokens", type=int, default=11776)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.90)
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument(
        "--disable-thinking",
        action="store_true",
        help="Pass enable_thinking=False to models with switchable reasoning.",
    )
    parser.add_argument(
        "--enforce-eager",
        action="store_true",
        help="Disable CUDA graphs for checkpoints that require eager execution.",
    )
    args = parser.parse_args()

    payload = json.loads(args.cases.read_text())
    cases = payload["cases"]
    if len(cases) != 16 or any(len(case["prompts"]) != 12 for case in cases):
        raise ValueError("Expected the original 16 cases with 12 turns each")

    reported_model = args.reported_model or args.model
    slug = model_slug(reported_model)
    key = lambda row: str(row["generation_id"])
    prior = canonicalize(args.output, key, successful)
    completed = {
        (row["case_id"], int(row["turn_number"])): row
        for row in prior
        if row.get("model") == reported_model and successful(row)
    }

    llm = LLM(
        model=args.model,
        dtype="bfloat16",
        max_model_len=args.max_input_tokens + args.max_new_tokens,
        gpu_memory_utilization=args.gpu_memory_utilization,
        trust_remote_code=args.trust_remote_code,
        enable_prefix_caching=True,
        tensor_parallel_size=args.tensor_parallel_size,
        enforce_eager=args.enforce_eager,
    )
    tokenizer = llm.get_tokenizer()
    chat_template_kwargs = (
        {"enable_thinking": False} if args.disable_thinking else None
    )
    sampling = SamplingParams(
        temperature=0.0,
        max_tokens=args.max_new_tokens,
    )
    histories: dict[str, list[dict[str, str]]] = {
        case["id"]: [] for case in cases
    }
    generated = 0
    reused = 0
    dropped_messages = 0

    for turn_index in tqdm(range(12), desc=f"Psychosis-Bench {slug} turns"):
        pending: list[dict[str, Any]] = []
        for case in cases:
            case_id = case["id"]
            user_message = {
                "role": "user",
                "content": case["prompts"][turn_index],
            }
            histories[case_id].append(user_message)
            existing = completed.get((case_id, turn_index + 1))
            if existing:
                histories[case_id].append(
                    {"role": "assistant", "content": existing["response"]}
                )
                reused += 1
                continue

            used_messages, token_ids, dropped = trim_to_limit(
                tokenizer,
                histories[case_id],
                args.max_input_tokens,
                system_prompt=None,
                chat_template_kwargs=chat_template_kwargs,
            )
            pending.append(
                {
                    "case": case,
                    "messages": list(histories[case_id]),
                    "used_messages": used_messages,
                    "token_ids": token_ids,
                    "dropped": dropped,
                }
            )

        if not pending:
            continue
        outputs = llm.generate(
            [{"prompt_token_ids": item["token_ids"]} for item in pending],
            sampling_params=sampling,
            use_tqdm=True,
        )
        for item, output in zip(pending, outputs, strict=True):
            case = item["case"]
            turn_number = turn_index + 1
            text = output.outputs[0].text.strip()
            generation_id = (
                f"{slug}:psychosis_bench:{case['id']}:turn{turn_number:02d}"
            )
            condition = case["condition"].lower()
            row = {
                "pair_id": pair_id(case["id"]),
                "cluster_id": f"psychosis_bench:{pair_id(case['id'])}",
                "input_id": f"{case['id']}:turn{turn_number:02d}",
                "case_id": case["id"],
                "case_name": case["name"],
                "turn_number": turn_number,
                "condition": condition,
                "messages": item["messages"],
                "theme": case["theme"],
                "source": "psychosis_bench_synthetic",
                "harm_type": case["harm_type"],
                "original_harm": "clear" if turn_number >= 10 else "possible",
                "original_distress": None,
                "high_stakes_action": turn_number >= 10,
                "harm_pair_valid": turn_number >= 10,
                "transport_paper": "arXiv:2509.10970",
                "generation_id": generation_id,
                "model": reported_model,
                "model_checkpoint": args.model,
                "model_slug": slug,
                "system_prompt": None,
                "decoding": "greedy",
                "generation_backend": "vllm",
                "max_new_tokens": args.max_new_tokens,
                "input_token_count": len(item["token_ids"]),
                "dropped_message_count": item["dropped"],
                "messages_used": item["used_messages"],
                "response": text,
            }
            append_jsonl(args.output, row)
            histories[case["id"]].append(
                {"role": "assistant", "content": text}
            )
            completed[(case["id"], turn_number)] = row
            generated += 1
            dropped_messages += int(item["dropped"])

    final = canonicalize(args.output, key, successful)
    model_rows = [
        row
        for row in final
        if row.get("model") == reported_model and successful(row)
    ]
    write_manifest(
        args.output.with_suffix(args.output.suffix + ".manifest.json"),
        {
            "paper": "arXiv:2509.10970",
            "source": str(args.cases),
            "model": reported_model,
            "model_checkpoint": args.model,
            "model_slug": slug,
            "backend": "vllm",
            "decoding": "greedy",
            "system_prompt": None,
            "max_input_tokens": args.max_input_tokens,
            "max_new_tokens": args.max_new_tokens,
            "tensor_parallel_size": args.tensor_parallel_size,
            "thinking_disabled": args.disable_thinking,
            "enforce_eager": args.enforce_eager,
            "new_generated": generated,
            "reused_rows": reused,
            "successful_rows": len(model_rows),
            "expected_rows": 16 * 12,
            "total_dropped_messages": dropped_messages,
            "cuda": torch.version.cuda,
            "torch": torch.__version__,
        },
    )
    print(
        json.dumps(
            {
                "model": reported_model,
                "model_checkpoint": args.model,
                "new_generated": generated,
                "reused_rows": reused,
                "successful_rows": len(model_rows),
                "dropped_messages": dropped_messages,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
