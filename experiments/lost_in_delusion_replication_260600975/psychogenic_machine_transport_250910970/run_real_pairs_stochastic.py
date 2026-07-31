#!/usr/bin/env python3
"""Generate matched real implicit/explicit responses under one locked protocol."""

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

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from io_utils import append_jsonl, canonicalize, read_jsonl, write_manifest  # noqa: E402
from run_local_models import model_slug, trim_to_limit  # noqa: E402


def successful(row: dict[str, Any]) -> bool:
    return isinstance(row.get("response"), str) and not row.get(
        "generation_error"
    )


def request_seed(
    replicate_seed: int,
    pair_id: str,
) -> int:
    digest = hashlib.sha256(
        f"{replicate_seed}:{pair_id}:shared".encode()
    ).digest()
    return int.from_bytes(digest[:4], "big") & 0x7FFFFFFF


def validate_inputs(rows: list[dict[str, Any]]) -> None:
    if len(rows) != 654:
        raise ValueError(f"expected 654 real pair rows, found {len(rows)}")
    grouped: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    for row in rows:
        grouped[row["pair_id"]][row["condition"]] = row
    if len(grouped) != 327:
        raise ValueError(f"expected 327 pairs, found {len(grouped)}")
    for pair_id, conditions in grouped.items():
        if set(conditions) != {"explicit", "implicit"}:
            raise ValueError(f"{pair_id}: incomplete condition pair")
        explicit = conditions["explicit"]["messages"]
        implicit = conditions["implicit"]["messages"]
        if explicit[:-1] != implicit[:-1]:
            raise ValueError(f"{pair_id}: conditions do not share history")
        if [item["role"] for item in explicit] != [
            item["role"] for item in implicit
        ]:
            raise ValueError(f"{pair_id}: conditions changed role sequence")
        if not explicit or explicit[-1]["role"] != "user":
            raise ValueError(f"{pair_id}: pair does not end in a user message")


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
    """Use the same retained-history boundary for both pair conditions."""
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
        raise ValueError("pair-shared truncation produced different histories")
    if len(first_tokens) > max_input_tokens or len(second_tokens) > max_input_tokens:
        raise ValueError("trimmed model input exceeds token budget")
    return (
        (first_used, first_tokens),
        (second_used, second_tokens),
        start,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--reported-model", required=True)
    parser.add_argument("--checkpoint-source", required=True)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--seeds", default="1101,2202,3303")
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

    seeds = [int(value) for value in args.seeds.split(",") if value.strip()]
    if len(seeds) != 3 or len(set(seeds)) != 3:
        raise ValueError("exactly three unique trajectory seeds are required")
    rows = read_jsonl(args.input)
    validate_inputs(rows)
    slug = model_slug(args.reported_model)
    prior = canonicalize(
        args.output,
        lambda row: str(row["generation_id"]),
        successful,
    )
    completed = {
        row["generation_id"] for row in prior if successful(row)
    }
    expected_rows = len(rows) * len(seeds)
    manifest_path = args.output.with_suffix(
        args.output.suffix + ".manifest.json"
    )
    if len(completed) == expected_rows and manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if (
            manifest.get("model") == args.reported_model
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

    grouped: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    for row in rows:
        grouped[row["pair_id"]][row["condition"]] = row
    encoded: dict[str, dict[str, Any]] = {}
    shared_drop_counts: list[int] = []
    for pair_id, conditions in tqdm(
        grouped.items(), desc=f"Tokenizing paired real inputs for {slug}"
    ):
        explicit = conditions["explicit"]
        implicit = conditions["implicit"]
        first, second, shared_drop = pair_shared_trim(
            tokenizer,
            explicit["messages"],
            implicit["messages"],
            args.max_input_tokens,
            chat_template_kwargs,
        )
        shared_drop_counts.append(shared_drop)
        for row, (used_messages, token_ids) in (
            (explicit, first),
            (implicit, second),
        ):
            encoded[row["input_id"]] = {
                "row": row,
                "messages_used": used_messages,
                "token_ids": token_ids,
                "pair_shared_dropped_message_count": shared_drop,
            }

    pending: list[dict[str, Any]] = []
    for seed in seeds:
        for item in encoded.values():
            row = item["row"]
            generation_id = (
                f"{slug}:psychosis_real_t1:seed{seed}:{row['input_id']}"
            )
            if generation_id in completed:
                continue
            pending.append(
                {
                    **item,
                    "replicate_seed": seed,
                    "request_seed": request_seed(seed, row["pair_id"]),
                    "generation_id": generation_id,
                }
            )
    pending.sort(key=lambda item: len(item["token_ids"]))

    generated = 0
    empty_responses = 0
    for start in tqdm(
        range(0, len(pending), args.chunk_size),
        desc=f"Stochastic real pairs {slug}",
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
            source = item["row"]
            text = output.outputs[0].text.strip()
            result = {
                **source,
                "generation_id": item["generation_id"],
                "model": args.reported_model,
                "model_checkpoint": args.model,
                "model_checkpoint_source": args.checkpoint_source,
                "model_revision": args.revision,
                "model_dtype": args.dtype,
                "model_slug": slug,
                "system_prompt": None,
                "decoding": "stochastic_api_default_sensitivity",
                "temperature": args.temperature,
                "top_p": args.top_p,
                "thinking_disabled": args.disable_thinking,
                "replicate_seed": item["replicate_seed"],
                "request_seed": item["request_seed"],
                "generation_backend": "vllm",
                "max_input_tokens": args.max_input_tokens,
                "max_new_tokens": args.max_new_tokens,
                "tensor_parallel_size": args.tensor_parallel_size,
                "cpu_offload_gb": args.cpu_offload_gb,
                "input_token_count": len(item["token_ids"]),
                "dropped_message_count": item[
                    "pair_shared_dropped_message_count"
                ],
                "pair_shared_dropped_message_count": item[
                    "pair_shared_dropped_message_count"
                ],
                "messages_used": item["messages_used"],
                "response": text,
                "finish_reason": output.outputs[0].finish_reason,
            }
            if not text:
                # Preserve immediate-stop outputs instead of resampling until
                # the model emits text, which would bias the paired estimate.
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
        "paper": "arXiv:2509.10970 real-data transport",
        "source": str(args.input),
        "source_sha256": hashlib.sha256(args.input.read_bytes()).hexdigest(),
        "model": args.reported_model,
        "model_checkpoint": args.model,
        "model_checkpoint_source": args.checkpoint_source,
        "model_revision": args.revision,
        "model_dtype": args.dtype,
        "backend": "vllm",
        "decoding": "stochastic_api_default_sensitivity",
        "temperature": args.temperature,
        "top_p": args.top_p,
        "seeds": seeds,
        "system_prompt": None,
        "max_input_tokens": args.max_input_tokens,
        "max_new_tokens": args.max_new_tokens,
        "chunk_size": args.chunk_size,
        "tensor_parallel_size": args.tensor_parallel_size,
        "thinking_disabled": args.disable_thinking,
        "pair_shared_history_truncation": True,
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
        "cuda": torch.version.cuda,
        "torch": torch.__version__,
    }
    write_manifest(manifest_path, manifest)
    print(json.dumps(manifest, indent=2))
    if len(model_rows) != expected_rows:
        raise SystemExit("Stochastic real-pair generation gate failed")


if __name__ == "__main__":
    main()
