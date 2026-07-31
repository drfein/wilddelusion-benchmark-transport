#!/usr/bin/env python3
"""Run resumable greedy generation with vLLM continuous batching."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch
from tqdm import tqdm
from vllm import LLM, SamplingParams

from io_utils import append_jsonl, canonicalize, read_jsonl, write_manifest
from wilddelusion_prompts import BASELINE_SYSTEM_PROMPT
from run_local_models import model_slug, successful, trim_to_limit


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument(
        "--reported-model",
        help="Canonical base-model ID to record when loading a quantized checkpoint.",
    )
    parser.add_argument("--max-input-tokens", type=int, default=12288)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--chunk-size", type=int, default=128)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.90)
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--no-system-prompt", action="store_true")
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

    reported_model = args.reported_model or args.model
    slug = model_slug(reported_model)
    generation_key = lambda row: str(row["generation_id"])
    prior = canonicalize(args.output, generation_key, successful)
    completed = {
        str(row["generation_id"]) for row in prior if successful(row)
    }
    inputs = read_jsonl(args.input)
    pending = [
        row
        for row in inputs
        if f"{slug}:{row['input_id']}" not in completed
    ]
    system_prompt = None if args.no_system_prompt else BASELINE_SYSTEM_PROMPT

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
    prepared: list[dict[str, Any]] = []
    for row in pending:
        used_messages, token_ids, dropped = trim_to_limit(
            tokenizer,
            row["messages"],
            args.max_input_tokens,
            system_prompt=system_prompt,
            chat_template_kwargs=chat_template_kwargs,
        )
        prepared.append(
            {
                "row": row,
                "used_messages": used_messages,
                "token_ids": token_ids,
                "dropped_messages": dropped,
            }
        )

    # Length-local chunks reduce scheduler churn while preserving continuous
    # batching within each chunk and durable writes between chunks.
    prepared.sort(key=lambda item: len(item["token_ids"]))
    sampling = SamplingParams(
        temperature=0.0,
        max_tokens=args.max_new_tokens,
    )
    generated = 0
    failures = 0
    chunks = [
        prepared[start : start + args.chunk_size]
        for start in range(0, len(prepared), args.chunk_size)
    ]
    for chunk in tqdm(chunks, desc=f"vLLM {slug} chunks"):
        prompts = [
            {"prompt_token_ids": item["token_ids"]} for item in chunk
        ]
        try:
            outputs = llm.generate(
                prompts,
                sampling_params=sampling,
                use_tqdm=True,
            )
            for item, output in zip(chunk, outputs, strict=True):
                source = item["row"]
                text = output.outputs[0].text.strip()
                append_jsonl(
                    args.output,
                    {
                        **source,
                        "generation_id": f"{slug}:{source['input_id']}",
                        "model": reported_model,
                        "model_checkpoint": args.model,
                        "model_slug": slug,
                        "system_prompt": system_prompt,
                        "decoding": "greedy",
                        "generation_backend": "vllm",
                        "max_new_tokens": args.max_new_tokens,
                        "input_token_count": len(item["token_ids"]),
                        "dropped_message_count": item["dropped_messages"],
                        "messages_used": item["used_messages"],
                        "response": text,
                    },
                )
                generated += 1
        except Exception as error:
            for item in chunk:
                source = item["row"]
                append_jsonl(
                    args.output,
                    {
                        **source,
                        "generation_id": f"{slug}:{source['input_id']}",
                        "model": reported_model,
                        "model_checkpoint": args.model,
                        "generation_backend": "vllm",
                        "generation_error": repr(error),
                    },
                )
                failures += 1

    final = canonicalize(args.output, generation_key, successful)
    model_rows = [row for row in final if row.get("model") == reported_model]
    successful_rows = sum(successful(row) for row in model_rows)
    write_manifest(
        args.output.with_suffix(args.output.suffix + ".manifest.json"),
        {
            "model": reported_model,
            "model_checkpoint": args.model,
            "model_slug": slug,
            "backend": "vllm",
            "decoding": "greedy",
            "system_prompt": system_prompt,
            "max_input_tokens": args.max_input_tokens,
            "max_new_tokens": args.max_new_tokens,
            "chunk_size": args.chunk_size,
            "tensor_parallel_size": args.tensor_parallel_size,
            "thinking_disabled": args.disable_thinking,
            "enforce_eager": args.enforce_eager,
            "new_generated": generated,
            "new_failures": failures,
            "successful_rows": successful_rows,
            "expected_rows": len(inputs),
            "cuda": torch.version.cuda,
            "torch": torch.__version__,
        },
    )
    print(
        json.dumps(
            {
                "model": reported_model,
                "model_checkpoint": args.model,
                "generated": generated,
                "failures": failures,
                "successful_rows": successful_rows,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
