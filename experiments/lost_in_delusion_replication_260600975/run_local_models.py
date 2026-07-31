#!/usr/bin/env python3
"""Run greedy batched chat generation with a local Hugging Face model."""

from __future__ import annotations

import argparse
import json
import os
import re
from pathlib import Path
from typing import Any

import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from io_utils import append_jsonl, canonicalize, read_jsonl, write_manifest
from wilddelusion_prompts import BASELINE_SYSTEM_PROMPT


def model_slug(model: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", model.lower()).strip("_")


def successful(row: dict[str, Any]) -> bool:
    return not row.get("generation_error") and bool(row.get("response"))


def trim_to_limit(
    tokenizer: Any,
    messages: list[dict[str, str]],
    max_input_tokens: int,
    system_prompt: str | None = BASELINE_SYSTEM_PROMPT,
    chat_template_kwargs: dict[str, Any] | None = None,
) -> tuple[list[dict[str, str]], list[int], int]:
    working = [dict(message) for message in messages]
    template_kwargs = chat_template_kwargs or {}

    def encode(current: list[dict[str, str]]) -> list[int]:
        chat = (
            [{"role": "system", "content": system_prompt}, *current]
            if system_prompt
            else current
        )
        encoded = tokenizer.apply_chat_template(
            chat,
            tokenize=True,
            add_generation_prompt=True,
            **template_kwargs,
        )
        # Transformers 5 returns a BatchEncoding for some fast tokenizers,
        # whereas earlier versions returned the input-ID list directly.
        if hasattr(encoded, "get") and encoded.get("input_ids") is not None:
            encoded = encoded["input_ids"]
        if hasattr(encoded, "tolist"):
            encoded = encoded.tolist()
        if encoded and isinstance(encoded[0], list):
            encoded = encoded[0]
        return list(encoded)

    token_ids = encode(working)
    dropped = 0
    if len(token_ids) > max_input_tokens and len(working) > 1:
        # Find the earliest user-turn suffix that fits without repeatedly
        # tokenizing every intermediate suffix in very long conversations.
        user_starts = [
            index
            for index, message in enumerate(working)
            if message["role"] == "user"
        ]
        low = 0
        high = len(user_starts) - 1
        best_start = user_starts[-1]
        best_tokens = encode(working[best_start:])
        while low <= high:
            middle = (low + high) // 2
            start = user_starts[middle]
            candidate_tokens = encode(working[start:])
            if len(candidate_tokens) <= max_input_tokens:
                best_start = start
                best_tokens = candidate_tokens
                high = middle - 1
            else:
                low = middle + 1
        dropped = best_start
        working = working[best_start:]
        token_ids = best_tokens
    if len(token_ids) > max_input_tokens:
        target_tokens = tokenizer.encode(
            working[-1]["content"], add_special_tokens=False
        )
        excess = len(token_ids) - max_input_tokens
        keep = max(64, len(target_tokens) - excess - 16)
        working[-1]["content"] = tokenizer.decode(
            target_tokens[-keep:], skip_special_tokens=True
        )
        token_ids = encode(working)
    if len(token_ids) > max_input_tokens:
        token_ids = token_ids[-max_input_tokens:]
    return working, token_ids, dropped


def packed_batches(
    items: list[dict[str, Any]], batch_size: int, max_batch_tokens: int
) -> list[list[dict[str, Any]]]:
    ordered = sorted(items, key=lambda item: len(item["token_ids"]))
    batches: list[list[dict[str, Any]]] = []
    current: list[dict[str, Any]] = []
    current_max = 0
    for item in ordered:
        proposed_max = max(current_max, len(item["token_ids"]))
        if current and (
            len(current) >= batch_size
            or proposed_max * (len(current) + 1) > max_batch_tokens
        ):
            batches.append(current)
            current = []
            current_max = 0
        current.append(item)
        current_max = max(current_max, len(item["token_ids"]))
    if current:
        batches.append(current)
    return batches


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-batch-tokens", type=int, default=32768)
    parser.add_argument("--max-input-tokens", type=int, default=12288)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument(
        "--no-system-prompt",
        action="store_true",
        help="Match benchmarks that send only the conversation messages.",
    )
    args = parser.parse_args()

    torch.set_float32_matmul_precision("high")
    slug = model_slug(args.model)
    generation_key = lambda row: str(row["generation_id"])
    prior = canonicalize(args.output, generation_key, successful)
    completed = {row["generation_id"] for row in prior if successful(row)}
    inputs = read_jsonl(args.input)
    pending = [
        row
        for row in inputs
        if f"{slug}:{row['input_id']}" not in completed
    ]

    tokenizer = AutoTokenizer.from_pretrained(
        args.model, trust_remote_code=args.trust_remote_code
    )
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.bfloat16,
        trust_remote_code=args.trust_remote_code,
        low_cpu_mem_usage=True,
    ).to(args.device)
    model.eval()

    prepared: list[dict[str, Any]] = []
    system_prompt = None if args.no_system_prompt else BASELINE_SYSTEM_PROMPT
    for row in pending:
        used_messages, token_ids, dropped = trim_to_limit(
            tokenizer,
            row["messages"],
            args.max_input_tokens,
            system_prompt=system_prompt,
        )
        prepared.append(
            {
                "row": row,
                "used_messages": used_messages,
                "token_ids": token_ids,
                "dropped_messages": dropped,
            }
        )
    batches = packed_batches(
        prepared, args.batch_size, args.max_batch_tokens
    )

    generated = 0
    failures = 0
    with torch.inference_mode():
        for batch in tqdm(batches, desc=f"Generating {slug}"):
            try:
                encoded = tokenizer.pad(
                    {"input_ids": [item["token_ids"] for item in batch]},
                    padding=True,
                    return_tensors="pt",
                )
                input_ids = encoded["input_ids"].to(args.device)
                attention_mask = encoded["attention_mask"].to(args.device)
                output = model.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    do_sample=False,
                    max_new_tokens=args.max_new_tokens,
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                    use_cache=True,
                )
                continuation = output[:, input_ids.shape[1] :]
                texts = tokenizer.batch_decode(
                    continuation, skip_special_tokens=True
                )
                for item, text in zip(batch, texts, strict=True):
                    source = item["row"]
                    result = {
                        **source,
                        "generation_id": f"{slug}:{source['input_id']}",
                        "model": args.model,
                        "model_slug": slug,
                        "system_prompt": system_prompt,
                        "decoding": "greedy",
                        "max_new_tokens": args.max_new_tokens,
                        "input_token_count": len(item["token_ids"]),
                        "dropped_message_count": item["dropped_messages"],
                        "messages_used": item["used_messages"],
                        "response": text.strip(),
                    }
                    append_jsonl(args.output, result)
                    generated += 1
            except torch.cuda.OutOfMemoryError as error:
                torch.cuda.empty_cache()
                for item in batch:
                    source = item["row"]
                    append_jsonl(
                        args.output,
                        {
                            **source,
                            "generation_id": f"{slug}:{source['input_id']}",
                            "model": args.model,
                            "generation_error": f"CUDA OOM in batch: {error}",
                        },
                    )
                    failures += 1

    final = canonicalize(args.output, generation_key, successful)
    model_rows = [row for row in final if row.get("model") == args.model]
    write_manifest(
        args.output.with_suffix(args.output.suffix + ".manifest.json"),
        {
            "model": args.model,
            "model_slug": slug,
            "device": args.device,
            "decoding": "greedy",
            "system_prompt": system_prompt,
            "max_input_tokens": args.max_input_tokens,
            "max_new_tokens": args.max_new_tokens,
            "new_generated": generated,
            "new_failures": failures,
            "successful_rows": sum(successful(row) for row in model_rows),
            "expected_rows": len(inputs),
            "cuda": torch.version.cuda,
            "torch": torch.__version__,
        },
    )
    print(
        json.dumps(
            {
                "model": args.model,
                "generated": generated,
                "failures": failures,
                "successful_rows": sum(successful(row) for row in model_rows),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
