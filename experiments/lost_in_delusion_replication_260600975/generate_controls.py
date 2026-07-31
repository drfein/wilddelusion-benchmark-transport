#!/usr/bin/env python3
"""Generate resumable paired grounded controls with an OpenAI mini model."""

from __future__ import annotations

import argparse
import asyncio
import fcntl
import json
from pathlib import Path
from typing import Any

from openai import AsyncOpenAI
from tqdm import tqdm

from io_utils import append_jsonl, canonicalize, read_jsonl, write_manifest
from wilddelusion_prompts import (
    CONTROL_SYSTEM_PROMPT,
    control_prompt,
    control_schema,
)


def valid_result(row: dict[str, Any]) -> bool:
    return not row.get("control_error") and bool(row.get("control_messages"))


async def run(args: argparse.Namespace) -> None:
    rows = read_jsonl(args.input)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    lock = args.output.with_suffix(args.output.suffix + ".lock").open("w")
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as error:
        raise RuntimeError(f"Another process owns {lock.name}") from error

    prior = canonicalize(args.output, lambda row: row["pair_id"], valid_result)
    completed = {row["pair_id"] for row in prior if valid_result(row)}
    pending = [row for row in rows if row["pair_id"] not in completed]
    if args.max_rows is not None:
        pending = pending[: args.max_rows]

    client = AsyncOpenAI(timeout=args.timeout)
    semaphore = asyncio.Semaphore(args.concurrency)
    write_lock = asyncio.Lock()

    async def generate(row: dict[str, Any]) -> dict[str, Any]:
        messages = row["history_messages"]
        character_count = sum(len(message["content"]) for message in messages)
        max_output_tokens = min(
            args.max_output_tokens,
            max(args.min_output_tokens, character_count // 3 + 800),
        )
        request = {
            "model": args.model,
            "input": [
                {"role": "system", "content": CONTROL_SYSTEM_PROMPT},
                {"role": "user", "content": control_prompt(messages)},
            ],
            "reasoning": {"effort": "none"},
            "text": {
                "format": {
                    "type": "json_schema",
                    "name": "matched_grounded_control",
                    "strict": True,
                    "schema": control_schema(len(messages)),
                }
            },
            "max_output_tokens": max_output_tokens,
        }
        last_error = ""
        for attempt in range(1, args.attempts + 1):
            try:
                async with semaphore:
                    response = await client.responses.create(**request)
                parsed = json.loads(response.output_text)
                controls = parsed["messages"]
                expected_roles = [message["role"] for message in messages]
                observed_roles = [message["role"] for message in controls]
                if observed_roles != expected_roles:
                    raise ValueError(
                        f"role mismatch: expected={expected_roles}, got={observed_roles}"
                    )
                if any(not message["content"].strip() for message in controls):
                    raise ValueError("empty control message")
                return {
                    "pair_id": row["pair_id"],
                    "control_messages": controls,
                    "edit_summary": parsed["edit_summary"],
                    "control_model": args.model,
                    "control_reasoning_effort": "none",
                    "control_input_tokens": response.usage.input_tokens,
                    "control_output_tokens": response.usage.output_tokens,
                    "control_attempt": attempt,
                }
            except Exception as error:
                last_error = repr(error)
                if attempt < args.attempts:
                    await asyncio.sleep(min(2**attempt, 12))
        return {
            "pair_id": row["pair_id"],
            "control_model": args.model,
            "control_error": last_error,
        }

    async def generate_and_save(row: dict[str, Any]) -> dict[str, Any]:
        result = await generate(row)
        async with write_lock:
            append_jsonl(args.output, result)
        return result

    tasks = [asyncio.create_task(generate_and_save(row)) for row in pending]
    totals = {"input_tokens": 0, "output_tokens": 0, "errors": 0}
    for task in tqdm(
        asyncio.as_completed(tasks), total=len(tasks), desc="Generating controls"
    ):
        result = await task
        totals["input_tokens"] += int(result.get("control_input_tokens") or 0)
        totals["output_tokens"] += int(result.get("control_output_tokens") or 0)
        totals["errors"] += int(not valid_result(result))

    final = canonicalize(args.output, lambda row: row["pair_id"], valid_result)
    write_manifest(
        args.output.with_suffix(args.output.suffix + ".manifest.json"),
        {
            "model": args.model,
            "reasoning_effort": "none",
            "requested_rows": len(rows),
            "new_rows": len(pending),
            "new_errors": totals["errors"],
            "new_input_tokens": totals["input_tokens"],
            "new_output_tokens": totals["output_tokens"],
            "successful_rows": sum(valid_result(row) for row in final),
        },
    )
    print(json.dumps(read_jsonl(args.output.with_suffix(args.output.suffix + ".manifest.json")) if False else {
        "successful_rows": sum(valid_result(row) for row in final),
        **totals,
    }, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input", type=Path, default=Path("artifacts/cohort_original.jsonl")
    )
    parser.add_argument(
        "--output", type=Path, default=Path("artifacts/generated_controls.jsonl")
    )
    parser.add_argument("--model", default="gpt-5.4-mini")
    parser.add_argument("--concurrency", type=int, default=8)
    parser.add_argument("--attempts", type=int, default=4)
    parser.add_argument("--timeout", type=float, default=300)
    parser.add_argument("--max-output-tokens", type=int, default=14000)
    parser.add_argument("--min-output-tokens", type=int, default=1200)
    parser.add_argument("--max-rows", type=int)
    asyncio.run(run(parser.parse_args()))


if __name__ == "__main__":
    main()
