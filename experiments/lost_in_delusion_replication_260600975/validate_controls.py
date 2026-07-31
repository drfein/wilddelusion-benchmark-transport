#!/usr/bin/env python3
"""Independently validate paired grounded controls."""

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
    STRICT_VALIDATION_SCHEMA,
    VALIDATION_SCHEMA,
    VALIDATION_SYSTEM_PROMPT,
    validation_prompt,
)


def successful(row: dict[str, Any]) -> bool:
    return not row.get("validation_error") and "usable" in row


async def run(args: argparse.Namespace) -> None:
    originals = {row["pair_id"]: row for row in read_jsonl(args.originals)}
    controls = {
        row["pair_id"]: row
        for row in read_jsonl(args.controls)
        if row.get("control_messages") and not row.get("control_error")
    }
    pair_ids = sorted(originals.keys() & controls.keys())
    lock = args.output.with_suffix(args.output.suffix + ".lock")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    lock_handle = lock.open("w")
    try:
        fcntl.flock(lock_handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as error:
        raise RuntimeError(f"Another process owns {lock}") from error

    prior = canonicalize(args.output, lambda row: row["pair_id"], successful)
    completed = {row["pair_id"] for row in prior if successful(row)}
    pending = [pair_id for pair_id in pair_ids if pair_id not in completed]
    if args.max_rows is not None:
        pending = pending[: args.max_rows]

    client = AsyncOpenAI(timeout=args.timeout)
    semaphore = asyncio.Semaphore(args.concurrency)
    write_lock = asyncio.Lock()

    async def validate(pair_id: str) -> dict[str, Any]:
        original = originals[pair_id]["history_messages"]
        control = controls[pair_id]["control_messages"]
        request = {
            "model": args.model,
            "input": [
                {"role": "system", "content": VALIDATION_SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": validation_prompt(
                        original,
                        control,
                        strict_alignment=args.strict_alignment,
                    ),
                },
            ],
            "reasoning": {"effort": "none"},
            "text": {
                "format": {
                    "type": "json_schema",
                    "name": "control_validation",
                    "strict": True,
                    "schema": (
                        STRICT_VALIDATION_SCHEMA
                        if args.strict_alignment
                        else VALIDATION_SCHEMA
                    ),
                }
            },
            "max_output_tokens": 900,
        }
        last_error = ""
        for attempt in range(1, args.attempts + 1):
            try:
                async with semaphore:
                    response = await client.responses.create(**request)
                parsed = json.loads(response.output_text)
                return {
                    "pair_id": pair_id,
                    **parsed,
                    "validation_model": args.model,
                    "validation_protocol": (
                        "strict_turn_alignment_v1"
                        if args.strict_alignment
                        else "matched_counterfactual_v1"
                    ),
                    "validation_reasoning_effort": "none",
                    "validation_input_tokens": response.usage.input_tokens,
                    "validation_output_tokens": response.usage.output_tokens,
                    "validation_attempt": attempt,
                }
            except Exception as error:
                last_error = repr(error)
                if attempt < args.attempts:
                    await asyncio.sleep(min(2**attempt, 12))
        return {
            "pair_id": pair_id,
            "validation_model": args.model,
            "validation_error": last_error,
        }

    async def validate_and_save(pair_id: str) -> dict[str, Any]:
        result = await validate(pair_id)
        async with write_lock:
            append_jsonl(args.output, result)
        return result

    tasks = [asyncio.create_task(validate_and_save(pair_id)) for pair_id in pending]
    totals = {"input_tokens": 0, "output_tokens": 0, "errors": 0}
    for task in tqdm(
        asyncio.as_completed(tasks), total=len(tasks), desc="Validating controls"
    ):
        result = await task
        totals["input_tokens"] += int(result.get("validation_input_tokens") or 0)
        totals["output_tokens"] += int(result.get("validation_output_tokens") or 0)
        totals["errors"] += int(not successful(result))

    final = canonicalize(args.output, lambda row: row["pair_id"], successful)
    valid = [row for row in final if successful(row)]
    write_manifest(
        args.output.with_suffix(args.output.suffix + ".manifest.json"),
        {
            "model": args.model,
            "reasoning_effort": "none",
            "strict_alignment": args.strict_alignment,
            "candidate_pairs": len(pair_ids),
            "new_rows": len(pending),
            "new_errors": totals["errors"],
            "new_input_tokens": totals["input_tokens"],
            "new_output_tokens": totals["output_tokens"],
            "validated_rows": len(valid),
            "usable_rows": sum(bool(row["usable"]) for row in valid),
        },
    )
    print(
        json.dumps(
            {
                "validated_rows": len(valid),
                "usable_rows": sum(bool(row["usable"]) for row in valid),
                **totals,
            },
            indent=2,
        )
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--originals", type=Path, default=Path("artifacts/cohort_original.jsonl")
    )
    parser.add_argument(
        "--controls", type=Path, default=Path("artifacts/generated_controls.jsonl")
    )
    parser.add_argument(
        "--output", type=Path, default=Path("artifacts/control_validation.jsonl")
    )
    parser.add_argument("--model", default="gpt-5.4-mini")
    parser.add_argument("--concurrency", type=int, default=12)
    parser.add_argument("--attempts", type=int, default=4)
    parser.add_argument("--timeout", type=float, default=240)
    parser.add_argument("--max-rows", type=int)
    parser.add_argument("--strict-alignment", action="store_true")
    asyncio.run(run(parser.parse_args()))


if __name__ == "__main__":
    main()
