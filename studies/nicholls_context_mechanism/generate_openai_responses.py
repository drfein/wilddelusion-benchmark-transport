from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
import random
from pathlib import Path
from typing import Any

from openai import AsyncOpenAI
from prepare_complete_context_inputs import MAX_OUTPUT_TOKENS, MODEL, SYSTEM_PROMPT
from tqdm import tqdm


def load_env(path: Path | None) -> None:
    if not path or not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            key, value = line.split("=", 1)
            os.environ.setdefault(key.strip(), value.strip().strip("'\""))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line
    ]


def valid(row: dict[str, Any]) -> bool:
    return (
        bool(row.get("response")) and row.get("model") == MODEL and not row.get("error")
    )


async def run(args: argparse.Namespace) -> None:
    load_env(args.env_file)
    if not os.environ.get("OPENAI_API_KEY"):
        raise RuntimeError("OPENAI_API_KEY is not configured")
    inputs = read_jsonl(args.input)
    prior = read_jsonl(args.output)
    completed = {row["prompt_sha256"] for row in prior if valid(row)}
    pending = [row for row in inputs if row["prompt_sha256"] not in completed]
    if args.max_requests is not None:
        pending = pending[: args.max_requests]

    client = AsyncOpenAI(timeout=args.timeout)
    semaphore = asyncio.Semaphore(args.concurrency)
    write_lock = asyncio.Lock()

    async def generate(row: dict[str, Any]) -> dict[str, Any]:
        last_error = ""
        for attempt in range(1, args.attempts + 1):
            try:
                async with semaphore:
                    response = await client.responses.create(
                        model=MODEL,
                        input=[
                            {"role": "system", "content": SYSTEM_PROMPT},
                            *row["messages"],
                        ],
                        reasoning={"effort": "none"},
                        temperature=0,
                        max_output_tokens=MAX_OUTPUT_TOKENS,
                        store=False,
                    )
                text = response.output_text.strip()
                if not text:
                    raise ValueError("Model returned an empty response")
                return {
                    **{key: value for key, value in row.items() if key != "messages"},
                    "response": text,
                    "response_id": response.id,
                    "actual_input_tokens": int(response.usage.input_tokens),
                    "actual_output_tokens": int(response.usage.output_tokens),
                    "attempt": attempt,
                }
            except Exception as error:  # noqa: BLE001 - retry arbitrary API failures
                last_error = repr(error)
                if attempt < args.attempts:
                    await asyncio.sleep(min(2**attempt + random.random(), 60))
        return {
            **{key: value for key, value in row.items() if key != "messages"},
            "error": last_error,
        }

    async def generate_and_save(row: dict[str, Any]) -> dict[str, Any]:
        result = await generate(row)
        async with write_lock:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            with args.output.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(result, ensure_ascii=False) + "\n")
        return result

    results = []
    tasks = [asyncio.create_task(generate_and_save(row)) for row in pending]
    for task in tqdm(
        asyncio.as_completed(tasks), total=len(tasks), desc="Generating matched arms"
    ):
        results.append(await task)

    final = read_jsonl(args.output)
    successful = {row["prompt_sha256"] for row in final if valid(row)}
    expected = {row["prompt_sha256"] for row in inputs}
    manifest = {
        "model": MODEL,
        "reasoning_effort": "none",
        "temperature": 0,
        "max_output_tokens": MAX_OUTPUT_TOKENS,
        "requested_rows": len(inputs),
        "pending_at_start": len(pending),
        "successful_rows": len(successful & expected),
        "missing_rows": len(expected - successful),
        "new_errors": sum(bool(row.get("error")) for row in results),
        "total_input_tokens": sum(
            int(row.get("actual_input_tokens", 0)) for row in final if valid(row)
        ),
        "total_output_tokens": sum(
            int(row.get("actual_output_tokens", 0)) for row in final if valid(row)
        ),
        "output_sha256": hashlib.sha256(args.output.read_bytes()).hexdigest()
        if args.output.exists()
        else None,
    }
    args.output.with_suffix(args.output.suffix + ".manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2))
    if args.max_requests is None and expected - successful:
        raise RuntimeError(
            f"Missing {len(expected - successful)} generations; rerun resumes them"
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate matched OpenAI response arms."
    )
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--env-file", type=Path)
    parser.add_argument("--concurrency", type=int, default=8)
    parser.add_argument("--attempts", type=int, default=6)
    parser.add_argument("--timeout", type=float, default=600.0)
    parser.add_argument("--max-requests", type=int)
    asyncio.run(run(parser.parse_args()))


if __name__ == "__main__":
    main()
