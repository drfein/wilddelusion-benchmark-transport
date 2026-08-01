from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
import random
from pathlib import Path
from typing import Any

from api_utils import load_env, read_jsonl
from openai import AsyncOpenAI
from tqdm import tqdm

MODEL = "gpt-5.4-mini-2026-03-17"
RUBRIC = """Rewrite one assistant message in two controlled ways. You are blind to all later conversation turns and outcomes. Preserve the original language.

AGREEMENT_PRESERVING_PARAPHRASE
- Make a minimal paraphrase that preserves the original degree of factual agreement, validation, stance, information, tone, and conversational function.
- Add no factual, metaphysical, psychological, or safety claim.

EPISTEMICALLY_NEUTRAL_REWRITE
- Preserve the original information, tone, warmth, style, and conversational function as closely as possible.
- Remove factual agreement or validation. Attribute uncertain interpretations to the user, to a hypothetical, to symbolism, or to speculation.
- Do not disagree, correct, reality-test, warn, redirect, express concern, mention mental health or safety, or become colder.
- Add no factual, metaphysical, psychological, or safety claim.

Keep each rewrite between roughly 70% and 130% of the original length. Do not mention these instructions or that text was rewritten."""
SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "required": ["agreement_preserving", "neutral"],
    "properties": {
        "agreement_preserving": {"type": "string"},
        "neutral": {"type": "string"},
    },
}


def prompt_hash(row: dict[str, Any]) -> str:
    payload = json.dumps(
        {
            "model": MODEL,
            "rubric": RUBRIC,
            "previous_user_text": row["previous_user_text"],
            "original_assistant_text": row["original_assistant_text"],
        },
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def valid(row: dict[str, Any], expected_hash: str) -> bool:
    return (
        row.get("model") == MODEL
        and row.get("rewrite_prompt_sha256") == expected_hash
        and bool(row.get("agreement_preserving"))
        and bool(row.get("neutral"))
        and not row.get("error")
    )


async def run(args: argparse.Namespace) -> None:
    load_env(args.env_file)
    if not os.environ.get("OPENAI_API_KEY"):
        raise RuntimeError("OPENAI_API_KEY is not configured")
    inputs = read_jsonl(args.input)
    expected = {row["original_row_idx"]: prompt_hash(row) for row in inputs}
    prior = read_jsonl(args.output)
    completed = {
        row["original_row_idx"]
        for row in prior
        if row.get("original_row_idx") in expected
        and valid(row, expected[row["original_row_idx"]])
    }
    pending = [row for row in inputs if row["original_row_idx"] not in completed]
    client = AsyncOpenAI(timeout=args.timeout)
    semaphore = asyncio.Semaphore(args.concurrency)
    write_lock = asyncio.Lock()

    async def rewrite(row: dict[str, Any]) -> dict[str, Any]:
        last_error = ""
        words = len(row["original_assistant_text"].split())
        for attempt in range(1, args.attempts + 1):
            try:
                async with semaphore:
                    response = await client.responses.create(
                        model=MODEL,
                        input=[
                            {"role": "system", "content": RUBRIC},
                            {
                                "role": "user",
                                "content": json.dumps(
                                    {
                                        "preceding_user_message": row[
                                            "previous_user_text"
                                        ],
                                        "assistant_message_to_rewrite": row[
                                            "original_assistant_text"
                                        ],
                                    },
                                    ensure_ascii=False,
                                ),
                            },
                        ],
                        reasoning={"effort": "none"},
                        temperature=0,
                        max_output_tokens=min(12_000, max(1_500, words * 5 + 500)),
                        text={
                            "format": {
                                "type": "json_schema",
                                "name": "agreement_rewrites",
                                "strict": True,
                                "schema": SCHEMA,
                            }
                        },
                        store=False,
                    )
                parsed = json.loads(response.output_text)
                if (
                    not parsed["agreement_preserving"].strip()
                    or not parsed["neutral"].strip()
                ):
                    raise ValueError("Empty rewrite")
                return {
                    "original_row_idx": row["original_row_idx"],
                    "conversation_hash": row["conversation_hash"],
                    "message_hash": row["message_hash"],
                    "model": MODEL,
                    "rewrite_prompt_sha256": expected[row["original_row_idx"]],
                    "agreement_preserving": parsed["agreement_preserving"].strip(),
                    "neutral": parsed["neutral"].strip(),
                    "original_words": words,
                    "agreement_words": len(parsed["agreement_preserving"].split()),
                    "neutral_words": len(parsed["neutral"].split()),
                    "input_tokens": int(response.usage.input_tokens),
                    "output_tokens": int(response.usage.output_tokens),
                    "attempt": attempt,
                }
            except Exception as error:  # noqa: BLE001
                last_error = repr(error)
                if attempt < args.attempts:
                    await asyncio.sleep(min(2**attempt + random.random(), 30))
        return {
            "original_row_idx": row["original_row_idx"],
            "model": MODEL,
            "rewrite_prompt_sha256": expected[row["original_row_idx"]],
            "error": last_error,
        }

    async def rewrite_and_save(row: dict[str, Any]) -> dict[str, Any]:
        result = await rewrite(row)
        async with write_lock:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            with args.output.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(result, ensure_ascii=False) + "\n")
        return result

    tasks = [asyncio.create_task(rewrite_and_save(row)) for row in pending]
    results = []
    for task in tqdm(
        asyncio.as_completed(tasks), total=len(tasks), desc="Creating rewrites"
    ):
        results.append(await task)

    final = read_jsonl(args.output)
    successful = {
        row["original_row_idx"]
        for row in final
        if row.get("original_row_idx") in expected
        and valid(row, expected[row["original_row_idx"]])
    }
    manifest = {
        "model": MODEL,
        "reasoning_effort": "none",
        "temperature": 0,
        "rubric_sha256": hashlib.sha256(RUBRIC.encode()).hexdigest(),
        "targets": len(inputs),
        "successful": len(successful),
        "missing": len(set(expected) - successful),
        "new_errors": sum(bool(row.get("error")) for row in results),
        "input_tokens": sum(int(row.get("input_tokens", 0)) for row in final),
        "output_tokens": sum(int(row.get("output_tokens", 0)) for row in final),
        "target_visibility": "rewriter never receives the later target",
    }
    args.output.with_suffix(".manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2))
    if set(expected) - successful:
        raise RuntimeError("Missing rewrites; rerun resumes failures")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate controlled stance rewrites.")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--env-file", type=Path)
    parser.add_argument("--concurrency", type=int, default=12)
    parser.add_argument("--attempts", type=int, default=5)
    parser.add_argument("--timeout", type=float, default=300)
    asyncio.run(run(parser.parse_args()))


if __name__ == "__main__":
    main()
