from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
import random
from pathlib import Path
from typing import Any

from config import ENDORSEMENT_THRESHOLD
from io_utils import append_jsonl, read_jsonl
from llm_delusions_annotations.annotation_prompts import (
    ANNOTATION_SYSTEM_PROMPT,
    build_prompt,
    load_annotations,
)
from openai import AsyncOpenAI
from tqdm import tqdm

ANNOTATION_ID = "bot-endorses-delusion"
JUDGE_MODEL = "gpt-5.4-mini-2026-03-17"
SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "required": ["score", "rationale", "quotes"],
    "properties": {
        "score": {"type": "integer", "minimum": 0, "maximum": 10},
        "rationale": {"type": "string"},
        "quotes": {"type": "array", "items": {"type": "string"}},
    },
}


def load_env(path: Path | None) -> None:
    if path is None or not path.exists():
        return
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if line and not line.startswith("#") and "=" in line:
            key, value = line.split("=", 1)
            os.environ.setdefault(key.strip(), value.strip().strip("'\""))


async def run(args: argparse.Namespace) -> None:
    load_env(args.env_file)
    if not os.environ.get("OPENAI_API_KEY"):
        raise RuntimeError("OPENAI_API_KEY is not configured")
    annotation = next(
        item for item in load_annotations() if item["id"] == ANNOTATION_ID
    )
    prompt_hash = hashlib.sha256(
        (ANNOTATION_SYSTEM_PROMPT + json.dumps(annotation, sort_keys=True)).encode()
    ).hexdigest()
    inputs = [
        row
        for row in read_jsonl(args.input)
        if row.get("response") and not row.get("error")
    ]
    prior = read_jsonl(args.output)
    completed = {
        row["generation_sha256"]
        for row in prior
        if row.get("judge_prompt_sha256") == prompt_hash
        and isinstance(row.get("annotation_score"), int)
        and not row.get("judge_error")
    }
    pending = [row for row in inputs if row["generation_sha256"] not in completed]
    client = AsyncOpenAI(timeout=args.timeout)
    semaphore = asyncio.Semaphore(args.concurrency)
    write_lock = asyncio.Lock()

    async def judge(row: dict[str, Any]) -> dict[str, Any]:
        prompt = build_prompt(
            annotation,
            row["response"],
            role="assistant",
            context_messages=[{"role": "user", "content": row["target_text"]}],
        )
        last_error = ""
        for attempt in range(1, args.attempts + 1):
            try:
                async with semaphore:
                    response = await client.responses.create(
                        model=JUDGE_MODEL,
                        input=[
                            {"role": "system", "content": ANNOTATION_SYSTEM_PROMPT},
                            {"role": "user", "content": prompt},
                        ],
                        reasoning={"effort": "none"},
                        temperature=0,
                        max_output_tokens=350,
                        text={
                            "format": {
                                "type": "json_schema",
                                "name": "endorsement_label",
                                "strict": True,
                                "schema": SCHEMA,
                            }
                        },
                        store=False,
                    )
                label = json.loads(response.output_text)
                return {
                    **row,
                    "annotation_id": ANNOTATION_ID,
                    "annotation_score": int(label["score"]),
                    "endorsement": int(label["score"] >= ENDORSEMENT_THRESHOLD),
                    "annotation_rationale": label["rationale"],
                    "annotation_quotes": label["quotes"],
                    "judge_model": JUDGE_MODEL,
                    "judge_reasoning_effort": "none",
                    "judge_prompt_sha256": prompt_hash,
                    "judge_input_tokens": int(response.usage.input_tokens),
                    "judge_output_tokens": int(response.usage.output_tokens),
                    "judge_attempt": attempt,
                }
            except Exception as error:  # noqa: BLE001 - retry API failures
                last_error = repr(error)
                if attempt < args.attempts:
                    await asyncio.sleep(min(2**attempt + random.random(), 30))
        return {**row, "judge_error": last_error, "judge_prompt_sha256": prompt_hash}

    async def judge_and_save(row: dict[str, Any]) -> None:
        result = await judge(row)
        async with write_lock:
            append_jsonl(args.output, result)

    tasks = [asyncio.create_task(judge_and_save(row)) for row in pending]
    for task in tqdm(asyncio.as_completed(tasks), total=len(tasks), desc="Judging"):
        await task

    final = read_jsonl(args.output)
    valid = [
        row
        for row in final
        if row.get("judge_prompt_sha256") == prompt_hash
        and isinstance(row.get("annotation_score"), int)
        and not row.get("judge_error")
    ]
    successful = {row["generation_sha256"] for row in valid}
    expected = {row["generation_sha256"] for row in inputs}
    manifest = {
        "annotation_id": ANNOTATION_ID,
        "threshold": ENDORSEMENT_THRESHOLD,
        "judge_model": JUDGE_MODEL,
        "reasoning_effort": "none",
        "judge_prompt_sha256": prompt_hash,
        "candidate_responses": len(inputs),
        "successful_rows": len(successful & expected),
        "missing_rows": len(expected - successful),
        "total_input_tokens": sum(
            int(row.get("judge_input_tokens", 0)) for row in valid
        ),
        "total_output_tokens": sum(
            int(row.get("judge_output_tokens", 0)) for row in valid
        ),
        "judge_context": "flagged target user message and candidate assistant response only",
    }
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))
    if expected - successful:
        raise RuntimeError(f"Missing {len(expected - successful)} judgments")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--env-file", type=Path)
    parser.add_argument("--concurrency", type=int, default=24)
    parser.add_argument("--attempts", type=int, default=5)
    parser.add_argument("--timeout", type=float, default=240)
    asyncio.run(run(parser.parse_args()))


if __name__ == "__main__":
    main()
