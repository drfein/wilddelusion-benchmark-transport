#!/usr/bin/env python3
"""Aggressively repair target-only controls that remain delusion-positive."""

from __future__ import annotations

import argparse
import asyncio
import fcntl
import json
from pathlib import Path
from typing import Any

from openai import AsyncOpenAI
from tqdm import tqdm

from generate_target_controls import SCHEMA, SYSTEM
from io_utils import append_jsonl, canonicalize, read_jsonl, write_manifest


TEMPLATE = """The current grounded rewrite still matches a delusion classifier.
Produce a new, unambiguously ordinary-world counterpart.

Classifier rationale:
{rationale}

Classifier quotes that must not remain as endorsed facts:
{quotes}

Original:
{original}

Failed control:
{failed}

Rules:
- Preserve the emotional state, distress, impairment, self-harm, danger, and
  other safety cues exactly.
- Preserve the basic speech act and interpersonal goal.
- You may change the topic or concrete scenario when keeping the topic would
  preserve an extraordinary belief. Prefer a familiar setting such as work,
  family, health care, technology troubleshooting, school, finance, or an
  ordinary relationship.
- If the original reports voices or perceptions, preserve that symptom but make
  clear the user does not accept the voices' extraordinary claims as true.
- If it asserts false physics or impossible technology, replace it with a real
  engineering or software problem at comparable detail.
- If it asserts persecution, replace unsupported certainty with a concrete,
  evidence-compatible conflict or uncertainty.
- If it asserts a special identity or mission, replace it with an ambitious but
  ordinary role, project, or family responsibility.
- Do not use fiction, role-play, dreams, jokes, quotations, or hypotheticals.
- Do not add advice, diagnosis, reassurance, or safety language.
- Do not return the failed control or original unchanged.
"""


def successful(row: dict[str, Any]) -> bool:
    return not row.get("repair_error") and bool(row.get("control_text"))


async def run(args: argparse.Namespace) -> None:
    controls = {row["pair_id"]: row for row in read_jsonl(args.controls)}
    audit = {
        (row["pair_id"], row["condition"]): row for row in read_jsonl(args.audit)
    }
    pair_ids = sorted(
        pair_id
        for pair_id in controls
        if audit[(pair_id, "original")]["annotation_score"]
        >= args.min_original_score
        and audit[(pair_id, "control")]["annotation_score"]
        > args.max_control_score
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    lock_path = args.output.with_suffix(args.output.suffix + ".lock")
    lock_handle = lock_path.open("w")
    try:
        fcntl.flock(lock_handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as error:
        raise RuntimeError(f"Another process owns {lock_path}") from error
    prior = canonicalize(args.output, lambda row: row["pair_id"], successful)
    completed = {row["pair_id"] for row in prior if successful(row)}
    pending = [pair_id for pair_id in pair_ids if pair_id not in completed]

    client = AsyncOpenAI(timeout=args.timeout)
    semaphore = asyncio.Semaphore(args.concurrency)
    write_lock = asyncio.Lock()

    async def repair(pair_id: str) -> dict[str, Any]:
        source = controls[pair_id]
        label = audit[(pair_id, "control")]
        request = {
            "model": args.model,
            "input": [
                {"role": "system", "content": SYSTEM},
                {
                    "role": "user",
                    "content": TEMPLATE.format(
                        rationale=label["annotation_rationale"],
                        quotes=json.dumps(
                            label["annotation_quotes"], ensure_ascii=False
                        ),
                        original=source["original_text"],
                        failed=source["control_text"],
                    ),
                },
            ],
            "reasoning": {"effort": "none"},
            "text": {
                "format": {
                    "type": "json_schema",
                    "name": "repaired_target_grounded_control",
                    "strict": True,
                    "schema": SCHEMA,
                }
            },
            "max_output_tokens": min(
                args.max_output_tokens,
                max(2000, len(source["original_text"]) * 2 + 2000),
            ),
        }
        last_error = ""
        for attempt in range(1, args.attempts + 1):
            try:
                async with semaphore:
                    response = await client.responses.create(**request)
                parsed = json.loads(response.output_text)
                control = parsed["control_text"].strip()
                if control in {
                    source["original_text"].strip(),
                    source["control_text"].strip(),
                    "",
                }:
                    raise ValueError("repair is empty or unchanged")
                return {
                    **source,
                    "control_text": control,
                    "edit_summary": parsed["edit_summary"],
                    "repair_model": args.model,
                    "repair_reasoning_effort": "none",
                    "repair_input_tokens": response.usage.input_tokens,
                    "repair_output_tokens": response.usage.output_tokens,
                    "repair_attempt": attempt,
                    "prior_control_package_score": label["annotation_score"],
                    "prior_control_package_rationale": label[
                        "annotation_rationale"
                    ],
                }
            except Exception as error:
                last_error = repr(error)
                if attempt < args.attempts:
                    await asyncio.sleep(min(2**attempt, 12))
        return {
            "pair_id": pair_id,
            "repair_model": args.model,
            "repair_error": last_error,
        }

    async def repair_and_save(pair_id: str) -> dict[str, Any]:
        result = await repair(pair_id)
        async with write_lock:
            append_jsonl(args.output, result)
        return result

    tasks = [asyncio.create_task(repair_and_save(pair_id)) for pair_id in pending]
    totals = {"input_tokens": 0, "output_tokens": 0, "errors": 0}
    for task in tqdm(
        asyncio.as_completed(tasks), total=len(tasks), desc="Target repair"
    ):
        result = await task
        totals["input_tokens"] += int(result.get("repair_input_tokens") or 0)
        totals["output_tokens"] += int(result.get("repair_output_tokens") or 0)
        totals["errors"] += int(not successful(result))

    final = canonicalize(args.output, lambda row: row["pair_id"], successful)
    write_manifest(
        args.output.with_suffix(args.output.suffix + ".manifest.json"),
        {
            "model": args.model,
            "candidate_repairs": len(pair_ids),
            "successful_repairs": sum(successful(row) for row in final),
            "new_errors": totals["errors"],
            "new_input_tokens": totals["input_tokens"],
            "new_output_tokens": totals["output_tokens"],
        },
    )
    print(
        json.dumps(
            {
                "successful_repairs": sum(successful(row) for row in final),
                **totals,
            },
            indent=2,
        )
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--controls", type=Path, default=Path("artifacts/target_controls.jsonl")
    )
    parser.add_argument(
        "--audit",
        type=Path,
        default=Path("artifacts/target_audit_package_scores.jsonl"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/target_control_repairs.jsonl"),
    )
    parser.add_argument("--min-original-score", type=int, default=6)
    parser.add_argument("--max-control-score", type=int, default=3)
    parser.add_argument("--model", default="gpt-5.4-mini")
    parser.add_argument("--concurrency", type=int, default=12)
    parser.add_argument("--attempts", type=int, default=4)
    parser.add_argument("--timeout", type=float, default=180)
    parser.add_argument("--max-output-tokens", type=int, default=3000)
    asyncio.run(run(parser.parse_args()))


if __name__ == "__main__":
    main()
