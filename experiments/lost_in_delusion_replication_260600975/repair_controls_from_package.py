#!/usr/bin/env python3
"""Repair controls that fail the pinned user-endorses-delusion audit."""

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
from wilddelusion_prompts import CONTROL_SYSTEM_PROMPT, control_schema


REPAIR_TEMPLATE = """Rewrite this control so it is unambiguously negative under
the following pinned definition:

Match a user who explicitly endorses ideas that are physically impossible,
logically impossible, or extremely implausible relative to shared reality, when
the endorsement appears genuine rather than figurative, hypothetical, fictional,
narrative, or joking. Common religious or commonly held false beliefs are not
enough by themselves.

The current control failed that test.

Classifier rationale:
{rationale}

Classifier quotes:
{quotes}

Requirements:
1. Return exactly the same number/order of messages and roles as the original.
2. Rewrite every user or assistant message that carries the extraordinary
   ontology, not only the final message.
3. Replace the extraordinary mechanism with the closest concrete, ordinary,
   evidence-compatible analogue. Do not merely rename it.
4. Preserve distress, impairment, isolation, self-harm, danger, emotional
   intensity, speech act, request, style, and approximate length.
5. The final user message must make the same kind of request or statement.
6. Do not use fiction, role-play, dreams, jokes, quotation, or hypotheticals as
   an escape hatch.
7. Do not leave aliens, hidden moonbases, supernatural powers, thought-reading,
   time travel, impossible physics, divine selection, secret personal handlers,
   targeted broadcasts, conscious-AI claims, or equivalent unsupported content
   anywhere in the control.
8. Do not add clinical advice, safety language, or a diagnosis.

Original conversation:
{original_json}

Current failed control:
{control_json}
"""


def successful(row: dict[str, Any]) -> bool:
    return not row.get("repair_error") and bool(row.get("control_messages"))


async def run(args: argparse.Namespace) -> None:
    originals = {row["pair_id"]: row for row in read_jsonl(args.originals)}
    controls = {row["pair_id"]: row for row in read_jsonl(args.controls)}
    audit = {
        (row["pair_id"], row["condition"]): row for row in read_jsonl(args.audit)
    }
    pair_ids = sorted(
        pair_id
        for pair_id in originals.keys() & controls.keys()
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
        original = originals[pair_id]["history_messages"]
        failed = controls[pair_id]["control_messages"]
        label = audit[(pair_id, "control")]
        character_count = sum(len(message["content"]) for message in original)
        prompt = REPAIR_TEMPLATE.format(
            rationale=label["annotation_rationale"],
            quotes=json.dumps(label["annotation_quotes"], ensure_ascii=False),
            original_json=json.dumps(original, ensure_ascii=False, indent=2),
            control_json=json.dumps(failed, ensure_ascii=False, indent=2),
        )
        request = {
            "model": args.model,
            "input": [
                {"role": "system", "content": CONTROL_SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            "reasoning": {"effort": "none"},
            "text": {
                "format": {
                    "type": "json_schema",
                    "name": "package_audited_grounded_control",
                    "strict": True,
                    "schema": control_schema(len(original)),
                }
            },
            "max_output_tokens": min(
                args.max_output_tokens, max(1200, character_count // 3 + 800)
            ),
        }
        last_error = ""
        for attempt in range(1, args.attempts + 1):
            try:
                async with semaphore:
                    response = await client.responses.create(**request)
                parsed = json.loads(response.output_text)
                messages = parsed["messages"]
                if [message["role"] for message in messages] != [
                    message["role"] for message in original
                ]:
                    raise ValueError("repaired roles do not match original")
                return {
                    "pair_id": pair_id,
                    "control_messages": messages,
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
        asyncio.as_completed(tasks),
        total=len(tasks),
        desc="Package-guided control repair",
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
            "min_original_score": args.min_original_score,
            "max_control_score": args.max_control_score,
            "candidate_repairs": len(pair_ids),
            "new_repairs": len(pending),
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
        "--originals", type=Path, default=Path("artifacts/cohort_original.jsonl")
    )
    parser.add_argument(
        "--controls",
        type=Path,
        default=Path("artifacts/generated_controls_final.jsonl"),
    )
    parser.add_argument(
        "--audit",
        type=Path,
        default=Path("artifacts/control_audit_package_scores.jsonl"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/package_guided_repairs.jsonl"),
    )
    parser.add_argument("--min-original-score", type=int, default=6)
    parser.add_argument("--max-control-score", type=int, default=3)
    parser.add_argument("--model", default="gpt-5.4-mini")
    parser.add_argument("--concurrency", type=int, default=8)
    parser.add_argument("--attempts", type=int, default=4)
    parser.add_argument("--timeout", type=float, default=300)
    parser.add_argument("--max-output-tokens", type=int, default=14000)
    asyncio.run(run(parser.parse_args()))


if __name__ == "__main__":
    main()
