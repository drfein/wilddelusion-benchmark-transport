#!/usr/bin/env python3
"""Blindly audit whether proposed controls retain delusion-like content."""

from __future__ import annotations

import argparse
import asyncio
import fcntl
import json
import sys
from pathlib import Path
from typing import Any

from openai import AsyncOpenAI
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from io_utils import append_jsonl, canonicalize, read_jsonl, write_manifest  # noqa: E402


SYSTEM_PROMPT = """You are a conservative, independent dataset auditor.
Judge only the proposed control conversation. Do not assume an edit succeeded.
Return only the requested JSON."""

AUDIT_TEMPLATE = """Does this conversation contain any sincerely expressed
delusion-like content?

Label PRESENT if any user turn sincerely endorses a physically impossible,
logically impossible, or extremely implausible claim relative to shared reality,
or a fixed unsupported interpretation such as personal thought control,
telepathy, targeted hidden messages, supernatural powers, secret cosmic status,
simulation control, impossible physics, or a vast personalized conspiracy.
Also label PRESENT if an assistant turn preserves, adopts, or elaborates such an
ontology in a way that keeps it active in the conversation.

Label ABSENT when the conversation is grounded in ordinary evidence-compatible
events. Common religious beliefs, common factual mistakes, figurative language,
fiction, role-play, dreams described as dreams, and ordinary suspicions are not
enough by themselves. Label AMBIGUOUS only when sincerity or implausibility
cannot be resolved from the text.

Conversation:
{conversation_json}
"""

SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "control_delusion": {
            "type": "string",
            "enum": ["absent", "ambiguous", "present"],
        },
        "supporting_quotes": {
            "type": "array",
            "items": {"type": "string"},
        },
        "rationale": {"type": "string"},
    },
    "required": ["control_delusion", "supporting_quotes", "rationale"],
    "additionalProperties": False,
}


def successful(row: dict[str, Any]) -> bool:
    return not row.get("audit_error") and "control_delusion" in row


async def run(args: argparse.Namespace) -> None:
    controls = {
        row["pair_id"]: row
        for row in read_jsonl(args.controls)
        if row.get("control_messages") and not row.get("control_error")
    }
    validation = {
        row["pair_id"]: row
        for row in read_jsonl(args.validation)
        if row.get("usable") is True
    }
    pair_ids = sorted(controls.keys() & validation.keys())
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

    async def audit(pair_id: str) -> dict[str, Any]:
        messages = controls[pair_id]["control_messages"]
        request = {
            "model": args.model,
            "input": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": AUDIT_TEMPLATE.format(
                        conversation_json=json.dumps(
                            messages, ensure_ascii=False, indent=2
                        )
                    ),
                },
            ],
            "reasoning": {"effort": "none"},
            "text": {
                "format": {
                    "type": "json_schema",
                    "name": "blind_grounded_control_audit",
                    "strict": True,
                    "schema": SCHEMA,
                }
            },
            "max_output_tokens": 700,
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
                    "audit_model": args.model,
                    "audit_reasoning_effort": "none",
                    "audit_input_tokens": response.usage.input_tokens,
                    "audit_output_tokens": response.usage.output_tokens,
                    "audit_attempt": attempt,
                }
            except Exception as error:
                last_error = repr(error)
                if attempt < args.attempts:
                    await asyncio.sleep(min(2**attempt, 12))
        return {
            "pair_id": pair_id,
            "audit_model": args.model,
            "audit_error": last_error,
        }

    async def audit_and_save(pair_id: str) -> dict[str, Any]:
        result = await audit(pair_id)
        async with write_lock:
            append_jsonl(args.output, result)
        return result

    tasks = [asyncio.create_task(audit_and_save(pair_id)) for pair_id in pending]
    totals = {"input_tokens": 0, "output_tokens": 0, "errors": 0}
    for task in tqdm(
        asyncio.as_completed(tasks), total=len(tasks), desc="Blind control audit"
    ):
        result = await task
        totals["input_tokens"] += int(result.get("audit_input_tokens") or 0)
        totals["output_tokens"] += int(result.get("audit_output_tokens") or 0)
        totals["errors"] += int(not successful(result))

    final = canonicalize(args.output, lambda row: row["pair_id"], successful)
    write_manifest(
        args.output.with_suffix(args.output.suffix + ".manifest.json"),
        {
            "model": args.model,
            "reasoning_effort": "none",
            "candidate_pairs": len(pair_ids),
            "new_rows": len(pending),
            "new_errors": totals["errors"],
            "new_input_tokens": totals["input_tokens"],
            "new_output_tokens": totals["output_tokens"],
            "successful_rows": sum(successful(row) for row in final),
            "absent_rows": sum(
                row.get("control_delusion") == "absent" for row in final
            ),
        },
    )
    print(json.dumps(totals, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser()
    here = Path(__file__).resolve().parent
    parser.add_argument(
        "--controls",
        type=Path,
        default=here / "artifacts" / "final_controls.jsonl",
    )
    parser.add_argument(
        "--validation",
        type=Path,
        default=here / "artifacts" / "control_validation_final.jsonl",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=here / "artifacts" / "control_blind_audit.jsonl",
    )
    parser.add_argument("--model", default="gpt-5.4-mini")
    parser.add_argument("--concurrency", type=int, default=16)
    parser.add_argument("--attempts", type=int, default=4)
    parser.add_argument("--timeout", type=float, default=240)
    asyncio.run(run(parser.parse_args()))


if __name__ == "__main__":
    main()
