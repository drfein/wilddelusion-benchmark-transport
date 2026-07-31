#!/usr/bin/env python3
"""Prepare target-only original/control messages for the pinned classifier."""

from __future__ import annotations

import argparse
from pathlib import Path

from io_utils import read_jsonl, write_jsonl


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--controls", type=Path, default=Path("artifacts/target_controls.jsonl")
    )
    parser.add_argument(
        "--turns-output",
        type=Path,
        default=Path("artifacts/target_audit_turns.jsonl"),
    )
    parser.add_argument(
        "--prompts-output",
        type=Path,
        default=Path("artifacts/target_audit_contexts.jsonl"),
    )
    args = parser.parse_args()
    turns = []
    prompts = []
    for row in read_jsonl(args.controls):
        if row.get("control_error"):
            continue
        for condition, text in (
            ("original", row["original_text"]),
            ("control", row["control_text"]),
        ):
            generation_id = f"{row['pair_id']}:target_only:{condition}"
            turns.append(
                {
                    "generation_id": generation_id,
                    "pair_id": row["pair_id"],
                    "condition": condition,
                    "user_text": text,
                }
            )
            prompts.append({"generation_id": generation_id, "messages": []})
    write_jsonl(args.turns_output, turns)
    write_jsonl(args.prompts_output, prompts)
    print(f"Wrote {len(turns)} target-only audit rows")


if __name__ == "__main__":
    main()
