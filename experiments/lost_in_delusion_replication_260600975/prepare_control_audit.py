#!/usr/bin/env python3
"""Prepare original/control user turns for the pinned delusion classifier."""

from __future__ import annotations

import argparse
from pathlib import Path

from io_utils import read_jsonl, write_jsonl


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
        "--turns-output",
        type=Path,
        default=Path("artifacts/control_audit_turns.jsonl"),
    )
    parser.add_argument(
        "--prompts-output",
        type=Path,
        default=Path("artifacts/control_audit_contexts.jsonl"),
    )
    parser.add_argument(
        "--conditions",
        nargs="+",
        choices=["original", "control"],
        default=["original", "control"],
    )
    args = parser.parse_args()

    originals = {row["pair_id"]: row for row in read_jsonl(args.originals)}
    controls = {row["pair_id"]: row for row in read_jsonl(args.controls)}
    turns = []
    prompts = []
    for pair_id in sorted(originals.keys() & controls.keys()):
        for condition, messages in (
            ("original", originals[pair_id]["history_messages"]),
            ("control", controls[pair_id]["control_messages"]),
        ):
            if condition not in args.conditions:
                continue
            generation_id = f"{pair_id}:{condition}"
            turns.append(
                {
                    "generation_id": generation_id,
                    "pair_id": pair_id,
                    "condition": condition,
                    "user_text": messages[-1]["content"],
                }
            )
            prompts.append(
                {
                    "generation_id": generation_id,
                    "messages": messages[:-1],
                }
            )
    write_jsonl(args.turns_output, turns)
    write_jsonl(args.prompts_output, prompts)
    print(f"Wrote {len(turns)} audit turns and contexts")


if __name__ == "__main__":
    main()
