#!/usr/bin/env python3
"""Neutralize prior assistant turns while preserving real user history."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from io_utils import read_jsonl, write_jsonl, write_manifest


NEUTRAL_ASSISTANT = "Please continue."


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--cohort", type=Path, default=Path("artifacts/cohort_original.jsonl")
    )
    parser.add_argument(
        "--target-inputs",
        type=Path,
        default=Path("artifacts/target_model_inputs_final.jsonl"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/assistant_neutralized_inputs_final.jsonl"),
    )
    args = parser.parse_args()

    cohort = {row["pair_id"]: row for row in read_jsonl(args.cohort)}
    pair_ids = sorted(
        {
            row["pair_id"]
            for row in read_jsonl(args.target_inputs)
            if row["condition"] == "delusion"
        }
    )
    rows: list[dict[str, Any]] = []
    for pair_id in pair_ids:
        original = cohort[pair_id]
        history = original["history_messages"]
        messages = [
            (
                {"role": "assistant", "content": NEUTRAL_ASSISTANT}
                if message["role"] == "assistant"
                else dict(message)
            )
            for message in history
        ]
        digest = hashlib.sha256(
            (
                pair_id
                + ":assistant_neutralized:"
                + json.dumps(messages, ensure_ascii=False, sort_keys=True)
            ).encode("utf-8")
        ).hexdigest()
        rows.append(
            {
                "pair_id": pair_id,
                "input_id": digest[:24],
                "condition": "assistant_neutralized",
                "messages": messages,
                "target_text": original["target_text"],
                "theme": original.get("theme"),
                "source": original.get("source"),
                "context_scope": "assistant_neutralized",
                "escalated_observed": original.get("escalated_observed"),
                "retained_message_count": original.get("retained_message_count"),
                "neutral_assistant_text": NEUTRAL_ASSISTANT,
            }
        )

    write_jsonl(args.output, rows)
    write_manifest(
        args.output.with_suffix(".manifest.json"),
        {
            "pairs": len(rows),
            "rows": len(rows),
            "context_scope": "assistant_neutralized",
            "neutral_assistant_text": NEUTRAL_ASSISTANT,
            "target_and_prior_user_turns_unchanged": True,
        },
    )
    print(f"Wrote {len(rows)} assistant-neutralized inputs to {args.output}")


if __name__ == "__main__":
    main()
