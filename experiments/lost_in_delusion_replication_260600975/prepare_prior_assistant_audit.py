#!/usr/bin/env python3
"""Prepare actual and grounded prior assistant turns for DCS auditing."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from io_utils import read_jsonl, write_jsonl, write_manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--cohort", type=Path, default=Path("artifacts/cohort_original.jsonl")
    )
    parser.add_argument(
        "--grounded",
        type=Path,
        default=Path("artifacts/grounded_assistant_histories.jsonl"),
    )
    parser.add_argument(
        "--target-inputs",
        type=Path,
        default=Path("artifacts/target_model_inputs_final.jsonl"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/prior_assistant_audit_inputs.jsonl"),
    )
    args = parser.parse_args()

    cohort = {row["pair_id"]: row for row in read_jsonl(args.cohort)}
    grounded = {
        row["pair_id"]: row
        for row in read_jsonl(args.grounded)
        if row.get("grounded_messages")
    }
    pair_ids = sorted(
        {
            row["pair_id"]
            for row in read_jsonl(args.target_inputs)
            if row["condition"] == "delusion"
        }
    )
    rows: list[dict[str, Any]] = []
    for pair_id in pair_ids:
        actual_messages = cohort[pair_id]["history_messages"]
        grounded_messages = grounded[pair_id]["grounded_messages"]
        assistant_indices = [
            index
            for index, message in enumerate(actual_messages[:-1])
            if message["role"] == "assistant"
        ]
        if not assistant_indices:
            continue
        index = assistant_indices[-1]
        for condition, messages in (
            ("actual_prior", actual_messages),
            ("grounded_prior", grounded_messages),
        ):
            rows.append(
                {
                    "generation_id": f"prior_audit:{condition}:{pair_id}",
                    "pair_id": pair_id,
                    "condition": condition,
                    "theme": cohort[pair_id].get("theme"),
                    "model": "observed_source_assistant",
                    "messages": messages[:index],
                    "messages_used": messages[:index],
                    "response": messages[index]["content"],
                    "escalated_observed": cohort[pair_id].get(
                        "escalated_observed"
                    ),
                }
            )
    write_jsonl(args.output, rows)
    write_manifest(
        args.output.with_suffix(".manifest.json"),
        {
            "rows": len(rows),
            "pairs": len({row["pair_id"] for row in rows}),
            "conditions": ["actual_prior", "grounded_prior"],
            "audited_turn": "last assistant response before flagged user message",
        },
    )
    print(f"Wrote {len(rows)} prior-assistant audit rows")


if __name__ == "__main__":
    main()
