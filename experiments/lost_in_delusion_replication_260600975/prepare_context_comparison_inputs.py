#!/usr/bin/env python3
"""Prepare full-history originals for the strict target-only cohort."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from io_utils import read_jsonl, write_jsonl, write_manifest


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
        "--prior-inputs",
        type=Path,
        default=Path("artifacts/model_inputs.jsonl"),
        help="Reuse prior input IDs when the exact full-history arm was already run.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/context_model_inputs_final.jsonl"),
    )
    args = parser.parse_args()

    cohort = {row["pair_id"]: row for row in read_jsonl(args.cohort)}
    prior = {
        row["pair_id"]: row
        for row in read_jsonl(args.prior_inputs)
        if row["condition"] == "delusion"
    }
    pair_ids = sorted(
        {
            row["pair_id"]
            for row in read_jsonl(args.target_inputs)
            if row["condition"] == "delusion"
        }
    )
    missing = sorted(set(pair_ids) - cohort.keys())
    if missing:
        raise ValueError(f"Missing {len(missing)} strict pairs from cohort: {missing}")

    rows: list[dict[str, Any]] = []
    for pair_id in pair_ids:
        original = cohort[pair_id]
        messages = original["history_messages"]
        digest = hashlib.sha256(
            (
                pair_id
                + ":delusion:"
                + json.dumps(messages, ensure_ascii=False, sort_keys=True)
            ).encode("utf-8")
        ).hexdigest()
        prior_row = prior.get(pair_id)
        input_id = digest[:24]
        if prior_row is not None:
            if prior_row["messages"] != messages:
                raise ValueError(f"Prior messages changed for pair {pair_id}")
            input_id = prior_row["input_id"]
        rows.append(
            {
                "pair_id": pair_id,
                "input_id": input_id,
                "condition": "delusion",
                "messages": messages,
                "target_text": original["target_text"],
                "theme": original.get("theme"),
                "source": original.get("source"),
                "context_scope": "full_history",
                "retained_message_count": original.get("retained_message_count"),
            }
        )

    write_jsonl(args.output, rows)
    write_manifest(
        args.output.with_suffix(".manifest.json"),
        {
            "pairs": len(rows),
            "rows": len(rows),
            "condition": "delusion",
            "context_scope": "full_history",
            "cohort": "strict target-only final cohort",
        },
    )
    print(f"Wrote {len(rows)} full-history inputs to {args.output}")


if __name__ == "__main__":
    main()
