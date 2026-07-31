#!/usr/bin/env python3
"""Materialize arm-blinded model inputs from validated pairs."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from pathlib import Path
from typing import Any

from io_utils import read_jsonl, write_jsonl, write_manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--originals", type=Path, default=Path("artifacts/cohort_original.jsonl")
    )
    parser.add_argument(
        "--controls", type=Path, default=Path("artifacts/generated_controls.jsonl")
    )
    parser.add_argument(
        "--validation", type=Path, default=Path("artifacts/control_validation.jsonl")
    )
    parser.add_argument(
        "--output", type=Path, default=Path("artifacts/model_inputs.jsonl")
    )
    parser.add_argument(
        "--include-validator-rejected",
        action="store_true",
        help="Use structurally valid controls even if the semantic validator rejected them.",
    )
    args = parser.parse_args()

    originals = {row["pair_id"]: row for row in read_jsonl(args.originals)}
    controls = {row["pair_id"]: row for row in read_jsonl(args.controls)}
    validations = {row["pair_id"]: row for row in read_jsonl(args.validation)}
    pair_ids = sorted(originals.keys() & controls.keys() & validations.keys())
    if not args.include_validator_rejected:
        pair_ids = [
            pair_id
            for pair_id in pair_ids
            if validations[pair_id].get("usable") is True
        ]

    rows: list[dict[str, Any]] = []
    for pair_id in pair_ids:
        original = originals[pair_id]
        control = controls[pair_id]
        validation = validations[pair_id]
        shared = {
            "pair_id": pair_id,
            "source": original.get("source"),
            "theme": original.get("theme"),
            "original_distress": validation.get("original_distress"),
            "control_distress": validation.get("control_distress"),
            "original_harm": validation.get("original_harm"),
            "control_harm": validation.get("control_harm"),
            "validation_usable": validation.get("usable"),
        }
        for condition, messages in (
            ("delusion", original["history_messages"]),
            ("grounded_control", control["control_messages"]),
        ):
            digest = hashlib.sha256(
                (
                    pair_id
                    + ":"
                    + condition
                    + ":"
                    + json.dumps(
                        messages, ensure_ascii=False, sort_keys=True
                    )
                ).encode("utf-8")
            ).hexdigest()
            rows.append(
                {
                    **shared,
                    "input_id": digest[:24],
                    "condition": condition,
                    "messages": messages,
                }
            )

    random.Random(260600975).shuffle(rows)
    write_jsonl(args.output, rows)
    write_manifest(
        args.output.with_suffix(".manifest.json"),
        {
            "pairs": len(pair_ids),
            "rows": len(rows),
            "conditions": ["delusion", "grounded_control"],
            "arm_order_seed": 260600975,
            "validator_rejected_included": args.include_validator_rejected,
        },
    )
    print(f"Wrote {len(rows)} inputs from {len(pair_ids)} pairs to {args.output}")


if __name__ == "__main__":
    main()
