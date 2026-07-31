#!/usr/bin/env python3
"""Merge repaired controls and seed accepted validations for the next round."""

from __future__ import annotations

import argparse
from pathlib import Path

from io_utils import read_jsonl, write_jsonl


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--controls", type=Path, default=Path("artifacts/generated_controls.jsonl")
    )
    parser.add_argument(
        "--repairs", type=Path, default=Path("artifacts/repaired_controls.jsonl")
    )
    parser.add_argument(
        "--validation", type=Path, default=Path("artifacts/control_validation.jsonl")
    )
    parser.add_argument(
        "--merged-output",
        type=Path,
        default=Path("artifacts/generated_controls_round2.jsonl"),
    )
    parser.add_argument(
        "--seed-validation-output",
        type=Path,
        default=Path("artifacts/control_validation_round2.jsonl"),
    )
    args = parser.parse_args()

    controls = {row["pair_id"]: row for row in read_jsonl(args.controls)}
    repairs = {
        row["pair_id"]: row
        for row in read_jsonl(args.repairs)
        if row.get("control_messages") and not row.get("repair_error")
    }
    controls.update(repairs)
    write_jsonl(args.merged_output, [controls[key] for key in sorted(controls)])

    accepted = [
        row for row in read_jsonl(args.validation) if row.get("usable") is True
    ]
    write_jsonl(args.seed_validation_output, accepted)
    print(
        f"Merged {len(repairs)} repairs into {len(controls)} controls; "
        f"seeded {len(accepted)} accepted validations"
    )


if __name__ == "__main__":
    main()
