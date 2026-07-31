#!/usr/bin/env python3
"""Merge target-only control repairs over their prior versions."""

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
        "--repairs",
        type=Path,
        default=Path("artifacts/target_control_repairs.jsonl"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/target_controls_final.jsonl"),
    )
    args = parser.parse_args()
    rows = {row["pair_id"]: row for row in read_jsonl(args.controls)}
    repairs = {
        row["pair_id"]: row
        for row in read_jsonl(args.repairs)
        if row.get("control_text") and not row.get("repair_error")
    }
    rows.update(repairs)
    write_jsonl(args.output, [rows[pair_id] for pair_id in sorted(rows)])
    print(f"Merged {len(repairs)} repairs into {len(rows)} target controls")


if __name__ == "__main__":
    main()
