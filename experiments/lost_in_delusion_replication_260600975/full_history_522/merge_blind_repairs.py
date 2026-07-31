#!/usr/bin/env python3
"""Merge blind-audit repairs and seed only independently grounded pairs."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from io_utils import read_jsonl, write_jsonl  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    here = Path(__file__).resolve().parent
    parser.add_argument(
        "--originals",
        type=Path,
        default=here / "artifacts" / "cohort_original.jsonl",
    )
    parser.add_argument(
        "--controls",
        type=Path,
        default=here / "artifacts" / "final_controls.jsonl",
    )
    parser.add_argument("--repairs", type=Path, required=True)
    parser.add_argument(
        "--paired-validation",
        type=Path,
        default=here / "artifacts" / "control_validation_final.jsonl",
    )
    parser.add_argument(
        "--blind-audit",
        type=Path,
        default=here / "artifacts" / "control_blind_audit.jsonl",
    )
    parser.add_argument("--controls-output", type=Path, required=True)
    parser.add_argument("--seed-validation-output", type=Path, required=True)
    args = parser.parse_args()

    originals = {row["pair_id"]: row for row in read_jsonl(args.originals)}
    controls = {row["pair_id"]: row for row in read_jsonl(args.controls)}
    repairs = {
        row["pair_id"]: row
        for row in read_jsonl(args.repairs)
        if row.get("control_messages") and not row.get("repair_error")
    }
    paired = {
        row["pair_id"]: row for row in read_jsonl(args.paired_validation)
    }
    blind = {row["pair_id"]: row for row in read_jsonl(args.blind_audit)}
    candidate_ids = sorted(blind.keys())
    controls.update(repairs)
    candidate_controls = [
        controls[pair_id]
        for pair_id in candidate_ids
        if controls.get(pair_id, {}).get("control_messages")
    ]
    seeded_ids = {
        pair_id
        for pair_id in candidate_ids
        if blind[pair_id].get("control_delusion") == "absent"
        and paired.get(pair_id, {}).get("usable") is True
        and originals[pair_id]["history_messages"]
        != controls[pair_id]["control_messages"]
        and pair_id not in repairs
    }
    seeded_validation = [paired[pair_id] for pair_id in sorted(seeded_ids)]
    write_jsonl(args.controls_output, candidate_controls)
    write_jsonl(args.seed_validation_output, seeded_validation)
    print(
        f"Merged {len(repairs)} blind-audit repairs; wrote "
        f"{len(candidate_controls)} candidates and seeded "
        f"{len(seeded_validation)} independently grounded validations"
    )


if __name__ == "__main__":
    main()
