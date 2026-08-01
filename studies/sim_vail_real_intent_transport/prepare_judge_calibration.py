#!/usr/bin/env python3
"""Build a deterministic score-stratified calibration set from SIM-VAIL turns."""

from __future__ import annotations

import argparse
import glob
import json
from collections import Counter
from pathlib import Path
from typing import Any

import pandas as pd

from design import STUDY_ID, stable_hash
from io_utils import sha256_file, write_jsonl


REFERENCE_MAP = {
    "overall_risk": "concerning",
    "belief_reinforcement": "mh_maladaptive_belief_reinforcement",
    "risky_action_enablement": "mh_risky_action_enablement",
    "boundary_dependence_risk": "mh_boundary_and_dependence_risk",
    "glorification_romanticization": "mh_glorification_or_romanticization",
}


def score_bin(score: int) -> str:
    if score == 1:
        return "1"
    if score <= 3:
        return "2-3"
    if score <= 6:
        return "4-6"
    return "7-10"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--turn-score-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--count", type=int, default=200)
    parser.add_argument("--exclude", type=Path)
    args = parser.parse_args()

    files = sorted(glob.glob(str(args.turn_score_dir / "*.csv")))
    if len(files) != 6367:
        raise ValueError(f"Expected 6367 released turn CSVs, found {len(files)}")
    excluded_ids = set()
    if args.exclude:
        with args.exclude.open(encoding="utf-8") as handle:
            excluded_ids = {
                json.loads(line)["turn_key"] for line in handle if line.strip()
            }
    rows = []
    incomplete_required_scores = 0
    for path in files:
        frame = pd.read_csv(path)
        if len(frame) != 1:
            raise ValueError(f"{path}: expected one row")
        row = frame.iloc[0].to_dict()
        turn_key = row.get("turn_key")
        if not isinstance(turn_key, str) or not turn_key:
            turn_key = (
                f"{row['transcript_id']}::user{int(row['user_idx'])}"
                f"::assistant{int(row['assistant_idx'])}"
            )
        item = {
            "calibration_id": stable_hash(STUDY_ID, turn_key, "calibration")[:24],
            "turn_key": turn_key,
            "transcript_id": str(row["transcript_id"]),
            "assistant_model": str(row["assistant_model"]),
            "user_text": str(row["user_text"]),
            "assistant_text": str(row["assistant_text"]),
            "reference_judge": "claude-sonnet-4.5 released SIM-VAIL turn score",
            "reference_file": str(path),
        }
        if any(pd.isna(row.get(source_name)) for source_name in REFERENCE_MAP.values()):
            incomplete_required_scores += 1
            continue
        for output_name, source_name in REFERENCE_MAP.items():
            score = int(row[source_name])
            if not 1 <= score <= 10:
                raise ValueError(f"{path}: invalid {source_name}={score}")
            item[f"reference_{output_name}"] = score
        if turn_key not in excluded_ids:
            rows.append(item)

    # Seek coverage for every dimension and score band, then fill by a stable
    # global hash. This tests the entire scale rather than only common score-1
    # rows while keeping selection independent of our future judge output.
    selected: dict[str, dict[str, Any]] = {}
    per_cell = 10
    for dimension in REFERENCE_MAP:
        for band in ("1", "2-3", "4-6", "7-10"):
            candidates = [
                row for row in rows
                if score_bin(row[f"reference_{dimension}"]) == band
            ]
            candidates.sort(
                key=lambda row: stable_hash(
                    STUDY_ID, dimension, band, row["turn_key"], "calibration-select"
                )
            )
            for row in candidates[:per_cell]:
                selected[row["calibration_id"]] = row
    remaining = sorted(
        (row for row in rows if row["calibration_id"] not in selected),
        key=lambda row: stable_hash(STUDY_ID, row["turn_key"], "calibration-fill"),
    )
    for row in remaining:
        if len(selected) >= args.count:
            break
        selected[row["calibration_id"]] = row
    output = list(selected.values())[: args.count]
    if len(output) != args.count:
        raise ValueError(f"Could only select {len(output)}/{args.count} calibration rows")
    output.sort(key=lambda row: row["calibration_id"])
    write_jsonl(args.output, output)

    manifest = {
        "study_id": STUDY_ID,
        "source": str(args.turn_score_dir),
        "released_turn_rows": len(rows),
        "released_turn_files": len(files),
        "incomplete_required_scores_excluded": incomplete_required_scores,
        "prior_calibration_turns_excluded": len(excluded_ids),
        "selected_rows": len(output),
        "selection": "10 per dimension-by-score-band cell, deduplicated, then stable-hash fill",
        "reference_map": REFERENCE_MAP,
        "score_bands": {
            dimension: Counter(
                score_bin(row[f"reference_{dimension}"]) for row in output
            )
            for dimension in REFERENCE_MAP
        },
        "output_sha256": sha256_file(args.output),
    }
    args.output.with_suffix(args.output.suffix + ".manifest.json").write_text(
        json.dumps(manifest, indent=2, default=dict) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2, default=dict))


if __name__ == "__main__":
    main()
