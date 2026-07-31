#!/usr/bin/env python3
"""Select the first strictly valid counterfactual pair for every target."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pairs", type=Path, nargs="+", required=True)
    parser.add_argument("--validations", type=Path, nargs="+", required=True)
    parser.add_argument("--output-pairs", type=Path, required=True)
    parser.add_argument("--output-validations", type=Path, required=True)
    args = parser.parse_args()
    if len(args.pairs) != len(args.validations):
        raise ValueError("--pairs and --validations must have the same length")

    selected_pairs: dict[str, dict[str, Any]] = {}
    selected_validations: dict[str, dict[str, Any]] = {}
    accepted_by_round: dict[str, int] = {}
    attempted: set[str] = set()
    for round_index, (pair_path, validation_path) in enumerate(
        zip(args.pairs, args.validations, strict=True)
    ):
        pair_rows = {row["pair_id"]: row for row in read_jsonl(pair_path)}
        validation_rows = {
            row["pair_id"]: row for row in read_jsonl(validation_path)
        }
        attempted.update(pair_rows)
        accepted = 0
        for pair_id in sorted(pair_rows.keys() & validation_rows.keys()):
            if pair_id in selected_pairs:
                continue
            if validation_rows[pair_id].get("usable") is not True:
                continue
            selected_pairs[pair_id] = pair_rows[pair_id]
            selected_validations[pair_id] = validation_rows[pair_id]
            accepted += 1
        accepted_by_round[str(round_index)] = accepted

    pair_rows = [selected_pairs[key] for key in sorted(selected_pairs)]
    validation_rows = [
        selected_validations[key] for key in sorted(selected_validations)
    ]
    write_jsonl(args.output_pairs, pair_rows)
    write_jsonl(args.output_validations, validation_rows)
    manifest = {
        "attempted_targets": len(attempted),
        "strictly_valid_targets": len(pair_rows),
        "accepted_by_round": accepted_by_round,
    }
    args.output_pairs.with_suffix(".manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
