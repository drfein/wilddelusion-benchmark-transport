from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import pandas as pd


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Freeze executable pre-outcome cohort."
    )
    parser.add_argument("--cohort", type=Path, required=True)
    parser.add_argument("--cumulative-rewrites", type=Path, required=True)
    parser.add_argument("--out-cohort", type=Path, required=True)
    parser.add_argument("--attrition-output", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    args = parser.parse_args()

    cohort = read_jsonl(args.cohort)
    valid_rewrites = {
        row["original_row_idx"]: row
        for row in read_jsonl(args.cumulative_rewrites)
        if isinstance(row.get("rewrites"), list) and not row.get("error")
    }
    included = [row for row in cohort if row["original_row_idx"] in valid_rewrites]
    excluded = [row for row in cohort if row["original_row_idx"] not in valid_rewrites]
    if len(included) != 216 or len(excluded) != 5:
        raise ValueError(
            f"Frozen execution cohort changed: {len(included)} included, {len(excluded)} excluded"
        )
    write_jsonl(args.out_cohort, included)
    attrition = pd.DataFrame(
        [
            {
                "original_row_idx": row["original_row_idx"],
                "source": row["source"],
                "conversation_hash": row["conversation_hash"],
                "message_hash": row["message_hash"],
                "estimated_full_input_tokens": row["estimated_full_input_tokens"],
                "earlier_assistant_messages": len(row["earlier_assistant_indices"]),
                "reason": "cumulative rewrite did not complete before user-directed stop",
            }
            for row in excluded
        ]
    ).sort_values("original_row_idx")
    args.attrition_output.parent.mkdir(parents=True, exist_ok=True)
    attrition.to_csv(args.attrition_output, index=False)
    manifest = {
        "frozen_before_target_model_outputs": True,
        "original_targets": len(cohort),
        "executable_targets": len(included),
        "excluded_targets": len(excluded),
        "excluded_original_row_indices": attrition["original_row_idx"].tolist(),
        "out_cohort_sha256": hashlib.sha256(args.out_cohort.read_bytes()).hexdigest(),
        "attrition_sha256": hashlib.sha256(
            args.attrition_output.read_bytes()
        ).hexdigest(),
    }
    args.manifest.write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
