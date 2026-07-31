#!/usr/bin/env python3
"""Report counterfactual retention from the complete 522-target release."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from io_utils import read_jsonl, write_manifest  # noqa: E402


def summarize(frame: pd.DataFrame, column: str) -> pd.DataFrame:
    return (
        frame.groupby(column, dropna=False)
        .agg(
            canonical_targets=("pair_id", "size"),
            source_conversations=("cluster_id", "nunique"),
            implicit_explicit_valid=("implicit_explicit_valid", "sum"),
            grounded_control_valid=("grounded_control_valid", "sum"),
        )
        .reset_index()
        .assign(
            implicit_explicit_retention=lambda data: (
                data["implicit_explicit_valid"] / data["canonical_targets"]
            ),
            grounded_control_retention=lambda data: (
                data["grounded_control_valid"] / data["canonical_targets"]
            ),
        )
        .sort_values("canonical_targets", ascending=False)
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    here = Path(__file__).resolve().parent
    parser.add_argument(
        "--cohort",
        type=Path,
        default=here / "artifacts" / "cohort_full522.jsonl",
    )
    parser.add_argument(
        "--implicit-explicit",
        type=Path,
        default=(
            here.parent
            / "psychogenic_machine_transport_250910970"
            / "full522"
            / "artifacts"
            / "final_pairs.jsonl"
        ),
    )
    parser.add_argument(
        "--grounded-controls",
        type=Path,
        default=here / "artifacts" / "final_grounded_controls.jsonl",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=here / "results" / "cohort_flow"
    )
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    frame = pd.DataFrame(read_jsonl(args.cohort))
    implicit_ids = {row["pair_id"] for row in read_jsonl(args.implicit_explicit)}
    grounded_ids = {row["pair_id"] for row in read_jsonl(args.grounded_controls)}
    frame["implicit_explicit_valid"] = frame["pair_id"].isin(implicit_ids)
    frame["grounded_control_valid"] = frame["pair_id"].isin(grounded_ids)
    frame[
        [
            "pair_id",
            "cluster_id",
            "source",
            "theme",
            "implicit_explicit_valid",
            "grounded_control_valid",
        ]
    ].to_csv(args.output_dir / "target_retention.csv", index=False)
    by_theme = summarize(frame, "theme")
    by_source = summarize(frame, "source")
    by_theme.to_csv(args.output_dir / "retention_by_theme.csv", index=False)
    by_source.to_csv(args.output_dir / "retention_by_source.csv", index=False)
    write_manifest(
        args.output_dir / "summary.json",
        {
            "canonical_targets": len(frame),
            "canonical_source_conversations": int(frame["cluster_id"].nunique()),
            "implicit_explicit_valid": int(frame["implicit_explicit_valid"].sum()),
            "grounded_control_valid": int(frame["grounded_control_valid"].sum()),
            "both_valid": int(
                (
                    frame["implicit_explicit_valid"]
                    & frame["grounded_control_valid"]
                ).sum()
            ),
            "retention_by_theme": json.loads(by_theme.to_json(orient="records")),
            "retention_by_source": json.loads(by_source.to_json(orient="records")),
        },
    )
    print((args.output_dir / "summary.json").read_text())


if __name__ == "__main__":
    main()
