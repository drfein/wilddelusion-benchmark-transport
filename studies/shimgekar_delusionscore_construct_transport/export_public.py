#!/usr/bin/env python3
"""Export auditable score tables without redistributing sensitive message text."""

from __future__ import annotations

import argparse
import hashlib
import shutil
from pathlib import Path

import pandas as pd


def group_hash(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def export_csv(source: Path, destination: Path, columns: list[str]) -> None:
    frame = pd.read_csv(source)
    if "conversation_id" in frame:
        frame["conversation_hash"] = frame.conversation_id.astype(str).map(group_hash)
    if "text" in frame:
        frame["text_chars"] = frame.text.fillna("").str.len()
    frame[[column for column in columns if column in frame]].to_csv(destination, index=False)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--work-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    shutil.copy2(args.work_dir / "results/results.json", args.output_dir / "paired_construct_results.json")
    shutil.copy2(
        args.work_dir / "random_control_results/results.json",
        args.output_dir / "random_control_results.json",
    )
    shutil.copy2(args.work_dir / "results/transport_summary.png", args.output_dir / "transport_summary.png")

    export_csv(
        args.work_dir / "results/endpoint_scores.csv",
        args.output_dir / "paired_endpoint_scores.csv",
        ["pair_id", "conversation_hash", "condition", "label", "source", "theme", "language", "fold", "score", "text_chars"],
    )
    export_csv(
        args.work_dir / "random_control_results/oof_scores.csv",
        args.output_dir / "random_control_oof_scores.csv",
        ["pair_id", "conversation_hash", "condition", "label", "source", "theme", "fold", "score", "length_only_score", "text_chars"],
    )
    export_csv(
        args.work_dir / "random_control_results/natural_hard_control_scores.csv",
        args.output_dir / "natural_hard_control_scores.csv",
        ["control_id", "conversation_hash", "source", "exclusion", "judge_confidence", "score", "length_only_score", "text_chars"],
    )
    export_csv(
        args.work_dir / "results/trajectory_scores.csv",
        args.output_dir / "paired_construct_trajectory_scores.csv",
        ["pair_id", "conversation_hash", "source", "theme", "message_index", "user_turn_index", "n_user_turns", "is_target", "fold", "score", "text_chars"],
    )
    export_csv(
        args.work_dir / "random_control_results/trajectory_case_summaries.csv",
        args.output_dir / "random_control_trajectory_case_summaries.csv",
        ["pair_id", "conversation_hash", "source", "theme", "n_user_turns", "all_slope", "prior_slope", "first_to_target", "target_jump", "first_to_last_prior"],
    )
    export_csv(
        args.work_dir / "random_control_results/trajectory_scores.csv",
        args.output_dir / "random_control_trajectory_scores.csv",
        ["pair_id", "conversation_hash", "source", "theme", "message_index", "user_turn_index", "n_user_turns", "is_target", "fold", "score", "text_chars"],
    )


if __name__ == "__main__":
    main()
