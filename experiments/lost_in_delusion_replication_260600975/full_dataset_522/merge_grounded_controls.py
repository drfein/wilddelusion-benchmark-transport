#!/usr/bin/env python3
"""Freeze one strictly validated grounded target rewrite per WildDelusion row."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from io_utils import read_jsonl, write_jsonl, write_manifest  # noqa: E402


def keyed(path: Path, success_key: str) -> dict[str, dict[str, Any]]:
    return {
        row["pair_id"]: row
        for row in read_jsonl(path)
        if not row.get(success_key)
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    artifact_dir = Path(__file__).resolve().parent / "artifacts"
    parser.add_argument(
        "--round0-controls",
        type=Path,
        default=artifact_dir / "grounded_controls_round0.jsonl",
    )
    parser.add_argument(
        "--round0-validation",
        type=Path,
        default=artifact_dir / "grounded_control_matching_validation.jsonl",
    )
    parser.add_argument(
        "--round0-package",
        type=Path,
        default=artifact_dir / "control_package_judgments.jsonl",
    )
    parser.add_argument(
        "--repair-controls",
        type=Path,
        default=artifact_dir / "grounded_controls_repair1.jsonl",
    )
    parser.add_argument(
        "--repair-validation",
        type=Path,
        default=artifact_dir / "repair1_matching_validation.jsonl",
    )
    parser.add_argument(
        "--repair-package",
        type=Path,
        default=artifact_dir / "repair1_package_judgments.jsonl",
    )
    parser.add_argument(
        "--repair2-controls",
        type=Path,
        default=artifact_dir / "grounded_controls_repair2.jsonl",
    )
    parser.add_argument(
        "--repair2-validation",
        type=Path,
        default=artifact_dir / "repair2_matching_validation.jsonl",
    )
    parser.add_argument(
        "--repair2-package",
        type=Path,
        default=artifact_dir / "repair2_package_judgments.jsonl",
    )
    parser.add_argument(
        "--repair3-controls",
        type=Path,
        default=artifact_dir / "grounded_controls_repair3.jsonl",
    )
    parser.add_argument(
        "--repair3-validation",
        type=Path,
        default=artifact_dir / "repair3_matching_validation.jsonl",
    )
    parser.add_argument(
        "--repair3-package",
        type=Path,
        default=artifact_dir / "repair3_package_judgments.jsonl",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=artifact_dir / "final_grounded_controls.jsonl",
    )
    parser.add_argument("--max-control-score", type=int, default=3)
    args = parser.parse_args()

    rounds = [
        (
            "round0",
            keyed(args.round0_controls, "control_error"),
            keyed(args.round0_validation, "validation_error"),
            keyed(args.round0_package, "judge_error"),
        ),
        (
            "repair1",
            keyed(args.repair_controls, "control_error"),
            keyed(args.repair_validation, "validation_error"),
            keyed(args.repair_package, "judge_error"),
        ),
        (
            "repair2",
            keyed(args.repair2_controls, "repair_error"),
            keyed(args.repair2_validation, "validation_error"),
            keyed(args.repair2_package, "judge_error"),
        ),
        (
            "repair3",
            keyed(args.repair3_controls, "repair_error"),
            keyed(args.repair3_validation, "validation_error"),
            keyed(args.repair3_package, "judge_error"),
        ),
    ]
    pair_ids = sorted(set().union(*(set(controls) for _, controls, _, _ in rounds)))
    selected: list[dict[str, Any]] = []
    counts = {name: 0 for name, *_ in rounds}
    for pair_id in pair_ids:
        for round_name, controls, validations, packages in rounds:
            control = controls.get(pair_id)
            validation = validations.get(pair_id)
            package = packages.get(pair_id)
            if not control or not validation or not package:
                continue
            if validation.get("usable") is not True:
                continue
            if int(package["annotation_score"]) > args.max_control_score:
                continue
            selected.append(
                {
                    **control,
                    "selection_round": round_name,
                    "control_package_score": int(package["annotation_score"]),
                    "control_package_rationale": package.get(
                        "annotation_rationale"
                    ),
                    "speech_act_match": validation["speech_act_match"],
                    "distress_match": validation["distress_match"],
                    "harm_match": validation["harm_match"],
                    "semantic_closeness": validation["semantic_closeness"],
                    "style_length_match": validation["style_length_match"],
                }
            )
            counts[round_name] += 1
            break

    write_jsonl(args.output, selected)
    write_manifest(
        args.output.with_suffix(args.output.suffix + ".manifest.json"),
        {
            "candidate_targets": len(pair_ids),
            "selected_pairs": len(selected),
            "selected_by_round": counts,
            "max_control_package_score": args.max_control_score,
            "matching_validation": "usable=true; every match score >=4/5",
            "selection_priority": [name for name, *_ in rounds],
        },
    )
    print(
        f"Wrote {len(selected)} strict controls "
        f"({', '.join(f'{name}={count}' for name, count in counts.items())})"
    )


if __name__ == "__main__":
    main()
