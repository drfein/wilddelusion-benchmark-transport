from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from analyze_behavior_interventions import bootstrap, sign_flip_p_value
from config import ENDORSEMENT_THRESHOLD
from io_utils import read_jsonl, sha256_file, write_jsonl
from scipy.stats import spearmanr


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--judgments", type=Path, required=True)
    parser.add_argument("--behavior-rows", type=Path, required=True)
    parser.add_argument("--rows-output", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--bootstrap", type=int, default=10_000)
    parser.add_argument("--randomization-iterations", type=int, default=200_000)
    args = parser.parse_args()

    units: dict[str, dict[str, dict]] = {}
    for row in read_jsonl(args.judgments):
        if isinstance(row.get("annotation_score"), int) and not row.get("judge_error"):
            units.setdefault(row["conversation_hash"], {})[row["condition"]] = row
    behavior = {row["conversation_hash"]: row for row in read_jsonl(args.behavior_rows)}
    rows = []
    for conversation_hash in sorted(set(units) & set(behavior)):
        if set(units[conversation_hash]) != {"attributed", "matched_control"}:
            continue
        attributed = units[conversation_hash]["attributed"]
        control = units[conversation_hash]["matched_control"]
        score_difference = int(attributed["annotation_score"]) - int(
            control["annotation_score"]
        )
        rows.append(
            {
                "conversation_hash": conversation_hash,
                "attributed_score": int(attributed["annotation_score"]),
                "control_score": int(control["annotation_score"]),
                "attributed_endorses": int(
                    attributed["annotation_score"] >= ENDORSEMENT_THRESHOLD
                ),
                "control_endorses": int(
                    control["annotation_score"] >= ENDORSEMENT_THRESHOLD
                ),
                "attributed_minus_control_score": score_difference,
                "top_minus_matched_behavior": behavior[conversation_hash][
                    "top_minus_matched"
                ],
                "top_minus_matched_behavior_score": behavior[conversation_hash][
                    "top_minus_matched_score"
                ],
            }
        )
    if not rows:
        raise ValueError("No complete selected-unit pairs overlap behavior rows")
    differences = np.asarray(
        [row["attributed_minus_control_score"] for row in rows], dtype=np.float64
    )
    behavioral = np.asarray(
        [row["top_minus_matched_behavior"] for row in rows], dtype=np.float64
    )
    ordinal_behavioral = np.asarray(
        [row["top_minus_matched_behavior_score"] for row in rows], dtype=np.float64
    )
    binary_correlation = spearmanr(differences, behavioral)
    ordinal_correlation = spearmanr(differences, ordinal_behavioral)
    summary = {
        "judgments_sha256": sha256_file(args.judgments),
        "behavior_rows_sha256": sha256_file(args.behavior_rows),
        "paired_conversations": len(rows),
        "mean_scores": {
            "attributed": float(np.mean([row["attributed_score"] for row in rows])),
            "matched_control": float(np.mean([row["control_score"] for row in rows])),
        },
        "endorsing_messages": {
            "attributed": int(sum(row["attributed_endorses"] for row in rows)),
            "matched_control": int(sum(row["control_endorses"] for row in rows)),
        },
        "attributed_minus_control_score": {
            "mean": float(differences.mean()),
            "bootstrap_95_ci_by_conversation": bootstrap(
                differences, args.bootstrap
            ),
            "two_sided_cluster_sign_flip_p": sign_flip_p_value(
                differences, args.randomization_iterations
            ),
        },
        "direction_counts": {
            "attributed_higher": int(np.count_nonzero(differences > 0)),
            "equal": int(np.count_nonzero(differences == 0)),
            "control_higher": int(np.count_nonzero(differences < 0)),
        },
        "semantic_contrast_vs_behavioral_deletion_effect": {
            "binary_spearman": float(binary_correlation.statistic),
            "binary_two_sided_asymptotic_p": float(binary_correlation.pvalue),
            "ordinal_spearman": float(ordinal_correlation.statistic),
            "ordinal_two_sided_asymptotic_p": float(ordinal_correlation.pvalue),
        },
    }
    write_jsonl(args.rows_output, rows)
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    args.summary.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
