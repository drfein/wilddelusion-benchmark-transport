from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from config import ENDORSEMENT_THRESHOLD, SEED
from io_utils import read_jsonl, sha256_file, write_jsonl
from scipy.special import expit
from scipy.stats import spearmanr
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score


def bootstrap_interval(values: np.ndarray, iterations: int) -> list[float]:
    rng = np.random.default_rng(SEED)
    estimates = np.empty(iterations, dtype=np.float64)
    for index in range(iterations):
        sample = rng.integers(0, len(values), len(values))
        estimates[index] = values[sample].mean()
    return [float(value) for value in np.quantile(estimates, [0.025, 0.975])]


def discrimination(truth: np.ndarray, prediction: np.ndarray) -> dict[str, float]:
    return {
        "auroc": float(roc_auc_score(truth, prediction)),
        "average_precision": float(average_precision_score(truth, prediction)),
        "brier": float(brier_score_loss(truth, prediction)),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cohort", type=Path, required=True)
    parser.add_argument("--judgments", type=Path, required=True)
    parser.add_argument("--full-activation-dir", type=Path, required=True)
    parser.add_argument("--target-activation-dir", type=Path, required=True)
    parser.add_argument("--checkpoint-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--bootstrap", type=int, default=10_000)
    args = parser.parse_args()

    cohort = {row["conversation_hash"]: row for row in read_jsonl(args.cohort)}
    labels: dict[str, list[int]] = defaultdict(list)
    for row in read_jsonl(args.judgments):
        conversation_hash = row.get("conversation_hash")
        if (
            conversation_hash in cohort
            and isinstance(row.get("annotation_score"), int)
            and not row.get("judge_error")
        ):
            labels[conversation_hash].append(
                int(row["annotation_score"] >= ENDORSEMENT_THRESHOLD)
            )

    rows = []
    for conversation_hash in sorted(cohort):
        row = cohort[conversation_hash]
        checkpoint_path = args.checkpoint_dir / f"outer_fold_{row['fold']}.npz"
        full_path = args.full_activation_dir / f"{conversation_hash}.npz"
        target_path = args.target_activation_dir / f"{conversation_hash}.npz"
        if (
            not checkpoint_path.exists()
            or not full_path.exists()
            or not target_path.exists()
        ):
            continue
        checkpoint = np.load(checkpoint_path)
        layer = str(checkpoint["layer"].item())
        direction = checkpoint["raw_direction"].astype(np.float64)
        intercept = float(checkpoint["raw_intercept"].item())
        full = np.load(full_path)[layer].astype(np.float64)
        target = np.load(target_path)[layer].astype(np.float64)
        full_logit = float(full @ direction + intercept)
        target_logit = float(target @ direction + intercept)
        rows.append(
            {
                "original_row_idx": row["original_row_idx"],
                "conversation_hash": conversation_hash,
                "fold": int(row["fold"]),
                "selected_layer": layer,
                "input_tokens": int(row["input_tokens"]),
                "prior_messages": len(row["messages"]) - 2,
                "endorsement_rate": float(np.mean(labels[conversation_hash])),
                "full_logit": full_logit,
                "target_only_logit": target_logit,
                "context_logit_shift": full_logit - target_logit,
                "full_projected_probability": float(expit(full_logit)),
                "target_only_projected_probability": float(expit(target_logit)),
                "context_probability_shift": float(
                    expit(full_logit) - expit(target_logit)
                ),
            }
        )
    if len(rows) != len(cohort):
        raise RuntimeError(
            f"Expected {len(cohort)} paired activations, found {len(rows)}"
        )
    write_jsonl(args.output, rows)

    logit_shifts = np.asarray([row["context_logit_shift"] for row in rows])
    probability_shifts = np.asarray([row["context_probability_shift"] for row in rows])
    rates = np.asarray([row["endorsement_rate"] for row in rows])
    full_probabilities = np.asarray([row["full_projected_probability"] for row in rows])
    target_probabilities = np.asarray(
        [row["target_only_projected_probability"] for row in rows]
    )
    binary = (rates >= 0.5).astype(np.int64)
    summary = {
        "design": (
            "Paired projection of full-history and target-user-only final prompt-token "
            "activations onto the same outer-fold-held-out probe direction"
        ),
        "cohort_sha256": sha256_file(args.cohort),
        "judgments_sha256": sha256_file(args.judgments),
        "conversations": len(rows),
        "mean_context_logit_shift": float(logit_shifts.mean()),
        "mean_context_logit_shift_bootstrap_95_ci": bootstrap_interval(
            logit_shifts, args.bootstrap
        ),
        "median_context_logit_shift": float(np.median(logit_shifts)),
        "fraction_context_increases_probe_logit": float((logit_shifts > 0).mean()),
        "fraction_context_increases_probe_logit_bootstrap_95_ci": bootstrap_interval(
            (logit_shifts > 0).astype(np.float64), args.bootstrap
        ),
        "mean_context_probability_shift": float(probability_shifts.mean()),
        "mean_context_probability_shift_bootstrap_95_ci": bootstrap_interval(
            probability_shifts, args.bootstrap
        ),
        "full_projection_discrimination": discrimination(binary, full_probabilities),
        "target_only_projection_discrimination": discrimination(
            binary, target_probabilities
        ),
        "spearman_context_shift_vs_endorsement_rate": float(
            spearmanr(logit_shifts, rates).statistic
        ),
        "scope": (
            "The shift is representational, not behavioral causation; paired regeneration "
            "is required to establish an endorsement-rate effect."
        ),
    }
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    args.summary.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
