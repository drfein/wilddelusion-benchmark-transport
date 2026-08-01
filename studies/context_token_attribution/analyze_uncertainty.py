from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from config import ENDORSEMENT_THRESHOLD, SEED
from io_utils import read_jsonl, sha256_file, write_jsonl
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.preprocessing import StandardScaler

WINDOWS = (8, 16, 32, 64, 128)
SIGNALS = ("entropy", "margin", "top_nll", "nucleus_0p9", "near_tie")


def features(row: dict, window: int | None) -> list[float]:
    trace = row["uncertainty"]
    length = (
        len(trace["entropy"]) if window is None else min(window, len(trace["entropy"]))
    )
    if length <= 0:
        raise ValueError("Empty uncertainty trace")
    result = []
    for signal in SIGNALS:
        values = np.asarray(trace[signal][:length], dtype=np.float64)
        result.extend((float(values.mean()), float(values.max())))
    return result


def grouped_oof(x: np.ndarray, y: np.ndarray, folds: np.ndarray) -> np.ndarray:
    prediction = np.full(len(y), np.nan)
    for fold in range(5):
        train = folds != fold
        test = folds == fold
        scaler = StandardScaler().fit(x[train])
        model = LogisticRegression(
            C=1.0,
            class_weight="balanced",
            solver="liblinear",
            max_iter=2_000,
            random_state=SEED,
        ).fit(scaler.transform(x[train]), y[train])
        prediction[test] = model.predict_proba(scaler.transform(x[test]))[:, 1]
    return prediction


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cohort", type=Path, required=True)
    parser.add_argument("--judgments", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--bootstrap", type=int, default=2_000)
    args = parser.parse_args()

    cohort = {row["conversation_hash"]: row for row in read_jsonl(args.cohort)}
    rows = [
        row
        for row in read_jsonl(args.judgments)
        if row.get("conversation_hash") in cohort
        and isinstance(row.get("annotation_score"), int)
        and row.get("uncertainty")
        and not row.get("judge_error")
    ]
    y = np.asarray(
        [int(row["annotation_score"] >= ENDORSEMENT_THRESHOLD) for row in rows]
    )
    groups = np.asarray([row["conversation_hash"] for row in rows])
    folds = np.asarray([int(cohort[group]["fold"]) for group in groups])
    labels: list[int | str] = [*WINDOWS, "full"]
    predictions: dict[int | str, np.ndarray] = {}
    results = {}
    for label in labels:
        window = None if label == "full" else int(label)
        x = np.asarray([features(row, window) for row in rows])
        prediction = grouped_oof(x, y, folds)
        predictions[label] = prediction
        results[str(label)] = {
            "auroc": float(roc_auc_score(y, prediction)),
            "average_precision": float(average_precision_score(y, prediction)),
            "mean_effective_tokens": float(
                np.mean(
                    [
                        len(row["uncertainty"]["entropy"])
                        if window is None
                        else min(window, len(row["uncertainty"]["entropy"]))
                        for row in rows
                    ]
                )
            ),
        }

    peak_label = max(
        labels[:-1], key=lambda label: results[str(label)]["average_precision"]
    )
    observed_delta = (
        results[str(peak_label)]["average_precision"]
        - results["full"]["average_precision"]
    )
    rng = np.random.default_rng(SEED)
    unique_groups = np.unique(groups)
    by_group = {group: np.flatnonzero(groups == group) for group in unique_groups}
    bootstrap_delta = []
    for _ in range(args.bootstrap):
        sampled = rng.choice(unique_groups, len(unique_groups), replace=True)
        indices = np.concatenate([by_group[group] for group in sampled])
        if len(np.unique(y[indices])) < 2:
            continue
        bootstrap_delta.append(
            average_precision_score(y[indices], predictions[peak_label][indices])
            - average_precision_score(y[indices], predictions["full"][indices])
        )
    delta_ci = [
        float(np.quantile(bootstrap_delta, 0.025)),
        float(np.quantile(bootstrap_delta, 0.975)),
    ]
    prediction_rows = []
    for index, row in enumerate(rows):
        prediction_rows.append(
            {
                "conversation_hash": row["conversation_hash"],
                "generation_sha256": row["generation_sha256"],
                "endorsement": int(y[index]),
                "generated_tokens": int(row["generated_tokens"]),
                **{
                    f"probability_{label}": float(predictions[label][index])
                    for label in labels
                },
            }
        )
    write_jsonl(args.predictions, prediction_rows)
    summary = {
        "adapted_method": "arXiv:2606.06635 token uncertainty features",
        "judgments_sha256": sha256_file(args.judgments),
        "responses": len(rows),
        "conversations": len(unique_groups),
        "prevalence": float(y.mean()),
        "windows": results,
        "best_genuine_early_window": peak_label,
        "early_minus_full_average_precision": observed_delta,
        "bootstrap_95_ci_by_conversation": delta_ci,
        "classification": (
            "committed" if delta_ci[0] > 0 else "not-established-as-committed"
        ),
        "scope_warning": (
            "The paper studies correctness failures in reasoning traces; this is a "
            "pre-registered adaptation to endorsement behavior in ordinary responses."
        ),
    }
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    args.summary.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
