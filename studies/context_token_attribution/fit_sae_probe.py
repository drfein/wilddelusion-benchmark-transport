from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from config import ENDORSEMENT_THRESHOLD, SEED
from fit_activation_probe import bootstrap_metrics, safe_metrics
from io_utils import read_jsonl, sha256_file, write_jsonl
from scipy import sparse
from sklearn.linear_model import LogisticRegression

CANDIDATE_C = (0.001, 0.01, 0.1, 1.0, 10.0)


def load_sparse(path: Path, d_sae: int) -> tuple[list[str], sparse.csr_matrix]:
    data = np.load(path)
    hashes = data["conversation_hashes"].tolist()
    indices = data["feature_indices"].astype(np.int32)
    values = data["feature_values"].astype(np.float32)
    rows = np.repeat(np.arange(len(hashes), dtype=np.int32), indices.shape[1])
    matrix = sparse.csr_matrix(
        (values.ravel(), (rows, indices.ravel())), shape=(len(hashes), d_sae)
    )
    matrix.eliminate_zeros()
    return hashes, matrix


def fit_one(x: sparse.csr_matrix, y: np.ndarray, c_value: float):
    return LogisticRegression(
        C=c_value,
        penalty="l1",
        solver="liblinear",
        intercept_scaling=100.0,
        max_iter=5_000,
        random_state=SEED,
    ).fit(x, y)


def nested_oof(
    x_by_conversation: sparse.csr_matrix,
    conversation_positions: np.ndarray,
    y: np.ndarray,
    folds: np.ndarray,
) -> tuple[np.ndarray, list[dict], np.ndarray]:
    x = x_by_conversation[conversation_positions]
    predictions = np.full(len(y), np.nan, dtype=np.float64)
    coefficients = np.zeros((5, x.shape[1]), dtype=np.float32)
    selections = []
    for outer_fold in range(5):
        outer_train = folds != outer_fold
        outer_test = folds == outer_fold
        candidates = []
        for c_value in CANDIDATE_C:
            inner_predictions = np.full(outer_train.sum(), np.nan)
            inner_truth = y[outer_train]
            train_indices = np.flatnonzero(outer_train)
            valid = True
            for inner_fold in sorted(set(folds[outer_train])):
                inner_test_global = train_indices[folds[outer_train] == inner_fold]
                inner_train_global = train_indices[folds[outer_train] != inner_fold]
                if len(np.unique(y[inner_train_global])) < 2:
                    valid = False
                    break
                model = fit_one(x[inner_train_global], y[inner_train_global], c_value)
                positions = np.flatnonzero(folds[outer_train] == inner_fold)
                inner_predictions[positions] = model.predict_proba(
                    x[inner_test_global]
                )[:, 1]
            if valid and np.isfinite(inner_predictions).all():
                candidates.append(
                    (
                        safe_metrics(inner_truth, inner_predictions)["log_loss"],
                        c_value,
                    )
                )
        if not candidates:
            raise RuntimeError(f"No valid SAE candidate for fold {outer_fold}")
        inner_loss, c_value = min(candidates)
        model = fit_one(x[outer_train], y[outer_train], c_value)
        predictions[outer_test] = model.predict_proba(x[outer_test])[:, 1]
        coefficients[outer_fold] = model.coef_[0].astype(np.float32)
        selections.append(
            {
                "outer_fold": outer_fold,
                "c_value": c_value,
                "inner_log_loss": inner_loss,
                "nonzero_features": int(np.count_nonzero(model.coef_[0])),
            }
        )
    if not np.isfinite(predictions).all():
        raise RuntimeError("Missing SAE out-of-fold predictions")
    return predictions, selections, coefficients


def stable_features(coefficients: np.ndarray, limit: int = 100) -> list[dict]:
    positive = (coefficients > 0).sum(axis=0)
    negative = (coefficients < 0).sum(axis=0)
    nonzero = positive + negative
    mean = coefficients.mean(axis=0)
    candidates = np.flatnonzero(nonzero >= 2)
    candidates = sorted(candidates, key=lambda index: abs(mean[index]), reverse=True)
    return [
        {
            "feature_index": int(index),
            "mean_coefficient": float(mean[index]),
            "selected_folds": int(nonzero[index]),
            "positive_folds": int(positive[index]),
            "negative_folds": int(negative[index]),
            "fold_coefficients": coefficients[:, index].astype(float).tolist(),
        }
        for index in candidates[:limit]
    ]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cohort", type=Path, required=True)
    parser.add_argument("--judgments", type=Path, required=True)
    parser.add_argument("--full-features", type=Path, required=True)
    parser.add_argument("--target-only-features", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--d-sae", type=int, default=65_536)
    parser.add_argument("--bootstrap", type=int, default=5_000)
    args = parser.parse_args()

    cohort_rows = sorted(
        read_jsonl(args.cohort), key=lambda row: row["conversation_hash"]
    )
    cohort = {row["conversation_hash"]: row for row in cohort_rows}
    hashes, full = load_sparse(args.full_features, args.d_sae)
    if hashes != [row["conversation_hash"] for row in cohort_rows]:
        raise ValueError("Full SAE feature order does not match cohort")
    representations = {"full": full}
    feature_hashes = {"full": sha256_file(args.full_features)}
    if args.target_only_features:
        target_hashes, target = load_sparse(args.target_only_features, args.d_sae)
        if target_hashes != hashes:
            raise ValueError(
                "Target-only SAE feature order does not match full features"
            )
        representations["target_only"] = target
        representations["context_delta"] = full - target
        feature_hashes["target_only"] = sha256_file(args.target_only_features)

    judgments = [
        row
        for row in read_jsonl(args.judgments)
        if row.get("conversation_hash") in cohort
        and isinstance(row.get("annotation_score"), int)
        and not row.get("judge_error")
    ]
    groups = np.asarray([row["conversation_hash"] for row in judgments])
    hash_to_position = {value: index for index, value in enumerate(hashes)}
    positions = np.asarray([hash_to_position[value] for value in groups])
    folds = np.asarray([int(cohort[value]["fold"]) for value in groups])
    y = np.asarray(
        [int(row["annotation_score"] >= ENDORSEMENT_THRESHOLD) for row in judgments]
    )
    baseline = np.asarray([y[folds != fold].mean() for fold in folds], dtype=np.float64)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    results = {}
    for name, matrix in representations.items():
        predictions, selections, coefficients = nested_oof(matrix, positions, y, folds)
        features = stable_features(coefficients)
        stable_count = int((np.count_nonzero(coefficients, axis=0) >= 2).sum())
        write_jsonl(args.output_dir / f"{name}_stable_features.jsonl", features)
        write_jsonl(
            args.output_dir / f"{name}_oof_predictions.jsonl",
            [
                {
                    "conversation_hash": row["conversation_hash"],
                    "generation_sha256": row["generation_sha256"],
                    "repetition": int(row["repetition"]),
                    "fold": int(folds[index]),
                    "endorsement": int(y[index]),
                    "oof_probability": float(predictions[index]),
                }
                for index, row in enumerate(judgments)
            ],
        )
        results[name] = {
            "oof_metrics": safe_metrics(y, predictions),
            "bootstrap_95_ci_by_conversation": bootstrap_metrics(
                y, predictions, groups, iterations=args.bootstrap
            ),
            "outer_fold_selection": selections,
            "stable_features_selected_in_at_least_two_folds": stable_count,
            "top_stable_features": features[:20],
            "matrix_nonzeros": int(matrix.nnz),
        }

    summary = {
        "design": "nested fixed five-fold CV grouped by conversation hash",
        "classifier": "L1 logistic regression on sparse TopK SAE features",
        "cohort_sha256": sha256_file(args.cohort),
        "judgments_sha256": sha256_file(args.judgments),
        "feature_files_sha256": feature_hashes,
        "responses": len(y),
        "conversations": len(hashes),
        "prevalence": float(y.mean()),
        "candidate_c": list(CANDIDATE_C),
        "fold_prevalence_baseline": safe_metrics(y, baseline),
        "representations": results,
    }
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    args.summary.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
