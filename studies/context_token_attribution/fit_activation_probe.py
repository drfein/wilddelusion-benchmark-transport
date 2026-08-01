from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from config import ENDORSEMENT_THRESHOLD, SEED
from io_utils import read_jsonl, sha256_file, write_jsonl
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    log_loss,
    roc_auc_score,
)
from sklearn.preprocessing import StandardScaler

CANDIDATE_C = (1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0)


def fit_one(x: np.ndarray, y: np.ndarray, c_value: float):
    scaler = StandardScaler().fit(x)
    model = LogisticRegression(
        C=c_value,
        penalty="l2",
        solver="liblinear",
        max_iter=2_000,
        random_state=SEED,
    ).fit(scaler.transform(x), y)
    return scaler, model


def predict_one(scaler, model, x: np.ndarray) -> np.ndarray:
    return model.predict_proba(scaler.transform(x))[:, 1]


def safe_metrics(y: np.ndarray, prediction: np.ndarray) -> dict[str, float]:
    result = {
        "n": len(y),
        "prevalence": float(y.mean()),
        "brier": float(brier_score_loss(y, prediction)),
        "log_loss": float(log_loss(y, prediction, labels=[0, 1])),
    }
    result["auroc"] = (
        float(roc_auc_score(y, prediction)) if len(np.unique(y)) == 2 else np.nan
    )
    result["average_precision"] = (
        float(average_precision_score(y, prediction))
        if len(np.unique(y)) == 2
        else np.nan
    )
    return result


def bootstrap_metrics(
    y: np.ndarray,
    prediction: np.ndarray,
    groups: np.ndarray,
    *,
    iterations: int,
) -> dict[str, list[float]]:
    rng = np.random.default_rng(SEED)
    unique_groups = np.unique(groups)
    values: dict[str, list[float]] = {"auroc": [], "average_precision": [], "brier": []}
    indices_by_group = {
        group: np.flatnonzero(groups == group) for group in unique_groups
    }
    for _ in range(iterations):
        sampled = rng.choice(unique_groups, size=len(unique_groups), replace=True)
        indices = np.concatenate([indices_by_group[group] for group in sampled])
        metrics = safe_metrics(y[indices], prediction[indices])
        for name, samples in values.items():
            if np.isfinite(metrics[name]):
                samples.append(metrics[name])
    return {
        name: [float(np.quantile(samples, 0.025)), float(np.quantile(samples, 0.975))]
        for name, samples in values.items()
        if samples
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cohort", type=Path, required=True)
    parser.add_argument("--judgments", type=Path, required=True)
    parser.add_argument("--activation-dir", type=Path, required=True)
    parser.add_argument("--checkpoint-dir", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--bootstrap", type=int, default=2_000)
    args = parser.parse_args()

    cohort = {row["conversation_hash"]: row for row in read_jsonl(args.cohort)}
    judgments = [
        row
        for row in read_jsonl(args.judgments)
        if row.get("conversation_hash") in cohort
        and isinstance(row.get("annotation_score"), int)
        and not row.get("judge_error")
    ]
    if not judgments:
        raise ValueError("No valid judgments")
    groups = np.asarray([row["conversation_hash"] for row in judgments])
    folds = np.asarray([int(cohort[group]["fold"]) for group in groups])
    y = np.asarray(
        [int(row["annotation_score"] >= ENDORSEMENT_THRESHOLD) for row in judgments],
        dtype=np.int64,
    )

    unique_groups = sorted(set(groups))
    activation_files = {
        group: np.load(args.activation_dir / f"{group}.npz") for group in unique_groups
    }
    layer_keys = sorted(
        set.intersection(*(set(value.files) for value in activation_files.values()))
    )
    if not layer_keys:
        raise ValueError("No common activation layers")
    group_x = {
        layer: {
            group: activation_files[group][layer].astype(np.float32)
            for group in unique_groups
        }
        for layer in layer_keys
    }
    x_by_layer = {
        layer: np.stack([group_x[layer][group] for group in groups])
        for layer in layer_keys
    }

    predictions = np.full(len(y), np.nan, dtype=np.float64)
    baseline_predictions = np.full(len(y), np.nan, dtype=np.float64)
    selected: dict[int, dict[str, float | str]] = {}
    args.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    for outer_fold in range(5):
        outer_train = folds != outer_fold
        outer_test = folds == outer_fold
        candidates = []
        for layer in layer_keys:
            x = x_by_layer[layer]
            for c_value in CANDIDATE_C:
                inner_predictions = np.full(outer_train.sum(), np.nan)
                inner_truth = y[outer_train]
                train_indices = np.flatnonzero(outer_train)
                valid_candidate = True
                for inner_fold in sorted(set(folds[outer_train])):
                    inner_test_global = train_indices[folds[outer_train] == inner_fold]
                    inner_train_global = train_indices[folds[outer_train] != inner_fold]
                    if len(np.unique(y[inner_train_global])) < 2:
                        valid_candidate = False
                        break
                    scaler, model = fit_one(
                        x[inner_train_global], y[inner_train_global], c_value
                    )
                    positions = np.flatnonzero(folds[outer_train] == inner_fold)
                    inner_predictions[positions] = predict_one(
                        scaler, model, x[inner_test_global]
                    )
                if valid_candidate and np.isfinite(inner_predictions).all():
                    candidates.append(
                        (
                            float(
                                log_loss(inner_truth, inner_predictions, labels=[0, 1])
                            ),
                            layer,
                            c_value,
                        )
                    )
        if not candidates:
            raise RuntimeError(f"No valid inner-CV candidate for fold {outer_fold}")
        inner_loss, layer, c_value = min(candidates)
        x = x_by_layer[layer]
        scaler, model = fit_one(x[outer_train], y[outer_train], c_value)
        predictions[outer_test] = predict_one(scaler, model, x[outer_test])
        baseline_predictions[outer_test] = y[outer_train].mean()
        raw_direction = model.coef_[0] / scaler.scale_
        raw_intercept = float(
            model.intercept_[0] - np.dot(model.coef_[0], scaler.mean_ / scaler.scale_)
        )
        np.savez_compressed(
            args.checkpoint_dir / f"outer_fold_{outer_fold}.npz",
            layer=np.asarray(layer),
            c_value=np.asarray(c_value),
            raw_direction=raw_direction.astype(np.float32),
            raw_intercept=np.asarray(raw_intercept),
            scaler_mean=scaler.mean_.astype(np.float32),
            scaler_scale=scaler.scale_.astype(np.float32),
            standardized_coef=model.coef_[0].astype(np.float32),
            standardized_intercept=model.intercept_.astype(np.float32),
        )
        selected[outer_fold] = {
            "layer": layer,
            "c_value": c_value,
            "inner_log_loss": inner_loss,
            "train_responses": int(outer_train.sum()),
            "test_responses": int(outer_test.sum()),
        }

    if not np.isfinite(predictions).all():
        raise RuntimeError("Missing outer-fold predictions")
    prediction_rows = []
    for index, row in enumerate(judgments):
        prediction_rows.append(
            {
                "conversation_hash": row["conversation_hash"],
                "generation_sha256": row["generation_sha256"],
                "repetition": int(row["repetition"]),
                "fold": int(folds[index]),
                "endorsement": int(y[index]),
                "endorsement_score": int(row["annotation_score"]),
                "oof_probability": float(predictions[index]),
                "fold_baseline_probability": float(baseline_predictions[index]),
                "selected_layer": selected[int(folds[index])]["layer"],
            }
        )
    write_jsonl(args.predictions, prediction_rows)

    metrics = safe_metrics(y, predictions)
    baseline_metrics = safe_metrics(y, baseline_predictions)
    summary = {
        "design": "nested five-fold CV grouped by conversation hash",
        "judgments_sha256": sha256_file(args.judgments),
        "responses": len(y),
        "conversations": len(unique_groups),
        "repetitions_per_conversation": sorted(
            {int(np.sum(groups == group)) for group in unique_groups}
        ),
        "layers_available": layer_keys,
        "candidate_c": list(CANDIDATE_C),
        "outer_fold_selection": selected,
        "oof_metrics": metrics,
        "fold_prevalence_baseline": baseline_metrics,
        "bootstrap_95_ci_by_conversation": bootstrap_metrics(
            y, predictions, groups, iterations=args.bootstrap
        ),
        "interpretation_gate": (
            "Probe attribution is interpreted only if OOF discrimination and "
            "proper-score improvement over the fold-prevalence baseline are present."
        ),
    }
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    args.summary.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
