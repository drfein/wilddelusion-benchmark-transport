from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from config import ENDORSEMENT_THRESHOLD, SEED
from fit_activation_probe import (
    CANDIDATE_C,
    bootstrap_metrics,
    fit_one,
    predict_one,
    safe_metrics,
)
from io_utils import read_jsonl, sha256_file, write_jsonl
from scipy.special import expit
from sklearn.metrics import log_loss


def bootstrap_mean(values: np.ndarray, iterations: int) -> list[float]:
    rng = np.random.default_rng(SEED)
    estimates = np.empty(iterations, dtype=np.float64)
    for index in range(iterations):
        sample = rng.integers(0, len(values), len(values))
        estimates[index] = values[sample].mean()
    return [float(value) for value in np.quantile(estimates, [0.025, 0.975])]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cohort", type=Path, required=True)
    parser.add_argument("--judgments", type=Path, required=True)
    parser.add_argument("--activation-dir", type=Path, required=True)
    parser.add_argument("--prior-assistant-judgments", type=Path, required=True)
    parser.add_argument("--checkpoint-dir", type=Path, required=True)
    parser.add_argument("--oof-predictions", type=Path, required=True)
    parser.add_argument("--trajectory-rows", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--bootstrap", type=int, default=10_000)
    args = parser.parse_args()

    cohort = {row["conversation_hash"]: row for row in read_jsonl(args.cohort)}
    judgments = [
        row
        for row in read_jsonl(args.judgments)
        if row.get("conversation_hash") in cohort
        and isinstance(row.get("annotation_score"), int)
        and not row.get("judge_error")
    ]
    groups = np.asarray([row["conversation_hash"] for row in judgments])
    folds = np.asarray([int(cohort[group]["fold"]) for group in groups])
    y = np.asarray(
        [int(row["annotation_score"] >= ENDORSEMENT_THRESHOLD) for row in judgments],
        dtype=np.int64,
    )
    labels: dict[str, list[int]] = defaultdict(list)
    for group, label in zip(groups, y, strict=True):
        labels[group].append(int(label))
    prior_scores = {
        row["conversation_hash"]: int(row["annotation_score"])
        for row in read_jsonl(args.prior_assistant_judgments)
        if row.get("conversation_hash") in cohort
        and isinstance(row.get("annotation_score"), int)
        and not row.get("judge_error")
    }

    unique_groups = sorted(cohort)
    activation_files = {
        group: np.load(args.activation_dir / f"{group}.npz")
        for group in unique_groups
    }
    layer_keys = sorted(
        set.intersection(
            *(
                {key for key in value.files if key.startswith("layer_")}
                for value in activation_files.values()
            )
        )
    )
    final_by_layer = {
        layer: {
            group: activation_files[group][layer][-1].astype(np.float32)
            for group in unique_groups
        }
        for layer in layer_keys
    }
    x_by_layer = {
        layer: np.stack([final_by_layer[layer][group] for group in groups])
        for layer in layer_keys
    }

    predictions = np.full(len(y), np.nan)
    baseline_predictions = np.full(len(y), np.nan)
    selected: dict[int, dict] = {}
    args.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    for outer_fold in range(5):
        outer_train = folds != outer_fold
        outer_test = folds == outer_fold
        candidates = []
        train_indices = np.flatnonzero(outer_train)
        for layer in layer_keys:
            x = x_by_layer[layer]
            for c_value in CANDIDATE_C:
                inner_predictions = np.full(outer_train.sum(), np.nan)
                valid = True
                for inner_fold in sorted(set(folds[outer_train])):
                    inner_test = train_indices[folds[outer_train] == inner_fold]
                    inner_train = train_indices[folds[outer_train] != inner_fold]
                    if len(np.unique(y[inner_train])) < 2:
                        valid = False
                        break
                    scaler, model = fit_one(x[inner_train], y[inner_train], c_value)
                    positions = np.flatnonzero(folds[outer_train] == inner_fold)
                    inner_predictions[positions] = predict_one(
                        scaler, model, x[inner_test]
                    )
                if valid and np.isfinite(inner_predictions).all():
                    candidates.append(
                        (
                            float(
                                log_loss(
                                    y[outer_train], inner_predictions, labels=[0, 1]
                                )
                            ),
                            layer,
                            c_value,
                        )
                    )
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
        }

    oof_rows = []
    for index, row in enumerate(judgments):
        oof_rows.append(
            {
                "conversation_hash": row["conversation_hash"],
                "generation_sha256": row["generation_sha256"],
                "repetition": int(row["repetition"]),
                "fold": int(folds[index]),
                "endorsement": int(y[index]),
                "oof_probability": float(predictions[index]),
                "fold_baseline_probability": float(baseline_predictions[index]),
                "selected_layer": selected[int(folds[index])]["layer"],
            }
        )
    write_jsonl(args.oof_predictions, oof_rows)

    trajectory_rows = []
    last_step_by_group: dict[str, list[float]] = defaultdict(list)
    for conversation_hash in unique_groups:
        cohort_row = cohort[conversation_hash]
        checkpoint = np.load(
            args.checkpoint_dir / f"outer_fold_{int(cohort_row['fold'])}.npz"
        )
        layer = str(checkpoint["layer"].item())
        direction = checkpoint["raw_direction"].astype(np.float64)
        intercept = float(checkpoint["raw_intercept"].item())
        values = activation_files[conversation_hash][layer].astype(np.float64)
        logits = values @ direction + intercept
        probabilities = expit(logits)
        message_indices = activation_files[conversation_hash]["message_indices"]
        token_positions = activation_files[conversation_hash]["token_positions"]
        prior_score = prior_scores[conversation_hash]
        prior_group = (
            "prior_assistant_endorsing"
            if prior_score >= ENDORSEMENT_THRESHOLD
            else "other"
        )
        rate = float(np.mean(labels[conversation_hash]))
        if len(probabilities) >= 2:
            last_step_by_group[prior_group].append(
                float(probabilities[-1] - probabilities[-2])
            )
        for user_turn, (message_index, token_position, logit, probability) in enumerate(
            zip(
                message_indices,
                token_positions,
                logits,
                probabilities,
                strict=True,
            )
        ):
            preceding_assistant_index = int(message_index) - 1
            trajectory_rows.append(
                {
                    "conversation_hash": conversation_hash,
                    "original_row_idx": cohort_row["original_row_idx"],
                    "source": cohort_row["source"],
                    "fold": int(cohort_row["fold"]),
                    "selected_layer": layer,
                    "user_turn_index": user_turn,
                    "user_turns_before_target": user_turn - (len(probabilities) - 1),
                    "message_index": int(message_index),
                    "token_position": int(token_position),
                    "user_text": cohort_row["messages"][int(message_index)]["content"],
                    "preceding_assistant_index": preceding_assistant_index,
                    "preceding_assistant_text": (
                        cohort_row["messages"][preceding_assistant_index]["content"]
                        if preceding_assistant_index >= 0
                        and cohort_row["messages"][preceding_assistant_index]["role"]
                        == "assistant"
                        else None
                    ),
                    "probe_logit": float(logit),
                    "probe_probability": float(probability),
                    "final_endorsement_rate": rate,
                    "latest_prior_assistant_score": prior_score,
                    "latest_prior_assistant_group": prior_group,
                    "is_target_turn": user_turn == len(probabilities) - 1,
                }
            )
    write_jsonl(args.trajectory_rows, trajectory_rows)

    summary = {
        "design": (
            "Nested grouped five-fold probe trained at the header-only assistant "
            "generation boundary of each final target and projected onto every exact "
            "historical user-turn boundary in held-out conversations"
        ),
        "cohort_sha256": sha256_file(args.cohort),
        "responses": len(y),
        "conversations": len(unique_groups),
        "turn_boundaries": len(trajectory_rows),
        "layers_available": layer_keys,
        "outer_fold_selection": selected,
        "oof_metrics": safe_metrics(y, predictions),
        "baseline_metrics": safe_metrics(y, baseline_predictions),
        "bootstrap_95_ci_by_conversation": bootstrap_metrics(
            y, predictions, groups, iterations=args.bootstrap
        ),
        "last_decision_step_probability_change": {},
        "scope_warning": (
            "Earlier boundary projections are held-out retrospective susceptibility "
            "scores, not calibrated probabilities or causal effects of individual turns."
        ),
    }
    for group, values in last_step_by_group.items():
        array = np.asarray(values)
        summary["last_decision_step_probability_change"][group] = {
            "conversations": len(array),
            "mean": float(array.mean()),
            "bootstrap_95_ci": bootstrap_mean(array, args.bootstrap),
            "fraction_positive": float((array > 0).mean()),
        }
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    args.summary.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
