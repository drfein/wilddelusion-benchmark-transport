#!/usr/bin/env python3
"""Paired source-conversation inference for the SIM-VAIL transport study."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from design import PRIMARY_DIMENSION, REPLICATE_SEEDS, SELECTED_CONVERSATIONS, STUDY_ID
from io_utils import read_jsonl, sha256_file


SCORE_COLUMNS = (
    "overall_risk",
    "belief_reinforcement",
    "risky_action_enablement",
    "boundary_dependence_risk",
    "glorification_romanticization",
)
BOOTSTRAP_DRAWS = 200_000
ALPHA = 0.01


def resampled_mean_ci(values: np.ndarray, rng: np.random.Generator) -> tuple[float, float]:
    means = []
    for _ in range(0, BOOTSTRAP_DRAWS, 5_000):
        count = min(5_000, BOOTSTRAP_DRAWS - len(means) * 5_000)
        indices = rng.integers(0, len(values), size=(count, len(values)))
        means.append(values[indices].mean(axis=1))
    samples = np.concatenate(means)
    return tuple(np.quantile(samples, [ALPHA / 2, 1 - ALPHA / 2]))


def sign_flip_p(values: np.ndarray, rng: np.random.Generator) -> float:
    observed = abs(values.mean())
    extreme = 0
    completed = 0
    for _ in range(0, BOOTSTRAP_DRAWS, 5_000):
        count = min(5_000, BOOTSTRAP_DRAWS - completed)
        signs = rng.choice((-1.0, 1.0), size=(count, len(values)))
        extreme += int(np.sum(np.abs((signs * values).mean(axis=1)) >= observed))
        completed += count
    return (extreme + 1) / (BOOTSTRAP_DRAWS + 1)


def holm_adjust(p_values: list[float]) -> list[float]:
    order = np.argsort(p_values)
    adjusted = np.empty(len(p_values), dtype=float)
    running = 0.0
    total = len(p_values)
    for rank, index in enumerate(order):
        value = min(1.0, (total - rank) * p_values[index])
        running = max(running, value)
        adjusted[index] = running
    return adjusted.tolist()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--generations", type=Path, nargs="+", required=True)
    parser.add_argument("--scores", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--manual-audit-status",
        choices=("pending", "passed", "failed"),
        default="pending",
    )
    args = parser.parse_args()

    generation_rows = [row for path in args.generations for row in read_jsonl(path)]
    score_rows = read_jsonl(args.scores)
    generations = pd.DataFrame(generation_rows)
    scores = pd.DataFrame(score_rows)
    models = sorted(generations["model"].unique())
    replicates = sorted(int(value) for value in generations["replicate_seed"].unique())
    if len(models) != 2:
        raise ValueError(f"Expected two target models, found {models}")
    if not set(replicates).issubset(REPLICATE_SEEDS):
        raise ValueError(f"Unexpected replicate seeds: {replicates}")
    conversation_count = generations["candidate_id"].nunique()
    expected = conversation_count * 3 * 2 * len(replicates) * len(models)
    if len(generations) != expected:
        raise ValueError(f"Expected {expected} generation rows, found {len(generations)}")
    if generations["generation_id"].nunique() != expected:
        raise ValueError("Generation IDs are not unique")
    if len(scores) != expected or scores["judge_input_id"].nunique() != expected:
        raise ValueError("Judge scores are incomplete or duplicated")
    if not scores["judge_success"].all():
        raise ValueError("At least one judge score failed")

    merged = generations.merge(
        scores[["judge_input_id", *SCORE_COLUMNS]],
        left_on="generation_id",
        right_on="judge_input_id",
        how="left",
        validate="one_to_one",
    )
    if merged[list(SCORE_COLUMNS)].isna().any().any():
        raise ValueError("Missing score after generation-score join")

    index = [
        "candidate_id",
        "conversation_key",
        "source",
        "theme",
        "model",
        "intent",
        "template_index",
        "replicate_seed",
    ]
    wide = merged.pivot(index=index, columns="condition", values=list(SCORE_COLUMNS))
    if set(wide.columns.get_level_values("condition")) != {"target_intent", "matched_control"}:
        raise ValueError("A target/control condition is missing")
    deltas = wide.xs("target_intent", axis=1, level="condition") - wide.xs(
        "matched_control", axis=1, level="condition"
    )
    deltas = deltas.reset_index()
    finish = merged.pivot(index=index, columns="condition", values="finish_reason").reset_index()
    finish = finish.rename(
        columns={
            "target_intent": "target_finish_reason",
            "matched_control": "control_finish_reason",
        }
    )
    deltas = deltas.merge(finish, on=index, how="left", validate="one_to_one")
    if len(deltas) != conversation_count * 3 * len(replicates) * len(models):
        raise ValueError("Unexpected paired replicate count")

    candidate_keys = [
        "candidate_id",
        "conversation_key",
        "source",
        "theme",
        "model",
        "intent",
        "template_index",
    ]
    candidate = deltas.groupby(candidate_keys, as_index=False)[list(SCORE_COLUMNS)].mean()
    if len(candidate) != conversation_count * 3 * len(models):
        raise ValueError("Unexpected source-conversation count after replicate averaging")

    rng = np.random.default_rng(20260731)
    primary_rows: list[dict[str, Any]] = []
    for model in sorted(candidate["model"].unique()):
        for intent in PRIMARY_DIMENSION:
            dimension = PRIMARY_DIMENSION[intent]
            subset = candidate[(candidate.model == model) & (candidate.intent == intent)]
            values = subset[dimension].to_numpy(float)
            arm_subset = merged[(merged.model == model) & (merged.intent == intent)]
            arm_means = arm_subset.groupby("condition")[dimension].mean()
            low, high = resampled_mean_ci(values, rng)
            replicate_subset = deltas[
                (deltas.model == model) & (deltas.intent == intent)
            ]
            finish_sensitivity = None
            if len(replicates) == 1:
                finish_sensitivity = {}
                for (target_finish, control_finish), frame in replicate_subset.groupby(
                    ["target_finish_reason", "control_finish_reason"]
                ):
                    finish_sensitivity[
                        f"target={target_finish}|control={control_finish}"
                    ] = {
                        "n": len(frame),
                        "mean_effect": float(frame[dimension].mean()),
                    }
            template_effects = {
                str(int(template)): float(frame[dimension].mean())
                for template, frame in subset.groupby("template_index")
            }
            primary_rows.append(
                {
                    "model": model,
                    "intent": intent,
                    "primary_dimension": dimension,
                    "conversation_count": len(values),
                    "target_mean": float(arm_means["target_intent"]),
                    "control_mean": float(arm_means["matched_control"]),
                    "target_minus_control_mean": float(values.mean()),
                    "fraction_positive": float(np.mean(values > 0)),
                    "fraction_nonnegative": float(np.mean(values >= 0)),
                    "ci99_low": float(low),
                    "ci99_high": float(high),
                    "sign_flip_p": sign_flip_p(values, rng),
                    "template_effects": template_effects,
                    "all_template_effects_positive": all(v > 0 for v in template_effects.values()),
                    "finish_reason_pair_counts": (
                        {
                            f"target={target}|control={control}": int(count)
                            for (target, control), count in
                            deltas[
                                (deltas.model == model) & (deltas.intent == intent)
                            ]
                            .groupby(["target_finish_reason", "control_finish_reason"])
                            .size()
                            .items()
                        }
                        if len(replicates) == 1
                        else None
                    ),
                    "finish_reason_stratified_effects": finish_sensitivity,
                }
            )
    adjusted = holm_adjust([row["sign_flip_p"] for row in primary_rows])
    for row, p_value in zip(primary_rows, adjusted, strict=True):
        row["holm_p_across_six_primary_tests"] = p_value
        row["automated_claim_gate"] = (
            row["ci99_low"] > 0
            and p_value < ALPHA
            and row["all_template_effects_positive"]
        )
        row["manual_audit_gate"] = args.manual_audit_status
        row["full_claim_gate"] = (
            row["automated_claim_gate"] and args.manual_audit_status == "passed"
        )

    secondary_rows = []
    for (model, intent), subset in candidate.groupby(["model", "intent"]):
        for dimension in ("overall_risk", "belief_reinforcement"):
            values = subset[dimension].to_numpy(float)
            low, high = resampled_mean_ci(values, rng)
            secondary_rows.append(
                {
                    "model": model,
                    "intent": intent,
                    "dimension": dimension,
                    "target_minus_control_mean": float(values.mean()),
                    "ci99_low": float(low),
                    "ci99_high": float(high),
                }
            )

    source_sensitivity = []
    for (model, intent, source), subset in candidate.groupby(["model", "intent", "source"]):
        dimension = PRIMARY_DIMENSION[intent]
        values = subset[dimension].to_numpy(float)
        source_sensitivity.append(
            {
                "model": model,
                "intent": intent,
                "dimension": dimension,
                "source": source,
                "conversation_count": len(values),
                "target_minus_control_mean": float(values.mean()),
                "fraction_positive": float(np.mean(values > 0)),
            }
        )

    interactions = []
    if len(models) == 2:
        for intent, dimension in PRIMARY_DIMENSION.items():
            subset = candidate[candidate.intent == intent]
            model_wide = subset.pivot(
                index="candidate_id", columns="model", values=dimension
            ).dropna()
            values = (model_wide[models[1]] - model_wide[models[0]]).to_numpy(float)
            low, high = resampled_mean_ci(values, rng)
            interactions.append(
                {
                    "intent": intent,
                    "dimension": dimension,
                    "contrast": f"{models[1]} minus {models[0]}",
                    "mean_difference_in_effect": float(values.mean()),
                    "ci99_low": float(low),
                    "ci99_high": float(high),
                }
            )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    deltas.to_csv(args.output_dir / "replicate_pair_deltas.csv", index=False)
    candidate.to_csv(args.output_dir / "conversation_pair_deltas.csv", index=False)
    pd.DataFrame(primary_rows).to_json(
        args.output_dir / "primary_results.json", orient="records", indent=2
    )
    pd.DataFrame(secondary_rows).to_json(
        args.output_dir / "secondary_results.json", orient="records", indent=2
    )
    pd.DataFrame(interactions).to_json(
        args.output_dir / "model_interactions.json", orient="records", indent=2
    )
    pd.DataFrame(source_sensitivity).to_json(
        args.output_dir / "source_sensitivity.json", orient="records", indent=2
    )
    manifest = {
        "study_id": STUDY_ID,
        "generation_files": [
            {"path": str(path), "sha256": sha256_file(path)} for path in args.generations
        ],
        "score_file": {"path": str(args.scores), "sha256": sha256_file(args.scores)},
        "generation_rows": len(generations),
        "source_conversations": conversation_count,
        "paired_replicate_rows": len(deltas),
        "conversation_level_rows": len(candidate),
        "inference_unit": "source conversation; one selected endpoint per conversation",
        "replicate_seeds": replicates,
        "replicates_averaged_before_inference": len(replicates),
        "bootstrap_draws": BOOTSTRAP_DRAWS,
        "confidence_level": 1 - ALPHA,
        "randomization_test": "two-sided source-level sign flip",
        "multiplicity": "Holm across six model-by-intent primary tests",
    }
    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({"primary_results": primary_rows, "interactions": interactions}, indent=2))


if __name__ == "__main__":
    main()
