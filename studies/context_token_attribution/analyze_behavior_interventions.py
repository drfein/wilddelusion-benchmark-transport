from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from config import ENDORSEMENT_THRESHOLD, SEED
from io_utils import read_jsonl, sha256_file, write_jsonl

CONDITIONS = ("top_assistant_deleted", "matched_assistant_deleted")


def judged_labels(path: Path) -> list[dict]:
    return [
        row
        for row in read_jsonl(path)
        if isinstance(row.get("annotation_score"), int) and not row.get("judge_error")
    ]


def bootstrap(values: np.ndarray, iterations: int) -> list[float]:
    rng = np.random.default_rng(SEED)
    estimates = np.empty(iterations, dtype=np.float64)
    for index in range(iterations):
        sample = rng.integers(0, len(values), len(values))
        estimates[index] = values[sample].mean()
    return [float(np.quantile(estimates, 0.025)), float(np.quantile(estimates, 0.975))]


def sign_flip_p_value(values: np.ndarray, iterations: int) -> float:
    """Cluster-level randomization test under a symmetric sharp null."""
    rng = np.random.default_rng(SEED + 1)
    observed = abs(float(values.mean()))
    extreme = 0
    completed = 0
    batch_size = 10_000
    while completed < iterations:
        current = min(batch_size, iterations - completed)
        signs = rng.choice((-1.0, 1.0), size=(current, len(values)))
        estimates = np.abs((signs * values).mean(axis=1))
        extreme += int(np.count_nonzero(estimates >= observed - 1e-12))
        completed += current
    return float((extreme + 1) / (iterations + 1))


def directional_sign_flip_p_value(values: np.ndarray, iterations: int) -> float:
    """One-sided test for the pre-specified hypothesis that deletion lowers score."""
    rng = np.random.default_rng(SEED + 2)
    observed = float(values.mean())
    extreme = 0
    completed = 0
    batch_size = 10_000
    while completed < iterations:
        current = min(batch_size, iterations - completed)
        signs = rng.choice((-1.0, 1.0), size=(current, len(values)))
        estimates = (signs * values).mean(axis=1)
        extreme += int(np.count_nonzero(estimates <= observed + 1e-12))
        completed += current
    return float((extreme + 1) / (iterations + 1))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--original-judgments", type=Path, required=True)
    parser.add_argument("--intervention-judgments", type=Path, required=True)
    parser.add_argument("--selections", type=Path, required=True)
    parser.add_argument("--cohort", type=Path)
    parser.add_argument("--rows-output", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--bootstrap", type=int, default=10_000)
    parser.add_argument("--randomization-iterations", type=int, default=200_000)
    args = parser.parse_args()

    selections = {row["conversation_hash"]: row for row in read_jsonl(args.selections)}
    cohort = (
        {row["conversation_hash"]: row for row in read_jsonl(args.cohort)}
        if args.cohort
        else {}
    )
    original_values: dict[str, dict[int, tuple[int, int]]] = defaultdict(dict)
    for row in judged_labels(args.original_judgments):
        if row["conversation_hash"] in selections:
            original_values[row["conversation_hash"]][int(row["repetition"])] = (
                int(row["annotation_score"] >= ENDORSEMENT_THRESHOLD),
                int(row["annotation_score"]),
            )

    intervention_values: dict[tuple[str, str], dict[int, tuple[int, int]]] = (
        defaultdict(dict)
    )
    for row in judged_labels(args.intervention_judgments):
        condition = row.get("condition")
        if condition in CONDITIONS and row["conversation_hash"] in selections:
            intervention_values[(row["conversation_hash"], condition)][
                int(row["repetition"])
            ] = (
                int(row["annotation_score"] >= ENDORSEMENT_THRESHOLD),
                int(row["annotation_score"]),
            )

    rows = []
    for conversation_hash, selection in sorted(selections.items()):
        original = original_values[conversation_hash]
        top = intervention_values[(conversation_hash, "top_assistant_deleted")]
        matched = intervention_values[(conversation_hash, "matched_assistant_deleted")]
        expected_repetitions = set(range(5))
        if not (set(original) == set(top) == set(matched) == expected_repetitions):
            raise ValueError(
                f"Incomplete intervention labels for {conversation_hash}: "
                f"{sorted(original)}, {sorted(top)}, {sorted(matched)}"
            )
        original_rate = float(np.mean([original[index][0] for index in range(5)]))
        top_rate = float(np.mean([top[index][0] for index in range(5)]))
        matched_rate = float(np.mean([matched[index][0] for index in range(5)]))
        original_score = float(np.mean([original[index][1] for index in range(5)]))
        top_score = float(np.mean([top[index][1] for index in range(5)]))
        matched_score = float(np.mean([matched[index][1] for index in range(5)]))
        rows.append(
            {
                **selection,
                **(
                    {"source": cohort[conversation_hash].get("source")}
                    if conversation_hash in cohort
                    else {}
                ),
                "original_rate": original_rate,
                "top_deleted_rate": top_rate,
                "matched_deleted_rate": matched_rate,
                "top_minus_matched": top_rate - matched_rate,
                "top_minus_original": top_rate - original_rate,
                "matched_minus_original": matched_rate - original_rate,
                "original_mean_score": original_score,
                "top_deleted_mean_score": top_score,
                "matched_deleted_mean_score": matched_score,
                "top_minus_matched_score": top_score - matched_score,
                "top_minus_original_score": top_score - original_score,
                "matched_minus_original_score": matched_score - original_score,
            }
        )

    def effect(key: str) -> dict:
        values = np.asarray([row[key] for row in rows], dtype=np.float64)
        return {
            "mean": float(values.mean()),
            "bootstrap_95_ci_by_conversation": bootstrap(values, args.bootstrap),
        }

    primary_values = np.asarray(
        [row["top_minus_matched"] for row in rows], dtype=np.float64
    )
    primary_score_values = np.asarray(
        [row["top_minus_matched_score"] for row in rows], dtype=np.float64
    )
    direction_counts = {
        "top_deletion_lower": int(np.count_nonzero(primary_values < 0)),
        "equal": int(np.count_nonzero(primary_values == 0)),
        "top_deletion_higher": int(np.count_nonzero(primary_values > 0)),
    }
    source_effects = {}
    for source in sorted({row.get("source") for row in rows if row.get("source")}):
        source_rows = [row for row in rows if row.get("source") == source]
        source_effects[source] = {
            "n_conversations": len(source_rows),
            "top_minus_matched": float(
                np.mean([row["top_minus_matched"] for row in source_rows])
            ),
            "top_minus_original": float(
                np.mean([row["top_minus_original"] for row in source_rows])
            ),
        }

    def distribution(key: str) -> dict:
        values = np.asarray([row[key] for row in rows], dtype=np.float64)
        return {
            "mean": float(values.mean()),
            "median": float(np.median(values)),
            "minimum": float(values.min()),
            "maximum": float(values.max()),
        }

    summary = {
        "original_judgments_sha256": sha256_file(args.original_judgments),
        "intervention_judgments_sha256": sha256_file(args.intervention_judgments),
        "selections_sha256": sha256_file(args.selections),
        "paired_conversations": len(rows),
        "samples_per_conversation_condition": 5,
        "estimand": (
            "Among selected conversations, change in endorsement caused by deleting "
            "the attributed assistant message versus deleting a within-conversation "
            "assistant control message. Conversation is the bootstrap and "
            "randomization unit."
        ),
        "matching_diagnostics": {
            "top_tokens": distribution("top_tokens"),
            "control_tokens": distribution("control_tokens"),
            "top_relative_position": distribution("top_relative_position"),
            "control_relative_position": distribution("control_relative_position"),
            "match_distance": distribution("match_distance"),
        },
        "endorsement_rates": {
            "original": float(np.mean([row["original_rate"] for row in rows])),
            "top_assistant_deleted": float(
                np.mean([row["top_deleted_rate"] for row in rows])
            ),
            "matched_assistant_deleted": float(
                np.mean([row["matched_deleted_rate"] for row in rows])
            ),
        },
        "primary_top_minus_matched": effect("top_minus_matched"),
        "primary_two_sided_cluster_sign_flip_p": sign_flip_p_value(
            primary_values, args.randomization_iterations
        ),
        "primary_directional_cluster_sign_flip_p": directional_sign_flip_p_value(
            primary_values, args.randomization_iterations
        ),
        "primary_direction_counts": direction_counts,
        "top_minus_original": effect("top_minus_original"),
        "matched_minus_original": effect("matched_minus_original"),
        "secondary_ordinal_score_effects": {
            "top_minus_matched": effect("top_minus_matched_score"),
            "top_minus_matched_two_sided_cluster_sign_flip_p": sign_flip_p_value(
                primary_score_values, args.randomization_iterations
            ),
            "top_minus_matched_directional_cluster_sign_flip_p": (
                directional_sign_flip_p_value(
                    primary_score_values, args.randomization_iterations
                )
            ),
            "top_minus_original": effect("top_minus_original_score"),
            "matched_minus_original": effect("matched_minus_original_score"),
        },
        "source_stratified_effects": source_effects,
        "interpretation": (
            "Negative top-minus-matched supports faithfulness of the assistant-message "
            "attribution; a confidence interval spanning zero falsifies the primary "
            "assistant-context hypothesis."
        ),
    }
    write_jsonl(args.rows_output, rows)
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    args.summary.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
