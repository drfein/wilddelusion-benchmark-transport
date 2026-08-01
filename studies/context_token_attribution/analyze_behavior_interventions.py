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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--original-judgments", type=Path, required=True)
    parser.add_argument("--intervention-judgments", type=Path, required=True)
    parser.add_argument("--selections", type=Path, required=True)
    parser.add_argument("--rows-output", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--bootstrap", type=int, default=10_000)
    args = parser.parse_args()

    selections = {row["conversation_hash"]: row for row in read_jsonl(args.selections)}
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

    summary = {
        "original_judgments_sha256": sha256_file(args.original_judgments),
        "intervention_judgments_sha256": sha256_file(args.intervention_judgments),
        "selections_sha256": sha256_file(args.selections),
        "paired_conversations": len(rows),
        "samples_per_conversation_condition": 5,
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
        "top_minus_original": effect("top_minus_original"),
        "matched_minus_original": effect("matched_minus_original"),
        "secondary_ordinal_score_effects": {
            "top_minus_matched": effect("top_minus_matched_score"),
            "top_minus_original": effect("top_minus_original_score"),
            "matched_minus_original": effect("matched_minus_original_score"),
        },
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
