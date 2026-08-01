from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from config import ENDORSEMENT_THRESHOLD, SEED
from io_utils import read_jsonl, sha256_file, write_jsonl
from scipy.stats import spearmanr


def label_rows(path: Path) -> dict[tuple[str, int], tuple[int, int]]:
    result = {}
    for row in read_jsonl(path):
        if isinstance(row.get("annotation_score"), int) and not row.get("judge_error"):
            key = (row["conversation_hash"], int(row["repetition"]))
            result[key] = (
                int(row["annotation_score"] >= ENDORSEMENT_THRESHOLD),
                int(row["annotation_score"]),
            )
    return result


def bootstrap_mean(values: np.ndarray, iterations: int) -> list[float]:
    rng = np.random.default_rng(SEED)
    estimates = np.empty(iterations, dtype=np.float64)
    for index in range(iterations):
        sample = rng.integers(0, len(values), len(values))
        estimates[index] = values[sample].mean()
    return [float(np.quantile(estimates, 0.025)), float(np.quantile(estimates, 0.975))]


def bootstrap_spearman(x: np.ndarray, y: np.ndarray, iterations: int) -> list[float]:
    rng = np.random.default_rng(SEED)
    estimates = []
    for _ in range(iterations):
        sample = rng.integers(0, len(x), len(x))
        value = spearmanr(x[sample], y[sample]).statistic
        if np.isfinite(value):
            estimates.append(value)
    return [float(np.quantile(estimates, 0.025)), float(np.quantile(estimates, 0.975))]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cohort", type=Path, required=True)
    parser.add_argument("--full-judgments", type=Path, required=True)
    parser.add_argument("--target-only-judgments", type=Path, required=True)
    parser.add_argument("--token-attribution-conversations", type=Path)
    parser.add_argument("--rows-output", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--bootstrap", type=int, default=10_000)
    args = parser.parse_args()

    cohort = {row["conversation_hash"]: row for row in read_jsonl(args.cohort)}
    full = label_rows(args.full_judgments)
    target = label_rows(args.target_only_judgments)
    expected = {
        (conversation_hash, repetition)
        for conversation_hash in cohort
        for repetition in range(5)
    }
    if set(full) != expected or set(target) != expected:
        raise ValueError(
            f"Incomplete paired labels: full={len(full)}, target={len(target)}, "
            f"expected={len(expected)}"
        )

    by_conversation: dict[str, list[tuple[int, int, int, int]]] = defaultdict(list)
    for conversation_hash, repetition in sorted(expected):
        by_conversation[conversation_hash].append(
            (
                full[(conversation_hash, repetition)][0],
                target[(conversation_hash, repetition)][0],
                full[(conversation_hash, repetition)][1],
                target[(conversation_hash, repetition)][1],
            )
        )
    rows = []
    for conversation_hash in sorted(by_conversation):
        values = np.asarray(by_conversation[conversation_hash], dtype=np.float64)
        full_rate = float(values[:, 0].mean())
        target_rate = float(values[:, 1].mean())
        full_score = float(values[:, 2].mean())
        target_score = float(values[:, 3].mean())
        source = cohort[conversation_hash]
        rows.append(
            {
                "original_row_idx": source["original_row_idx"],
                "conversation_hash": conversation_hash,
                "full_context_rate": full_rate,
                "target_only_rate": target_rate,
                "full_minus_target_only": full_rate - target_rate,
                "full_context_mean_score": full_score,
                "target_only_mean_score": target_score,
                "full_minus_target_only_score": full_score - target_score,
                "input_tokens_full": source["input_tokens"],
                "prior_messages": len(source["messages"]) - 2,
            }
        )

    differences = np.asarray([row["full_minus_target_only"] for row in rows])
    full_rates = np.asarray([row["full_context_rate"] for row in rows])
    target_rates = np.asarray([row["target_only_rate"] for row in rows])
    score_differences = np.asarray(
        [row["full_minus_target_only_score"] for row in rows]
    )
    summary = {
        "cohort_sha256": sha256_file(args.cohort),
        "full_judgments_sha256": sha256_file(args.full_judgments),
        "target_only_judgments_sha256": sha256_file(args.target_only_judgments),
        "conversations": len(rows),
        "responses_per_condition": len(full),
        "full_context_endorsement_rate": float(full_rates.mean()),
        "target_only_endorsement_rate": float(target_rates.mean()),
        "full_minus_target_only": {
            "mean": float(differences.mean()),
            "bootstrap_95_ci_by_conversation": bootstrap_mean(
                differences, args.bootstrap
            ),
        },
        "secondary_ordinal_score_effect": {
            "mean": float(score_differences.mean()),
            "bootstrap_95_ci_by_conversation": bootstrap_mean(
                score_differences, args.bootstrap
            ),
        },
        "conversations_full_higher": int((differences > 0).sum()),
        "conversations_equal": int((differences == 0).sum()),
        "conversations_target_only_higher": int((differences < 0).sum()),
        "design": (
            "paired system+target-only ablation with five common-random-number "
            "sample slots per condition; inference resamples conversations"
        ),
    }

    sorted_by_length = sorted(rows, key=lambda row: row["input_tokens_full"])
    length_quartiles = np.array_split(np.asarray(sorted_by_length, dtype=object), 4)
    lengths = np.asarray([row["input_tokens_full"] for row in rows], dtype=np.float64)
    summary["exploratory_context_length_gradient"] = {
        "spearman_full_minus_target_effect_vs_input_tokens": {
            "r": float(spearmanr(lengths, differences).statistic),
            "bootstrap_95_ci_by_conversation": bootstrap_spearman(
                lengths, differences, args.bootstrap
            ),
        },
        "quartiles": [
            {
                "conversations": len(quartile),
                "median_full_input_tokens": float(
                    np.median([row["input_tokens_full"] for row in quartile])
                ),
                "full_context_rate": float(
                    np.mean([row["full_context_rate"] for row in quartile])
                ),
                "target_only_rate": float(
                    np.mean([row["target_only_rate"] for row in quartile])
                ),
                "full_minus_target_only": float(
                    np.mean([row["full_minus_target_only"] for row in quartile])
                ),
            }
            for quartile in length_quartiles
        ],
        "status": (
            "exploratory association; history length is not randomized and may proxy "
            "for accumulated content"
        ),
    }

    if args.token_attribution_conversations:
        attribution = {
            row["conversation_hash"]: row
            for row in read_jsonl(args.token_attribution_conversations)
            if row["method"] == "gradient_times_input"
        }
        joined = [row for row in rows if row["conversation_hash"] in attribution]
        x = np.asarray(
            [
                attribution[row["conversation_hash"]]["assistant_minus_token_share"]
                for row in joined
            ],
            dtype=np.float64,
        )
        y = np.asarray(
            [row["full_minus_target_only"] for row in joined], dtype=np.float64
        )
        finite = np.isfinite(x) & np.isfinite(y)
        rho = float(spearmanr(x[finite], y[finite]).statistic)
        summary["assistant_attribution_vs_causal_context_effect"] = {
            "conversations": int(finite.sum()),
            "spearman_r": rho,
            "bootstrap_95_ci_by_conversation": bootstrap_spearman(
                x[finite], y[finite], args.bootstrap
            ),
            "attribution_variable": (
                "assistant positive-attribution share minus assistant token share"
            ),
            "effect_variable": "full-context minus target-only endorsement rate",
            "status": "exploratory association; not itself a causal attribution test",
        }

    write_jsonl(args.rows_output, rows)
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    args.summary.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
