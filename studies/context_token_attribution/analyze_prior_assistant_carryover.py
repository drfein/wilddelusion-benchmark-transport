from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from config import ENDORSEMENT_THRESHOLD, SEED
from io_utils import read_jsonl, sha256_file, write_jsonl
from scipy.stats import spearmanr


def bootstrap_mean(values: np.ndarray, iterations: int) -> list[float]:
    rng = np.random.default_rng(SEED)
    estimates = np.empty(iterations, dtype=np.float64)
    for index in range(iterations):
        sample = rng.integers(0, len(values), len(values))
        estimates[index] = values[sample].mean()
    return [float(value) for value in np.quantile(estimates, [0.025, 0.975])]


def bootstrap_spearman(x: np.ndarray, y: np.ndarray, iterations: int) -> list[float]:
    rng = np.random.default_rng(SEED)
    estimates = []
    for _ in range(iterations):
        sample = rng.integers(0, len(x), len(x))
        value = spearmanr(x[sample], y[sample]).statistic
        if np.isfinite(value):
            estimates.append(value)
    return [float(value) for value in np.quantile(estimates, [0.025, 0.975])]


def group_summary(rows: list[dict], iterations: int) -> dict:
    effects = np.asarray([row["full_minus_target_only"] for row in rows])
    return {
        "conversations": len(rows),
        "mean_context_effect": float(effects.mean()),
        "bootstrap_95_ci_by_conversation": bootstrap_mean(effects, iterations),
        "mean_prior_assistant_score": float(
            np.mean([row["prior_assistant_score"] for row in rows])
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prior-assistant-judgments", type=Path, required=True)
    parser.add_argument("--behavior-rows", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--bootstrap", type=int, default=10_000)
    args = parser.parse_args()

    prior = {
        row["conversation_hash"]: row
        for row in read_jsonl(args.prior_assistant_judgments)
        if isinstance(row.get("annotation_score"), int) and not row.get("judge_error")
    }
    behavior = {row["conversation_hash"]: row for row in read_jsonl(args.behavior_rows)}
    if set(prior) != set(behavior):
        raise ValueError(
            f"Mismatched rows: prior={len(prior)}, behavior={len(behavior)}"
        )
    rows = []
    for conversation_hash in sorted(prior):
        label = prior[conversation_hash]
        effect = behavior[conversation_hash]
        rows.append(
            {
                "conversation_hash": conversation_hash,
                "source": label["source"],
                "prior_assistant_score": int(label["annotation_score"]),
                "prior_assistant_endorsement": int(
                    label["annotation_score"] >= ENDORSEMENT_THRESHOLD
                ),
                "full_context_rate": effect["full_context_rate"],
                "target_only_rate": effect["target_only_rate"],
                "full_minus_target_only": effect["full_minus_target_only"],
                "input_tokens_full": effect["input_tokens_full"],
            }
        )

    scores = np.asarray([row["prior_assistant_score"] for row in rows])
    effects = np.asarray([row["full_minus_target_only"] for row in rows])
    endorsed = [row for row in rows if row["prior_assistant_endorsement"]]
    not_endorsed = [row for row in rows if not row["prior_assistant_endorsement"]]
    by_source: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        by_source[row["source"]].append(row)
    summary = {
        "prior_assistant_judgments_sha256": sha256_file(args.prior_assistant_judgments),
        "behavior_rows_sha256": sha256_file(args.behavior_rows),
        "conversations": len(rows),
        "spearman_prior_assistant_score_vs_causal_context_effect": {
            "r": float(spearmanr(scores, effects).statistic),
            "bootstrap_95_ci_by_conversation": bootstrap_spearman(
                scores, effects, args.bootstrap
            ),
        },
        "by_prior_assistant_endorsement": {
            "endorsing": group_summary(endorsed, args.bootstrap),
            "not_endorsing": group_summary(not_endorsed, args.bootstrap),
        },
        "by_source": {
            source: group_summary(values, args.bootstrap)
            for source, values in sorted(by_source.items())
        },
        "by_source_and_prior_assistant_endorsement": {
            source: {
                "endorsing": group_summary(
                    [row for row in values if row["prior_assistant_endorsement"]],
                    args.bootstrap,
                ),
                "not_endorsing": group_summary(
                    [row for row in values if not row["prior_assistant_endorsement"]],
                    args.bootstrap,
                ),
            }
            for source, values in sorted(by_source.items())
            if any(row["prior_assistant_endorsement"] for row in values)
            and any(not row["prior_assistant_endorsement"] for row in values)
        },
        "interpretation": (
            "Association of the preceding assistant's endorsement score with the "
            "paired full-minus-target behavior effect. The matched message-deletion "
            "experiment is required for assistant-specific causation."
        ),
    }
    write_jsonl(args.output, rows)
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    args.summary.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
