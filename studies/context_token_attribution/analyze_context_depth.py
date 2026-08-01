from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from config import ENDORSEMENT_THRESHOLD, SEED
from io_utils import read_jsonl, sha256_file, write_jsonl


def judged(path: Path) -> dict[tuple[str, int], tuple[int, int]]:
    result = {}
    for row in read_jsonl(path):
        if isinstance(row.get("annotation_score"), int) and not row.get("judge_error"):
            result[(row["conversation_hash"], int(row["repetition"]))] = (
                int(row["annotation_score"] >= ENDORSEMENT_THRESHOLD),
                int(row["annotation_score"]),
            )
    return result


def bootstrap(values: np.ndarray, iterations: int) -> list[float]:
    rng = np.random.default_rng(SEED)
    estimates = np.empty(iterations, dtype=np.float64)
    for index in range(iterations):
        sample = rng.integers(0, len(values), len(values))
        estimates[index] = values[sample].mean()
    return [float(value) for value in np.quantile(estimates, [0.025, 0.975])]


def effect(rows: list[dict], key: str, iterations: int) -> dict:
    values = np.asarray([row[key] for row in rows], dtype=np.float64)
    return {
        "mean": float(values.mean()),
        "bootstrap_95_ci_by_conversation": bootstrap(values, iterations),
        "conversations_positive": int((values > 0).sum()),
        "conversations_zero": int((values == 0).sum()),
        "conversations_negative": int((values < 0).sum()),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cohort", type=Path, required=True)
    parser.add_argument("--full", type=Path, required=True)
    parser.add_argument("--last-exchange", type=Path, required=True)
    parser.add_argument("--target-only", type=Path, required=True)
    parser.add_argument("--rows-output", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--bootstrap", type=int, default=10_000)
    args = parser.parse_args()

    cohort = {row["conversation_hash"]: row for row in read_jsonl(args.cohort)}
    conditions = {
        "full": judged(args.full),
        "last_exchange": judged(args.last_exchange),
        "target_only": judged(args.target_only),
    }
    expected = {
        (conversation_hash, repetition)
        for conversation_hash in cohort
        for repetition in range(5)
    }
    for name, values in conditions.items():
        if set(values) != expected:
            raise ValueError(f"{name}: expected {len(expected)}, found {len(values)}")

    rows = []
    for conversation_hash in sorted(cohort):
        result = {
            "original_row_idx": cohort[conversation_hash]["original_row_idx"],
            "conversation_hash": conversation_hash,
            "input_tokens_full": cohort[conversation_hash]["input_tokens"],
            "prior_messages": len(cohort[conversation_hash]["messages"]) - 2,
        }
        for name, values in conditions.items():
            result[f"{name}_rate"] = float(
                np.mean([values[(conversation_hash, index)][0] for index in range(5)])
            )
            result[f"{name}_score"] = float(
                np.mean([values[(conversation_hash, index)][1] for index in range(5)])
            )
        for suffix in ("rate", "score"):
            result[f"full_minus_last_exchange_{suffix}"] = (
                result[f"full_{suffix}"] - result[f"last_exchange_{suffix}"]
            )
            result[f"last_exchange_minus_target_only_{suffix}"] = (
                result[f"last_exchange_{suffix}"] - result[f"target_only_{suffix}"]
            )
            result[f"full_minus_target_only_{suffix}"] = (
                result[f"full_{suffix}"] - result[f"target_only_{suffix}"]
            )
        rows.append(result)

    summary = {
        "design": (
            "Three paired prompt conditions with common random-number sample slots: "
            "full history, latest prior user-assistant exchange plus target, and target only"
        ),
        "cohort_sha256": sha256_file(args.cohort),
        "judgment_sha256": {
            "full": sha256_file(args.full),
            "last_exchange": sha256_file(args.last_exchange),
            "target_only": sha256_file(args.target_only),
        },
        "conversations": len(rows),
        "samples_per_conversation_condition": 5,
        "endorsement_rates": {
            name: float(np.mean([row[f"{name}_rate"] for row in rows]))
            for name in conditions
        },
        "binary_rate_effects": {
            name: effect(rows, f"{name}_rate", args.bootstrap)
            for name in (
                "full_minus_last_exchange",
                "last_exchange_minus_target_only",
                "full_minus_target_only",
            )
        },
        "secondary_ordinal_score_effects": {
            name: effect(rows, f"{name}_score", args.bootstrap)
            for name in (
                "full_minus_last_exchange",
                "last_exchange_minus_target_only",
                "full_minus_target_only",
            )
        },
        "interpretation": (
            "Last-exchange minus target-only is the total effect of the latest prior "
            "user-assistant exchange, not the assistant message alone. The matched "
            "assistant-deletion experiment isolates assistant-message attribution."
        ),
    }
    write_jsonl(args.rows_output, rows)
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    args.summary.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
