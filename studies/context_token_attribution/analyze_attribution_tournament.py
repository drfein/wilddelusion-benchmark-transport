from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from config import SEED
from io_utils import read_jsonl, sha256_file, write_jsonl


def bootstrap(values: np.ndarray, iterations: int) -> list[float]:
    rng = np.random.default_rng(SEED)
    estimates = np.empty(iterations, dtype=np.float64)
    for index in range(iterations):
        sample = rng.integers(0, len(values), len(values))
        estimates[index] = values[sample].mean()
    return [float(value) for value in np.quantile(estimates, [0.025, 0.975])]


def summarize(rows: list[dict], key: str, iterations: int) -> dict:
    values = np.asarray([row[key] for row in rows], dtype=np.float64)
    return {
        "mean": float(values.mean()),
        "bootstrap_95_ci_by_conversation": bootstrap(values, iterations),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gradient-rows", type=Path, required=True)
    parser.add_argument("--integrated-rows", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--bootstrap", type=int, default=10_000)
    args = parser.parse_args()

    gradient = {row["conversation_hash"]: row for row in read_jsonl(args.gradient_rows)}
    integrated = {
        row["conversation_hash"]: row for row in read_jsonl(args.integrated_rows)
    }
    rows = []
    for conversation_hash in sorted(set(gradient) & set(integrated)):
        gradient_row = gradient[conversation_hash]
        integrated_row = integrated[conversation_hash]
        same_top = (
            gradient_row["top_message_index"] == integrated_row["top_message_index"]
        )
        rows.append(
            {
                "conversation_hash": conversation_hash,
                "same_top_assistant_message": same_top,
                "gradient_top_message_index": gradient_row["top_message_index"],
                "integrated_top_message_index": integrated_row["top_message_index"],
                "gradient_top_deleted_rate": gradient_row["top_deleted_rate"],
                "integrated_top_deleted_rate": integrated_row["top_deleted_rate"],
                "gradient_minus_integrated_top_deleted_rate": (
                    gradient_row["top_deleted_rate"]
                    - integrated_row["top_deleted_rate"]
                ),
                "gradient_top_minus_matched": gradient_row["top_minus_matched"],
                "integrated_top_minus_matched": integrated_row["top_minus_matched"],
                "gradient_minus_integrated_matched_effect": (
                    gradient_row["top_minus_matched"]
                    - integrated_row["top_minus_matched"]
                ),
                "gradient_minus_integrated_top_deleted_score": (
                    gradient_row["top_deleted_mean_score"]
                    - integrated_row["top_deleted_mean_score"]
                ),
            }
        )
    if not rows:
        raise ValueError("No overlapping intervention conversations")
    disagree = [row for row in rows if not row["same_top_assistant_message"]]
    summary = {
        "gradient_rows_sha256": sha256_file(args.gradient_rows),
        "integrated_rows_sha256": sha256_file(args.integrated_rows),
        "overlapping_conversations": len(rows),
        "same_top_assistant_messages": int(
            sum(row["same_top_assistant_message"] for row in rows)
        ),
        "different_top_assistant_messages": len(disagree),
        "sign_convention": (
            "Negative differences favor gradient-times-input because deleting its "
            "selected message produced less endorsement."
        ),
        "all_overlap": {
            key: summarize(rows, key, args.bootstrap)
            for key in (
                "gradient_minus_integrated_top_deleted_rate",
                "gradient_minus_integrated_matched_effect",
                "gradient_minus_integrated_top_deleted_score",
            )
        },
        "different-selection_sensitivity": {
            key: summarize(disagree, key, args.bootstrap)
            for key in (
                "gradient_minus_integrated_top_deleted_rate",
                "gradient_minus_integrated_matched_effect",
                "gradient_minus_integrated_top_deleted_score",
            )
        },
    }
    write_jsonl(args.output, rows)
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    args.summary.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
