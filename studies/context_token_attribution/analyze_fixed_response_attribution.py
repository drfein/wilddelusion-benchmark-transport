from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from analyze_behavior_interventions import bootstrap, sign_flip_p_value
from io_utils import read_jsonl, sha256_file, write_jsonl


def response_features(row: dict) -> dict:
    units = row["units"]
    assistants = [unit for unit in units if unit["role"] == "assistant"]
    users = [unit for unit in units if unit["role"] == "user"]
    latest_assistant = max(assistants, key=lambda unit: unit["message_index"])
    top = max(units, key=lambda unit: unit["attricot_mean_contribution"])
    top_assistant = max(
        assistants, key=lambda unit: unit["attricot_mean_contribution"]
    )
    return {
        "latest_assistant_contribution": latest_assistant[
            "attricot_mean_contribution"
        ],
        "assistant_contribution_sum": float(
            sum(unit["attricot_mean_contribution"] for unit in assistants)
        ),
        "assistant_positive_contribution_sum": float(
            sum(max(0.0, unit["attricot_mean_contribution"]) for unit in assistants)
        ),
        "user_contribution_sum": float(
            sum(unit["attricot_mean_contribution"] for unit in users)
        ),
        "top_message_index": int(top["message_index"]),
        "top_message_role": top["role"],
        "top_assistant_message_index": int(top_assistant["message_index"]),
        "top_assistant_contribution": float(
            top_assistant["attricot_mean_contribution"]
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--rows-output", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--bootstrap", type=int, default=10_000)
    parser.add_argument("--randomization-iterations", type=int, default=200_000)
    args = parser.parse_args()

    grouped: dict[str, dict[int, dict]] = defaultdict(dict)
    for row in read_jsonl(args.input):
        if row.get("complete") and not row.get("error"):
            grouped[row["conversation_hash"]][int(row["endorsement"])] = row
    rows = []
    for conversation_hash, classes in sorted(grouped.items()):
        if set(classes) != {0, 1}:
            continue
        nonendorsing = response_features(classes[0])
        endorsing = response_features(classes[1])
        row = {
            "conversation_hash": conversation_hash,
            "endorsing_generation_sha256": classes[1]["generation_sha256"],
            "nonendorsing_generation_sha256": classes[0]["generation_sha256"],
            "same_top_message": endorsing["top_message_index"]
            == nonendorsing["top_message_index"],
            "same_top_assistant_message": endorsing["top_assistant_message_index"]
            == nonendorsing["top_assistant_message_index"],
        }
        for key in (
            "latest_assistant_contribution",
            "assistant_contribution_sum",
            "assistant_positive_contribution_sum",
            "user_contribution_sum",
            "top_assistant_contribution",
        ):
            row[f"endorsing_{key}"] = endorsing[key]
            row[f"nonendorsing_{key}"] = nonendorsing[key]
            row[f"endorsing_minus_nonendorsing_{key}"] = (
                endorsing[key] - nonendorsing[key]
            )
        row["endorsing_top_message_role"] = endorsing["top_message_role"]
        row["nonendorsing_top_message_role"] = nonendorsing["top_message_role"]
        rows.append(row)
    if not rows:
        raise ValueError("No complete within-conversation response-class pairs")

    contrasts = {}
    for key in (
        "latest_assistant_contribution",
        "assistant_contribution_sum",
        "assistant_positive_contribution_sum",
        "user_contribution_sum",
        "top_assistant_contribution",
    ):
        values = np.asarray(
            [row[f"endorsing_minus_nonendorsing_{key}"] for row in rows],
            dtype=np.float64,
        )
        contrasts[key] = {
            "mean_endorsing_minus_nonendorsing": float(values.mean()),
            "bootstrap_95_ci_by_conversation": bootstrap(values, args.bootstrap),
            "two_sided_cluster_sign_flip_p": sign_flip_p_value(
                values, args.randomization_iterations
            ),
        }
    summary = {
        "input_sha256": sha256_file(args.input),
        "paired_conversations": len(rows),
        "fixed_responses": 2 * len(rows),
        "estimand": (
            "Within the same prompt, difference in fixed-response log-probability "
            "support from each deleted message for an endorsing versus non-endorsing "
            "sample. Contributions are response-token mean log-probability changes."
        ),
        "same_top_message_rate": float(
            np.mean([row["same_top_message"] for row in rows])
        ),
        "same_top_assistant_message_rate": float(
            np.mean([row["same_top_assistant_message"] for row in rows])
        ),
        "top_message_role_counts": {
            "endorsing_assistant": int(
                sum(row["endorsing_top_message_role"] == "assistant" for row in rows)
            ),
            "nonendorsing_assistant": int(
                sum(
                    row["nonendorsing_top_message_role"] == "assistant"
                    for row in rows
                )
            ),
        },
        "paired_contrasts": contrasts,
    }
    write_jsonl(args.rows_output, rows)
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    args.summary.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
