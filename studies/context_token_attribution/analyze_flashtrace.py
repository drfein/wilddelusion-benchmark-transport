from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from analyze_behavior_interventions import bootstrap, sign_flip_p_value
from io_utils import read_jsonl, sha256_file, write_jsonl


def trace_features(path: Path, messages: list[dict]) -> dict:
    data = np.load(path)
    message_indices = data["message_indices"].astype(np.int64)
    scores = data["flashtrace_scores"].astype(np.float64)
    message_scores = {
        index: float(scores[message_indices == index].sum())
        for index in range(len(messages))
    }
    content_total = sum(message_scores.values())
    if content_total <= 0:
        raise ValueError(f"Non-positive content attribution total in {path}")
    normalized = {index: score / content_total for index, score in message_scores.items()}
    prior_indices = list(range(1, len(messages) - 1))
    assistant_indices = [
        index for index in prior_indices if messages[index]["role"] == "assistant"
    ]
    user_indices = [
        index for index in prior_indices if messages[index]["role"] == "user"
    ]
    latest_assistant = max(assistant_indices)
    top = max(prior_indices, key=normalized.__getitem__)
    top_assistant = max(assistant_indices, key=normalized.__getitem__)
    return {
        "latest_assistant_share": normalized[latest_assistant],
        "assistant_share": float(sum(normalized[index] for index in assistant_indices)),
        "user_share": float(sum(normalized[index] for index in user_indices)),
        "template_share": float(scores[message_indices == -1].sum() / scores.sum()),
        "top_message_index": top,
        "top_message_role": messages[top]["role"],
        "top_assistant_message_index": top_assistant,
        "top_assistant_share": normalized[top_assistant],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cohort", type=Path, required=True)
    parser.add_argument("--index", type=Path, required=True)
    parser.add_argument("--attribution-dir", type=Path, required=True)
    parser.add_argument("--rows-output", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--bootstrap", type=int, default=10_000)
    parser.add_argument("--randomization-iterations", type=int, default=200_000)
    args = parser.parse_args()

    cohort = {row["conversation_hash"]: row for row in read_jsonl(args.cohort)}
    grouped: dict[str, dict[int, tuple[dict, dict]]] = defaultdict(dict)
    for record in read_jsonl(args.index):
        if not record.get("complete") or record.get("error"):
            continue
        features = trace_features(
            args.attribution_dir / record["attribution_file"],
            cohort[record["conversation_hash"]]["messages"],
        )
        grouped[record["conversation_hash"]][int(record["endorsement"])] = (
            record,
            features,
        )

    rows = []
    for conversation_hash, classes in sorted(grouped.items()):
        if set(classes) != {0, 1}:
            continue
        non_record, nonendorsing = classes[0]
        end_record, endorsing = classes[1]
        row = {
            "conversation_hash": conversation_hash,
            "endorsing_generation_sha256": end_record["generation_sha256"],
            "nonendorsing_generation_sha256": non_record["generation_sha256"],
            "same_top_message": endorsing["top_message_index"]
            == nonendorsing["top_message_index"],
            "same_top_assistant_message": endorsing["top_assistant_message_index"]
            == nonendorsing["top_assistant_message_index"],
            "endorsing_top_message_role": endorsing["top_message_role"],
            "nonendorsing_top_message_role": nonendorsing["top_message_role"],
        }
        for key in (
            "latest_assistant_share",
            "assistant_share",
            "user_share",
            "template_share",
            "top_assistant_share",
        ):
            row[f"endorsing_{key}"] = endorsing[key]
            row[f"nonendorsing_{key}"] = nonendorsing[key]
            row[f"endorsing_minus_nonendorsing_{key}"] = (
                endorsing[key] - nonendorsing[key]
            )
        rows.append(row)
    if not rows:
        raise ValueError("No complete within-conversation response-class pairs")

    contrasts = {}
    for key in (
        "latest_assistant_share",
        "assistant_share",
        "user_share",
        "template_share",
        "top_assistant_share",
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
        "cohort_sha256": sha256_file(args.cohort),
        "index_sha256": sha256_file(args.index),
        "paired_conversations": len(rows),
        "fixed_responses": 2 * len(rows),
        "normalization": (
            "message score divided by total score on message-content tokens within "
            "each fixed-response trace"
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
