from __future__ import annotations

import argparse
import json
from collections import defaultdict
from collections.abc import Callable
from pathlib import Path

import numpy as np
from config import ENDORSEMENT_THRESHOLD, SEED
from io_utils import read_jsonl, sha256_file, write_jsonl
from scipy.stats import spearmanr


def role_masks(
    message_indices: np.ndarray, messages: list[dict]
) -> dict[str, np.ndarray]:
    target_index = len(messages) - 1
    labels = np.full(len(message_indices), "template", dtype=object)
    for index, message in enumerate(messages):
        if index == 0 and message["role"] == "system":
            label = "system"
        elif index == target_index:
            label = "target_user"
        elif message["role"] == "assistant":
            label = "prior_assistant"
        elif message["role"] == "user":
            label = "prior_user"
        else:
            label = f"prior_{message['role']}"
        labels[message_indices == index] = label
    return {label: labels == label for label in sorted(set(labels))}


def safe_ratio(numerator: float, denominator: float) -> float:
    return numerator / denominator if denominator > 0 else float("nan")


def summarize_one(
    row: dict,
    scores: np.ndarray,
    message_indices: np.ndarray,
    endorsement_rate: float,
    method: str,
) -> tuple[dict, list[dict]]:
    masks = role_masks(message_indices, row["messages"])
    positive = np.maximum(scores, 0)
    absolute = np.abs(scores)
    role_values = {}
    for role, mask in masks.items():
        role_values[role] = {
            "tokens": int(mask.sum()),
            "signed_sum": float(scores[mask].sum()),
            "positive_sum": float(positive[mask].sum()),
            "absolute_sum": float(absolute[mask].sum()),
        }

    assistant = role_values.get("prior_assistant", {})
    user = role_values.get("prior_user", {})
    assistant_tokens = int(assistant.get("tokens", 0))
    user_tokens = int(user.get("tokens", 0))
    assistant_positive = float(assistant.get("positive_sum", 0))
    user_positive = float(user.get("positive_sum", 0))
    prior_tokens = assistant_tokens + user_tokens
    prior_positive = assistant_positive + user_positive
    assistant_token_share = safe_ratio(assistant_tokens, prior_tokens)
    assistant_positive_share = safe_ratio(assistant_positive, prior_positive)
    enrichment = safe_ratio(assistant_positive_share, assistant_token_share)

    message_rows = []
    for message_index in range(1, len(row["messages"]) - 1):
        mask = message_indices == message_index
        if not mask.any():
            continue
        message_rows.append(
            {
                "original_row_idx": row["original_row_idx"],
                "conversation_hash": row["conversation_hash"],
                "method": method,
                "message_index": message_index,
                "role": row["messages"][message_index]["role"],
                "relative_position": message_index / (len(row["messages"]) - 1),
                "tokens": int(mask.sum()),
                "signed_sum": float(scores[mask].sum()),
                "positive_sum": float(positive[mask].sum()),
                "absolute_sum": float(absolute[mask].sum()),
                "endorsement_rate": endorsement_rate,
                "message": row["messages"][message_index]["content"],
                "target_user_message": row["messages"][-1]["content"],
            }
        )
    top_positive = max(message_rows, key=lambda item: item["positive_sum"])
    top_signed = max(message_rows, key=lambda item: item["signed_sum"])
    assistant_indices = [
        item["message_index"] for item in message_rows if item["role"] == "assistant"
    ]
    most_recent_assistant = max(assistant_indices) if assistant_indices else None
    return (
        {
            "original_row_idx": row["original_row_idx"],
            "conversation_hash": row["conversation_hash"],
            "method": method,
            "endorsement_rate": endorsement_rate,
            "input_tokens": row["input_tokens"],
            "prior_messages": len(row["messages"]) - 2,
            "assistant_token_share": assistant_token_share,
            "assistant_positive_share": assistant_positive_share,
            "assistant_positive_enrichment": enrichment,
            "assistant_minus_token_share": (
                assistant_positive_share - assistant_token_share
            ),
            "assistant_positive_per_token": safe_ratio(
                assistant_positive, assistant_tokens
            ),
            "user_positive_per_token": safe_ratio(user_positive, user_tokens),
            "top_positive_role": top_positive["role"],
            "top_positive_message_index": top_positive["message_index"],
            "top_signed_role": top_signed["role"],
            "top_signed_message_index": top_signed["message_index"],
            "top_positive_is_most_recent_assistant": bool(
                top_positive["role"] == "assistant"
                and top_positive["message_index"] == most_recent_assistant
            ),
            "role_values": role_values,
        },
        message_rows,
    )


def bootstrap_interval(
    rows: list[dict], statistic: Callable[[list[dict]], float], iterations: int
) -> list[float]:
    rng = np.random.default_rng(SEED)
    values = []
    for _ in range(iterations):
        sample = [rows[index] for index in rng.integers(0, len(rows), len(rows))]
        value = statistic(sample)
        if np.isfinite(value):
            values.append(value)
    return [float(np.quantile(values, 0.025)), float(np.quantile(values, 0.975))]


def finite_mean(rows: list[dict], key: str) -> float:
    values = np.asarray([row[key] for row in rows], dtype=np.float64)
    return float(np.nanmean(values))


def aggregate(rows: list[dict], iterations: int) -> dict:
    result = {"conversations": len(rows)}
    for key in (
        "assistant_token_share",
        "assistant_positive_share",
        "assistant_minus_token_share",
        "assistant_positive_enrichment",
    ):
        result[key] = {
            "mean": finite_mean(rows, key),
            "bootstrap_95_ci": bootstrap_interval(
                rows, lambda sample, name=key: finite_mean(sample, name), iterations
            ),
        }
    for key in (
        "top_positive_role",
        "top_signed_role",
    ):
        result[f"fraction_{key}_assistant"] = {
            "mean": float(np.mean([row[key] == "assistant" for row in rows])),
            "bootstrap_95_ci": bootstrap_interval(
                rows,
                lambda sample, name=key: float(
                    np.mean([row[name] == "assistant" for row in sample])
                ),
                iterations,
            ),
        }
    result["fraction_top_positive_most_recent_assistant"] = {
        "mean": float(
            np.mean([row["top_positive_is_most_recent_assistant"] for row in rows])
        ),
        "bootstrap_95_ci": bootstrap_interval(
            rows,
            lambda sample: float(
                np.mean(
                    [row["top_positive_is_most_recent_assistant"] for row in sample]
                )
            ),
            iterations,
        ),
    }
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cohort", type=Path, required=True)
    parser.add_argument("--judgments", type=Path, required=True)
    parser.add_argument("--attribution-dir", type=Path, required=True)
    parser.add_argument("--index", type=Path, required=True)
    parser.add_argument("--conversation-output", type=Path, required=True)
    parser.add_argument("--message-output", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--bootstrap", type=int, default=10_000)
    args = parser.parse_args()

    cohort = {row["conversation_hash"]: row for row in read_jsonl(args.cohort)}
    label_values: dict[str, list[int]] = defaultdict(list)
    for row in read_jsonl(args.judgments):
        if isinstance(row.get("annotation_score"), int) and not row.get("judge_error"):
            label_values[row["conversation_hash"]].append(
                int(row["annotation_score"] >= ENDORSEMENT_THRESHOLD)
            )
    endorsement_rates = {
        key: float(np.mean(values)) for key, values in label_values.items()
    }
    index = {
        row["conversation_hash"]: row
        for row in read_jsonl(args.index)
        if row.get("attribution_file") and not row.get("error")
    }
    conversation_rows = []
    message_rows = []
    methods: dict[str, list[dict]] = defaultdict(list)
    agreement_rows = []
    for conversation_hash in sorted(index):
        row = cohort[conversation_hash]
        data = np.load(
            args.attribution_dir / index[conversation_hash]["attribution_file"]
        )
        message_indices = data["message_indices"]
        if "integrated_gradients" in data:
            gradient = data["gradient_times_input"].astype(np.float64)
            integrated = data["integrated_gradients"].astype(np.float64)
            prior_context = (message_indices > 0) & (
                message_indices < len(row["messages"]) - 1
            )
            message_ids = sorted(set(message_indices[prior_context].tolist()))
            gradient_messages = np.asarray(
                [gradient[message_indices == value].sum() for value in message_ids]
            )
            integrated_messages = np.asarray(
                [integrated[message_indices == value].sum() for value in message_ids]
            )
            assistant_ids = [
                value
                for value in message_ids
                if row["messages"][value]["role"] == "assistant"
            ]
            gradient_assistant = np.asarray(
                [gradient[message_indices == value].sum() for value in assistant_ids]
            )
            integrated_assistant = np.asarray(
                [integrated[message_indices == value].sum() for value in assistant_ids]
            )
            agreement_rows.append(
                {
                    "conversation_hash": conversation_hash,
                    "input_tokens": int(row["input_tokens"]),
                    "token_spearman": float(
                        spearmanr(
                            gradient[prior_context], integrated[prior_context]
                        ).statistic
                    ),
                    "message_sum_spearman": float(
                        spearmanr(gradient_messages, integrated_messages).statistic
                    ),
                    "top_signed_message_agreement": int(
                        int(np.argmax(gradient_messages))
                        == int(np.argmax(integrated_messages))
                    ),
                    "assistant_message_sum_spearman": (
                        float(
                            spearmanr(
                                gradient_assistant, integrated_assistant
                            ).statistic
                        )
                        if len(assistant_ids) > 1
                        else float("nan")
                    ),
                    "top_signed_assistant_agreement": (
                        int(
                            int(np.argmax(gradient_assistant))
                            == int(np.argmax(integrated_assistant))
                        )
                        if assistant_ids
                        else float("nan")
                    ),
                }
            )
        for key, method in (
            ("gradient_times_input", "gradient_times_input"),
            ("integrated_gradients", "integrated_gradients"),
        ):
            if key not in data:
                continue
            summary, messages = summarize_one(
                row,
                data[key].astype(np.float64),
                message_indices,
                endorsement_rates[conversation_hash],
                method,
            )
            conversation_rows.append(summary)
            message_rows.extend(messages)
            methods[method].append(summary)

    write_jsonl(args.conversation_output, conversation_rows)
    write_jsonl(args.message_output, message_rows)
    summary = {
        "cohort_sha256": sha256_file(args.cohort),
        "judgments_sha256": sha256_file(args.judgments),
        "index_sha256": sha256_file(args.index),
        "unit_of_inference": "conversation",
        "positive_attribution_definition": "maximum(signed token attribution, 0)",
        "role_scope": "prior user and assistant content tokens; target/system/template excluded",
        "methods": {
            method: aggregate(rows, args.bootstrap) for method, rows in methods.items()
        },
        "gradient_times_input_vs_integrated_gradients": {
            "conversations": len(agreement_rows),
            "availability_note": (
                "Frozen smallest-hash subset with exact no-truncation IG; the longest "
                "planned cases were not run to prioritize causal regeneration."
            ),
            **{
                key: {
                    "mean": finite_mean(agreement_rows, key),
                    "bootstrap_95_ci": bootstrap_interval(
                        agreement_rows,
                        lambda sample, name=key: finite_mean(sample, name),
                        args.bootstrap,
                    ),
                }
                for key in (
                    "token_spearman",
                    "message_sum_spearman",
                    "top_signed_message_agreement",
                    "assistant_message_sum_spearman",
                    "top_signed_assistant_agreement",
                )
            },
        },
    }
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    args.summary.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
