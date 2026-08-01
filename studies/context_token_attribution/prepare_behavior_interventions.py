from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
from config import MODEL_ID, MODEL_REVISION, stable_hash
from io_utils import read_jsonl, sha256_file, write_jsonl
from transformers import AutoTokenizer


def choose_matched_control(candidates: list[dict], top: dict) -> dict | None:
    alternatives = [
        item for item in candidates if item["message_index"] != top["message_index"]
    ]
    if not alternatives:
        return None
    for item in alternatives:
        item["match_distance"] = abs(
            math.log1p(item["tokens"]) - math.log1p(top["tokens"])
        ) + 2 * abs(item["relative_position"] - top["relative_position"])
    return min(
        alternatives, key=lambda item: (item["match_distance"], item["message_index"])
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cohort", type=Path, required=True)
    parser.add_argument("--attribution-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--count", type=int, default=60)
    parser.add_argument("--model", default=MODEL_ID)
    parser.add_argument("--revision", default=MODEL_REVISION)
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(
        args.model, revision=args.revision, trust_remote_code=False
    )
    cohort_rows = sorted(
        read_jsonl(args.cohort), key=lambda row: row["conversation_hash"]
    )[: args.count]
    intervention_rows = []
    exclusions = []
    selections = []
    for row in cohort_rows:
        path = args.attribution_dir / f"{row['conversation_hash']}.npz"
        if not path.exists():
            exclusions.append(
                {
                    "conversation_hash": row["conversation_hash"],
                    "reason": "missing_token_attribution",
                }
            )
            continue
        attribution = np.load(path)
        token_messages = attribution["message_indices"]
        scores = attribution["gradient_times_input"]
        assistant_units = []
        for message_index in range(1, len(row["messages"]) - 1):
            if row["messages"][message_index]["role"] != "assistant":
                continue
            mask = token_messages == message_index
            assistant_units.append(
                {
                    "message_index": message_index,
                    "tokens": int(mask.sum()),
                    "relative_position": message_index / (len(row["messages"]) - 1),
                    "gradient_times_input_sum": float(scores[mask].sum()),
                }
            )
        if len(assistant_units) < 2:
            exclusions.append(
                {
                    "conversation_hash": row["conversation_hash"],
                    "reason": "fewer_than_two_prior_assistant_messages",
                }
            )
            continue
        top = max(
            assistant_units,
            key=lambda item: (
                item["gradient_times_input_sum"],
                -item["message_index"],
            ),
        )
        if top["gradient_times_input_sum"] <= 0:
            exclusions.append(
                {
                    "conversation_hash": row["conversation_hash"],
                    "reason": "no_positive_assistant_attribution",
                }
            )
            continue
        control = choose_matched_control(assistant_units, top)
        if control is None:
            raise RuntimeError("Control matching failed after eligibility check")
        selection = {
            "original_row_idx": row["original_row_idx"],
            "conversation_hash": row["conversation_hash"],
            "top_message_index": top["message_index"],
            "top_tokens": top["tokens"],
            "top_relative_position": top["relative_position"],
            "top_attribution": top["gradient_times_input_sum"],
            "control_message_index": control["message_index"],
            "control_tokens": control["tokens"],
            "control_relative_position": control["relative_position"],
            "control_attribution": control["gradient_times_input_sum"],
            "match_distance": control["match_distance"],
        }
        selections.append(selection)
        for condition, deletion in (
            ("top_assistant_deleted", top),
            ("matched_assistant_deleted", control),
        ):
            messages = [
                message
                for index, message in enumerate(row["messages"])
                if index != deletion["message_index"]
            ]
            prompt_hash = stable_hash(
                {
                    "parent_prompt": row["prompt_content_sha256"],
                    "condition": condition,
                    "deleted_message_index": deletion["message_index"],
                    "messages": messages,
                }
            )
            input_tokens = len(
                tokenizer.apply_chat_template(
                    messages,
                    tokenize=True,
                    add_generation_prompt=True,
                    enable_thinking=False,
                )
            )
            intervention_rows.append(
                {
                    **{key: value for key, value in row.items() if key != "messages"},
                    "condition": condition,
                    "deleted_message_index": deletion["message_index"],
                    "deleted_role": "assistant",
                    "deleted_message_tokens": deletion["tokens"],
                    "deleted_relative_position": deletion["relative_position"],
                    "deleted_attribution": deletion["gradient_times_input_sum"],
                    "messages": messages,
                    "input_tokens": input_tokens,
                    "parent_prompt_content_sha256": row["prompt_content_sha256"],
                    "prompt_content_sha256": prompt_hash,
                }
            )
    write_jsonl(args.output, intervention_rows)
    write_jsonl(
        args.output.with_name("behavior_intervention_selections.jsonl"), selections
    )
    manifest = {
        "cohort_sha256": sha256_file(args.cohort),
        "attribution_subset_rule": "smallest conversation SHA-256 hashes",
        "requested_conversations": args.count,
        "eligible_paired_conversations": len(selections),
        "intervention_prompts": len(intervention_rows),
        "conditions": ["top_assistant_deleted", "matched_assistant_deleted"],
        "top_rule": "largest positive signed gradient-times-input sum over assistant content tokens",
        "control_rule": (
            "same-conversation assistant message nearest in log token length plus "
            "twice relative-position distance; ties use earliest message index"
        ),
        "common_random_numbers": "generation seeds depend on conversation, not condition",
        "excluded": exclusions,
    }
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
