from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

from config import ENDORSEMENT_THRESHOLD, MODEL_ID, MODEL_REVISION, stable_hash
from io_utils import read_jsonl, sha256_file, write_jsonl
from transformers import AutoTokenizer


def matched_control(candidates: list[dict], latest: dict) -> dict | None:
    alternatives = [
        item for item in candidates if item["message_index"] != latest["message_index"]
    ]
    if not alternatives:
        return None
    for item in alternatives:
        item["match_distance"] = abs(
            math.log1p(item["tokens"]) - math.log1p(latest["tokens"])
        ) + 2 * abs(item["relative_position"] - latest["relative_position"])
    return min(
        alternatives, key=lambda item: (item["match_distance"], item["message_index"])
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cohort", type=Path, required=True)
    parser.add_argument("--prior-assistant-judgments", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--model", default=MODEL_ID)
    parser.add_argument("--revision", default=MODEL_REVISION)
    args = parser.parse_args()

    eligible = {
        row["conversation_hash"]: row
        for row in read_jsonl(args.prior_assistant_judgments)
        if isinstance(row.get("annotation_score"), int)
        and row["annotation_score"] >= ENDORSEMENT_THRESHOLD
        and not row.get("judge_error")
    }
    tokenizer = AutoTokenizer.from_pretrained(
        args.model, revision=args.revision, trust_remote_code=False
    )
    rows = []
    selections = []
    exclusions = []
    for source in read_jsonl(args.cohort):
        conversation_hash = source["conversation_hash"]
        if conversation_hash not in eligible:
            continue
        assistant_units = []
        for index, message in enumerate(source["messages"][1:-1], start=1):
            if message["role"] != "assistant":
                continue
            assistant_units.append(
                {
                    "message_index": index,
                    "tokens": len(
                        tokenizer(message["content"], add_special_tokens=False)[
                            "input_ids"
                        ]
                    ),
                    "relative_position": index / (len(source["messages"]) - 1),
                }
            )
        latest = max(assistant_units, key=lambda item: item["message_index"])
        control = matched_control(assistant_units, latest)
        if control is None:
            exclusions.append(
                {
                    "conversation_hash": conversation_hash,
                    "reason": "fewer_than_two_prior_assistant_messages",
                }
            )
            continue
        selection = {
            "original_row_idx": source["original_row_idx"],
            "conversation_hash": conversation_hash,
            "prior_assistant_score": eligible[conversation_hash]["annotation_score"],
            "top_message_index": latest["message_index"],
            "top_tokens": latest["tokens"],
            "top_relative_position": latest["relative_position"],
            "control_message_index": control["message_index"],
            "control_tokens": control["tokens"],
            "control_relative_position": control["relative_position"],
            "match_distance": control["match_distance"],
        }
        selections.append(selection)
        for condition, deletion in (
            ("top_assistant_deleted", latest),
            ("matched_assistant_deleted", control),
        ):
            messages = [
                message
                for index, message in enumerate(source["messages"])
                if index != deletion["message_index"]
            ]
            prompt_hash = stable_hash(
                {
                    "parent_prompt": source["prompt_content_sha256"],
                    "selection": "latest_endorsing_assistant",
                    "condition": condition,
                    "deleted_message_index": deletion["message_index"],
                    "messages": messages,
                }
            )
            rows.append(
                {
                    **{
                        key: value for key, value in source.items() if key != "messages"
                    },
                    "condition": condition,
                    "selection_rule": "latest_endorsing_assistant",
                    "deleted_message_index": deletion["message_index"],
                    "deleted_role": "assistant",
                    "deleted_message_tokens": deletion["tokens"],
                    "deleted_relative_position": deletion["relative_position"],
                    "messages": messages,
                    "input_tokens": len(
                        tokenizer.apply_chat_template(
                            messages,
                            tokenize=True,
                            add_generation_prompt=True,
                            enable_thinking=False,
                        )
                    ),
                    "parent_prompt_content_sha256": source["prompt_content_sha256"],
                    "prompt_content_sha256": prompt_hash,
                }
            )

    write_jsonl(args.output, rows)
    selections_path = args.output.with_name("behavior_intervention_selections.jsonl")
    write_jsonl(selections_path, selections)
    manifest = {
        "cohort_sha256": sha256_file(args.cohort),
        "prior_assistant_judgments_sha256": sha256_file(args.prior_assistant_judgments),
        "selection": (
            "all conversations whose immediately preceding assistant response scored "
            f">={ENDORSEMENT_THRESHOLD} on the frozen endorsement rubric"
        ),
        "eligible_conversations": len(eligible),
        "paired_conversations": len(selections),
        "intervention_prompts": len(rows),
        "top_rule": "delete the immediately preceding assistant response",
        "control_rule": (
            "delete the same-conversation earlier assistant response nearest in log "
            "token length plus twice relative-position distance"
        ),
        "common_random_numbers": "generation seeds depend on conversation, not condition",
        "excluded": exclusions,
        "output_sha256": sha256_file(args.output),
        "selections_sha256": sha256_file(selections_path),
    }
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
