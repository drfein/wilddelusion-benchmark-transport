from __future__ import annotations

import argparse
import json
from pathlib import Path

from config import ENDORSEMENT_THRESHOLD, MODEL_ID, MODEL_REVISION, stable_hash
from io_utils import read_jsonl, sha256_file, write_jsonl
from transformers import AutoTokenizer


def preceding_user(messages: list[dict], assistant_index: int) -> str:
    return next(
        messages[index]["content"]
        for index in range(assistant_index - 1, -1, -1)
        if messages[index]["role"] == "user"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cohort", type=Path, required=True)
    parser.add_argument("--selections", type=Path, required=True)
    parser.add_argument("--candidate-judgments", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--selection-output", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--model", default=MODEL_ID)
    parser.add_argument("--revision", default=MODEL_REVISION)
    args = parser.parse_args()

    selections = {row["conversation_hash"]: row for row in read_jsonl(args.selections)}
    nonendorsing_candidates: dict[str, list[dict]] = {}
    for row in read_jsonl(args.candidate_judgments):
        if (
            row.get("condition") == "candidate"
            and isinstance(row.get("annotation_score"), int)
            and row["annotation_score"] < ENDORSEMENT_THRESHOLD
            and not row.get("judge_error")
        ):
            nonendorsing_candidates.setdefault(row["conversation_hash"], []).append(row)
    tokenizer = AutoTokenizer.from_pretrained(
        args.model, revision=args.revision, trust_remote_code=False
    )
    rows = []
    swap_selections = []
    exclusions = []
    for source in read_jsonl(args.cohort):
        selection = selections.get(source["conversation_hash"])
        if selection is None:
            continue
        top_index = int(selection["top_message_index"])
        candidates = nonendorsing_candidates.get(source["conversation_hash"], [])
        if not candidates:
            exclusions.append(
                {
                    "conversation_hash": source["conversation_hash"],
                    "reason": "no_scored_nonendorsing_control_candidate",
                }
            )
            continue
        control = min(
            candidates,
            key=lambda row: (float(row["match_distance"]), int(row["message_index"])),
        )
        control_index = int(control["message_index"])
        messages = [dict(message) for message in source["messages"]]
        if not (
            messages[top_index]["role"] == messages[control_index]["role"] == "assistant"
        ):
            raise ValueError("Selected swap units must both be assistant messages")
        top_content = messages[top_index]["content"]
        control_content = messages[control_index]["content"]
        messages[top_index]["content"] = control_content
        messages[control_index]["content"] = top_content
        input_tokens = len(
            tokenizer.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=True,
                enable_thinking=False,
            )
        )
        if input_tokens != int(source["input_tokens"]):
            raise ValueError(
                f"Position swap changed token count for {source['conversation_hash']}: "
                f"{input_tokens} != {source['input_tokens']}"
            )
        prompt_hash = stable_hash(
            {
                "parent_prompt": source["prompt_content_sha256"],
                "condition": "assistant_positions_swapped",
                "top_message_index": top_index,
                "control_message_index": control_index,
                "messages": messages,
            }
        )
        rows.append(
            {
                **{key: value for key, value in source.items() if key != "messages"},
                "condition": "assistant_positions_swapped",
                "selection_rule": "latest_endorsing_assistant",
                "top_message_index": top_index,
                "control_message_index": control_index,
                "messages": messages,
                "input_tokens": input_tokens,
                "parent_prompt_content_sha256": source["prompt_content_sha256"],
                "prompt_content_sha256": prompt_hash,
            }
        )
        swap_selections.append(
            {
                **selection,
                "control_message_index": control_index,
                "control_tokens": int(control["tokens"]),
                "control_relative_position": float(control["relative_position"]),
                "match_distance": float(control["match_distance"]),
                "control_assistant_score": int(control["annotation_score"]),
            }
        )

    write_jsonl(args.output, rows)
    write_jsonl(args.selection_output, swap_selections)
    manifest = {
        "cohort_sha256": sha256_file(args.cohort),
        "selections_sha256": sha256_file(args.selections),
        "candidate_judgments_sha256": sha256_file(args.candidate_judgments),
        "paired_conversations": len(rows),
        "intervention_prompts": len(rows),
        "intervention": (
            "swap the complete contents of the latest endorsing assistant turn and "
            "its matched earlier assistant turn"
        ),
        "invariants": (
            "all content tokens, message roles, message count, and total rendered "
            "prompt-token count are preserved; only assistant-content position changes"
        ),
        "common_random_numbers": "generation seeds depend on conversation, not condition",
        "control_rule": (
            "nearest of up to eight earlier assistant candidates that scored below "
            f"the frozen endorsement threshold of {ENDORSEMENT_THRESHOLD}"
        ),
        "excluded": exclusions,
        "output_sha256": sha256_file(args.output),
        "selection_output_sha256": sha256_file(args.selection_output),
    }
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
