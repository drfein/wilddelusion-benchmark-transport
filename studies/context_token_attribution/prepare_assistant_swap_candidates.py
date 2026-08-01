from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

from config import MODEL_ID, MODEL_REVISION, stable_hash
from io_utils import read_jsonl, sha256_file, write_jsonl
from prepare_assistant_swap_interventions import preceding_user
from transformers import AutoTokenizer


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cohort", type=Path, required=True)
    parser.add_argument("--selections", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--candidate-limit", type=int, default=8)
    parser.add_argument("--model", default=MODEL_ID)
    parser.add_argument("--revision", default=MODEL_REVISION)
    args = parser.parse_args()

    selections = {row["conversation_hash"]: row for row in read_jsonl(args.selections)}
    tokenizer = AutoTokenizer.from_pretrained(
        args.model, revision=args.revision, trust_remote_code=False
    )
    rows = []
    candidate_counts = []
    for source in read_jsonl(args.cohort):
        selection = selections.get(source["conversation_hash"])
        if selection is None:
            continue
        latest_index = int(selection["top_message_index"])
        latest_tokens = int(selection["top_tokens"])
        latest_position = float(selection["top_relative_position"])
        candidates = []
        for index, message in enumerate(source["messages"]):
            if message["role"] != "assistant" or index == latest_index:
                continue
            tokens = len(
                tokenizer(message["content"], add_special_tokens=False)["input_ids"]
            )
            position = index / (len(source["messages"]) - 1)
            distance = abs(math.log1p(tokens) - math.log1p(latest_tokens)) + 2 * abs(
                position - latest_position
            )
            candidates.append(
                {
                    "message_index": index,
                    "tokens": tokens,
                    "relative_position": position,
                    "match_distance": distance,
                }
            )
        candidates.sort(key=lambda item: (item["match_distance"], item["message_index"]))
        candidates = candidates[: args.candidate_limit]
        candidate_counts.append(len(candidates))
        units = [
            {
                "condition": "latest",
                "message_index": latest_index,
                "tokens": latest_tokens,
                "relative_position": latest_position,
                "match_distance": 0.0,
            },
            *({"condition": "candidate", **item} for item in candidates),
        ]
        for unit in units:
            message_index = unit["message_index"]
            content = source["messages"][message_index]["content"]
            identity = stable_hash(
                {
                    "conversation_hash": source["conversation_hash"],
                    "condition": unit["condition"],
                    "message_index": message_index,
                    "content": content,
                }
            )
            rows.append(
                {
                    "original_row_idx": source["original_row_idx"],
                    "source": source["source"],
                    "conversation_hash": source["conversation_hash"],
                    "condition": unit["condition"],
                    "repetition": 0,
                    **{key: value for key, value in unit.items() if key != "condition"},
                    "target_text": preceding_user(source["messages"], message_index),
                    "response": content,
                    "generation_sha256": identity,
                }
            )

    write_jsonl(args.output, rows)
    manifest = {
        "cohort_sha256": sha256_file(args.cohort),
        "selections_sha256": sha256_file(args.selections),
        "conversations": len(candidate_counts),
        "candidate_limit": args.candidate_limit,
        "scoring_rows": len(rows),
        "minimum_candidates": min(candidate_counts),
        "mean_candidates": sum(candidate_counts) / len(candidate_counts),
        "selection_plan": (
            "after frozen endorsement scoring, choose the nearest earlier assistant "
            "candidate scoring below the endorsement threshold"
        ),
        "output_sha256": sha256_file(args.output),
    }
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
