from __future__ import annotations

import argparse
import json
from pathlib import Path

from config import stable_hash
from io_utils import read_jsonl, sha256_file, write_jsonl
from prepare_assistant_swap_interventions import preceding_user


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cohort", type=Path, required=True)
    parser.add_argument("--selections", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    args = parser.parse_args()

    cohort = {row["conversation_hash"]: row for row in read_jsonl(args.cohort)}
    selections = read_jsonl(args.selections)
    rows = []
    for selection in selections:
        source = cohort[selection["conversation_hash"]]
        for condition, key in (
            ("attributed", "top_message_index"),
            ("matched_control", "control_message_index"),
        ):
            message_index = int(selection[key])
            message = source["messages"][message_index]
            if message["role"] != "assistant":
                raise ValueError(f"Selected unit {message_index} is not assistant")
            identity = stable_hash(
                {
                    "conversation_hash": source["conversation_hash"],
                    "condition": condition,
                    "message_index": message_index,
                    "content": message["content"],
                }
            )
            rows.append(
                {
                    "original_row_idx": source["original_row_idx"],
                    "source": source["source"],
                    "conversation_hash": source["conversation_hash"],
                    "condition": condition,
                    "repetition": 0,
                    "message_index": message_index,
                    "target_text": preceding_user(source["messages"], message_index),
                    "response": message["content"],
                    "generation_sha256": identity,
                }
            )

    write_jsonl(args.output, rows)
    manifest = {
        "cohort_sha256": sha256_file(args.cohort),
        "selections_sha256": sha256_file(args.selections),
        "selected_conversations": len(selections),
        "scoring_rows": len(rows),
        "unit": "complete attributed or matched-control assistant message",
        "output_sha256": sha256_file(args.output),
    }
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
