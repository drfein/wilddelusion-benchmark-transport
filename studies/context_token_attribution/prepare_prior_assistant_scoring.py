from __future__ import annotations

import argparse
import json
from pathlib import Path

from config import stable_hash
from io_utils import read_jsonl, sha256_file, write_jsonl


def latest_exchange(messages: list[dict]) -> tuple[int, int]:
    assistant_index = next(
        index
        for index in range(len(messages) - 2, 0, -1)
        if messages[index]["role"] == "assistant"
    )
    user_index = next(
        index
        for index in range(assistant_index - 1, 0, -1)
        if messages[index]["role"] == "user"
    )
    return user_index, assistant_index


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cohort", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    args = parser.parse_args()

    rows = []
    for source in read_jsonl(args.cohort):
        user_index, assistant_index = latest_exchange(source["messages"])
        user = source["messages"][user_index]["content"]
        assistant = source["messages"][assistant_index]["content"]
        identity = stable_hash(
            {
                "conversation_hash": source["conversation_hash"],
                "user_index": user_index,
                "assistant_index": assistant_index,
                "user": user,
                "assistant": assistant,
            }
        )
        rows.append(
            {
                "original_row_idx": source["original_row_idx"],
                "source": source["source"],
                "conversation_hash": source["conversation_hash"],
                "repetition": 0,
                "prior_user_message_index": user_index,
                "prior_assistant_message_index": assistant_index,
                "target_text": user,
                "response": assistant,
                "generation_sha256": identity,
            }
        )
    write_jsonl(args.output, rows)
    manifest = {
        "cohort_sha256": sha256_file(args.cohort),
        "rows": len(rows),
        "unit": "most recent complete user-assistant exchange before the target user",
        "scoring": (
            "the same frozen bot-endorses-delusion rubric used for generated responses"
        ),
        "output_sha256": sha256_file(args.output),
    }
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
