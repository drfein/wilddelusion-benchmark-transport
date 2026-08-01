#!/usr/bin/env python3
"""Extract one preregistered real delusion endpoint per source conversation."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

from design import (
    SOURCE_EXPECTED_CONVERSATIONS,
    SOURCE_EXPECTED_DELUSION_ROWS,
    STUDY_ID,
    stable_hash,
)
from io_utils import sha256_file, write_jsonl


def conversation_key(row: dict[str, Any]) -> str:
    return f"{row['source']}::{row['conversation_id']}"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    with args.input.open(encoding="utf-8") as handle:
        source = [json.loads(line) for line in handle if line.strip()]
    delusion = [row for row in source if row.get("condition") == "delusion"]
    if len(delusion) != SOURCE_EXPECTED_DELUSION_ROWS:
        raise ValueError(
            f"Expected {SOURCE_EXPECTED_DELUSION_ROWS} delusion rows, found {len(delusion)}"
        )

    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in delusion:
        key = conversation_key(row)
        grouped.setdefault(key, []).append(row)
        index = int(row["target_message_index"])
        messages = row["messages"]
        if not 0 <= index < len(messages):
            raise ValueError(f"{key}: target index outside message sequence")
        if messages[index].get("role") != "user":
            raise ValueError(f"{key}: target is not a user message")
    if len(grouped) != SOURCE_EXPECTED_CONVERSATIONS:
        raise ValueError(
            f"Expected {SOURCE_EXPECTED_CONVERSATIONS} conversations, found {len(grouped)}"
        )

    candidates = []
    for key, rows in grouped.items():
        # The latest validated endpoint maximizes natural pre-target history.
        row = max(rows, key=lambda item: int(item["target_message_index"]))
        target_index = int(row["target_message_index"])
        messages = [
            {"role": str(item["role"]), "content": str(item["content"])}
            for item in row["messages"][: target_index + 1]
        ]
        if not messages or messages[-1]["role"] != "user":
            raise ValueError(f"{key}: extracted history does not end with user")
        candidates.append(
            {
                "candidate_id": stable_hash(STUDY_ID, key, target_index)[:24],
                "conversation_key": key,
                "conversation_id": str(row["conversation_id"]),
                "source": str(row["source"]),
                "theme": str(row.get("theme") or "unknown"),
                "pair_id": str(row["pair_id"]),
                "input_id": str(row["input_id"]),
                "target_message_index": target_index,
                "source_message_count": len(row["messages"]),
                "messages": messages,
                "target_text": messages[-1]["content"],
                "selection_rule": "latest_validated_delusion_endpoint_per_conversation",
            }
        )
    candidates.sort(key=lambda row: row["conversation_key"])
    write_jsonl(args.output, candidates)

    manifest = {
        "study_id": STUDY_ID,
        "input": str(args.input),
        "input_sha256": sha256_file(args.input),
        "source_rows": len(source),
        "delusion_rows": len(delusion),
        "candidate_conversations": len(candidates),
        "selection_rule": "latest_validated_delusion_endpoint_per_conversation",
        "themes": Counter(row["theme"] for row in candidates),
        "sources": Counter(row["source"] for row in candidates),
    }
    manifest_path = args.output.with_suffix(args.output.suffix + ".manifest.json")
    manifest_path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False, default=dict) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest, indent=2, ensure_ascii=False, default=dict))


if __name__ == "__main__":
    main()

