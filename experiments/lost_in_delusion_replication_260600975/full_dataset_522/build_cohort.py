#!/usr/bin/env python3
"""Build the canonical 522-target WildDelusion transport cohort."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_INPUT = (
    PROJECT_ROOT
    / "results/merged_delusion_taxonomy_v2/wilddelusion_combined_with_themes.jsonl"
)


def normalized_history(row: dict[str, Any]) -> list[dict[str, str]]:
    target_index = int(row["target_message_index"])
    history: list[dict[str, str]] = []
    for raw in row["messages"][: target_index + 1]:
        role = str(raw.get("role", "")).strip().lower()
        if role == "llm":
            role = "assistant"
        content = str(raw.get("content", "")).strip()
        if role in {"user", "assistant"} and content:
            history.append({"role": role, "content": content})
    while history and history[0]["role"] != "user":
        history.pop(0)
    if not history or history[-1]["role"] != "user":
        raise ValueError(f"{row['message_hash']}: history does not end in user")
    if history[-1]["content"] != str(row["target_text"]).strip():
        raise ValueError(f"{row['message_hash']}: normalized target mismatch")
    return history


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument(
        "--output", type=Path, default=Path("artifacts/cohort_full522.jsonl")
    )
    args = parser.parse_args()

    source_rows = [
        json.loads(line)
        for line in args.input.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    rows: list[dict[str, Any]] = []
    for source in source_rows:
        history = normalized_history(source)
        pair_id = str(source["message_hash"])
        cluster_id = f"{source['source']}:{source['conversation_id']}"
        rows.append(
            {
                "pair_id": pair_id,
                "cluster_id": cluster_id,
                "source": source["source"],
                "conversation_id": str(source["conversation_id"]),
                "message_hash": pair_id,
                "target_message_index": int(source["target_message_index"]),
                "target_text": history[-1]["content"],
                "history_messages": history,
                "theme": source.get("delusion_theme_primary"),
                "theme_secondary": source.get("delusion_theme_secondary") or [],
                "theme_confidence": source.get("delusion_theme_confidence"),
                "discovery_split": source.get("discovery_split"),
                "annotation_score": source.get("annotation_score"),
                "annotation_rationale": source.get("annotation_rationale"),
                "annotation_quotes": source.get("annotation_quotes") or [],
                "judge_rationale": source.get("judge_rationale"),
                "judge_supporting_quotes": source.get("judge_supporting_quotes")
                or [],
            }
        )

    rows.sort(key=lambda row: row["pair_id"])
    if len(rows) != 522:
        raise ValueError(f"expected 522 targets, found {len(rows)}")
    if len({row["pair_id"] for row in rows}) != len(rows):
        raise ValueError("pair IDs are not unique")

    write_jsonl(args.output, rows)
    digest = hashlib.sha256(args.output.read_bytes()).hexdigest()
    manifest = {
        "rows": len(rows),
        "source_conversations": len({row["cluster_id"] for row in rows}),
        "target_integrity_errors": 0,
        "sources": dict(Counter(row["source"] for row in rows)),
        "themes": dict(Counter(row["theme"] for row in rows)),
        "discovery_splits": dict(
            Counter(row["discovery_split"] for row in rows)
        ),
        "max_history_messages": max(len(row["history_messages"]) for row in rows),
        "max_history_characters": max(
            sum(len(message["content"]) for message in row["history_messages"])
            for row in rows
        ),
        "sha256": digest,
    }
    manifest_path = args.output.with_suffix(args.output.suffix + ".manifest.json")
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
