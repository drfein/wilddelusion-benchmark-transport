from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq


ALLOWED_CHANGED_COLUMNS = {"messages", "target_message_index"}
EXPECTED_HISTORY_COLUMNS = {
    "history_all_message_text_available",
    "history_complete",
    "history_message_count",
    "history_original_target_message_index",
    "history_prefix_has_assistant",
    "history_rehydration_method",
    "history_sha256",
    "history_source_message_count_matches_metadata",
    "history_source_reported_message_count",
    "history_source_revision",
    "history_unavailable_message_count",
}


def canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def canonical_digest(rows: list[dict[str, Any]]) -> str:
    ordered = sorted(rows, key=lambda row: (str(row["source"]), str(row["message_hash"])))
    return hashlib.sha256(canonical_json(ordered).encode("utf-8")).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate a rehydrated release against its input.")
    parser.add_argument("--original", type=Path, required=True)
    parser.add_argument("--rehydrated", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()

    original_table = pq.read_table(args.original)
    rehydrated_table = pq.read_table(args.rehydrated)
    original = original_table.to_pylist()
    rehydrated = rehydrated_table.to_pylist()
    if len(original) != 522 or len(rehydrated) != len(original):
        raise ValueError(f"Unexpected row counts: original={len(original)}, rehydrated={len(rehydrated)}")
    original_columns = set(original[0])
    rehydrated_columns = set(rehydrated[0])
    if rehydrated_columns - original_columns != EXPECTED_HISTORY_COLUMNS:
        raise ValueError("Unexpected added or missing history columns")

    unchanged_columns = sorted(original_columns - ALLOWED_CHANGED_COLUMNS)
    for column in unchanged_columns:
        if original_table.schema.field(column).type != rehydrated_table.schema.field(column).type:
            raise ValueError(f"Arrow type changed for untouched column {column}")
    target_errors = 0
    unchanged_errors: list[dict[str, Any]] = []
    transcript_by_conversation: dict[tuple[str, str], set[str]] = {}
    for index, (before, after) in enumerate(zip(original, rehydrated, strict=True)):
        if before["message_hash"] != after["message_hash"]:
            raise ValueError(f"Row identity changed at position {index}")
        for column in unchanged_columns:
            if canonical_json(before[column]) != canonical_json(after[column]):
                unchanged_errors.append({"row": index, "column": column})
        target_index = int(after["target_message_index"])
        messages = after["messages"]
        target = messages[target_index]
        if target["role"] != "user" or target["content"] != after["target_text"]:
            target_errors += 1
        if int(after["history_original_target_message_index"]) != int(before["target_message_index"]):
            raise ValueError(f"Original target index not preserved at row {index}")
        if int(after["history_message_count"]) != len(messages):
            raise ValueError(f"Message count mismatch at row {index}")
        if not any(message["role"] == "assistant" for message in messages):
            raise ValueError(f"No assistant message at row {index}")
        key = (str(after["source"]), str(after["conversation_id"]))
        transcript_by_conversation.setdefault(key, set()).add(str(after["history_sha256"]))

    if unchanged_errors:
        raise ValueError(f"Unrelated source columns changed: {unchanged_errors[:10]}")
    if target_errors:
        raise ValueError(f"Target integrity errors: {target_errors}")
    inconsistent = [key for key, hashes in transcript_by_conversation.items() if len(hashes) != 1]
    if inconsistent:
        raise ValueError(f"Source conversations map to multiple transcripts: {inconsistent[:10]}")
    if not all(bool(row["history_complete"]) for row in rehydrated):
        raise ValueError("At least one transcript is not structurally complete")

    text_complete = sum(bool(row["history_all_message_text_available"]) for row in rehydrated)
    unavailable_nodes = sum(int(row["history_unavailable_message_count"]) for row in rehydrated)
    report = {
        "passed": True,
        "rows": len(rehydrated),
        "source_conversations": len(transcript_by_conversation),
        "unchanged_original_columns": len(unchanged_columns),
        "changed_original_columns": sorted(ALLOWED_CHANGED_COLUMNS),
        "added_history_columns": sorted(EXPECTED_HISTORY_COLUMNS),
        "target_integrity_errors": 0,
        "transcript_consistency_errors": 0,
        "rows_with_assistant_messages": len(rehydrated),
        "rows_with_all_message_text_available": text_complete,
        "rows_with_unavailable_message_text": len(rehydrated) - text_complete,
        "unavailable_source_message_nodes": unavailable_nodes,
        "source_metadata_count_mismatches": sum(
            row["history_source_message_count_matches_metadata"] is False for row in rehydrated
        ),
        "original_parquet_sha256": sha256(args.original),
        "rehydrated_parquet_sha256": sha256(args.rehydrated),
        "canonical_sha256": canonical_digest(rehydrated),
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
