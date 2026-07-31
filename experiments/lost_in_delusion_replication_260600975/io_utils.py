#!/usr/bin/env python3
"""Small JSONL helpers shared by the replication scripts."""

from __future__ import annotations

import json
import os
import hashlib
from pathlib import Path
from typing import Any, Callable


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    os.replace(temporary, path)


def append_jsonl(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def canonicalize(
    path: Path,
    key: Callable[[dict[str, Any]], str],
    successful: Callable[[dict[str, Any]], bool],
) -> list[dict[str, Any]]:
    """Keep one row per key, preferring the newest successful result."""
    selected: dict[str, dict[str, Any]] = {}
    order: list[str] = []
    for row in read_jsonl(path):
        row_key = key(row)
        if row_key not in selected:
            order.append(row_key)
            selected[row_key] = row
        elif successful(row) or not successful(selected[row_key]):
            selected[row_key] = row
    rows = [selected[row_key] for row_key in order]
    if rows:
        write_jsonl(path, rows)
    return rows


def write_manifest(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")


def canonical_json_sha256(value: Any) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def messages_for_judging(row: dict[str, Any]) -> list[dict[str, str]]:
    """Resolve the explicitly frozen judge context before legacy fallbacks."""
    messages = (
        row.get("judge_messages")
        or row.get("messages_used")
        or row.get("messages")
    )
    if not isinstance(messages, list) or not messages:
        raise ValueError("A non-empty judge conversation is required")
    return messages
