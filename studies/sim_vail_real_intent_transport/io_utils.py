"""Small auditable JSONL and hashing helpers for this study."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any, Callable


def read_jsonl(path: Path) -> list[dict[str, Any]]:
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
    if not path.exists():
        return []
    selected: dict[str, dict[str, Any]] = {}
    order: list[str] = []
    for row in read_jsonl(path):
        row_key = key(row)
        if row_key not in selected:
            order.append(row_key)
            selected[row_key] = row
        elif not successful(selected[row_key]) and successful(row):
            selected[row_key] = row
    rows = [selected[row_key] for row_key in order]
    write_jsonl(path, rows)
    return rows


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()

