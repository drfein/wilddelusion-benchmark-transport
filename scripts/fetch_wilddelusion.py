#!/usr/bin/env python3
"""Fetch the immutable 522-row public release and emit canonical JSONL."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from datasets import load_dataset

DEFAULT_REPOSITORY = "danielfein/WildDelusionCombined"
DEFAULT_REVISION = "f547346d490fe7b5be9553a24eebe02aee777015"


def normalize(value: Any) -> Any:
    if hasattr(value, "tolist"):
        return normalize(value.tolist())
    if isinstance(value, dict):
        return {str(key): normalize(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [normalize(item) for item in value]
    return value


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repository", default=DEFAULT_REPOSITORY)
    parser.add_argument("--revision", default=DEFAULT_REVISION)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--theme-labels", type=Path, required=True)
    args = parser.parse_args()

    dataset = load_dataset(
        args.repository,
        split="train",
        revision=args.revision,
    )
    rows = [normalize(dict(row)) for row in dataset]
    if len(rows) != 522:
        raise ValueError(f"expected 522 rows, received {len(rows)}")
    required = {"source", "conversation_id", "messages", "target_message_index"}
    missing = required - set(rows[0])
    if missing:
        raise ValueError(f"release schema is missing {sorted(missing)}")
    labels = [
        json.loads(line)
        for line in args.theme_labels.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    labels_by_hash = {row["message_hash"]: row for row in labels}
    if len(labels) != 522 or len(labels_by_hash) != 522:
        raise ValueError("theme-label map must contain 522 unique message hashes")
    for row in rows:
        message_hash = row["message_hash"]
        if message_hash not in labels_by_hash:
            raise ValueError(f"missing frozen theme label for {message_hash}")
        row.update(
            {
                key: value
                for key, value in labels_by_hash[message_hash].items()
                if key != "message_hash"
            }
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    temporary.replace(args.output)
    manifest = {
        "repository": args.repository,
        "revision": args.revision,
        "split": "train",
        "rows": len(rows),
        "theme_labels_sha256": sha256(args.theme_labels),
        "sha256": sha256(args.output),
    }
    args.output.with_suffix(args.output.suffix + ".manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
