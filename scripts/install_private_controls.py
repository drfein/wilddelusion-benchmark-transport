#!/usr/bin/env python3
"""Install licensed natural-control files without adding them to Git."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path

REQUIRED = (
    "strict_nonsincere_controls.jsonl",
    "high_confidence_strict_controls.jsonl",
    "human_confirmed_controls.jsonl",
    "human_verifier_disagreements.jsonl",
)


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--destination", type=Path, required=True)
    args = parser.parse_args()
    source = args.source.expanduser().resolve()
    missing = [name for name in REQUIRED if not (source / name).is_file()]
    if missing:
        raise FileNotFoundError(f"private control bundle lacks {missing}")
    args.destination.mkdir(parents=True, exist_ok=True)
    rows = []
    for name in REQUIRED:
        destination = args.destination / name
        shutil.copy2(source / name, destination)
        rows.append({"file": name, "sha256": digest(destination)})
    manifest = {"source": str(source), "files": rows}
    (args.destination / "install.manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
