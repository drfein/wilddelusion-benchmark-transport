#!/usr/bin/env python3
"""Copy the generated claim registry and completion audit to the repo root."""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

FILES = (
    "README.md",
    "claim_registry.json",
    "claim_registry.csv",
    "COMPLETION_AUDIT.json",
    "COMPLETION_AUDIT.md",
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--destination", type=Path, required=True)
    args = parser.parse_args()
    args.destination.mkdir(parents=True, exist_ok=True)
    for name in FILES:
        source = args.source / name
        if not source.is_file():
            raise FileNotFoundError(source)
        shutil.copy2(source, args.destination / name)


if __name__ == "__main__":
    main()
