#!/usr/bin/env python3
"""Install immutable validated counterfactuals into the exact study layout."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def copy_group(source: Path, destination: Path) -> list[dict[str, object]]:
    destination.mkdir(parents=True, exist_ok=True)
    copied = []
    for path in sorted(source.iterdir()):
        if not path.is_file():
            continue
        target = destination / path.name
        shutil.copy2(path, target)
        copied.append(
            {"source": str(path), "output": str(target), "sha256": sha256(target)}
        )
    return copied


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, required=True)
    args = parser.parse_args()
    repo = args.repo.resolve()
    experiment = (
        repo / "experiments" / "lost_in_delusion_replication_260600975"
    )
    copied = []
    copied.extend(
        copy_group(
            repo / "frozen" / "generated_controls",
            experiment / "full_history_522" / "artifacts",
        )
    )
    copied.extend(
        copy_group(
            repo / "frozen" / "psychogenic_pairs",
            experiment
            / "psychogenic_machine_transport_250910970"
            / "full522"
            / "artifacts",
        )
    )
    manifest = {"files": copied, "count": len(copied)}
    output = repo / ".repro" / "frozen_examples.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
