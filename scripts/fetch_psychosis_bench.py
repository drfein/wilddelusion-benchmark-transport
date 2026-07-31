#!/usr/bin/env python3
"""Fetch the exact upstream Psychosis-Bench commit used by the study."""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

REPOSITORY = "https://github.com/w-is-h/psychosis-bench.git"
COMMIT = "73966f95be2e40f1ceee76dfe08eebe939ad6c21"


def run(*args: str, cwd: Path | None = None) -> str:
    result = subprocess.run(
        list(args), cwd=cwd, check=True, text=True, capture_output=True
    )
    return result.stdout.strip()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    if not (output / ".git").exists():
        output.mkdir(parents=True, exist_ok=True)
        run("git", "init", str(output))
        run("git", "remote", "add", "origin", REPOSITORY, cwd=output)
    run("git", "fetch", "--depth", "1", "origin", COMMIT, cwd=output)
    run("git", "checkout", "--detach", "FETCH_HEAD", cwd=output)
    resolved = run("git", "rev-parse", "HEAD", cwd=output)
    if resolved != COMMIT:
        raise ValueError(f"expected {COMMIT}, checked out {resolved}")
    manifest = {"repository": REPOSITORY, "commit": resolved}
    (output / ".wilddelusion-source.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
