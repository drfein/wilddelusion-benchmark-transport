#!/usr/bin/env python3
"""Download one model at an exact Hugging Face revision."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from huggingface_hub import snapshot_download


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    path = snapshot_download(
        repo_id=args.model,
        revision=args.revision,
        local_dir=args.output,
        token=os.getenv("HF_TOKEN"),
    )
    manifest = {
        "model": args.model,
        "revision": args.revision,
        "local_dir": str(Path(path).resolve()),
    }
    (args.output / ".wilddelusion-model.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
