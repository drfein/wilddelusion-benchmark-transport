#!/usr/bin/env python3
"""Stage a pinned Hugging Face model across disk and RAM-backed storage."""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
import time
from pathlib import Path
from typing import Any

from huggingface_hub import HfApi, hf_hub_download


def needed_metadata(filename: str) -> bool:
    name = filename.lower()
    basename = Path(name).name
    if name.endswith(".safetensors"):
        return False
    if any(
        name.endswith(suffix)
        for suffix in (".bin", ".pt", ".pth", ".onnx", ".gguf")
    ):
        return False
    return (
        basename.startswith(("config", "tokenizer", "generation_config"))
        or basename
        in {
            "added_tokens.json",
            "chat_template.jinja",
            "merges.txt",
            "model.safetensors.index.json",
            "preprocessor_config.json",
            "special_tokens_map.json",
            "tokenizer.model",
            "vocab.json",
        }
        or name.endswith((".py", ".jinja", ".tiktoken"))
    )


def sibling_record(sibling: Any) -> dict[str, Any]:
    lfs = sibling.lfs
    return {
        "filename": sibling.rfilename,
        "size": int(sibling.size or (lfs.size if lfs else 0) or 0),
        "sha256": lfs.sha256 if lfs else None,
    }


def download_file(
    model: str,
    revision: str,
    filename: str,
    root: Path,
) -> str:
    root.mkdir(parents=True, exist_ok=True)
    return hf_hub_download(
        repo_id=model,
        filename=filename,
        revision=revision,
        local_dir=root,
    )


def run(args: argparse.Namespace) -> None:
    api = HfApi()
    metadata = api.model_info(
        args.model,
        revision=args.revision,
        files_metadata=True,
    )
    if metadata.sha != args.revision:
        raise RuntimeError(
            f"resolved revision {metadata.sha} != pinned {args.revision}"
        )
    siblings = [sibling_record(item) for item in metadata.siblings]
    weights = sorted(
        (
            item
            for item in siblings
            if item["filename"].endswith(".safetensors")
        ),
        key=lambda item: item["filename"],
    )
    if not weights:
        raise RuntimeError("No safetensor weights found")
    if not 0 <= args.disk_shards <= len(weights):
        raise ValueError(
            f"disk_shards must be in [0, {len(weights)}], "
            f"got {args.disk_shards}"
        )
    metadata_files = [
        item for item in siblings if needed_metadata(item["filename"])
    ]
    args.model_dir.mkdir(parents=True, exist_ok=True)
    args.ram_dir.mkdir(parents=True, exist_ok=True)

    placements: dict[str, str] = {}
    jobs: list[tuple[dict[str, Any], Path]] = []
    for index, item in enumerate(weights):
        root = args.model_dir if index < args.disk_shards else args.ram_dir
        placements[item["filename"]] = (
            "disk" if root == args.model_dir else "ram"
        )
        jobs.append((item, root))
    for item in metadata_files:
        placements[item["filename"]] = "disk_metadata"
        jobs.append((item, args.model_dir))

    started = time.time()
    with concurrent.futures.ThreadPoolExecutor(
        max_workers=args.concurrency
    ) as executor:
        futures = {
            executor.submit(
                download_file,
                args.model,
                args.revision,
                item["filename"],
                root,
            ): (item, root)
            for item, root in jobs
        }
        for number, future in enumerate(
            concurrent.futures.as_completed(futures), start=1
        ):
            item, root = futures[future]
            path = Path(future.result())
            if item["size"] and path.stat().st_size != item["size"]:
                raise RuntimeError(
                    f"{item['filename']}: {path.stat().st_size} bytes, "
                    f"expected {item['size']}"
                )
            print(
                f"[{number:02d}/{len(jobs):02d}] "
                f"{item['filename']} -> {root}",
                flush=True,
            )

    for item in weights[args.disk_shards :]:
        link = args.model_dir / item["filename"]
        target = args.ram_dir / item["filename"]
        if link.is_symlink() or link.exists():
            link.unlink()
        link.parent.mkdir(parents=True, exist_ok=True)
        link.symlink_to(target)

    manifest = {
        "model_id": args.model,
        "revision": args.revision,
        "resolved_revision": metadata.sha,
        "model_dir": str(args.model_dir),
        "ram_dir": str(args.ram_dir),
        "weight_bytes": sum(item["size"] for item in weights),
        "disk_weight_bytes": sum(
            item["size"] for item in weights[: args.disk_shards]
        ),
        "ram_weight_bytes": sum(
            item["size"] for item in weights[args.disk_shards :]
        ),
        "disk_shards": args.disk_shards,
        "ram_shards": len(weights) - args.disk_shards,
        "elapsed_seconds": time.time() - started,
        "files": [
            {**item, "placement": placements[item["filename"]]}
            for item in weights + metadata_files
        ],
    }
    (args.model_dir / "staging_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--ram-dir", type=Path, required=True)
    parser.add_argument("--disk-shards", type=int, required=True)
    parser.add_argument("--concurrency", type=int, default=6)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
