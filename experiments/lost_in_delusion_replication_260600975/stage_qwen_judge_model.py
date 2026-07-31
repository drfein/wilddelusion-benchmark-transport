#!/usr/bin/env python3
"""Stage a pinned Hugging Face snapshot across disk and RAM storage."""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import os
import subprocess
import time
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(16 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def fetch_metadata(model: str, revision: str) -> dict[str, Any]:
    encoded = urllib.parse.quote(model, safe="/")
    url = (
        f"https://huggingface.co/api/models/{encoded}/revision/"
        f"{revision}?blobs=true"
    )
    with urllib.request.urlopen(url) as response:
        metadata = json.load(response)
    if metadata.get("sha") != revision:
        raise RuntimeError(
            f"resolved revision {metadata.get('sha')} does not match {revision}"
        )
    return metadata


def download(
    *,
    model: str,
    revision: str,
    filename: str,
    destination: Path,
    expected_size: int,
    expected_sha256: str | None,
) -> dict[str, Any]:
    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_name(destination.name + ".part")
    if destination.exists() and destination.stat().st_size == expected_size:
        if not expected_sha256 or sha256(destination) == expected_sha256:
            return {"filename": filename, "status": "existing"}
        destination.unlink()
    quoted_name = urllib.parse.quote(filename, safe="/")
    encoded_model = urllib.parse.quote(model, safe="/")
    url = (
        f"https://huggingface.co/{encoded_model}/resolve/{revision}/"
        f"{quoted_name}?download=true"
    )
    subprocess.run(
        [
            "curl",
            "--location",
            "--fail",
            "--retry",
            "12",
            "--retry-all-errors",
            "--retry-delay",
            "5",
            "--continue-at",
            "-",
            "--silent",
            "--show-error",
            "--output",
            str(partial),
            url,
        ],
        check=True,
    )
    if partial.stat().st_size != expected_size:
        raise RuntimeError(
            f"{filename}: got {partial.stat().st_size}, expected {expected_size}"
        )
    actual_sha256 = sha256(partial) if expected_sha256 else None
    if expected_sha256 and actual_sha256 != expected_sha256:
        raise RuntimeError(
            f"{filename}: sha256 {actual_sha256}, expected {expected_sha256}"
        )
    os.replace(partial, destination)
    return {"filename": filename, "status": "downloaded"}


def run(args: argparse.Namespace) -> None:
    metadata = fetch_metadata(args.model, args.revision)
    args.model_dir.mkdir(parents=True, exist_ok=True)
    args.ram_dir.mkdir(parents=True, exist_ok=True)

    siblings = []
    for entry in metadata["siblings"]:
        lfs = entry.get("lfs") or {}
        siblings.append(
            {
                "filename": entry["rfilename"],
                "size": int(lfs.get("size", entry.get("size") or 0)),
                "sha256": lfs.get("sha256"),
            }
        )
    weight_files = sorted(
        (
            entry
            for entry in siblings
            if entry["filename"].endswith(".safetensors")
        ),
        key=lambda entry: entry["filename"],
    )
    metadata_files = [
        entry for entry in siblings if not entry["filename"].endswith(".safetensors")
    ]
    if not weight_files:
        raise RuntimeError("snapshot has no safetensor files")

    jobs = []
    placement = {}
    for index, entry in enumerate(weight_files):
        root = args.model_dir if index < args.disk_shards else args.ram_dir
        destination = root / entry["filename"]
        placement[entry["filename"]] = (
            "disk" if root == args.model_dir else "ram"
        )
        jobs.append((entry, destination))
    jobs.extend((entry, args.model_dir / entry["filename"]) for entry in metadata_files)

    started = time.time()
    with concurrent.futures.ThreadPoolExecutor(
        max_workers=args.concurrency
    ) as executor:
        future_to_job = {
            executor.submit(
                download,
                model=args.model,
                revision=args.revision,
                filename=entry["filename"],
                destination=destination,
                expected_size=entry["size"],
                expected_sha256=entry["sha256"],
            ): (entry, destination)
            for entry, destination in jobs
        }
        complete = 0
        for future in concurrent.futures.as_completed(future_to_job):
            entry, destination = future_to_job[future]
            result = future.result()
            complete += 1
            print(
                f"[{complete:02d}/{len(jobs):02d}] {result['status']}: "
                f"{entry['filename']} -> {destination}",
                flush=True,
            )

    for entry in weight_files[args.disk_shards :]:
        link = args.model_dir / entry["filename"]
        target = args.ram_dir / entry["filename"]
        if link.is_symlink() or link.exists():
            link.unlink()
        link.symlink_to(target)

    manifest = {
        "model_id": args.model,
        "revision": args.revision,
        "resolved_revision": metadata["sha"],
        "model_dir": str(args.model_dir),
        "ram_dir": str(args.ram_dir),
        "weight_bytes": sum(entry["size"] for entry in weight_files),
        "disk_shards": args.disk_shards,
        "ram_shards": len(weight_files) - args.disk_shards,
        "elapsed_seconds": time.time() - started,
        "files": [
            {
                **entry,
                "placement": placement.get(entry["filename"], "disk_metadata"),
            }
            for entry in siblings
        ],
    }
    manifest_path = args.model_dir / "staging_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model", default="Qwen/Qwen3-30B-A3B-Thinking-2507"
    )
    parser.add_argument(
        "--revision",
        default="144afc2f379b542fdd4e85a1fcd5e1f79112d95d",
    )
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--ram-dir", type=Path, required=True)
    parser.add_argument("--disk-shards", type=int, default=5)
    parser.add_argument("--concurrency", type=int, default=6)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
