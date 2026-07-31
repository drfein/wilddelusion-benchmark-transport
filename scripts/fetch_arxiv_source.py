#!/usr/bin/env python3
"""Download and safely extract an arXiv source bundle."""

from __future__ import annotations

import argparse
import hashlib
import inspect
import io
import json
import tarfile
import urllib.request
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--paper", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expected-sha256")
    args = parser.parse_args()
    url = f"https://arxiv.org/e-print/{args.paper}"
    with urllib.request.urlopen(url, timeout=120) as response:
        payload = response.read()
    payload_sha256 = hashlib.sha256(payload).hexdigest()
    if args.expected_sha256 and payload_sha256 != args.expected_sha256:
        raise ValueError(
            f"source hash mismatch: expected {args.expected_sha256}, "
            f"received {payload_sha256}"
        )
    args.output.mkdir(parents=True, exist_ok=True)
    with tarfile.open(fileobj=io.BytesIO(payload), mode="r:*") as archive:
        destination = args.output.resolve()
        members = []
        for member in archive.getmembers():
            if member.issym() or member.islnk():
                raise ValueError(f"archive links are not allowed: {member.name}")
            if not member.isfile() and not member.isdir():
                raise ValueError(f"unsupported archive member: {member.name}")
            target = (destination / member.name).resolve()
            if destination != target and destination not in target.parents:
                raise ValueError(f"unsafe archive member: {member.name}")
            members.append(member)
        kwargs = {"filter": "data"} if "filter" in inspect.signature(
            archive.extractall
        ).parameters else {}
        archive.extractall(destination, members=members, **kwargs)
    manifest = {"paper": args.paper, "url": url, "bytes": len(payload)}
    manifest["payload_sha256"] = payload_sha256
    (args.output / ".wilddelusion-source.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
