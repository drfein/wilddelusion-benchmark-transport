#!/usr/bin/env python3
"""Verify and reproduce the archived SIM-VAIL real-context analysis."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import shutil
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent
RAW = ROOT / "raw"
DEFAULT_WORK = ROOT.parents[1] / ".repro" / "sim_vail_real_intent_transport"


def sha256_stream(stream) -> str:
    digest = hashlib.sha256()
    for chunk in iter(lambda: stream.read(1024 * 1024), b""):
        digest.update(chunk)
    return digest.hexdigest()


def load_manifest() -> dict:
    return json.loads((RAW / "manifest.json").read_text(encoding="utf-8"))


def verify_archives() -> None:
    for entry in load_manifest()["files"]:
        archive = RAW / entry["archive"]
        with archive.open("rb") as stream:
            archive_hash = sha256_stream(stream)
        if archive_hash != entry["archive_sha256"]:
            raise RuntimeError(f"Compressed hash mismatch: {archive}")
        with gzip.open(archive, "rb") as stream:
            raw_hash = sha256_stream(stream)
        if raw_hash != entry["raw_sha256"]:
            raise RuntimeError(f"Raw hash mismatch: {archive}")
    print(f"Verified {len(load_manifest()['files'])} compressed raw artifacts.")


def materialize(output: Path) -> Path:
    data = output / "data"
    data.mkdir(parents=True, exist_ok=True)
    for entry in load_manifest()["files"]:
        archive = RAW / entry["archive"]
        target = data / entry["raw_file"]
        if target.exists():
            with target.open("rb") as stream:
                if sha256_stream(stream) == entry["raw_sha256"]:
                    continue
        temporary = target.with_suffix(target.suffix + ".tmp")
        with gzip.open(archive, "rb") as source, temporary.open("wb") as destination:
            shutil.copyfileobj(source, destination)
        with temporary.open("rb") as stream:
            if sha256_stream(stream) != entry["raw_sha256"]:
                temporary.unlink(missing_ok=True)
                raise RuntimeError(f"Materialized hash mismatch: {target}")
        temporary.replace(target)
    return data


def run(command: list[str]) -> None:
    print("+", " ".join(command))
    subprocess.run(command, check=True, cwd=ROOT)


def assert_same(path_a: Path, path_b: Path) -> None:
    if path_a.read_bytes() != path_b.read_bytes():
        raise RuntimeError(f"Reproduced artifact differs: {path_a.name}")


def analyze(work: Path) -> None:
    data = materialize(work)
    output = work / "results"
    calibration = work / "judge_calibration_exact"
    output.mkdir(parents=True, exist_ok=True)
    calibration.mkdir(parents=True, exist_ok=True)

    run(
        [
            sys.executable,
            str(ROOT / "analyze_judge_calibration.py"),
            "--calibration",
            str(data / "judge_calibration_exact_200.jsonl"),
            "--scores",
            str(data / "judge_calibration_exact_scores_gpt52.jsonl"),
            "--output",
            str(calibration),
        ]
    )
    run(
        [
            sys.executable,
            str(ROOT / "analyze_outcomes.py"),
            "--generations",
            str(data / "primary_judge_input_960_1024.jsonl"),
            "--scores",
            str(data / "primary_scores_exact_gpt52_960.jsonl"),
            "--output-dir",
            str(output),
            "--manual-audit-status",
            "passed",
        ]
    )
    run(
        [
            sys.executable,
            str(ROOT / "plot_primary_results.py"),
            "--input",
            str(output / "primary_results.json"),
            "--output",
            str(output / "primary_effects.png"),
        ]
    )

    published = ROOT / "results" / "primary_exact"
    for name in (
        "primary_results.json",
        "secondary_results.json",
        "model_interactions.json",
        "source_sensitivity.json",
        "conversation_pair_deltas.csv",
        "replicate_pair_deltas.csv",
    ):
        assert_same(output / name, published / name)
    for name in ("summary.json", "dimension_metrics.csv", "paired_scores.csv"):
        assert_same(calibration / name, ROOT / "results" / "judge_calibration_exact" / name)
    print(f"Reproduced all numeric results exactly under {work}.")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("verify", "materialize", "analyze"), nargs="?", default="verify")
    parser.add_argument("--work-dir", type=Path, default=DEFAULT_WORK)
    args = parser.parse_args()

    verify_archives()
    if args.command == "materialize":
        print(materialize(args.work_dir))
    elif args.command == "analyze":
        analyze(args.work_dir)


if __name__ == "__main__":
    main()
