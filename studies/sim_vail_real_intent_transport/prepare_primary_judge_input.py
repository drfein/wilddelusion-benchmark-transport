#!/usr/bin/env python3
"""Freeze the budgeted primary replicate before outcome scoring."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
from pathlib import Path

from lingua import Language, LanguageDetectorBuilder

from design import REPLICATE_SEEDS, SELECTED_CONVERSATIONS, STUDY_ID, stable_hash
from io_utils import read_jsonl, sha256_file, write_jsonl


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--generations", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--replicate-seed", type=int, default=REPLICATE_SEEDS[0])
    parser.add_argument("--conversation-count", type=int, default=80)
    args = parser.parse_args()

    rows = [row for path in args.generations for row in read_jsonl(path)]
    expected_full = SELECTED_CONVERSATIONS * 3 * 2 * len(REPLICATE_SEEDS) * 2
    if len(rows) != expected_full or len({row["generation_id"] for row in rows}) != expected_full:
        raise ValueError(f"Expected {expected_full} unique complete generations, found {len(rows)}")
    if args.replicate_seed not in REPLICATE_SEEDS:
        raise ValueError(f"Replicate seed must be one of {REPLICATE_SEEDS}")
    candidate_text = {
        row["candidate_id"]: row["original_target_text"] for row in rows
    }
    detector = LanguageDetectorBuilder.from_all_languages().build()
    detected_languages = {
        candidate_id: detector.detect_language_of(text)
        for candidate_id, text in candidate_text.items()
    }
    english_candidates = {
        candidate_id
        for candidate_id, language in detected_languages.items()
        if language == Language.ENGLISH
    }
    if len(english_candidates) < args.conversation_count:
        raise ValueError(
            f"Only {len(english_candidates)} English candidates for "
            f"requested count {args.conversation_count}"
        )
    candidate_ids = sorted(
        english_candidates,
        key=lambda candidate_id: stable_hash(
            STUDY_ID, candidate_id, "primary-budget-cohort"
        ),
    )[: args.conversation_count]
    retained = set(candidate_ids)
    selected = [
        row for row in rows
        if row["replicate_seed"] == args.replicate_seed
        and row["candidate_id"] in retained
    ]
    expected_selected = args.conversation_count * 3 * 2 * 2
    if len(selected) != expected_selected:
        raise ValueError(f"Expected {expected_selected} selected rows, found {len(selected)}")
    selected.sort(key=lambda row: row["generation_id"])
    write_jsonl(args.output, selected)
    manifest = {
        "study_id": STUDY_ID,
        "purpose": "budget-preserving primary exact-judge analysis",
        "selection_timing": "frozen before any study outcome was judged",
        "replicate_seed": args.replicate_seed,
        "conversation_count": args.conversation_count,
        "conversation_selection": (
            "Lingua-detected English endpoint, then stable SHA-256 hash before "
            "outcome scoring"
        ),
        "language_detector": {
            "package": "lingua-language-detector",
            "version": importlib.metadata.version("lingua-language-detector"),
            "scope": "original unmodified endpoint text",
            "english_candidates": len(english_candidates),
            "non_english_candidates": len(candidate_text) - len(english_candidates),
        },
        "available_replicate_seeds": REPLICATE_SEEDS,
        "generation_files": [
            {"path": str(path), "sha256": sha256_file(path)} for path in args.generations
        ],
        "full_generation_rows": len(rows),
        "selected_rows": len(selected),
        "output_sha256": sha256_file(args.output),
    }
    args.output.with_suffix(args.output.suffix + ".manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
