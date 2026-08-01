#!/usr/bin/env python3
"""Prepare leakage-auditable inputs for the DelusionScore construct stress test."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path

from lingua import Language, LanguageDetectorBuilder


def read_jsonl(path: Path):
    with path.open() as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def write_jsonl(path: Path, rows) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def text_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def detect_language(detector, text: str) -> tuple[str, float]:
    confidences = detector.compute_language_confidence_values(text[:10000])
    if not confidences:
        return "UNKNOWN", 0.0
    top = confidences[0]
    return top.language.name, float(top.value)


def target_text(row: dict) -> str:
    message = row["messages"][int(row["target_message_index"])]
    if message["role"] != "user":
        raise ValueError(f"Target is not user: {row['input_id']}")
    return message["content"].strip()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--paired-inputs", type=Path, required=True)
    parser.add_argument("--natural-controls", type=Path, required=True)
    parser.add_argument("--broad-controls", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    detector = LanguageDetectorBuilder.from_all_languages().build()
    by_pair: dict[str, dict[str, dict]] = defaultdict(dict)
    all_rows = list(read_jsonl(args.paired_inputs))
    for row in all_rows:
        by_pair[row["pair_id"]][row["condition"]] = row

    endpoints = []
    complete_pairs = 0
    for pair_id, conditions in sorted(by_pair.items()):
        if set(conditions) != {"delusion", "grounded_control"}:
            continue
        complete_pairs += 1
        for condition in ("delusion", "grounded_control"):
            row = conditions[condition]
            text = target_text(row)
            language, confidence = detect_language(detector, text)
            endpoints.append(
                {
                    "pair_id": pair_id,
                    "conversation_id": row["conversation_id"],
                    "condition": condition,
                    "label": int(condition == "delusion"),
                    "source": row["source"],
                    "theme": row["theme"],
                    "text": text,
                    "text_hash": text_hash(text),
                    "language": language,
                    "language_confidence": confidence,
                    "target_message_index": int(row["target_message_index"]),
                }
            )

    trajectories = []
    for row in all_rows:
        if row["condition"] != "delusion":
            continue
        user_turn_index = 0
        target_idx = int(row["target_message_index"])
        user_messages = [
            (idx, message["content"].strip())
            for idx, message in enumerate(row["messages"][: target_idx + 1])
            if message["role"] == "user" and message["content"].strip()
        ]
        for idx, text in user_messages:
            language, confidence = detect_language(detector, text)
            trajectories.append(
                {
                    "pair_id": row["pair_id"],
                    "conversation_id": row["conversation_id"],
                    "source": row["source"],
                    "theme": row["theme"],
                    "message_index": idx,
                    "target_message_index": target_idx,
                    "user_turn_index": user_turn_index,
                    "n_user_turns": len(user_messages),
                    "is_target": idx == target_idx,
                    "text": text,
                    "text_hash": text_hash(text),
                    "language": language,
                    "language_confidence": confidence,
                }
            )
            user_turn_index += 1

    natural_controls = []
    for row in read_jsonl(args.natural_controls):
        text = row["target_text"].strip()
        language, confidence = detect_language(detector, text)
        natural_controls.append(
            {
                "control_id": row["control_id"],
                "conversation_id": row["conversation_id"],
                "source": row["source"],
                "exclusion": row["judge_exclusion"],
                "judge_confidence": row["judge_confidence"],
                "text": text,
                "text_hash": text_hash(text),
                "language": language,
                "language_confidence": confidence,
            }
        )

    broad_controls = []
    if args.broad_controls:
        for row in read_jsonl(args.broad_controls):
            if row.get("judge_label") != "negative":
                continue
            if float(row.get("judge_confidence", 0)) < 0.9:
                continue
            if row.get("same_conversation_as_release_positive"):
                continue
            if row.get("target_text_matches_release_positive_elsewhere"):
                continue
            text = row["target_text"].strip()
            language, confidence = detect_language(detector, text)
            broad_controls.append(
                {
                    "control_id": row["control_id"],
                    "conversation_id": row["conversation_id"],
                    "source": row["source"],
                    "exclusion": row["judge_exclusion"],
                    "judge_confidence": row["judge_confidence"],
                    "text": text,
                    "text_hash": text_hash(text),
                    "language": language,
                    "language_confidence": confidence,
                }
            )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(args.output_dir / "endpoints.jsonl", endpoints)
    write_jsonl(args.output_dir / "trajectories.jsonl", trajectories)
    write_jsonl(args.output_dir / "natural_controls.jsonl", natural_controls)
    write_jsonl(args.output_dir / "broad_controls.jsonl", broad_controls)

    summary = {
        "complete_pairs": complete_pairs,
        "endpoint_rows": len(endpoints),
        "trajectory_rows": len(trajectories),
        "trajectory_cases": len({row["pair_id"] for row in trajectories}),
        "trajectory_conversations": len({row["conversation_id"] for row in trajectories}),
        "natural_controls": len(natural_controls),
        "broad_controls": len(broad_controls),
        "english": {
            "endpoint_rows": sum(row["language"] == "ENGLISH" for row in endpoints),
            "trajectory_rows": sum(row["language"] == "ENGLISH" for row in trajectories),
            "natural_controls": sum(row["language"] == "ENGLISH" for row in natural_controls),
            "broad_controls": sum(row["language"] == "ENGLISH" for row in broad_controls),
        },
    }
    (args.output_dir / "cohort_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
