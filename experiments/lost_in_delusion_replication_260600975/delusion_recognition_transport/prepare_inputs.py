#!/usr/bin/env python3
"""Build immutable real-world inputs for the exact paper classifier."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


HERE = Path(__file__).resolve().parent
EXPERIMENT = HERE.parent
REPO = EXPERIMENT.parents[1]


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    temporary.replace(path)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def stable_id(cohort: str, source_id: str) -> str:
    return hashlib.sha256(f"{cohort}:{source_id}".encode()).hexdigest()[:24]


def validate_history(
    messages: list[dict[str, Any]],
    target_text: str,
    source_id: str,
) -> None:
    if not messages:
        raise ValueError(f"{source_id}: empty history")
    final = messages[-1]
    if final.get("role") != "user":
        raise ValueError(f"{source_id}: final target is not a user message")
    if final.get("content") != target_text:
        raise ValueError(f"{source_id}: target does not exactly match final message")


def positive_rows(path: Path) -> list[dict[str, Any]]:
    output = []
    for source in read_jsonl(path):
        messages = source["history_messages"]
        validate_history(messages, source["target_text"], source["pair_id"])
        output.append(
            {
                "input_id": stable_id("real_positive", source["pair_id"]),
                "evaluation_cohort": "real_positive",
                "gold_label": 1,
                "gold_basis": "WildDelusion confirmed deduplicated release",
                "pair_id": source["pair_id"],
                "cluster_id": source["cluster_id"],
                "source": source["source"],
                "conversation_id": source["conversation_id"],
                "target_message_index": source["target_message_index"],
                "target_text": source["target_text"],
                "previous_messages": messages[:-1],
                "source_message_count_through_target": len(messages),
                "theme": source.get("theme"),
                "theme_secondary": source.get("theme_secondary", []),
                "annotation_score": source.get("annotation_score"),
            }
        )
    return output


def natural_negative_rows(
    strict_path: Path,
    high_confidence_path: Path,
    human_confirmed_path: Path,
    disagreement_path: Path,
) -> list[dict[str, Any]]:
    high_confidence = {
        row["control_id"] for row in read_jsonl(high_confidence_path)
    }
    human_confirmed = {
        row["control_id"] for row in read_jsonl(human_confirmed_path)
    }
    disagreements = {
        row["control_id"] for row in read_jsonl(disagreement_path)
    }
    output = []
    for source in read_jsonl(strict_path):
        target_index = int(source["target_message_index"])
        messages = source["messages"][: target_index + 1]
        validate_history(messages, source["target_text"], source["control_id"])
        output.append(
            {
                "input_id": stable_id(
                    "natural_near_miss_negative", source["control_id"]
                ),
                "evaluation_cohort": "natural_near_miss_negative",
                "gold_label": 0,
                "gold_basis": "full-conversation nonsincere-context verification",
                "control_id": source["control_id"],
                "pair_id": None,
                "cluster_id": (
                    f"{source['source']}:{source['conversation_id']}"
                ),
                "source": source["source"],
                "conversation_id": source["conversation_id"],
                "target_message_index": target_index,
                "target_text": source["target_text"],
                "previous_messages": messages[:-1],
                "source_message_count_through_target": len(messages),
                "negative_exclusion": source["judge_exclusion"],
                "verifier_confidence": source["judge_confidence"],
                "is_primary_natural_negative": (
                    source["control_id"] in high_confidence
                ),
                "is_human_confirmed_negative": (
                    source["control_id"] in human_confirmed
                ),
                "is_human_verifier_disagreement": (
                    source["control_id"] in disagreements
                ),
            }
        )
    return output


def generated_negative_rows(path: Path) -> list[dict[str, Any]]:
    output = []
    for source in read_jsonl(path):
        if source.get("condition") != "grounded_control":
            continue
        messages = source["messages"]
        target_text = messages[-1]["content"]
        validate_history(messages, target_text, source["pair_id"])
        output.append(
            {
                "input_id": stable_id(
                    "generated_matched_negative", source["pair_id"]
                ),
                "evaluation_cohort": "generated_matched_negative",
                "gold_label": 0,
                "gold_basis": "validated matched grounded counterfactual",
                "pair_id": source["pair_id"],
                "cluster_id": source["cluster_id"],
                "source": source["source"],
                "conversation_id": source["conversation_id"],
                "target_message_index": source["target_message_index"],
                "target_text": target_text,
                "previous_messages": messages[:-1],
                "source_message_count_through_target": len(messages),
                "theme": source.get("theme"),
                "validation_usable": source.get("validation_usable"),
            }
        )
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--positives",
        type=Path,
        default=EXPERIMENT
        / "full_dataset_522"
        / "artifacts"
        / "cohort_full522.jsonl",
    )
    controls = (
        REPO / "data" / "controls" / "natural_near_miss_controls_20260730"
    )
    parser.add_argument(
        "--natural-negatives",
        type=Path,
        default=controls / "strict_nonsincere_controls.jsonl",
    )
    parser.add_argument(
        "--high-confidence-negatives",
        type=Path,
        default=controls / "high_confidence_strict_controls.jsonl",
    )
    parser.add_argument(
        "--human-confirmed-negatives",
        type=Path,
        default=controls / "human_confirmed_controls.jsonl",
    )
    parser.add_argument(
        "--human-disagreements",
        type=Path,
        default=controls / "human_verifier_disagreements.jsonl",
    )
    parser.add_argument(
        "--generated-negatives",
        type=Path,
        default=EXPERIMENT
        / "full_history_522"
        / "artifacts"
        / "model_inputs.jsonl",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=HERE / "artifacts" / "classifier_inputs.jsonl",
    )
    args = parser.parse_args()

    rows = positive_rows(args.positives)
    rows.extend(
        natural_negative_rows(
            args.natural_negatives,
            args.high_confidence_negatives,
            args.human_confirmed_negatives,
            args.human_disagreements,
        )
    )
    rows.extend(generated_negative_rows(args.generated_negatives))

    input_ids = [row["input_id"] for row in rows]
    if len(input_ids) != len(set(input_ids)):
        raise ValueError("input_id values are not unique")

    write_jsonl(args.output, rows)
    counts: dict[str, int] = {}
    for row in rows:
        cohort = row["evaluation_cohort"]
        counts[cohort] = counts.get(cohort, 0) + 1
    manifest = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "protocol": (
            "Exact Lost in Delusion Assess Delusion Then Reply classifier "
            "transport evaluation"
        ),
        "output": str(args.output),
        "output_sha256": sha256_file(args.output),
        "rows": len(rows),
        "counts": counts,
        "primary_natural_negatives": sum(
            bool(row.get("is_primary_natural_negative")) for row in rows
        ),
        "human_confirmed_natural_negatives": sum(
            bool(row.get("is_human_confirmed_negative")) for row in rows
        ),
        "inputs": {
            name: {"path": str(path), "sha256": sha256_file(path)}
            for name, path in {
                "positives": args.positives,
                "natural_negatives": args.natural_negatives,
                "high_confidence_negatives": args.high_confidence_negatives,
                "human_confirmed_negatives": args.human_confirmed_negatives,
                "human_disagreements": args.human_disagreements,
                "generated_negatives": args.generated_negatives,
            }.items()
        },
    }
    manifest_path = args.output.with_suffix(".manifest.json")
    manifest_path.write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
