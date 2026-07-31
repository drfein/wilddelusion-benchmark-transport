#!/usr/bin/env python3
"""Insert validated grounded target rewrites into unchanged real histories."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from io_utils import read_jsonl, write_jsonl, write_manifest  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    here = Path(__file__).resolve().parent
    parser.add_argument(
        "--cohort",
        type=Path,
        default=here / "artifacts" / "cohort_full522.jsonl",
    )
    parser.add_argument(
        "--controls",
        type=Path,
        default=here / "artifacts" / "final_grounded_controls.jsonl",
    )
    parser.add_argument(
        "--safety-labels",
        type=Path,
        default=(
            here.parent
            / "psychogenic_machine_transport_250910970"
            / "full522"
            / "artifacts"
            / "safety_labels.jsonl"
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=here / "artifacts" / "grounded_model_inputs.jsonl",
    )
    args = parser.parse_args()

    cohort = {row["pair_id"]: row for row in read_jsonl(args.cohort)}
    controls = {row["pair_id"]: row for row in read_jsonl(args.controls)}
    safety = {
        row["pair_id"]: row
        for row in read_jsonl(args.safety_labels)
        if not row.get("label_error")
    }
    missing = set(controls) - set(cohort)
    if missing:
        raise ValueError(f"{len(missing)} controls are absent from the cohort")

    rows: list[dict[str, Any]] = []
    for pair_id in sorted(cohort):
        source = cohort[pair_id]
        control = controls.get(pair_id)
        label = safety.get(pair_id, {})
        original_messages = source["history_messages"]
        if (
            not original_messages
            or original_messages[-1]["role"] != "user"
            or original_messages[-1]["content"] != source["target_text"]
        ):
            raise ValueError(f"Target alignment failed for {pair_id}")
        conditions = [("delusion", source["target_text"])]
        if control:
            conditions.append(("grounded_control", control["control_text"]))
        for condition, target_text in conditions:
            messages = [
                {"role": message["role"], "content": message["content"]}
                for message in original_messages
            ]
            messages[-1] = {"role": "user", "content": target_text}
            digest = hashlib.sha256(
                (
                    pair_id
                    + ":full_history:"
                    + condition
                    + ":"
                    + json.dumps(messages, ensure_ascii=False, sort_keys=True)
                ).encode()
            ).hexdigest()
            rows.append(
                {
                    "pair_id": pair_id,
                    "cluster_id": source["cluster_id"],
                    "input_id": digest[:24],
                    "condition": condition,
                    "messages": messages,
                    "theme": source.get("theme"),
                    "source": source.get("source"),
                    "conversation_id": source.get("conversation_id"),
                    "target_message_index": source.get("target_message_index"),
                    "selection_round": (
                        control.get("selection_round") if control else None
                    ),
                    "control_package_score": (
                        control.get("control_package_score") if control else None
                    ),
                    "speech_act_match": (
                        control.get("speech_act_match") if control else None
                    ),
                    "distress_match": (
                        control.get("distress_match") if control else None
                    ),
                    "harm_match": (
                        control.get("harm_match") if control else None
                    ),
                    "semantic_closeness": (
                        control.get("semantic_closeness") if control else None
                    ),
                    "style_length_match": (
                        control.get("style_length_match") if control else None
                    ),
                    "original_harm": label.get("original_harm"),
                    "original_distress": label.get("original_distress"),
                    "high_stakes_action": bool(
                        label.get("high_stakes_action", False)
                    ),
                    "harm_pair_valid": bool(
                        label.get("high_stakes_action", False)
                    ),
                    "harm_type": label.get("harm_type"),
                    "control_harm": label.get("original_harm"),
                    "transport_paper": "arXiv:2606.00975",
                }
            )
    random.Random(260600975).shuffle(rows)
    write_jsonl(args.output, rows)
    write_manifest(
        args.output.with_suffix(args.output.suffix + ".manifest.json"),
        {
            "canonical_cohort_targets": len(cohort),
            "selected_pairs": len(controls),
            "original_condition_targets": len(cohort),
            "source_conversation_clusters": len(
                {cohort[pair_id]["cluster_id"] for pair_id in controls}
            ),
            "rows": len(rows),
            "paired_harm_valid_pairs": sum(
                bool(safety.get(pair_id, {}).get("high_stakes_action"))
                for pair_id in controls
            ),
            "full_cohort_harm_valid_targets": sum(
                bool(safety.get(pair_id, {}).get("high_stakes_action"))
                for pair_id in cohort
            ),
            "conditions": ["delusion", "grounded_control"],
            "history_design": (
                "full real history through target; only final target text differs"
            ),
        },
    )
    print(
        f"Wrote {len(rows)} rows: all {len(cohort)} original targets and "
        f"{len(controls)} paired grounded controls"
    )


if __name__ == "__main__":
    main()
