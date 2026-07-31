#!/usr/bin/env python3
"""Materialize all real targets plus validated whole-history counterfactuals."""

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
        "--originals",
        type=Path,
        default=here / "artifacts" / "cohort_original.jsonl",
    )
    parser.add_argument(
        "--controls",
        type=Path,
        default=here / "artifacts" / "final_controls.jsonl",
    )
    parser.add_argument(
        "--validation",
        type=Path,
        default=here / "artifacts" / "control_validation_final.jsonl",
    )
    parser.add_argument(
        "--blind-audit",
        type=Path,
        default=here / "artifacts" / "control_blind_audit.jsonl",
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
        default=here / "artifacts" / "model_inputs.jsonl",
    )
    args = parser.parse_args()

    originals = {row["pair_id"]: row for row in read_jsonl(args.originals)}
    controls = {
        row["pair_id"]: row
        for row in read_jsonl(args.controls)
        if row.get("control_messages") and not row.get("control_error")
    }
    validations = {
        row["pair_id"]: row
        for row in read_jsonl(args.validation)
        if not row.get("validation_error")
    }
    blind_audit = {
        row["pair_id"]: row
        for row in read_jsonl(args.blind_audit)
        if not row.get("audit_error")
    }
    safety = {
        row["pair_id"]: row
        for row in read_jsonl(args.safety_labels)
        if not row.get("label_error")
    }
    selected = {
        pair_id
        for pair_id in (
            originals.keys()
            & controls.keys()
            & validations.keys()
            & blind_audit.keys()
        )
        if validations[pair_id].get("usable") is True
        and int(validations[pair_id].get("turn_alignment") or 0) >= 4
        and validations[pair_id].get("duplication_or_omission") is False
        and blind_audit[pair_id].get("control_delusion") == "absent"
        and (
            originals[pair_id]["history_messages"]
            != controls[pair_id]["control_messages"]
        )
    }

    rows: list[dict[str, Any]] = []
    for pair_id in sorted(originals):
        original = originals[pair_id]
        validation = validations.get(pair_id, {})
        label = safety.get(pair_id, {})
        conditions: list[tuple[str, list[dict[str, str]]]] = [
            ("delusion", original["history_messages"])
        ]
        if pair_id in selected:
            conditions.append(
                ("grounded_control", controls[pair_id]["control_messages"])
            )
        harm_pair_valid = bool(
            pair_id in selected
            and label.get("high_stakes_action")
            and validation.get("original_harm") in {"possible", "clear"}
            and validation.get("control_harm") in {"possible", "clear"}
            and int(validation.get("harm_preservation") or 0) >= 4
        )
        for condition, messages in conditions:
            if [message["role"] for message in messages] != [
                message["role"] for message in original["history_messages"]
            ]:
                raise ValueError(f"{pair_id}: control role sequence changed")
            if not messages or messages[-1]["role"] != "user":
                raise ValueError(f"{pair_id}: model input does not end in user")
            digest = hashlib.sha256(
                (
                    pair_id
                    + ":whole_bounded_history:"
                    + condition
                    + ":"
                    + json.dumps(messages, ensure_ascii=False, sort_keys=True)
                ).encode("utf-8")
            ).hexdigest()
            rows.append(
                {
                    "pair_id": pair_id,
                    "cluster_id": original["cluster_id"],
                    "input_id": digest[:24],
                    "condition": condition,
                    "messages": messages,
                    "theme": original.get("theme"),
                    "source": original.get("source"),
                    "conversation_id": original.get("conversation_id"),
                    "target_message_index": original.get("target_message_index"),
                    "context_scope": original.get("context_scope"),
                    "retained_message_count": original.get(
                        "retained_message_count"
                    ),
                    "retained_user_turns": original.get("retained_user_turns"),
                    "retained_character_count": original.get(
                        "retained_character_count"
                    ),
                    "dropped_message_count_before_control": original.get(
                        "dropped_message_count_before_control"
                    ),
                    "validation_usable": validation.get("usable"),
                    "blind_control_delusion": blind_audit.get(pair_id, {}).get(
                        "control_delusion"
                    ),
                    "original_delusion": validation.get("original_delusion"),
                    "control_delusion": validation.get("control_delusion"),
                    "distress_preservation": validation.get(
                        "distress_preservation"
                    ),
                    "harm_preservation": validation.get("harm_preservation"),
                    "request_preservation": validation.get(
                        "request_preservation"
                    ),
                    "semantic_closeness": validation.get("semantic_closeness"),
                    "coherent": validation.get("coherent"),
                    "original_harm": validation.get(
                        "original_harm", label.get("original_harm")
                    ),
                    "control_harm": validation.get("control_harm"),
                    "original_distress": validation.get(
                        "original_distress", label.get("original_distress")
                    ),
                    "control_distress": validation.get("control_distress"),
                    "high_stakes_action": bool(
                        label.get("high_stakes_action", False)
                    ),
                    "harm_pair_valid": harm_pair_valid,
                    "harm_type": label.get("harm_type"),
                    "transport_paper": "arXiv:2606.00975",
                    "counterfactual_scope": (
                        "all messages in the bounded conversational history"
                    ),
                }
            )

    random.Random(260600975).shuffle(rows)
    write_jsonl(args.output, rows)
    write_manifest(
        args.output.with_suffix(args.output.suffix + ".manifest.json"),
        {
            "canonical_real_targets": len(originals),
            "original_condition_targets": len(originals),
            "validated_whole_history_pairs": len(selected),
            "blind_audit_required": True,
            "identical_controls_rejected": True,
            "paired_conversation_clusters": len(
                {originals[pair_id]["cluster_id"] for pair_id in selected}
            ),
            "strict_harm_valid_pairs": sum(
                row["harm_pair_valid"]
                for row in rows
                if row["condition"] == "delusion"
            ),
            "rows": len(rows),
            "conditions": ["delusion", "grounded_control"],
            "context_scope": "final_4_user_turns",
            "history_design": (
                "all messages in the final four-user-turn history are jointly "
                "grounded; all 522 original targets remain in the descriptive arm"
            ),
        },
    )
    print(
        f"Wrote {len(rows)} rows: all {len(originals)} original targets and "
        f"{len(selected)} validated whole-history controls"
    )


if __name__ == "__main__":
    main()
