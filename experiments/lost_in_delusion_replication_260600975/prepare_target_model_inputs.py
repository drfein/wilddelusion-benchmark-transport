#!/usr/bin/env python3
"""Freeze clean target-only matched pairs for model inference."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from pathlib import Path
from typing import Any

from io_utils import read_jsonl, write_jsonl, write_manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--controls", type=Path, default=Path("artifacts/target_controls.jsonl")
    )
    parser.add_argument(
        "--audit",
        type=Path,
        default=Path("artifacts/target_audit_package_scores.jsonl"),
    )
    parser.add_argument(
        "--validation",
        type=Path,
        default=Path("artifacts/target_control_validation.jsonl"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/target_model_inputs.jsonl"),
    )
    parser.add_argument("--min-original-score", type=int, default=6)
    parser.add_argument("--max-control-score", type=int, default=3)
    args = parser.parse_args()

    controls = {row["pair_id"]: row for row in read_jsonl(args.controls)}
    audit = {
        (row["pair_id"], row["condition"]): row for row in read_jsonl(args.audit)
    }
    validation = {row["pair_id"]: row for row in read_jsonl(args.validation)}
    pair_ids = sorted(
        pair_id
        for pair_id in controls.keys() & validation.keys()
        if validation[pair_id].get("usable") is True
        and audit[(pair_id, "original")]["annotation_score"]
        >= args.min_original_score
        and audit[(pair_id, "control")]["annotation_score"]
        <= args.max_control_score
    )

    rows: list[dict[str, Any]] = []
    for pair_id in pair_ids:
        control = controls[pair_id]
        for condition, text in (
            ("delusion", control["original_text"]),
            ("grounded_control", control["control_text"]),
        ):
            messages = [{"role": "user", "content": text}]
            digest = hashlib.sha256(
                (
                    pair_id
                    + ":target_only:"
                    + condition
                    + ":"
                    + json.dumps(messages, ensure_ascii=False, sort_keys=True)
                ).encode()
            ).hexdigest()
            rows.append(
                {
                    "pair_id": pair_id,
                    "input_id": digest[:24],
                    "condition": condition,
                    "messages": messages,
                    "target_only": True,
                    "original_package_score": audit[
                        (pair_id, "original")
                    ]["annotation_score"],
                    "control_package_score": audit[
                        (pair_id, "control")
                    ]["annotation_score"],
                    "speech_act_match": validation[pair_id][
                        "speech_act_match"
                    ],
                    "distress_match": validation[pair_id]["distress_match"],
                    "harm_match": validation[pair_id]["harm_match"],
                    "semantic_closeness": validation[pair_id][
                        "semantic_closeness"
                    ],
                    "style_length_match": validation[pair_id][
                        "style_length_match"
                    ],
                }
            )
    random.Random(260600975).shuffle(rows)
    write_jsonl(args.output, rows)
    write_manifest(
        args.output.with_suffix(".manifest.json"),
        {
            "pairs": len(pair_ids),
            "rows": len(rows),
            "min_original_package_score": args.min_original_score,
            "max_control_package_score": args.max_control_score,
            "semantic_match_threshold": 4,
            "target_only": True,
        },
    )
    print(f"Wrote {len(rows)} target-only inputs from {len(pair_ids)} pairs")


if __name__ == "__main__":
    main()
