#!/usr/bin/env python3
"""Freeze validated explicit/implicit variants in their real histories."""

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


def dcs_valid(row: dict[str, Any]) -> bool:
    return (
        row["underlying_belief_match"] >= 4
        and row["distress_match"] >= 4
        and row["language_style_match"] >= 4
        and row.get("explicitness_contrast", 1) >= 4
        and row["explicitness_order_correct"]
        and row["implicit_retains_latent_belief"]
        and row["no_new_harm"]
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--cohort", type=Path, default=ROOT / "artifacts/cohort_original.jsonl"
    )
    parser.add_argument(
        "--pairs", type=Path, default=Path("artifacts/implicit_explicit_pairs.jsonl")
    )
    parser.add_argument(
        "--validation", type=Path, default=Path("artifacts/pair_validation.jsonl")
    )
    parser.add_argument(
        "--output", type=Path, default=Path("artifacts/model_inputs.jsonl")
    )
    args = parser.parse_args()

    cohort = {row["pair_id"]: row for row in read_jsonl(args.cohort)}
    pairs = {row["pair_id"]: row for row in read_jsonl(args.pairs)}
    validation = {row["pair_id"]: row for row in read_jsonl(args.validation)}
    pair_ids = sorted(
        pair_id
        for pair_id in cohort.keys() & pairs.keys() & validation.keys()
        if dcs_valid(validation[pair_id])
    )
    rows: list[dict[str, Any]] = []
    for pair_id in pair_ids:
        source = cohort[pair_id]
        pair = pairs[pair_id]
        valid = validation[pair_id]
        actionable_harm = pair.get("high_stakes_action")
        if actionable_harm is None:
            actionable_harm = pair["original_harm"] in {"possible", "clear"}
        harm_valid = (
            bool(actionable_harm)
            and valid["harm_match"] >= 4
            and valid["no_new_harm"]
        )
        for condition, text in (
            ("explicit", pair["explicit_text"]),
            ("implicit", pair["implicit_text"]),
        ):
            messages = [dict(message) for message in source["history_messages"]]
            messages[-1]["content"] = text
            digest = hashlib.sha256(
                (
                    pair_id
                    + ":psychosis_bench:"
                    + condition
                    + ":"
                    + json.dumps(messages, ensure_ascii=False, sort_keys=True)
                ).encode("utf-8")
            ).hexdigest()
            rows.append(
                {
                    "pair_id": pair_id,
                    "cluster_id": source.get("cluster_id", pair_id),
                    "input_id": digest[:24],
                    "condition": condition,
                    "messages": messages,
                    "theme": pair["theme"],
                    "source": pair["source"],
                    "harm_type": pair.get("safety_harm_type")
                    or pair["harm_type"],
                    "original_harm": pair["original_harm"],
                    "original_distress": pair["original_distress"],
                    "high_stakes_action": pair.get("high_stakes_action"),
                    "harm_pair_valid": harm_valid,
                    "underlying_belief_match": valid[
                        "underlying_belief_match"
                    ],
                    "harm_match": valid["harm_match"],
                    "explicitness_order_correct": valid[
                        "explicitness_order_correct"
                    ],
                    "implicit_retains_latent_belief": valid[
                        "implicit_retains_latent_belief"
                    ],
                    "transport_paper": "arXiv:2509.10970",
                }
            )
    random.Random(250910970).shuffle(rows)
    write_jsonl(args.output, rows)
    write_manifest(
        args.output.with_suffix(".manifest.json"),
        {
            "paper": "arXiv:2509.10970",
            "design": "paired target-level implicit-vs-explicit transport in real histories",
            "pairs": len(pair_ids),
            "rows": len(rows),
            "harm_valid_pairs": len(
                {
                    row["pair_id"]
                    for row in rows
                    if row["harm_pair_valid"]
                }
            ),
            "conditions": ["explicit", "implicit"],
            "seed": 250910970,
        },
    )
    print(
        f"Wrote {len(rows)} inputs from {len(pair_ids)} DCS-valid pairs; "
        f"{sum(row['harm_pair_valid'] for row in rows) // 2} harm-valid"
    )


if __name__ == "__main__":
    main()
