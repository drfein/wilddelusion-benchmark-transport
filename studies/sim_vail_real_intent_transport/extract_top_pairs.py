#!/usr/bin/env python3
"""Export the largest paired primary-dimension effects for inspection."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

from design import PRIMARY_DIMENSION
from io_utils import read_jsonl, sha256_file


SCORE_COLUMNS = (
    "overall_risk",
    "belief_reinforcement",
    "risky_action_enablement",
    "boundary_dependence_risk",
    "glorification_romanticization",
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--generations", type=Path, required=True)
    parser.add_argument("--scores", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--count", type=int, default=5)
    args = parser.parse_args()

    scores = {row["judge_input_id"]: row for row in read_jsonl(args.scores)}
    groups: dict[tuple[str, str, str], dict[str, dict[str, Any]]] = defaultdict(dict)
    for row in read_jsonl(args.generations):
        score = scores.get(row["generation_id"])
        if not score or not score.get("judge_success"):
            raise ValueError(f"Missing score for {row['generation_id']}")
        enriched = {
            **row,
            "scores": {name: score[name] for name in SCORE_COLUMNS},
        }
        groups[(row["model"], row["intent"], row["candidate_id"])][
            row["condition"]
        ] = enriched

    by_cell: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for (model, intent, candidate_id), pair in groups.items():
        if set(pair) != {"target_intent", "matched_control"}:
            raise ValueError(f"Incomplete pair for {model}/{intent}/{candidate_id}")
        target = pair["target_intent"]
        control = pair["matched_control"]
        dimension = PRIMARY_DIMENSION[intent]
        item = {
            "model": model,
            "intent": intent,
            "primary_dimension": dimension,
            "candidate_id": candidate_id,
            "conversation_key": target["conversation_key"],
            "source": target["source"],
            "theme": target["theme"],
            "original_user_message": target["original_target_text"],
            "target_cue": target["cue"],
            "control_cue": control["cue"],
            "target_response": target["response"],
            "control_response": control["response"],
            "target_scores": target["scores"],
            "control_scores": control["scores"],
            "primary_score_difference": (
                target["scores"][dimension] - control["scores"][dimension]
            ),
            "target_finish_reason": target["finish_reason"],
            "control_finish_reason": control["finish_reason"],
        }
        by_cell[f"{model}::{intent}"].append(item)

    output = {}
    for cell, items in sorted(by_cell.items()):
        items.sort(
            key=lambda row: (row["primary_score_difference"], row["candidate_id"]),
            reverse=True,
        )
        output[cell] = items[: args.count]
    payload = {
        "generation_file": str(args.generations),
        "generation_sha256": sha256_file(args.generations),
        "score_file": str(args.scores),
        "score_sha256": sha256_file(args.scores),
        "top_pairs_per_model_intent": args.count,
        "cells": output,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps({cell: len(items) for cell, items in output.items()}, indent=2))


if __name__ == "__main__":
    main()
