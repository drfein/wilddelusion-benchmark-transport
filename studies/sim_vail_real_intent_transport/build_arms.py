#!/usr/bin/env python3
"""Select screened conversations and construct frozen matched cue arms."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from design import (
    CUE_PAIRS,
    SELECTED_CONVERSATIONS,
    STUDY_ID,
    stable_hash,
    template_index,
)
from io_utils import read_jsonl, sha256_file, write_jsonl


def eligible(row: dict[str, Any]) -> bool:
    return (
        row.get("screen_success") is True
        and row.get("language") == "English"
        and all(
            int(row.get(f"{intent}_presence", 2)) <= 1
            for intent in CUE_PAIRS
        )
    )


def proportional_theme_sample(
    rows: list[dict[str, Any]], count: int
) -> list[dict[str, Any]]:
    """Deterministic proportional allocation with largest remainders."""
    by_theme: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_theme[row["theme"]].append(row)
    for theme in by_theme:
        by_theme[theme].sort(
            key=lambda row: stable_hash(STUDY_ID, row["conversation_key"], "select")
        )
    total = len(rows)
    quotas = {theme: count * len(group) / total for theme, group in by_theme.items()}
    allocated = {theme: min(len(by_theme[theme]), int(value)) for theme, value in quotas.items()}
    remaining = count - sum(allocated.values())
    order = sorted(
        by_theme,
        key=lambda theme: (-(quotas[theme] - int(quotas[theme])), theme),
    )
    while remaining:
        changed = False
        for theme in order:
            if allocated[theme] < len(by_theme[theme]):
                allocated[theme] += 1
                remaining -= 1
                changed = True
                if remaining == 0:
                    break
        if not changed:
            raise ValueError("Not enough eligible rows to satisfy selection count")
    return [row for theme in sorted(by_theme) for row in by_theme[theme][: allocated[theme]]]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidates", type=Path, required=True)
    parser.add_argument("--screens", type=Path, required=True)
    parser.add_argument("--selected-output", type=Path, required=True)
    parser.add_argument("--arms-output", type=Path, required=True)
    args = parser.parse_args()

    candidates = read_jsonl(args.candidates)
    screens = {row["candidate_id"]: row for row in read_jsonl(args.screens)}
    if set(screens) != {row["candidate_id"] for row in candidates}:
        raise ValueError("Screen coverage does not exactly match candidates")
    merged = [{**row, **screens[row["candidate_id"]]} for row in candidates]
    available = [row for row in merged if eligible(row)]
    if len(available) < SELECTED_CONVERSATIONS:
        raise ValueError(
            f"Only {len(available)} eligible conversations; "
            f"need {SELECTED_CONVERSATIONS}"
        )
    selected = proportional_theme_sample(available, SELECTED_CONVERSATIONS)
    selected.sort(key=lambda row: row["conversation_key"])
    if len({row["conversation_key"] for row in selected}) != len(selected):
        raise ValueError("Selected rows are not conversation-disjoint")
    write_jsonl(args.selected_output, selected)

    arms = []
    for row in selected:
        original_messages = row["messages"]
        original_target = original_messages[-1]["content"]
        for intent, pairs in CUE_PAIRS.items():
            index = template_index(row["conversation_key"], intent)
            target_cue, control_cue = pairs[index]
            for arm, cue in (("target_intent", target_cue), ("matched_control", control_cue)):
                messages = [dict(message) for message in original_messages]
                messages[-1] = {
                    "role": "user",
                    "content": f"{original_target.rstrip()}\n\n{cue}",
                }
                arm_id = stable_hash(
                    STUDY_ID, row["candidate_id"], intent, arm, index
                )[:28]
                arms.append(
                    {
                        "study_id": STUDY_ID,
                        "arm_id": arm_id,
                        "candidate_id": row["candidate_id"],
                        "conversation_key": row["conversation_key"],
                        "conversation_id": row["conversation_id"],
                        "source": row["source"],
                        "theme": row["theme"],
                        "intent": intent,
                        "condition": arm,
                        "template_index": index,
                        "cue": cue,
                        "cue_word_count": len(cue.split()),
                        "original_target_text": original_target,
                        "messages": messages,
                    }
                )
    expected = SELECTED_CONVERSATIONS * len(CUE_PAIRS) * 2
    if len(arms) != expected or len({row["arm_id"] for row in arms}) != expected:
        raise ValueError("Arm construction is incomplete or duplicated")
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in arms:
        grouped[(row["candidate_id"], row["intent"])].append(row)
    for key, pair in grouped.items():
        if {row["condition"] for row in pair} != {"target_intent", "matched_control"}:
            raise ValueError(f"{key}: missing paired condition")
        if pair[0]["messages"][:-1] != pair[1]["messages"][:-1]:
            raise ValueError(f"{key}: history changed across conditions")
        if pair[0]["cue_word_count"] != pair[1]["cue_word_count"]:
            raise ValueError(f"{key}: cue lengths differ")
        for item in pair:
            prefix, suffix = item["messages"][-1]["content"].rsplit("\n\n", 1)
            if prefix != item["original_target_text"].rstrip() or suffix != item["cue"]:
                raise ValueError(f"{key}: original target was not preserved")
    write_jsonl(args.arms_output, arms)

    manifest = {
        "study_id": STUDY_ID,
        "candidates": len(candidates),
        "eligible": len(available),
        "selected": len(selected),
        "arms": len(arms),
        "selection": "deterministic proportional-by-theme largest-remainder sample",
        "sources": Counter(row["source"] for row in selected),
        "themes": Counter(row["theme"] for row in selected),
        "template_counts": Counter(
            f"{row['intent']}:{row['template_index']}" for row in arms
            if row["condition"] == "target_intent"
        ),
        "candidates_sha256": sha256_file(args.candidates),
        "screens_sha256": sha256_file(args.screens),
        "selected_sha256": sha256_file(args.selected_output),
        "arms_sha256": sha256_file(args.arms_output),
    }
    manifest_path = args.arms_output.with_suffix(args.arms_output.suffix + ".manifest.json")
    manifest_path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False, default=dict) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest, indent=2, ensure_ascii=False, default=dict))


if __name__ == "__main__":
    main()

