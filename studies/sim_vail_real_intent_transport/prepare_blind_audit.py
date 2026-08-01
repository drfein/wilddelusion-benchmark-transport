#!/usr/bin/env python3
"""Create a fixed condition-blinded response audit before outcome scoring."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

from design import STUDY_ID, stable_hash
from io_utils import read_jsonl, sha256_file


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--key-output", type=Path, required=True)
    parser.add_argument("--pairs-per-cell", type=int, default=5)
    args = parser.parse_args()

    rows = read_jsonl(args.input)
    cells: dict[tuple[str, str], dict[str, dict[str, dict[str, Any]]]] = defaultdict(
        lambda: defaultdict(dict)
    )
    for row in rows:
        cells[(row["model"], row["intent"])][row["candidate_id"]][row["condition"]] = row

    audit_rows = []
    key_rows = []
    for (model, intent), candidates in sorted(cells.items()):
        eligible = [
            (candidate_id, pair)
            for candidate_id, pair in candidates.items()
            if set(pair) == {"target_intent", "matched_control"}
        ]
        eligible.sort(
            key=lambda item: stable_hash(STUDY_ID, model, intent, item[0], "manual-audit")
        )
        for candidate_id, pair in eligible[: args.pairs_per_cell]:
            swapped = int(stable_hash(STUDY_ID, model, intent, candidate_id, "blind")[:2], 16) % 2
            order = ["target_intent", "matched_control"]
            if swapped:
                order.reverse()
            audit_id = stable_hash(STUDY_ID, model, intent, candidate_id, "audit-id")[:20]
            for label, condition in zip(("A", "B"), order, strict=True):
                row = pair[condition]
                audit_rows.append(
                    {
                        "audit_id": audit_id,
                        "response_label": label,
                        "model": model,
                        "intent": intent,
                        "theme": row["theme"],
                        "original_user_message": row["original_target_text"],
                        "appended_user_sentence": row["cue"],
                        "assistant_response": row["response"],
                        "malformed_or_incoherent": "",
                        "mentions_evaluation_or_condition": "",
                        "review_notes": "",
                    }
                )
                key_rows.append(
                    {
                        "audit_id": audit_id,
                        "response_label": label,
                        "generation_id": row["generation_id"],
                        "condition": condition,
                    }
                )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(audit_rows[0]))
        writer.writeheader()
        writer.writerows(audit_rows)
    args.key_output.write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in key_rows) + "\n",
        encoding="utf-8",
    )
    manifest = {
        "study_id": STUDY_ID,
        "input": str(args.input),
        "input_sha256": sha256_file(args.input),
        "pairs_per_model_intent_cell": args.pairs_per_cell,
        "paired_audits": len(audit_rows) // 2,
        "response_rows": len(audit_rows),
        "selection": "stable hash; condition labels A/B assigned by separate stable hash",
        "outcome_scores_available_at_selection": False,
        "output_sha256": sha256_file(args.output),
        "key_output_sha256": sha256_file(args.key_output),
    }
    args.output.with_suffix(args.output.suffix + ".manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
