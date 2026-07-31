#!/usr/bin/env python3
"""Summarize control construction, validation, and paired-cohort selection."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from statistics import median
from typing import Any


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def describe(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "targets": len(rows),
        "clusters": len({row["cluster_id"] for row in rows}),
        "median_retained_characters": median(
            row["retained_character_count"] for row in rows
        ),
        "median_retained_messages": median(
            row["retained_message_count"] for row in rows
        ),
        "median_annotation_score": median(
            float(row["annotation_score"])
            for row in rows
            if row.get("annotation_score") is not None
        ),
        "sources": dict(Counter(row.get("source") for row in rows)),
        "themes": dict(Counter(row.get("theme") for row in rows)),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    here = Path(__file__).resolve().parent
    parser.add_argument(
        "--cohort",
        type=Path,
        default=here / "artifacts" / "cohort_original.jsonl",
    )
    parser.add_argument(
        "--generated-controls",
        type=Path,
        default=here / "artifacts" / "generated_controls.jsonl",
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
        "--output",
        type=Path,
        default=here / "artifacts" / "cohort_flow_summary.json",
    )
    args = parser.parse_args()

    cohort = {row["pair_id"]: row for row in read_jsonl(args.cohort)}
    generated_controls = {
        row["pair_id"]: row
        for row in read_jsonl(args.generated_controls)
        if row.get("control_messages") and not row.get("control_error")
    }
    controls = {
        row["pair_id"]: row
        for row in read_jsonl(args.controls)
        if row.get("control_messages") and not row.get("control_error")
    }
    final_validation = {
        row["pair_id"]: row for row in read_jsonl(args.validation)
    }
    blind_audit = {
        row["pair_id"]: row for row in read_jsonl(args.blind_audit)
    }
    selected_ids = {
        pair_id
        for pair_id, row in final_validation.items()
        if row.get("usable") is True
        and int(row.get("turn_alignment") or 0) >= 4
        and row.get("duplication_or_omission") is False
        and blind_audit.get(pair_id, {}).get("control_delusion") == "absent"
        and pair_id in cohort
        and pair_id in controls
        and cohort[pair_id]["history_messages"]
        != controls[pair_id]["control_messages"]
    }
    selected = [cohort[pair_id] for pair_id in selected_ids]
    excluded = [
        row for pair_id, row in cohort.items() if pair_id not in selected_ids
    ]

    payload = {
        "canonical_real_targets": len(cohort),
        "canonical_conversation_clusters": len(
            {row["cluster_id"] for row in cohort.values()}
        ),
        "initial_generated_control_successes": len(generated_controls),
        "candidate_controls_after_repairs": len(controls),
        "blind_audit_labels": dict(
            Counter(row.get("control_delusion") for row in blind_audit.values())
        ),
        "blind_absent_candidates": sum(
            row.get("control_delusion") == "absent"
            for row in blind_audit.values()
        ),
        "pairwise_usable_candidates": sum(
            row.get("usable") is True for row in final_validation.values()
        ),
        "strict_alignment_candidates": sum(
            row.get("usable") is True
            and int(row.get("turn_alignment") or 0) >= 4
            and row.get("duplication_or_omission") is False
            for row in final_validation.values()
        ),
        "final_validated_pairs": len(selected_ids),
        "final_validated_clusters": len(
            {row["cluster_id"] for row in selected}
        ),
        "final_control_delusion_labels": dict(
            Counter(
                row.get("control_delusion")
                for row in final_validation.values()
            )
        ),
        "selected": describe(selected),
        "excluded_from_paired_but_retained_descriptively": describe(excluded),
        "selection_rule": (
            "blind control-only audit labels delusion absent; independent paired "
            "validator sets usable=true; strict turn alignment >=4/5 with no "
            "duplication/omission; roles match; control differs from original"
        ),
    }
    args.output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
