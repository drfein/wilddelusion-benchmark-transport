#!/usr/bin/env python3
"""Recompute the visible-context claim using only its two primary models."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--bootstrap-draws", type=int, default=100_000)
    parser.add_argument("--seed", type=int, default=260600976)
    args = parser.parse_args()
    experiment = args.experiment.resolve()
    cross = experiment / "cross_domain_comparison"
    sys.path[:0] = [str(experiment), str(cross)]

    from analyze_lost_exact_claims import (  # type: ignore
        EXPECTED_COHORT_SHA256,
        EXPECTED_POSSIBLE_MOD_HARM_COUNT,
        audit_generation_protocol,
        audit_judgments,
        audit_mod_harm_superset,
    )
    from analyze_visible_context_mod_harm import (  # type: ignore
        PRIMARY_MODELS,
        audit_visible_labels,
        coverage_rows,
        gate_ready,
        load_visible_inputs,
        pair_ids_sha256,
        visible_lexical_ids,
    )

    history = experiment / "full_history_522"
    psychogenic = experiment / "psychogenic_machine_transport_250910970"
    inputs_path = history / "artifacts" / "model_inputs.jsonl"
    cohort_path = (
        experiment / "full_dataset_522" / "artifacts" / "cohort_full522.jsonl"
    )
    generations = [
        history / "results" / "generations_llama31_8b.jsonl",
        history / "results" / "generations_olmo.jsonl",
    ]
    judgments = history / "results" / "judgments.qwen3-30b-thinking.jsonl"

    inputs = load_visible_inputs(inputs_path)
    base_ids, base_audit = audit_mod_harm_superset(
        cohort_path,
        {
            rubric: history / "artifacts" / f"mod_harm_labels_{rubric.lower()}.jsonl"
            for rubric in ("A", "B", "C")
        },
        psychogenic / "full522" / "artifacts" / "safety_labels.jsonl",
        expected_cohort_sha256=EXPECTED_COHORT_SHA256,
        expected_possible_count=EXPECTED_POSSIBLE_MOD_HARM_COUNT,
    )
    visible_a, audit_a = audit_visible_labels(
        history / "artifacts" / "mod_harm_visible_context_a.jsonl",
        "A",
        inputs,
        inputs_path,
    )
    visible_b, audit_b = audit_visible_labels(
        history / "artifacts" / "mod_harm_visible_context_b.jsonl",
        "B",
        inputs,
        inputs_path,
    )
    lexical_ids, lexical_counts = visible_lexical_ids(inputs)
    possible_ids = base_ids | visible_a | visible_b | lexical_ids
    if len(possible_ids) != 161:
        raise ValueError(f"expected 161 possible Mod+Harm endpoints, got {len(possible_ids)}")

    generation_protocol = audit_generation_protocol(generations)
    completion, eligible = audit_judgments(
        judgments, generations, set(PRIMARY_MODELS)
    )
    rows = coverage_rows(
        eligible[eligible["model"].isin(PRIMARY_MODELS)],
        possible_ids,
        args.bootstrap_draws,
        args.seed,
    )
    gate = {
        "claim_id": "lost_confirmation_failures_mostly_outside_mod_harm",
        "ready": gate_ready(rows),
        "required_evidence": (
            "Both official-BF16 primary models must be complete. After unioning "
            "every target-level and exact-visible-context possible-Mod+Harm "
            "screen, each model's conversation-clustered 99% lower bound must "
            "place a majority of validation failures outside the screen."
        ),
        "allowed_claim": (
            "Even a deliberately overinclusive possible-Mod+Harm screen over "
            "the exact assistant-visible context captured only "
            f"{min(row['validate_amplify_inside_share'] for row in rows) * 100:.1f}-"
            f"{max(row['validate_amplify_inside_share'] for row in rows) * 100:.1f}% "
            "of real responses scored Validate/Amplify."
        ),
        "causal_guardrail": (
            "This is a conservative corpus-composition result, not evidence "
            "that adding crisis language causes a safer response."
        ),
        "screen": {
            "possible_n": len(possible_ids),
            "outside_n": 522 - len(possible_ids),
            "possible_share": len(possible_ids) / 522,
            "pair_ids_sha256": pair_ids_sha256(possible_ids),
            "construction": (
                "Union of the frozen target-level superset, both visible-context "
                "rubrics with UNCLEAR treated as possible, and deterministic "
                "harm-language matches in any exact-visible user message."
            ),
            "base_target_screen_n": len(base_ids),
            "visible_rubric_a_possible_n": len(visible_a),
            "visible_rubric_b_possible_n": len(visible_b),
            "visible_lexical_n": len(lexical_ids),
        },
        "evidence": rows,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "claim_gate.json").write_text(
        json.dumps(gate, indent=2) + "\n", encoding="utf-8"
    )
    with (args.output_dir / "coverage_by_model.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    audit = {
        "reproduction_scope": "two primary models used by the headline claim",
        "base_target_screen": base_audit,
        "visible_context_labels": [audit_a, audit_b],
        "visible_lexical_counts": lexical_counts,
        "generation_protocol": generation_protocol,
        "model_completion": completion.to_dict(orient="records"),
    }
    (args.output_dir / "audit.json").write_text(
        json.dumps(audit, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(gate, indent=2))


if __name__ == "__main__":
    main()
