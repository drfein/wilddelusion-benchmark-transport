#!/usr/bin/env python3
"""Audit natural near-miss versus generated-control classifier errors."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd


HERE = Path(__file__).resolve().parent
EXPERIMENT = HERE.parent
RECOGNITION = EXPERIMENT / "delusion_recognition_transport"
sys.path.insert(0, str(RECOGNITION))

from analyze_recognition import (  # noqa: E402
    MODEL_SPECS,
    audit_outputs,
    natural_control_consensus,
    partial_control_claim_gate,
    partial_control_gap_rows,
    validate_generation_artifacts,
)
from prepare_inputs import read_jsonl, sha256_file, write_jsonl  # noqa: E402


PRIMARY_MODELS = {
    "allenai/Olmo-3-7B-Instruct",
    "meta-llama/Llama-3.1-8B-Instruct",
}


def run(args: argparse.Namespace) -> dict[str, object]:
    expected = pd.DataFrame(read_jsonl(args.inputs))
    if len(expected) != 932 or expected["input_id"].nunique() != 932:
        raise ValueError("Frozen 932-row classifier input cohort changed")

    source_rows = [
        row for path in args.generation_files for row in read_jsonl(path)
    ]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    combined = args.output_dir / "official_bf16_classifications.jsonl"
    write_jsonl(combined, source_rows)
    generation_audit = validate_generation_artifacts(
        args.generation_files,
        combined,
        required_models=PRIMARY_MODELS,
    )

    (
        consensus_ids,
        explicit_ids,
        full_context_ids,
        control_audit,
        _,
    ) = natural_control_consensus(
        expected,
        args.control_adjudication,
        args.control_retention,
        args.retained_control_adjudication,
    )
    outputs = pd.DataFrame(source_rows)
    outputs["is_consensus_natural_negative"] = outputs["control_id"].isin(
        consensus_ids
    )
    outputs["is_explicit_consensus_natural_negative"] = outputs[
        "control_id"
    ].isin(explicit_ids)
    outputs["is_full_context_consensus_natural_negative"] = outputs[
        "control_id"
    ].isin(full_context_ids)
    outputs["is_paper_theme_consensus"] = False

    integrity, _ = audit_outputs(outputs, expected)
    rows = partial_control_gap_rows(
        outputs,
        integrity,
        args.bootstrap_repetitions,
        args.seed,
    )
    gate = partial_control_claim_gate(rows)
    if set(gate["primary_models"]) != {
        MODEL_SPECS[model]["label"] for model in PRIMARY_MODELS
    }:
        raise ValueError("Primary-model gate did not bind both official runs")

    result: dict[str, object] = {
        "estimand": (
            "Natural-minus-generated control false-positive-rate gap for the "
            "exact disclosed direct-delusion classifier. Every unparsable "
            "natural output is assigned non-delusional and every unparsable "
            "generated output delusional for the lower bound."
        ),
        "input_sha256": sha256_file(args.inputs),
        "generation_audit": generation_audit,
        "control_audit": control_audit,
        "run_integrity": integrity.to_dict(orient="records"),
        "partial_identification_rows": rows,
        "claim_gate": gate,
        "guardrail": (
            "Both cohorts are non-delusion ground truth for a recognition "
            "classifier; this result says nothing about assistant validation. "
            "The natural controls are deliberately difficult explicit fiction, "
            "role-play, quotation, joke, third-party, and text-task near misses. "
            "This is a hard non-delusion stress test, not a population false-"
            "positive rate."
        ),
    }
    pd.DataFrame(rows).to_csv(
        args.output_dir / "partial_identification_control_gaps.csv",
        index=False,
    )
    integrity.to_csv(args.output_dir / "run_integrity.csv", index=False)
    (args.output_dir / "claim_gate.json").write_text(
        json.dumps(gate, indent=2) + "\n", encoding="utf-8"
    )
    (args.output_dir / "summary.json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, indent=2))
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--inputs",
        type=Path,
        default=RECOGNITION / "artifacts" / "classifier_inputs.jsonl",
    )
    parser.add_argument(
        "--generation-files",
        type=Path,
        nargs=2,
        default=[
            RECOGNITION / "results" / "classifications.olmo3_7b.jsonl",
            RECOGNITION / "results" / "classifications.llama31_8b.jsonl",
        ],
    )
    parser.add_argument(
        "--control-adjudication",
        type=Path,
        default=RECOGNITION
        / "artifacts"
        / "natural_controls_independent_adjudication_mini.jsonl",
    )
    parser.add_argument(
        "--control-retention",
        type=Path,
        default=RECOGNITION
        / "artifacts"
        / "natural_control_context_retention.jsonl",
    )
    parser.add_argument(
        "--retained-control-adjudication",
        type=Path,
        default=RECOGNITION
        / "artifacts"
        / "natural_controls_retained_context_adjudication.jsonl",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=HERE / "control_realism",
    )
    parser.add_argument("--bootstrap-repetitions", type=int, default=100_000)
    parser.add_argument("--seed", type=int, default=260600975)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
