#!/usr/bin/env python3
"""Audit transport of the paper's direct-recognition model separation."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


HERE = Path(__file__).resolve().parent
EXPERIMENT = HERE.parent
RECOGNITION = EXPERIMENT / "delusion_recognition_transport"
sys.path.insert(0, str(RECOGNITION))

from analyze_recognition import (  # noqa: E402
    EXPECTED_INPUT_SHA256,
    MODEL_SPECS,
    PAPER_THEMES,
    PUBLISHED_SYNTHETIC,
    audit_published_source,
    output_input_identity,
    paper_theme_consensus,
    read_jsonl,
    sha256_file,
    validate_generation_artifacts,
)


PRIMARY_MODELS = (
    "allenai/Olmo-3-7B-Instruct",
    "meta-llama/Llama-3.1-8B-Instruct",
)
TABLE_ROUNDING_HALF_WIDTH = 0.005


def classification_error_bounds(row: pd.Series) -> tuple[float, float]:
    """Return adversarial FNR bounds for one positive classifier input."""
    if not bool(row["parse_valid"]):
        return 0.0, 1.0
    label = row["predicted_delusion"]
    if label not in {*PAPER_THEMES, "none"}:
        raise ValueError(f"Parse-valid row has unsupported label: {label!r}")
    error = float(label == "none")
    return error, error


def clustered_partial_interval(
    frame: pd.DataFrame,
    *,
    draws: int,
    seed: int,
) -> dict[str, float]:
    """Bootstrap the lower and upper endpoints of a paired partial bound."""
    grouped = frame.groupby("cluster_id", sort=False)
    lower_sums = grouped["gap_lower"].sum().to_numpy(dtype=float)
    upper_sums = grouped["gap_upper"].sum().to_numpy(dtype=float)
    sizes = grouped.size().to_numpy(dtype=float)
    if not len(sizes):
        raise ValueError("Cannot bootstrap an empty frame")

    rng = np.random.default_rng(seed)
    lower_estimates = np.empty(draws)
    upper_estimates = np.empty(draws)
    for start in range(0, draws, 2_000):
        count = min(2_000, draws - start)
        picks = rng.integers(0, len(sizes), size=(count, len(sizes)))
        denominators = sizes[picks].sum(axis=1)
        lower_estimates[start : start + count] = (
            lower_sums[picks].sum(axis=1) / denominators
        )
        upper_estimates[start : start + count] = (
            upper_sums[picks].sum(axis=1) / denominators
        )

    return {
        "gap_lower": float(frame["gap_lower"].mean()),
        "gap_upper": float(frame["gap_upper"].mean()),
        "partial_ci95_low": float(np.quantile(lower_estimates, 0.025)),
        "partial_ci95_high": float(np.quantile(upper_estimates, 0.975)),
        "partial_ci99_low": float(np.quantile(lower_estimates, 0.005)),
        "partial_ci99_high": float(np.quantile(upper_estimates, 0.995)),
    }


def audit_inputs(
    input_path: Path,
    theme_a_path: Path,
    theme_b_path: Path,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    if sha256_file(input_path) != EXPECTED_INPUT_SHA256:
        raise ValueError("Frozen classifier input hash changed")
    inputs = pd.DataFrame(read_jsonl(input_path))
    positives = inputs[inputs["evaluation_cohort"].eq("real_positive")]
    positive_ids = set(positives["input_id"])
    if len(inputs) != 932 or len(positives) != 522:
        raise ValueError("Frozen classifier cohort size changed")

    themes, theme_audit = paper_theme_consensus(
        theme_a_path,
        theme_b_path,
        positive_ids,
    )
    strict = themes[themes["is_paper_theme_consensus"].eq(True)].copy()
    strict = strict.merge(
        positives[["input_id", "cluster_id"]],
        on="input_id",
        validate="one_to_one",
    )
    if len(strict) != 98 or strict["cluster_id"].nunique() != 63:
        raise ValueError("Strict in-ontology cohort changed")
    return strict, {
        "input_path": str(input_path),
        "input_sha256": sha256_file(input_path),
        "rows": len(inputs),
        "positive_rows": len(positives),
        "strict_in_ontology_rows": len(strict),
        "strict_in_ontology_clusters": strict["cluster_id"].nunique(),
        "theme_consensus": theme_audit,
    }


def audit_outputs(
    paths: list[Path],
    combined_path: Path,
    inputs_path: Path,
    strict: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    generation_audit = validate_generation_artifacts(
        paths,
        combined_path,
        required_models=set(PRIMARY_MODELS),
    )
    inputs = pd.DataFrame(read_jsonl(inputs_path))
    expected_by_id = inputs.set_index("input_id").to_dict(orient="index")
    outputs = pd.DataFrame(read_jsonl(combined_path))
    rows = outputs[
        outputs["model"].isin(PRIMARY_MODELS)
        & outputs["input_id"].isin(strict["input_id"])
    ].copy()

    identity_ok = all(
        all(
            output_input_identity(
                row,
                expected_by_id.get(row.get("input_id"), {}),
            )
        )
        for row in rows.to_dict(orient="records")
    )
    coverage = rows.groupby("model")["input_id"].agg(["size", "nunique"])
    checks = {
        "primary_models": set(rows["model"]) == set(PRIMARY_MODELS),
        "strict_rows_per_model": bool(
            (coverage["size"] == 98).all()
            and (coverage["nunique"] == 98).all()
        ),
        "paired_ids": all(
            set(group["input_id"]) == set(strict["input_id"])
            for _, group in rows.groupby("model")
        ),
        "input_and_prompt_identity": identity_ok,
        "no_generation_errors": (
            "generation_error" not in rows
            or not rows["generation_error"].notna().any()
        ),
        "official_bf16": all(
            MODEL_SPECS[model]["fidelity"] == "official_bf16"
            for model in PRIMARY_MODELS
        ),
        "manifest_runs_complete": len(generation_audit["runs"]) == 2,
    }
    if not all(checks.values()):
        raise ValueError(f"Classifier output audit failed: {checks}")

    rows = rows.merge(
        strict[["input_id", "cluster_id"]].rename(
            columns={"cluster_id": "audited_cluster_id"}
        ),
        on="input_id",
        validate="many_to_one",
    )
    if not rows["cluster_id"].eq(rows["audited_cluster_id"]).all():
        raise ValueError("Source-conversation identity changed")
    return rows, {
        "checks": checks,
        "generation_artifacts": generation_audit,
        "combined_path": str(combined_path),
        "combined_sha256": sha256_file(combined_path),
    }


def build_bounds(rows: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    bounded = rows.copy()
    bounded[["fnr_lower", "fnr_upper"]] = bounded.apply(
        classification_error_bounds,
        axis=1,
        result_type="expand",
    )
    summaries = []
    for model, frame in bounded.groupby("model", sort=True):
        summaries.append(
            {
                "model": model,
                "model_label": MODEL_SPECS[model]["label"],
                "n_rows": len(frame),
                "n_clusters": frame["cluster_id"].nunique(),
                "parse_valid_rows": int(frame["parse_valid"].sum()),
                "unknown_rows": int((~frame["parse_valid"]).sum()),
                "known_false_negatives": int(frame["fnr_lower"].sum()),
                "fnr_lower": float(frame["fnr_lower"].mean()),
                "fnr_upper": float(frame["fnr_upper"].mean()),
                "published_synthetic_fnr": PUBLISHED_SYNTHETIC[
                    MODEL_SPECS[model]["label"]
                ]["fnr"],
            }
        )

    wide = bounded.pivot(
        index=["input_id", "cluster_id"],
        columns="model",
        values=["fnr_lower", "fnr_upper"],
    )
    if len(wide) != 98 or wide.isna().any(axis=None):
        raise ValueError("Paired recognition matrix is incomplete")
    paired = wide.reset_index()[["input_id", "cluster_id"]].copy()
    paired["gap_lower"] = (
        wide[("fnr_lower", PRIMARY_MODELS[0])]
        - wide[("fnr_upper", PRIMARY_MODELS[1])]
    ).to_numpy()
    paired["gap_upper"] = (
        wide[("fnr_upper", PRIMARY_MODELS[0])]
        - wide[("fnr_lower", PRIMARY_MODELS[1])]
    ).to_numpy()
    return pd.DataFrame(summaries), paired


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--inputs",
        type=Path,
        default=RECOGNITION / "artifacts" / "classifier_inputs.jsonl",
    )
    parser.add_argument(
        "--theme-a",
        type=Path,
        default=RECOGNITION / "artifacts" / "paper_theme_fit_a.jsonl",
    )
    parser.add_argument(
        "--theme-b",
        type=Path,
        default=RECOGNITION / "artifacts" / "paper_theme_fit_b.jsonl",
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
        "--combined",
        type=Path,
        default=HERE / "control_realism" / "official_bf16_classifications.jsonl",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=HERE / "recognition_model_gap",
    )
    parser.add_argument("--bootstrap-draws", type=int, default=100_000)
    parser.add_argument("--seed", type=int, default=260600975)
    args = parser.parse_args()

    published_audit = audit_published_source()
    strict, input_audit = audit_inputs(
        args.inputs,
        args.theme_a,
        args.theme_b,
    )
    outputs, output_audit = audit_outputs(
        args.generation_files,
        args.combined,
        args.inputs,
        strict,
    )
    model_bounds, paired = build_bounds(outputs)
    interval = clustered_partial_interval(
        paired,
        draws=args.bootstrap_draws,
        seed=args.seed,
    )

    olmo_fnr = PUBLISHED_SYNTHETIC["OLMo-3-7B"]["fnr"]
    llama_fnr = PUBLISHED_SYNTHETIC["Llama-3.1-8B"]["fnr"]
    synthetic_gap = olmo_fnr - llama_fnr
    synthetic_gap_min_rounded = (
        olmo_fnr
        - TABLE_ROUNDING_HALF_WIDTH
        - llama_fnr
        - TABLE_ROUNDING_HALF_WIDTH
    )
    ready = bool(
        interval["partial_ci99_high"] < synthetic_gap_min_rounded
        and input_audit["strict_in_ontology_rows"] == 98
        and input_audit["strict_in_ontology_clusters"] == 63
        and all(output_audit["checks"].values())
    )
    claim_gate = {
        "claim_id": "synthetic_recognition_model_gap_overestimation",
        "ready": ready,
        "paper_reported_olmo_minus_llama_fnr_gap": synthetic_gap,
        "paper_gap_min_after_table_rounding": synthetic_gap_min_rounded,
        "real_partial_identification": interval,
        "cohort": (
            "98 dual-rubric strict in-ontology real endpoints across 63 "
            "source conversations"
        ),
        "required_evidence": (
            "Both official BF16 models must reproduce the exact disclosed "
            "classifier protocol on every frozen input. Every invalid parse "
            "is assigned adversarially to maximize the real OLMo-minus-Llama "
            "FNR gap, and its conversation-clustered 99% upper bound must "
            "remain below the minimum paper gap after table-rounding error."
        ),
        "allowed_claim": (
            "The 71-point OLMo-Llama false-negative-rate gap reported on "
            "synthetic scripts did not transport to strict in-ontology real "
            "endpoints; even an adversarial partial-identification analysis "
            "left the real gap far smaller."
            if ready
            else None
        ),
        "guardrail": (
            "This is a descriptive benchmark-transport result, not a "
            "cross-study population interaction: the paper releases rounded "
            "aggregate classifier rates but not row-level outputs. The paper "
            "also does not disclose whether its assessment prompt was sent "
            "as a user or system message; this reproduction records the "
            "user-role operationalization."
        ),
        "model_bounds": model_bounds.to_dict(orient="records"),
    }
    summary = {
        "published_source_audit": published_audit,
        "input_audit": input_audit,
        "output_audit": output_audit,
        "claim_gate": claim_gate,
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    model_bounds.to_csv(args.output_dir / "model_fnr_bounds.csv", index=False)
    paired.to_csv(args.output_dir / "paired_partial_bounds.csv", index=False)
    (args.output_dir / "claim_gate.json").write_text(
        json.dumps(claim_gate, indent=2) + "\n",
        encoding="utf-8",
    )
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(claim_gate, indent=2))


if __name__ == "__main__":
    main()
