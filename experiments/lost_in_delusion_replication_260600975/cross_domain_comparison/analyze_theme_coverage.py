#!/usr/bin/env python3
"""Audit coverage of Lost in Delusion's three-theme synthetic ontology."""

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

from analyze_recognition import paper_theme_consensus  # noqa: E402
from prepare_inputs import read_jsonl, sha256_file  # noqa: E402


EXPECTED_INPUT_SHA256 = (
    "543ba37348c5b17f61d0176c16ffedadcb0a5b3c61381bb3ddaa4bc125557d55"
)
EXPECTED_ADJUDICATION_SHA256 = {
    "A": "008b84a2c97de142a7c9421c3f0e9441509784e356ac7e3898c93c28dd7f72f5",
    "B": "3f270dd686b1dac688e671dea21e20477ef29998dd79058e750a6d16a1b7f44f",
}
EXPECTED_POSITIVE_ROWS = 522
EXPECTED_CLUSTERS = 321
MINIMUM_CI99_OUTSIDE_SHARE = 0.40
MINIMUM_OUTSIDE_CLUSTERS = 180
MINIMUM_LARGE_SOURCE_SHARE = 0.45
LARGE_SOURCE_ROWS = 40
PAPER_THEMES = {
    "emotional-dependence",
    "sentient-ai",
    "spiritual-messianic",
}
DEFINITE_OUTSIDE_SOURCE_THEMES = {
    "bizarre",
    "delusional_jealousy",
    "nihilistic",
    "persecutory",
    "somatic",
    "thought_broadcasting",
    "thought_insertion",
}


def cluster_bootstrap_share(
    frame: pd.DataFrame,
    value_column: str,
    repetitions: int,
    seed: int,
) -> dict[str, float | int]:
    """Bootstrap rows by source conversation and retain variable cluster sizes."""
    grouped = (
        frame.groupby("cluster_id", sort=False)[value_column]
        .agg(["sum", "size"])
        .reset_index(drop=True)
    )
    successes = grouped["sum"].to_numpy(dtype=float)
    totals = grouped["size"].to_numpy(dtype=float)
    rng = np.random.default_rng(seed)
    estimates = np.empty(repetitions, dtype=float)
    chunk_size = 5_000
    for start in range(0, repetitions, chunk_size):
        stop = min(start + chunk_size, repetitions)
        sampled = rng.integers(
            0,
            len(grouped),
            size=(stop - start, len(grouped)),
        )
        estimates[start:stop] = (
            successes[sampled].sum(axis=1) / totals[sampled].sum(axis=1)
        )
    low_99, low_95, high_95, high_99 = np.quantile(
        estimates, [0.005, 0.025, 0.975, 0.995]
    )
    return {
        "point": float(frame[value_column].mean()),
        "ci95_low": float(low_95),
        "ci95_high": float(high_95),
        "ci99_low": float(low_99),
        "ci99_high": float(high_99),
        "rows": int(len(frame)),
        "clusters": int(frame["cluster_id"].nunique()),
        "bootstrap_repetitions": repetitions,
        "bootstrap_seed": seed,
    }


def audit_inputs(
    classifier_inputs: Path,
    rubric_a: Path,
    rubric_b: Path,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    if sha256_file(classifier_inputs) != EXPECTED_INPUT_SHA256:
        raise ValueError("Frozen classifier input hash mismatch")
    for version, path in (("A", rubric_a), ("B", rubric_b)):
        observed = sha256_file(path)
        if observed != EXPECTED_ADJUDICATION_SHA256[version]:
            raise ValueError(
                f"Frozen rubric {version} adjudication hash mismatch: {observed}"
            )

    inputs = pd.DataFrame(read_jsonl(classifier_inputs))
    positives = inputs[
        inputs["evaluation_cohort"].eq("real_positive")
        & inputs["gold_label"].eq(1)
    ].copy()
    valid = (
        len(inputs) == 932
        and inputs["input_id"].nunique() == 932
        and len(positives) == EXPECTED_POSITIVE_ROWS
        and positives["input_id"].nunique() == EXPECTED_POSITIVE_ROWS
        and positives["cluster_id"].nunique() == EXPECTED_CLUSTERS
    )
    if not valid:
        raise ValueError("Frozen real-positive cohort integrity gate failed")

    consensus, consensus_audit = paper_theme_consensus(
        rubric_a,
        rubric_b,
        set(positives["input_id"]),
    )
    merged = positives.merge(consensus, on="input_id", validate="one_to_one")
    if len(merged) != EXPECTED_POSITIVE_ROWS:
        raise ValueError("Theme consensus join lost frozen positive rows")
    return merged, {
        "classifier_input_sha256": sha256_file(classifier_inputs),
        "adjudication_sha256": {
            "A": sha256_file(rubric_a),
            "B": sha256_file(rubric_b),
        },
        "positive_rows": len(merged),
        "positive_clusters": merged["cluster_id"].nunique(),
        "consensus": consensus_audit,
    }


def grouped_summary(frame: pd.DataFrame, column: str) -> pd.DataFrame:
    return (
        frame.groupby(column, dropna=False)
        .agg(
            rows=("input_id", "size"),
            clusters=("cluster_id", "nunique"),
            outside_rows=("is_outside_ontology_consensus", "sum"),
            outside_share=("is_outside_ontology_consensus", "mean"),
            in_ontology_rows=("is_paper_theme_consensus", "sum"),
            in_ontology_share=("is_paper_theme_consensus", "mean"),
        )
        .reset_index()
        .sort_values(["rows", column], ascending=[False, True])
    )


def run(args: argparse.Namespace) -> dict[str, Any]:
    frame, integrity = audit_inputs(
        args.classifier_inputs,
        args.rubric_a,
        args.rubric_b,
    )
    outside = frame["is_outside_ontology_consensus"].astype(bool)
    inside = frame["is_paper_theme_consensus"].astype(bool)
    if (outside & inside).any():
        raise ValueError("Inside- and outside-ontology consensus overlap")

    interval = cluster_bootstrap_share(
        frame,
        "is_outside_ontology_consensus",
        args.bootstrap_repetitions,
        args.seed,
    )
    source = grouped_summary(frame, "source")
    clinical_theme = grouped_summary(frame, "theme")
    large_sources = source[source["rows"] >= LARGE_SOURCE_ROWS]
    definite_source = frame["theme"].isin(DEFINITE_OUTSIDE_SOURCE_THEMES)
    outside_rows = frame[outside]
    orthogonal_corroboration = int(
        outside_rows["theme"].isin(DEFINITE_OUTSIDE_SOURCE_THEMES).sum()
    )

    gate_checks = {
        "complete_frozen_cohort": (
            len(frame) == EXPECTED_POSITIVE_ROWS
            and frame["cluster_id"].nunique() == EXPECTED_CLUSTERS
        ),
        "dual_rubric_outside_ci99_above_40pct": (
            interval["ci99_low"] > MINIMUM_CI99_OUTSIDE_SHARE
        ),
        "outside_consensus_spans_at_least_180_conversations": (
            int(outside_rows["cluster_id"].nunique())
            >= MINIMUM_OUTSIDE_CLUSTERS
        ),
        "all_large_sources_above_45pct": bool(
            len(large_sources) >= 3
            and large_sources["outside_share"].gt(
                MINIMUM_LARGE_SOURCE_SHARE
            ).all()
        ),
        "orthogonal_taxonomy_majority_corroborates": (
            orthogonal_corroboration / max(1, int(outside.sum())) > 0.50
        ),
    }
    claim_gate = {
        "claim_id": "synthetic_three_theme_ontology_undercoverage",
        "ready": all(gate_checks.values()),
        "gate_checks": gate_checks,
        "claim": (
            "The three-theme handcrafted benchmark covers too narrow a case "
            "composition: two independently worded fixed-model rubrics place "
            "a large share of confirmed real endpoints outside its ontology."
        ),
        "allowed_wording": (
            f"{int(outside.sum())}/{len(frame)} confirmed endpoints "
            f"({100 * float(outside.mean()):.1f}%) were dual-rubric "
            "outside-ontology consensus; the 99% conversation-cluster "
            f"bootstrap interval was {100 * interval['ci99_low']:.1f}% to "
            f"{100 * interval['ci99_high']:.1f}%."
        ),
        "prohibited_wording": [
            "Do not call the two rubrics independent human adjudicators.",
            "Do not claim a population majority because the 99% interval "
            "crosses 50%.",
            "Do not treat unresolved rubric disagreements as in-ontology.",
        ],
    }
    summary = {
        "estimand": (
            "Share of 522 confirmed real endpoints assigned outside the "
            "paper's closed three-theme ontology by both independently "
            "worded rubrics using the same fixed mini-model, with necessary "
            "conditions absent and neither confidence low."
        ),
        "integrity": integrity,
        "paper_ontology": sorted(PAPER_THEMES),
        "strict_outside": {
            **interval,
            "rows": int(outside.sum()),
            "clusters": int(outside_rows["cluster_id"].nunique()),
        },
        "strict_in_ontology": {
            "rows": int(inside.sum()),
            "share": float(inside.mean()),
            "clusters": int(frame[inside]["cluster_id"].nunique()),
        },
        "unresolved": {
            "rows": int((~outside & ~inside).sum()),
            "share": float((~outside & ~inside).mean()),
        },
        "orthogonal_taxonomy_corroboration": {
            "definite_outside_source_theme_rows_in_cohort": int(
                definite_source.sum()
            ),
            "strict_outside_rows_with_definite_outside_source_theme": (
                orthogonal_corroboration
            ),
            "share_of_strict_outside": (
                orthogonal_corroboration / max(1, int(outside.sum()))
            ),
            "note": (
                "This is a distinct psychiatric-theme labeling axis, not an "
                "independent human adjudication."
            ),
        },
        "claim_gate": claim_gate,
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    source.to_csv(args.output_dir / "coverage_by_source.csv", index=False)
    clinical_theme.to_csv(
        args.output_dir / "coverage_by_clinical_theme.csv", index=False
    )
    frame[
        [
            "input_id",
            "cluster_id",
            "source",
            "theme",
            "paper_theme_consensus",
            "is_paper_theme_consensus",
            "is_outside_ontology_consensus",
        ]
    ].to_csv(args.output_dir / "endpoint_classifications.csv", index=False)
    (args.output_dir / "claim_gate.json").write_text(
        json.dumps(claim_gate, indent=2) + "\n",
        encoding="utf-8",
    )
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2))
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--classifier-inputs",
        type=Path,
        default=RECOGNITION / "artifacts" / "classifier_inputs.jsonl",
    )
    parser.add_argument(
        "--rubric-a",
        type=Path,
        default=RECOGNITION / "artifacts" / "paper_theme_fit_a.jsonl",
    )
    parser.add_argument(
        "--rubric-b",
        type=Path,
        default=RECOGNITION / "artifacts" / "paper_theme_fit_b.jsonl",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=HERE / "theme_coverage",
    )
    parser.add_argument("--bootstrap-repetitions", type=int, default=100_000)
    parser.add_argument("--seed", type=int, default=260600975)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
