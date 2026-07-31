#!/usr/bin/env python3
"""Fail-closed completion audit for the five abstract claims."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Callable


HERE = Path(__file__).resolve().parent
REGISTRY = HERE / "abstract_claim_registry" / "claim_registry.json"

SOURCE_PATHS = {
    "lost_confirmation_failures_mostly_outside_mod_harm": (
        HERE / "visible_context_mod_harm" / "claim_gate.json"
    ),
    "natural_near_misses_are_harder_than_generated_controls": (
        HERE / "control_realism" / "claim_gate.json"
    ),
    "psychogenic_implicitness_model_generalization": (
        HERE / "psychogenic_robustness" / "claim_gate.json"
    ),
    "synthetic_three_theme_ontology_undercoverage": (
        HERE / "theme_coverage" / "claim_gate.json"
    ),
    "synthetic_recognition_model_gap_overestimation": (
        HERE / "recognition_model_gap" / "claim_gate.json"
    ),
    "paired_context_relocalizes_non_grounding_failures": (
        HERE.parent
        / "full_history_522"
        / "results"
        / "context_ablation_paired"
        / "analysis_shared_context"
        / "claim_gate.json"
    ),
}


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def close(left: float, right: float) -> bool:
    return math.isclose(left, right, rel_tol=0, abs_tol=1e-12)


def verify_visible_mod_harm(gate: dict[str, Any]) -> dict[str, Any]:
    require(gate.get("ready") is True, "Visible-context gate is closed")
    screen = gate["screen"]
    evidence = gate["evidence"]
    require(screen["possible_n"] == 161, "Visible screen denominator changed")
    require(len(evidence) == 2, "Expected two official-BF16 response models")
    captured = []
    for row in evidence:
        require(row["n_endpoints"] == 522, "Visible response cohort changed")
        require(row["n_clusters"] == 321, "Visible response clusters changed")
        require(
            row["validate_amplify_outside_share_ci99_low"] > 0.5,
            "Validate/Amplify outside share is not resolved above one half",
        )
        require(
            row["non_grounding_outside_share_ci99_low"] > 0.5,
            "Non-grounding outside share is not resolved above one half",
        )
        require(
            row["dcs_mass_outside_share_ci99_low"] > 0.5,
            "Ordinal DCS outside mass is not resolved above one half",
        )
        captured.append(row["validate_amplify_inside_share"])
    return {
        "models": len(evidence),
        "endpoints_per_model": 522,
        "possible_mod_harm_endpoints": 161,
        "validate_amplify_captured_range": [min(captured), max(captured)],
        "guardrail": gate["causal_guardrail"],
    }


def verify_control_realism(gate: dict[str, Any]) -> dict[str, Any]:
    require(gate.get("ready") is True, "Control-realism gate is closed")
    require(
        gate.get("all_primary_worst_case_gaps_positive_99") is True,
        "Worst-case control gap is not positive at 99%",
    )
    evidence = gate["evidence"]
    require(len(evidence) == 2, "Expected two official-BF16 classifiers")
    gaps = []
    lower_bounds = []
    for row in evidence:
        require(row["model_fidelity"] == "official_bf16", "Non-BF16 model")
        require(row["natural_n_rows"] >= 66, "Natural-control cohort shrank")
        require(row["natural_n_clusters"] >= 50, "Natural clusters shrank")
        require(row["generated_n_rows"] >= 100, "Generated controls shrank")
        require(row["generated_n_clusters"] >= 90, "Generated clusters shrank")
        require(row["gap_lower_ci99_low"] > 0, "Adversarial 99% gap crosses zero")
        require(
            row["unknown_assignment_for_gap_lower"]
            == "natural unknowns are non-positive; generated unknowns are positive",
            "Malformed outputs were not assigned against the claim",
        )
        gaps.append(row["gap_lower"])
        lower_bounds.append(row["gap_lower_ci99_low"])
    return {
        "models": len(evidence),
        "worst_case_gap_range": [min(gaps), max(gaps)],
        "ci99_lower_bound_range": [min(lower_bounds), max(lower_bounds)],
        "guardrail": gate["negative_label_definition"],
    }


def verify_psychogenic(gate: dict[str, Any]) -> dict[str, Any]:
    require(
        gate.get("model_generalization_claim_ready") is True,
        "Psychogenic model-generalization gate is closed",
    )
    for key in (
        "real_ci99",
        "synthetic_full_trajectory_ci99",
        "synthetic_matched_local_ci99",
    ):
        require(gate[key][1] < 0, f"{key} does not exclude zero negatively")
    require(gate["all_seed_effects_negative"] is True, "Seed reversal failed")
    require(
        gate["all_model_effects_nonpositive"] is True,
        "Model-level reversal failed",
    )
    require(
        gate["all_eight_leave_one_scenario_out_effects_negative"] is True,
        "Leave-one-scenario reversal failed",
    )
    require(gate["real_pair_input_audit"]["eligible"] is True, "Real pairs failed")
    require(gate["synthetic_input_audit"]["eligible"] is True, "Cases failed")
    flips = gate["scenario_sign_flip_sensitivity"]
    full = flips["synthetic_full_trajectory"]
    local = flips["synthetic_matched_local"]
    require(full["n_scenarios"] == 8, "Synthetic scenario count changed")
    require(full["all_scenario_effects_negative"] is True, "A scenario did not reverse")
    require(full["p_two_sided"] <= 0.01, "Full sign-flip sensitivity failed")
    require(local["observed_delta"] < 0, "Matched-local direction changed")
    require(local["p_two_sided"] <= 0.05, "Matched-local sensitivity failed")
    require(
        close(gate["synthetic_full_trajectory_delta"], -0.28858024691358025),
        "Psychogenic headline effect changed",
    )
    return {
        "models": len(gate["primary_models"]),
        "released_scenarios": full["n_scenarios"],
        "synthetic_full_delta": gate["synthetic_full_trajectory_delta"],
        "synthetic_full_ci99": gate["synthetic_full_trajectory_ci99"],
        "scenario_sign_flip_two_sided": full["p_two_sided"],
        "matched_real_pairs": gate["real_pair_input_audit"]["pairs"],
        "guardrail": full["interpretation"],
    }


def verify_theme_coverage(gate: dict[str, Any]) -> dict[str, Any]:
    require(gate.get("ready") is True, "Theme-coverage gate is closed")
    require(all(gate["gate_checks"].values()), "A theme-coverage check failed")
    summary = json.loads(
        (HERE / "theme_coverage" / "summary.json").read_text(encoding="utf-8")
    )
    outside = summary["strict_outside"]
    require(outside["rows"] == 274, "Outside-ontology count changed")
    require(summary["integrity"]["positive_rows"] == 522, "Theme cohort changed")
    require(outside["clusters"] >= 180, "Outside-ontology clusters shrank")
    require(outside["ci99_low"] < 0.5 < outside["ci99_high"], "Majority caveat changed")
    return {
        "outside_rows": outside["rows"],
        "total_rows": summary["integrity"]["positive_rows"],
        "outside_share": outside["point"],
        "ci99": [outside["ci99_low"], outside["ci99_high"]],
        "adjudicator": summary["integrity"]["consensus"]["adjudicator"],
        "guardrail": gate["prohibited_wording"],
    }


def verify_recognition_gap(gate: dict[str, Any]) -> dict[str, Any]:
    require(gate.get("ready") is True, "Recognition-gap gate is closed")
    real = gate["real_partial_identification"]
    require(len(gate["model_bounds"]) == 2, "Recognition model pair changed")
    require(
        real["partial_ci99_high"] < gate["paper_gap_min_after_table_rounding"],
        "Adversarial real-gap upper bound reaches the paper gap",
    )
    require(real["gap_upper"] < 0.25, "Point partial-identification gap widened")
    require(
        all(row["n_rows"] == 98 and row["n_clusters"] == 63 for row in gate["model_bounds"]),
        "Strict in-ontology recognition cohort changed",
    )
    return {
        "paper_gap": gate["paper_reported_olmo_minus_llama_fnr_gap"],
        "real_point_bounds": [real["gap_lower"], real["gap_upper"]],
        "real_clustered_ci99_upper": real["partial_ci99_high"],
        "cohort": gate["cohort"],
        "guardrail": gate["guardrail"],
    }


def verify_context(gate: dict[str, Any]) -> dict[str, Any]:
    require(gate.get("ready") is True, "Selected context gate is closed")
    evidence = gate["evidence"]
    require(len(evidence) == 2, "Context model count changed")
    checks = (
        "exact_aggregate_ci99_within_10pp",
        "secondary_aggregate_ci99_within_10pp",
        "exact_flip_ci99_above_15pct",
        "secondary_flip_ci99_above_15pct",
        "exact_excess_flip_ci99_above_zero",
        "secondary_excess_flip_ci99_above_zero",
        "same_direction_flip_ci99_above_7_5pct",
    )
    require(
        all(row[key] is True for row in evidence for key in checks),
        "A selected context confirmation check failed",
    )
    return {
        "models": len(evidence),
        "paired_endpoints_per_model": evidence[0]["paired_n"],
        "source_conversations_per_model": evidence[0]["cluster_n"],
        "guardrail": gate["prohibited_wording"],
    }


VERIFIERS: dict[str, Callable[[dict[str, Any]], dict[str, Any]]] = {
    "lost_confirmation_failures_mostly_outside_mod_harm": verify_visible_mod_harm,
    "natural_near_misses_are_harder_than_generated_controls": verify_control_realism,
    "psychogenic_implicitness_model_generalization": verify_psychogenic,
    "synthetic_three_theme_ontology_undercoverage": verify_theme_coverage,
    "synthetic_recognition_model_gap_overestimation": verify_recognition_gap,
    "paired_context_relocalizes_non_grounding_failures": verify_context,
}


def verify_rows(
    rows: list[dict[str, Any]],
    registry_path: Path = REGISTRY,
) -> dict[str, Any]:
    selected = [row for row in rows if row.get("selected_for_abstract")]
    require(len(selected) == 5, "Exactly five claims must be selected")
    families = {row["evidence_family"] for row in selected}
    require(len(families) == 5, "Selected claims must use distinct evidence families")
    audits = []
    for row in selected:
        claim_id = row["claim_id"]
        require(row["status"] == "selected", f"{claim_id}: status mismatch")
        require(row["gate_ready"] is True, f"{claim_id}: embedded gate closed")
        require(row["headline_eligible"] is True, f"{claim_id}: headline disabled")
        require(claim_id in VERIFIERS, f"{claim_id}: no completion verifier")
        source_path = SOURCE_PATHS[claim_id]
        require(source_path.exists(), f"{claim_id}: source gate is missing")
        source_gate = json.loads(source_path.read_text(encoding="utf-8"))
        require(row["gate"] == source_gate, f"{claim_id}: registry gate is stale")
        audits.append(
            {
                "claim_id": claim_id,
                "headline": row["headline"],
                "evidence_family": row["evidence_family"],
                "source_gate": source_path.relative_to(HERE.parent).as_posix(),
                "source_gate_sha256": sha256_file(source_path),
                "verification": VERIFIERS[claim_id](source_gate),
                "claim_guardrail": row["guardrail"],
            }
        )
    return {
        "complete": True,
        "selected_claims": len(selected),
        "distinct_evidence_families": len(families),
        "registry_sha256": sha256_file(registry_path),
        "claims": audits,
    }


def markdown_report(audit: dict[str, Any]) -> str:
    lines = [
        "# Abstract claim completion audit",
        "",
        "**PASS: exactly five non-overlapping claims satisfy every claim-specific gate.**",
        "",
    ]
    for index, row in enumerate(audit["claims"], 1):
        lines.extend(
            [
                f"## {index}. {row['headline']}",
                "",
                f"- Evidence family: `{row['evidence_family']}`",
                f"- Source gate SHA-256: `{row['source_gate_sha256']}`",
                f"- Verified evidence: `{json.dumps(row['verification'], sort_keys=True)}`",
                f"- Guardrail: {row['claim_guardrail']}",
                "",
            ]
        )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--registry", type=Path, default=REGISTRY)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=HERE / "abstract_claim_registry",
    )
    args = parser.parse_args()
    rows = json.loads(args.registry.read_text(encoding="utf-8"))
    audit = verify_rows(rows, args.registry)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "COMPLETION_AUDIT.json").write_text(
        json.dumps(audit, indent=2) + "\n", encoding="utf-8"
    )
    (args.output_dir / "COMPLETION_AUDIT.md").write_text(
        markdown_report(audit), encoding="utf-8"
    )
    print(json.dumps(audit, indent=2))


if __name__ == "__main__":
    main()
