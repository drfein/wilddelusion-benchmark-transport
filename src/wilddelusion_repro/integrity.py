"""Zero-cost checks for the frozen headline-result package."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .pipeline import sha256_file

EXPERIMENT = Path("experiments/lost_in_delusion_replication_260600975")


def verify_packaged_results(root: Path) -> dict[str, Any]:
    registry_path = root / "results" / "claim_registry.json"
    registry = json.loads(registry_path.read_text(encoding="utf-8"))
    selected = [row for row in registry if row.get("selected_for_abstract")]
    errors: list[str] = []
    if len(selected) != 5:
        errors.append(f"expected 5 selected claims, found {len(selected)}")
    families = {row.get("evidence_family") for row in selected}
    if len(families) != len(selected):
        errors.append("selected claims do not have distinct evidence families")
    if any(row.get("gate_ready") is not True for row in selected):
        errors.append("a selected claim has a closed gate")

    experiment_registry = (
        root
        / EXPERIMENT
        / "cross_domain_comparison"
        / "abstract_claim_registry"
        / "claim_registry.json"
    )
    if registry_path.read_bytes() != experiment_registry.read_bytes():
        errors.append("top-level and exact-layout claim registries differ")

    audit_path = root / "results" / "COMPLETION_AUDIT.json"
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    expected_hashes = {
        row["claim_id"]: row["source_gate_sha256"] for row in audit["claims"]
    }
    for row in selected:
        gate_source = Path(row["gate_source"])
        source_map = {
            "visible_mod_harm": "visible_context_mod_harm",
            "recognition": "control_realism",
            "psychogenic": "psychogenic_robustness",
            "theme_coverage": "theme_coverage",
            "recognition_gap": "recognition_model_gap",
        }
        directory = source_map.get(gate_source.name)
        if directory is None:
            errors.append(f"unknown gate source {gate_source}")
            continue
        gate_path = (
            root
            / EXPERIMENT
            / "cross_domain_comparison"
            / directory
            / "claim_gate.json"
        )
        if sha256_file(gate_path) != expected_hashes.get(row["claim_id"]):
            errors.append(f"gate hash mismatch for {row['claim_id']}")

    if errors:
        raise ValueError("; ".join(errors))
    return {
        "status": "pass",
        "selected_claims": len(selected),
        "evidence_families": sorted(families),
        "registry_sha256": sha256_file(registry_path),
        "completion_audit_sha256": sha256_file(audit_path),
    }
