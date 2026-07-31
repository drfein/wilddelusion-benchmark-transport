#!/usr/bin/env python3
"""Tests for non-overlapping abstract claim selection."""

from __future__ import annotations

import unittest

from build_claim_registry import build_registry, index_gates


class ClaimRegistryTests(unittest.TestCase):
    def test_missing_gate_cannot_select_claim(self) -> None:
        rows = build_registry({})
        self.assertFalse(any(row["selected_for_abstract"] for row in rows))

    def test_only_one_claim_per_evidence_family(self) -> None:
        gates = {
            "lost_confirmed_endpoint_underestimation": {
                "descriptive_ready": True
            },
            "lost_matched_effect_underestimation": {"ready": True},
        }
        rows = build_registry(gates)
        selected = [
            row
            for row in rows
            if row["evidence_family"] == "lost_confirmation_severity"
            and row["selected_for_abstract"]
        ]
        self.assertEqual(len(selected), 1)
        alternate = next(
            row
            for row in rows
            if row["claim_id"] == "lost_matched_effect_underestimation"
        )
        self.assertEqual(alternate["status"], "ready_but_duplicate")

    def test_psychogenic_uses_abstract_gate_not_supporting_result(self) -> None:
        rows = build_registry(
            {
                "psychogenic_implicitness_domain_drop": {
                    "published_point_outside_real_ci": True,
                    "abstract_claim_ready": False,
                }
            }
        )
        row = next(
            item
            for item in rows
            if item["claim_id"] == "psychogenic_implicitness_domain_drop"
        )
        self.assertFalse(row["selected_for_abstract"])

    def test_psychogenic_model_generalization_uses_separate_gate(self) -> None:
        rows = build_registry(
            {
                "psychogenic_implicitness_domain_drop": {
                    "abstract_claim_ready": False,
                },
                "psychogenic_implicitness_model_generalization": {
                    "model_generalization_claim_ready": True,
                },
            }
        )
        row = next(
            item
            for item in rows
            if item["claim_id"]
            == "psychogenic_implicitness_model_generalization"
        )
        self.assertTrue(row["selected_for_abstract"])

    def test_audited_mod_harm_result_can_be_selected(self) -> None:
        rows = build_registry(
            {
                "lost_confirmation_failures_mostly_outside_mod_harm": {
                    "ready": True
                }
            }
        )
        row = next(
            item
            for item in rows
            if item["claim_id"]
            == "lost_confirmation_failures_mostly_outside_mod_harm"
        )
        self.assertEqual(row["status"], "selected")
        self.assertTrue(row["selected_for_abstract"])

    def test_visible_context_gate_overrides_weaker_endpoint_gate(self) -> None:
        gates = index_gates(
            lost={
                "claim_gates": [
                    {
                        "claim_id": (
                            "lost_confirmation_failures_mostly_outside_mod_harm"
                        ),
                        "ready": True,
                    }
                ]
            },
            context=None,
            psychogenic=None,
            recognition=None,
            control_realism=None,
            theme_coverage=None,
            recognition_gap=None,
            visible_mod_harm={"ready": False},
        )
        rows = build_registry(gates)
        row = next(
            item
            for item in rows
            if item["claim_id"]
            == "lost_confirmation_failures_mostly_outside_mod_harm"
        )
        self.assertFalse(row["selected_for_abstract"])

    def test_partial_identification_control_result_can_be_selected(self) -> None:
        rows = build_registry(
            {
                "natural_near_misses_are_harder_than_generated_controls": {
                    "ready": True
                }
            }
        )
        row = next(
            item
            for item in rows
            if item["claim_id"]
            == "natural_near_misses_are_harder_than_generated_controls"
        )
        self.assertTrue(row["selected_for_abstract"])
        self.assertEqual(row["evidence_family"], "recognition_specificity")

    def test_registry_selects_at_most_five_claims(self) -> None:
        rows = build_registry(
            {
                "lost_confirmed_endpoint_underestimation": {
                    "descriptive_ready": True
                },
                "paired_context_relocalizes_non_grounding_failures": {
                    "ready": True
                },
                "lost_primary_model_gap_overestimation": {
                    "descriptive_ready": True
                },
                "lost_confirmation_failures_mostly_outside_mod_harm": {
                    "ready": True
                },
                "psychogenic_implicitness_domain_drop": {
                    "abstract_claim_ready": True
                },
                "synthetic_overestimates_direct_recognition_sensitivity": {
                    "ready": True
                },
            }
        )
        selected = [row for row in rows if row["selected_for_abstract"]]
        self.assertEqual(len(selected), 5)
        deferred = next(
            row
            for row in rows
            if row["claim_id"]
            == "lost_confirmed_endpoint_underestimation"
        )
        self.assertEqual(deferred["status"], "ready_below_top_five")

    def test_theme_coverage_is_preferred_to_descriptive_backups(self) -> None:
        rows = build_registry(
            {
                "synthetic_three_theme_ontology_undercoverage": {
                    "ready": True
                }
            }
        )
        row = next(
            item
            for item in rows
            if item["claim_id"]
            == "synthetic_three_theme_ontology_undercoverage"
        )
        self.assertTrue(row["selected_for_abstract"])
        self.assertEqual(row["priority"], 5)

    def test_context_severity_uses_confirmation_gate(self) -> None:
        rows = build_registry(
            {
                "paired_context_increases_non_grounding_severity": {
                    "severity_ready": True,
                    "ready": False,
                },
                "paired_context_relocalizes_non_grounding_failures": {
                    "severity_ready": True,
                    "ready": False,
                },
            }
        )
        severity = next(
            row
            for row in rows
            if row["claim_id"]
            == "paired_context_increases_non_grounding_severity"
        )
        relocalization = next(
            row
            for row in rows
            if row["claim_id"]
            == "paired_context_relocalizes_non_grounding_failures"
        )
        self.assertTrue(severity["selected_for_abstract"])
        self.assertFalse(relocalization["selected_for_abstract"])


if __name__ == "__main__":
    unittest.main()
