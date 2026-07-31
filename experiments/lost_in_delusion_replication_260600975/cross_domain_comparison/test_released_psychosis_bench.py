#!/usr/bin/env python3
"""Tests for the released Psychosis-Bench log audit."""

from __future__ import annotations

import unittest

from analyze_released_psychosis_bench import (
    DEFAULT_SOURCE,
    EXPECTED_SOURCE_SHA256,
    analyze,
    parse_released_log,
    sha256_file,
    validate_release,
)


class ReleasedPsychosisBenchTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.frame = parse_released_log(
            DEFAULT_SOURCE.read_text(encoding="utf-8")
        )

    def test_frozen_release_integrity(self) -> None:
        self.assertEqual(sha256_file(DEFAULT_SOURCE), EXPECTED_SOURCE_SHA256)
        audit = validate_release(self.frame)
        self.assertEqual(audit["experiments"], 128)
        self.assertEqual(audit["models"], 8)
        self.assertEqual(audit["scenario_pairs"], 8)
        self.assertEqual(audit["available_turn_level_scores"], 1151)
        self.assertEqual(audit["incomplete_experiment"], 91)
        self.assertEqual(
            audit[
                "scenario_pairs_with_condition_specific_theme_or_harm_metadata"
            ],
            [8],
        )

    def test_recomputes_published_pooled_effect(self) -> None:
        models, scenarios, summary = analyze(
            self.frame,
            bootstrap_repetitions=20_000,
            seed=250910970,
        )
        self.assertEqual(len(models), 8)
        self.assertEqual(len(scenarios), 8)
        self.assertAlmostEqual(
            summary["recomputed_from_released_summary_means"],
            0.310140625,
        )
        self.assertLess(summary["absolute_reproduction_error"], 0.001)
        self.assertGreater(
            summary["fixed_model_scenario_cluster_bootstrap_ci99"][0],
            0,
        )

    def test_gpt4o_effect_is_positive_in_all_scenario_pairs(self) -> None:
        _, _, summary = analyze(
            self.frame,
            bootstrap_repetitions=2_000,
            seed=250910970,
        )
        self.assertAlmostEqual(
            summary["gpt4o"]["delta_implicit_minus_explicit"],
            0.222,
        )
        self.assertEqual(summary["gpt4o"]["positive_scenario_pairs"], 8)
        self.assertEqual(summary["gpt4o"]["negative_scenario_pairs"], 0)
        self.assertEqual(summary["gpt4o"]["tied_scenario_pairs"], 0)


if __name__ == "__main__":
    unittest.main()
