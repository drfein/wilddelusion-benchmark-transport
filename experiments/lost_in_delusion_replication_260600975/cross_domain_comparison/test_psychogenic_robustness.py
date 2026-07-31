#!/usr/bin/env python3
"""Tests for the fail-closed Psychogenic Machine transport gate."""

from __future__ import annotations

import unittest
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from analyze_psychogenic_robustness import (
    EXPERIMENT,
    MODEL_SPECS,
    build_claim_gate,
    exact_scenario_sign_flip,
    judge_error_mask,
    paired_units,
    validate_real_pair_inputs,
    validate_stochastic_provenance,
)


def stochastic_frame(domain: str = "synthetic") -> pd.DataFrame:
    rows = []
    per_model = {
        "synthetic": 432,
        "synthetic_local": 864,
        "real": 1962,
    }[domain]
    per_seed = per_model // 3
    for model, spec in MODEL_SPECS.items():
        for seed in (1101, 2202, 3303):
            for pair_index in range(per_seed // 2):
                if domain == "synthetic_local":
                    pair_id = f"pair-{pair_index // 18}"
                    history_anchor = (
                        "explicit"
                        if pair_index % 18 < 9
                        else "implicit"
                    )
                    turn_number = pair_index % 9 + 4
                else:
                    pair_id = f"pair-{pair_index}"
                    history_anchor = None
                    turn_number = None
                for condition in ("explicit", "implicit"):
                    rows.append(
                        {
                            "model": model,
                            "model_checkpoint_source": spec["source"],
                            "model_revision": spec["revision"],
                            "model_dtype": spec["dtype"],
                            "generation_backend": "vllm",
                            "decoding": (
                                "stochastic_api_default_sensitivity_matched_local"
                                if domain == "synthetic_local"
                                else "stochastic_api_default_sensitivity"
                            ),
                            "temperature": 1.0,
                            "top_p": 1.0,
                            "thinking_disabled": spec[
                                "thinking_disabled"
                            ],
                            "max_input_tokens": 11776,
                            "max_new_tokens": 512,
                            "replicate_seed": seed,
                            "request_seed": seed * 1000 + pair_index,
                            "pair_id": pair_id,
                            "history_anchor": history_anchor,
                            "turn_number": turn_number,
                            "condition": condition,
                            "pair_shared_dropped_message_count": 0,
                        }
                    )
    return pd.DataFrame(rows)


def pooled_frame(
    synthetic_low: float = 0.20,
    synthetic_high: float = 0.40,
) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "analysis": "real_matched_local",
                "delta_implicit_minus_explicit": 0.01,
                "ci99_low": -0.06,
                "ci99_high": 0.07,
            },
            {
                "analysis": "synthetic_full_trajectory",
                "delta_implicit_minus_explicit": 0.30,
                "ci99_low": synthetic_low,
                "ci99_high": synthetic_high,
            },
            {
                "analysis": "synthetic_matched_local",
                "delta_implicit_minus_explicit": 0.24,
                "ci99_low": 0.14,
                "ci99_high": 0.34,
            },
        ]
    )


def synthetic_paired_frame() -> pd.DataFrame:
    rows = []
    for model in MODEL_SPECS:
        for seed in (1101, 2202, 3303):
            for pair_index in range(8):
                for turn_number in range(4, 13):
                    for condition in ("explicit", "implicit"):
                        rows.append(
                            {
                                "model": model,
                                "pair_id": f"pair-{pair_index}",
                                "replicate_seed": seed,
                                "turn_number": turn_number,
                                "condition": condition,
                                "DCS": int(condition == "implicit"),
                            }
                        )
    return pd.DataFrame(rows)


def synthetic_local_paired_frame() -> pd.DataFrame:
    rows = []
    for model in MODEL_SPECS:
        for seed in (1101, 2202, 3303):
            for pair_index in range(8):
                for history_anchor in ("explicit", "implicit"):
                    for turn_number in range(4, 13):
                        for condition in ("explicit", "implicit"):
                            rows.append(
                                {
                                    "model": model,
                                    "pair_id": f"pair-{pair_index}",
                                    "history_anchor": history_anchor,
                                    "replicate_seed": seed,
                                    "turn_number": turn_number,
                                    "condition": condition,
                                    "DCS": int(condition == "implicit"),
                                }
                            )
    return pd.DataFrame(rows)


def interaction() -> list[dict[str, float | str]]:
    return [
        {
            "comparison": "real_local_minus_synthetic_local",
            "difference": -0.23,
            "ci99_low": -0.37,
            "ci99_high": -0.09,
        }
    ]


def seed_rows() -> list[dict[str, float | int]]:
    return [
        {
            "replicate_seed": seed,
            "synthetic_full_delta": synthetic,
            "synthetic_local_delta": synthetic - 0.04,
            "real_local_delta": real,
            "real_local_minus_synthetic_local": (
                real - (synthetic - 0.04)
            ),
        }
        for seed, synthetic, real in (
            (1101, 0.27, 0.01),
            (2202, 0.31, 0.02),
            (3303, 0.29, 0.00),
        )
    ]


def model_rows() -> list[dict[str, float | str]]:
    rows = []
    for model in ("OLMo-3-7B", "Llama-3.1-8B", "Qwen3-4B"):
        rows.extend(
            [
                {
                    "model": model,
                    "analysis": "synthetic_full_trajectory",
                    "delta_implicit_minus_explicit": 0.30,
                },
                {
                    "model": model,
                    "analysis": "synthetic_matched_local",
                    "delta_implicit_minus_explicit": 0.24,
                },
                {
                    "model": model,
                    "analysis": "real_matched_local",
                    "delta_implicit_minus_explicit": 0.01,
                },
            ]
        )
    return rows


def leave_one_out_rows() -> list[dict[str, float | str]]:
    return [
        {
            "omitted_synthetic_scenario": f"pair-{index}",
            "synthetic_full_delta": 0.30,
            "synthetic_local_delta": 0.24,
        }
        for index in range(8)
    ]


def negative_pooled_frame() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "analysis": "real_matched_local",
                "delta_implicit_minus_explicit": -0.05,
                "ci99_low": -0.09,
                "ci99_high": -0.01,
            },
            {
                "analysis": "synthetic_full_trajectory",
                "delta_implicit_minus_explicit": -0.29,
                "ci99_low": -0.48,
                "ci99_high": -0.11,
            },
            {
                "analysis": "synthetic_matched_local",
                "delta_implicit_minus_explicit": -0.10,
                "ci99_low": -0.16,
                "ci99_high": -0.01,
            },
        ]
    )


def negative_seed_rows() -> list[dict[str, float | int]]:
    return [
        {
            "replicate_seed": seed,
            "synthetic_full_delta": full,
            "synthetic_local_delta": local,
            "real_local_delta": real,
            "real_local_minus_synthetic_local": real - local,
        }
        for seed, full, local, real in (
            (1101, -0.25, -0.09, -0.01),
            (2202, -0.40, -0.11, -0.08),
            (3303, -0.22, -0.09, -0.06),
        )
    ]


def negative_model_rows() -> list[dict[str, float | str]]:
    rows = []
    for index, model in enumerate(
        ("OLMo-3-7B", "Llama-3.1-8B", "Qwen3-4B")
    ):
        rows.extend(
            [
                {
                    "model": model,
                    "analysis": "synthetic_full_trajectory",
                    "delta_implicit_minus_explicit": -0.30 + index * 0.02,
                },
                {
                    "model": model,
                    "analysis": "synthetic_matched_local",
                    "delta_implicit_minus_explicit": -0.10 + index * 0.05,
                },
                {
                    "model": model,
                    "analysis": "real_matched_local",
                    "delta_implicit_minus_explicit": -0.05,
                },
            ]
        )
    return rows


def negative_leave_one_out_rows() -> list[dict[str, float | str]]:
    return [
        {
            "omitted_synthetic_scenario": f"pair-{index}",
            "synthetic_full_delta": -0.29,
            "synthetic_local_delta": -0.10,
        }
        for index in range(8)
    ]


def sign_flip_results(direction: str) -> dict[str, dict[str, object]]:
    sign = -1.0 if direction == "negative" else 1.0
    common = {
        "observed_delta": 0.25 * sign,
        "all_scenario_effects_negative": direction == "negative",
        "all_scenario_effects_positive": direction == "positive",
        "p_two_sided": 0.0078125,
    }
    local = dict(common)
    local["p_two_sided"] = 0.03125
    return {
        "synthetic_full_trajectory": dict(common),
        "synthetic_matched_local": local,
    }


class PsychogenicRobustnessTests(unittest.TestCase):
    def test_absent_judge_error_column_is_clean(self) -> None:
        frame = pd.DataFrame([{"DCS": 0}, {"DCS": 1}])
        self.assertFalse(judge_error_mask(frame).any())

    def test_nonempty_judge_error_is_rejected(self) -> None:
        frame = pd.DataFrame(
            [{"judge_error": None}, {"judge_error": "parse failure"}]
        )
        self.assertEqual(judge_error_mask(frame).tolist(), [False, True])

    def test_frozen_real_pair_inputs_pass_identity_audit(self) -> None:
        root = (
            EXPERIMENT
            / "psychogenic_machine_transport_250910970"
            / "full522"
            / "artifacts"
        )
        audit = validate_real_pair_inputs(
            EXPERIMENT
            / "full_dataset_522"
            / "artifacts"
            / "cohort_full522.jsonl",
            root / "final_pairs.jsonl",
            root / "final_validations.jsonl",
            root / "model_inputs.jsonl",
        )
        self.assertTrue(audit["eligible"])
        self.assertEqual(audit["pairs"], 327)

    def test_locked_synthetic_provenance_passes(self) -> None:
        validate_stochastic_provenance(stochastic_frame(), "synthetic")

    def test_locked_real_provenance_passes(self) -> None:
        validate_stochastic_provenance(stochastic_frame("real"), "real")

    def test_locked_synthetic_local_provenance_passes(self) -> None:
        validate_stochastic_provenance(
            stochastic_frame("synthetic_local"),
            "synthetic_local",
        )

    def test_synthetic_pairing_has_eight_scenario_pairs(self) -> None:
        paired = paired_units(synthetic_paired_frame(), "synthetic")
        self.assertEqual(len(paired), 3 * 8 * 9 * 3)
        self.assertEqual(paired["pair_id"].nunique(), 8)

    def test_exact_sign_flip_uses_eight_scenario_units(self) -> None:
        paired = paired_units(synthetic_paired_frame(), "synthetic")
        result = exact_scenario_sign_flip(paired)
        self.assertEqual(result["n_scenarios"], 8)
        self.assertEqual(result["enumerated_sign_assignments"], 256)
        self.assertTrue(result["all_scenario_effects_positive"])
        self.assertAlmostEqual(result["p_two_sided"], 2 / 256)

    def test_incomplete_synthetic_pairing_fails(self) -> None:
        frame = synthetic_paired_frame().iloc[:-1]
        with self.assertRaisesRegex(ValueError, "Incomplete paired synthetic"):
            paired_units(frame, "synthetic")

    def test_synthetic_local_pairing_includes_both_history_anchors(self) -> None:
        paired = paired_units(
            synthetic_local_paired_frame(),
            "synthetic_local",
        )
        self.assertEqual(len(paired), 3 * 8 * 2 * 9 * 3)
        self.assertEqual(set(paired["history_anchor"]), {"explicit", "implicit"})

    def test_wrong_checkpoint_revision_fails(self) -> None:
        frame = stochastic_frame()
        frame.loc[0, "model_revision"] = "wrong"
        with self.assertRaisesRegex(ValueError, "model_revision"):
            validate_stochastic_provenance(frame, "synthetic")

    def test_unpaired_sampling_seed_fails_local_provenance(self) -> None:
        frame = stochastic_frame("synthetic_local")
        frame.loc[0, "request_seed"] += 1
        with self.assertRaisesRegex(ValueError, "shared sampling seed"):
            validate_stochastic_provenance(frame, "synthetic_local")

    def test_abstract_gate_requires_stochastic_bridge(self) -> None:
        gate = build_claim_gate(
            pd.DataFrame(),
            interactions=[],
            primary_seed_rows=[],
            stochastic_available=False,
        )
        self.assertFalse(gate["abstract_claim_ready"])
        self.assertIsNone(gate["allowed_claim"])

    def test_abstract_gate_opens_only_for_replicated_domain_drop(self) -> None:
        gate = build_claim_gate(
            pooled_frame(),
            interaction(),
            seed_rows(),
            stochastic_available=True,
            model_rows=model_rows(),
            leave_one_out_rows=leave_one_out_rows(),
            scenario_sign_flips=sign_flip_results("positive"),
        )
        self.assertTrue(gate["abstract_claim_ready"])
        self.assertIsNotNone(gate["allowed_claim"])

    def test_model_generalization_failure_gate_requires_robust_reversal(
        self,
    ) -> None:
        gate = build_claim_gate(
            negative_pooled_frame(),
            interaction(),
            negative_seed_rows(),
            stochastic_available=True,
            model_rows=negative_model_rows(),
            leave_one_out_rows=negative_leave_one_out_rows(),
            scenario_sign_flips=sign_flip_results("negative"),
        )
        self.assertTrue(gate["model_generalization_claim_ready"])
        self.assertIsNotNone(gate["allowed_model_generalization_claim"])
        self.assertFalse(gate["abstract_claim_ready"])

    def test_model_generalization_gate_requires_scenario_sign_flip(self) -> None:
        gate = build_claim_gate(
            negative_pooled_frame(),
            interaction(),
            negative_seed_rows(),
            stochastic_available=True,
            model_rows=negative_model_rows(),
            leave_one_out_rows=negative_leave_one_out_rows(),
        )
        self.assertFalse(gate["model_generalization_claim_ready"])

    def test_weak_scenario_sign_flip_closes_generalization_gate(self) -> None:
        sign_flips = sign_flip_results("negative")
        sign_flips["synthetic_full_trajectory"]["p_two_sided"] = 0.02
        gate = build_claim_gate(
            negative_pooled_frame(),
            interaction(),
            negative_seed_rows(),
            stochastic_available=True,
            model_rows=negative_model_rows(),
            leave_one_out_rows=negative_leave_one_out_rows(),
            scenario_sign_flips=sign_flips,
        )
        self.assertFalse(gate["model_generalization_claim_ready"])

    def test_negative_synthetic_seed_closes_abstract_gate(self) -> None:
        seeds = seed_rows()
        seeds[1]["synthetic_local_delta"] = -0.01
        seeds[1]["real_local_minus_synthetic_local"] = 0.03
        gate = build_claim_gate(
            pooled_frame(),
            interaction(),
            seeds,
            stochastic_available=True,
            model_rows=model_rows(),
            leave_one_out_rows=leave_one_out_rows(),
            scenario_sign_flips=sign_flip_results("positive"),
        )
        self.assertFalse(gate["abstract_claim_ready"])

    def test_nonnegative_seedwise_domain_drop_closes_gate(self) -> None:
        seeds = seed_rows()
        seeds[2]["real_local_minus_synthetic_local"] = 0.01
        gate = build_claim_gate(
            pooled_frame(),
            interaction(),
            seeds,
            stochastic_available=True,
            model_rows=model_rows(),
            leave_one_out_rows=leave_one_out_rows(),
            scenario_sign_flips=sign_flip_results("positive"),
        )
        self.assertFalse(gate["abstract_claim_ready"])

    def test_missing_model_consistency_closes_gate(self) -> None:
        gate = build_claim_gate(
            pooled_frame(),
            interaction(),
            seed_rows(),
            stochastic_available=True,
            model_rows=model_rows()[:-1],
            leave_one_out_rows=leave_one_out_rows(),
            scenario_sign_flips=sign_flip_results("positive"),
        )
        self.assertFalse(gate["abstract_claim_ready"])

    def test_model_level_full_trajectory_reversal_closes_gate(self) -> None:
        models = model_rows()
        for row in models:
            if (
                row["model"] == "Qwen3-4B"
                and row["analysis"] == "synthetic_full_trajectory"
            ):
                row["delta_implicit_minus_explicit"] = -0.01
        gate = build_claim_gate(
            pooled_frame(),
            interaction(),
            seed_rows(),
            stochastic_available=True,
            model_rows=models,
            leave_one_out_rows=leave_one_out_rows(),
            scenario_sign_flips=sign_flip_results("positive"),
        )
        self.assertFalse(gate["abstract_claim_ready"])

    def test_leave_one_scenario_out_reversal_closes_gate(self) -> None:
        leave_one_out = leave_one_out_rows()
        leave_one_out[3]["synthetic_local_delta"] = -0.01
        gate = build_claim_gate(
            pooled_frame(),
            interaction(),
            seed_rows(),
            stochastic_available=True,
            model_rows=model_rows(),
            leave_one_out_rows=leave_one_out,
            scenario_sign_flips=sign_flip_results("positive"),
        )
        self.assertFalse(gate["abstract_claim_ready"])


if __name__ == "__main__":
    unittest.main()
