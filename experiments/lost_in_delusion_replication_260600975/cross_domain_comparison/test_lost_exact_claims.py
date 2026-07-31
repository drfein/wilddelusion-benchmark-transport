#!/usr/bin/env python3
"""Unit tests for fail-closed Lost transport claim analysis."""

from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from analyze_lost_exact_claims import (
    EXPECTED_JUDGE_MODEL,
    EXPECTED_JUDGE_REVISION,
    EXPECTED_JUDGE_TEMPLATE_SHA256,
    EXPECTED_PROTOCOL,
    MOD_HARM_DEFINITION,
    MOD_HARM_LABEL_MODEL,
    MOD_HARM_PROTOCOL,
    audit_judgment_manifest,
    audit_judgments,
    audit_mod_harm_superset,
    audit_published_source,
    bootstrap_binary_group_difference,
    bootstrap_ratio,
    claim_gates,
    endpoint_rows,
    synthetic_uncertainty_envelope,
)


def write_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_mod_harm_fixture(root: Path) -> tuple[Path, dict[str, Path], Path]:
    cohort = root / "cohort.jsonl"
    cohort_rows = [
        {
            "pair_id": f"pair-{index:03d}",
            "cluster_id": f"cluster-{index // 2:03d}",
            "target_text": f"target {index}",
            "history_messages": [
                {"role": "user", "content": f"target {index}"}
            ],
        }
        for index in range(522)
    ]
    write_jsonl(cohort, cohort_rows)
    cohort.with_suffix(".jsonl.manifest.json").write_text(
        json.dumps(
            {
                "rows": 522,
                "target_integrity_errors": 0,
                "sha256": sha256_file(cohort),
            }
        ),
        encoding="utf-8",
    )

    labels: dict[str, Path] = {}
    sets = {
        "A": (set(range(7)), set(range(7, 10))),
        "B": (set(range(6)), set()),
        "C": (set(range(20)), set(range(20, 45))),
    }
    for rubric, (yes_indices, unclear_indices) in sets.items():
        path = root / f"labels_{rubric}.jsonl"
        rows = []
        for index, cohort_row in enumerate(cohort_rows):
            label = (
                "yes"
                if index in yes_indices
                else "unclear"
                if index in unclear_indices
                else "no"
            )
            rows.append(
                {
                    "pair_id": cohort_row["pair_id"],
                    "cluster_id": cohort_row["cluster_id"],
                    "label": label,
                    "evidence_quote": (
                        cohort_row["target_text"] if label == "yes" else ""
                    ),
                    "rubric_version": rubric,
                    "operational_definition": MOD_HARM_DEFINITION,
                    "label_model": MOD_HARM_LABEL_MODEL,
                    "reasoning_effort": "none",
                    "target_sha256": hashlib.sha256(
                        cohort_row["target_text"].encode("utf-8")
                    ).hexdigest(),
                }
            )
        write_jsonl(path, rows)
        counts = {
            label: sum(row["label"] == label for row in rows)
            for label in ("yes", "no", "unclear")
        }
        path.with_suffix(".jsonl.manifest.json").write_text(
            json.dumps(
                {
                    "protocol": MOD_HARM_PROTOCOL,
                    "operational_definition": MOD_HARM_DEFINITION,
                    "rubric_version": rubric,
                    "model": MOD_HARM_LABEL_MODEL,
                    "reasoning_effort": "none",
                    "input_sha256": sha256_file(cohort),
                    "expected_rows": 522,
                    "successful_rows": 522,
                    "new_errors": 0,
                    "label_counts": counts,
                    "output_sha256": sha256_file(path),
                }
            ),
            encoding="utf-8",
        )
        labels[rubric] = path

    legacy = root / "legacy.jsonl"
    write_jsonl(
        legacy,
        [
            {
                "pair_id": row["pair_id"],
                "cluster_id": row["cluster_id"],
                "label_model": "gpt-5.4-mini",
                "high_stakes_action": 31 <= index < 73,
            }
            for index, row in enumerate(cohort_rows)
        ],
    )
    legacy.with_suffix(".jsonl.manifest.json").write_text(
        json.dumps(
            {
                "candidate_targets": 522,
                "successful_labels": 522,
                "model": "gpt-5.4-mini",
                "new_errors": 0,
            }
        ),
        encoding="utf-8",
    )
    return cohort, labels, legacy


class ClaimAuditTests(unittest.TestCase):
    def test_published_values_are_bound_to_source_table(self) -> None:
        audit = audit_published_source()
        self.assertEqual(
            audit["verified_values"]["allenai/Olmo-3-7B-Instruct"],
            {
                "control_mean": 0.12,
                "delusion_mean": 1.08,
                "paired_difference": 0.96,
            },
        )

    def test_exact_manifest_binds_inputs_and_output(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            inputs = [root / "first.jsonl", root / "second.jsonl"]
            for index, path in enumerate(inputs):
                write_jsonl(path, [{"generation_id": f"g-{index}"}])
            output = root / "judgments.jsonl"
            write_jsonl(output, [{}, {}])
            manifest = {
                "model_id": EXPECTED_JUDGE_MODEL,
                "model_revision": EXPECTED_JUDGE_REVISION,
                "dtype": "bfloat16",
                "judge_protocol": EXPECTED_PROTOCOL,
                "previous_exchanges": 3,
                "input_roles": ["user"],
                "extra_system_prompt": None,
                "temperature": 0.6,
                "top_p": 0.95,
                "top_k": 20,
                "rubric_template_sha256": EXPECTED_JUDGE_TEMPLATE_SHA256,
                "candidate_responses": 2,
                "successful_rows": 2,
                "new_errors": 0,
                "input_files": [
                    {
                        "path": str(path),
                        "sha256": sha256_file(path),
                        "rows": 1,
                    }
                    for path in inputs
                ],
                "output_sha256": sha256_file(output),
            }
            output.with_suffix(".jsonl.manifest.json").write_text(
                json.dumps(manifest),
                encoding="utf-8",
            )
            self.assertTrue(
                audit_judgment_manifest(output, inputs)["eligible"]
            )
            output.write_text("{}\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "manifest integrity"):
                audit_judgment_manifest(output, inputs)

    def test_mod_harm_superset_is_fail_closed_and_conservative(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            cohort, labels, legacy = write_mod_harm_fixture(Path(directory))
            possible, audit = audit_mod_harm_superset(
                cohort, labels, legacy
            )
            self.assertEqual(len(possible), 73)
            self.assertEqual(
                audit["screen_positive_counts"],
                {
                    "A": 10,
                    "B": 6,
                    "C": 45,
                    "legacy_broad_danger": 42,
                    "lexical_harm_sensitivity": 0,
                },
            )
            self.assertTrue(audit["eligible"])

    def test_mod_harm_superset_rejects_target_hash_drift(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            cohort, labels, legacy = write_mod_harm_fixture(Path(directory))
            rows = [
                json.loads(line)
                for line in labels["C"].read_text(encoding="utf-8").splitlines()
            ]
            rows[0]["target_sha256"] = "0" * 64
            write_jsonl(labels["C"], rows)
            manifest_path = labels["C"].with_suffix(
                ".jsonl.manifest.json"
            )
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            manifest["output_sha256"] = sha256_file(labels["C"])
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
            with self.assertRaisesRegex(
                ValueError, "Mod.Harm rubric C integrity gate failed"
            ):
                audit_mod_harm_superset(cohort, labels, legacy)

    def test_synthetic_envelope_clusters_by_persona(self) -> None:
        _, high_30 = synthetic_uncertainty_envelope(0.69, 0.0, 2.0)
        _, high_90 = synthetic_uncertainty_envelope(
            0.69, 0.0, 2.0, clusters=90
        )
        self.assertGreater(high_30, high_90)
        self.assertGreater(high_30, 1.2)

    def test_point_only_transport_differences_remain_descriptive(self) -> None:
        models = [
            "allenai/Olmo-3-7B-Instruct",
            "meta-llama/Llama-3.1-8B-Instruct",
        ]
        audit = pd.DataFrame(
            {
                "model": models,
                "model_fidelity": [
                    "official_bf16_temporally_attested_revision"
                ]
                * 2,
                "complete_and_eligible": [True, True],
            }
        )
        endpoint = pd.DataFrame(
            {
                "model": models,
                "published_synthetic_target_mean_dcs": [1.08, 0.69],
                "published_target_below_real_ci99": [True, True],
                "real_target_above_synthetic_envelope99": [False, False],
                "published_effect_below_real_ci99": [False, True],
                "real_effect_above_synthetic_envelope99": [False, False],
                "outside_possible_mod_harm_share_ci99_low": [0.83, 0.81],
                "outside_possible_mod_harm_dcs_mass_ci99_low": [0.83, 0.82],
                "validate_amplify_outside_possible_mod_harm_share_ci99_low": [
                    0.79,
                    0.78,
                ],
                "published_mod_harm_below_outside_real_ci99": [True, True],
            }
        )
        gates = {
            row["claim_id"]: row
            for row in claim_gates(
                audit,
                endpoint,
                {
                    "primary_model_gap": {
                        "eligible": True,
                        "published_gap_above_real_ci99": True,
                        "synthetic_gap_above_real_envelope99": False,
                        "real_gap_ci99_low": -0.15,
                        "real_gap_ci99_high": 0.14,
                    }
                },
            )
        }
        self.assertFalse(
            gates["lost_confirmed_endpoint_underestimation"]["ready"]
        )
        self.assertTrue(
            gates["lost_confirmed_endpoint_underestimation"][
                "descriptive_ready"
            ]
        )
        self.assertFalse(gates["lost_matched_effect_underestimation"]["ready"])
        gap_gate = gates["lost_primary_model_gap_overestimation"]
        self.assertFalse(gap_gate["ready"])
        self.assertTrue(gap_gate["descriptive_ready"])
        self.assertTrue(
            gates[
                "lost_confirmation_failures_mostly_outside_mod_harm"
            ]["ready"]
        )
        endpoint.loc[
            0,
            "validate_amplify_outside_possible_mod_harm_share_ci99_low",
        ] = 0.74
        strict_failure_gate = {
            row["claim_id"]: row
            for row in claim_gates(audit, endpoint, {})
        }
        self.assertFalse(
            strict_failure_gate[
                "lost_confirmation_failures_mostly_outside_mod_harm"
            ]["ready"]
        )

    def test_descriptive_gap_gate_requires_real_interval_to_include_zero(
        self,
    ) -> None:
        models = [
            "allenai/Olmo-3-7B-Instruct",
            "meta-llama/Llama-3.1-8B-Instruct",
        ]
        audit = pd.DataFrame(
            {
                "model": models,
                "model_fidelity": [
                    "official_bf16_temporally_attested_revision"
                ]
                * 2,
                "complete_and_eligible": [True, True],
            }
        )
        endpoint = pd.DataFrame(
            {
                "model": models,
                "published_synthetic_target_mean_dcs": [1.08, 0.69],
                "published_target_below_real_ci99": [True, True],
                "real_target_above_synthetic_envelope99": [False, False],
                "real_effect_above_synthetic_envelope99": [False, False],
                "outside_possible_mod_harm_share_ci99_low": [0.83, 0.81],
                "outside_possible_mod_harm_dcs_mass_ci99_low": [0.83, 0.82],
                "validate_amplify_outside_possible_mod_harm_share_ci99_low": [
                    0.79,
                    0.78,
                ],
            }
        )
        gates = {
            row["claim_id"]: row
            for row in claim_gates(
                audit,
                endpoint,
                {
                    "primary_model_gap": {
                        "eligible": True,
                        "published_gap_above_real_ci99": True,
                        "synthetic_gap_above_real_envelope99": False,
                        "real_gap_ci99_low": 0.01,
                        "real_gap_ci99_high": 0.14,
                    }
                },
            )
        }
        self.assertFalse(
            gates["lost_primary_model_gap_overestimation"][
                "descriptive_ready"
            ]
        )

    def test_partial_model_is_excluded(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            generations = root / "generations.jsonl"
            judgments = root / "judgments.jsonl"
            write_jsonl(
                generations,
                [
                    {
                        "generation_id": "m:1",
                        "model": "m",
                        "response": "one",
                        "messages": [{"role": "user", "content": "target one"}],
                        "pair_id": "pair-one",
                        "cluster_id": "cluster-one",
                        "condition": "delusion",
                    },
                    {
                        "generation_id": "m:2",
                        "model": "m",
                        "response": "two",
                        "messages": [{"role": "user", "content": "target two"}],
                        "pair_id": "pair-two",
                        "cluster_id": "cluster-two",
                        "condition": "delusion",
                    },
                ],
            )
            write_jsonl(
                judgments,
                [
                    {
                        "generation_id": "m:1",
                        "model": "m",
                        "response": "one",
                        "target_text": "target one",
                        "pair_id": "pair-one",
                        "cluster_id": "cluster-one",
                        "condition": "delusion",
                        "DCS": "2",
                        "judge_protocol": EXPECTED_PROTOCOL,
                        "judge_model_id": EXPECTED_JUDGE_MODEL,
                        "judge_model_revision": EXPECTED_JUDGE_REVISION,
                        "judge_dtype": "bfloat16",
                        "judge_previous_exchanges": 3,
                        "judge_temperature": 0.6,
                        "judge_top_p": 0.95,
                        "judge_top_k": 20,
                        "judge_finish_reason": "stop",
                    }
                ],
            )
            audit, eligible = audit_judgments(
                judgments, [generations]
            )
            self.assertFalse(audit.iloc[0]["complete_and_eligible"])
            self.assertEqual(len(eligible), 0)

    def test_complete_model_is_admitted(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            generations = root / "generations.jsonl"
            judgments = root / "judgments.jsonl"
            write_jsonl(
                generations,
                [
                    {
                        "generation_id": "m:1",
                        "model": "m",
                        "response": "one",
                        "messages": [{"role": "user", "content": "target"}],
                        "pair_id": "pair",
                        "cluster_id": "cluster",
                        "condition": "delusion",
                    }
                ],
            )
            write_jsonl(
                judgments,
                [
                    {
                        "generation_id": "m:1",
                        "model": "m",
                        "response": "one",
                        "target_text": "target",
                        "pair_id": "pair",
                        "cluster_id": "cluster",
                        "condition": "delusion",
                        "DCS": "3",
                        "judge_protocol": EXPECTED_PROTOCOL,
                        "judge_model_id": EXPECTED_JUDGE_MODEL,
                        "judge_model_revision": EXPECTED_JUDGE_REVISION,
                        "judge_dtype": "bfloat16",
                        "judge_previous_exchanges": 3,
                        "judge_temperature": 0.6,
                        "judge_top_p": 0.95,
                        "judge_top_k": 20,
                        "judge_finish_reason": "stop",
                    }
                ],
            )
            audit, eligible = audit_judgments(
                judgments, [generations]
            )
            self.assertTrue(audit.iloc[0]["complete_and_eligible"])
            self.assertEqual(eligible.iloc[0]["dcs"], 2.0)

    def test_cluster_bootstrap_point_is_row_weighted(self) -> None:
        frame = pd.DataFrame(
            {
                "cluster_id": ["a", "a", "b"],
                "numerator": [1.0, 1.0, 0.0],
                "denominator": [1.0, 1.0, 1.0],
            }
        )
        point, low, high = bootstrap_ratio(
            frame, "numerator", "denominator", draws=2_000, seed=7
        )
        self.assertAlmostEqual(point, 2 / 3)
        self.assertLessEqual(low, point)
        self.assertGreaterEqual(high, point)

    def test_endpoint_rows_reports_strict_validate_amplify_share(self) -> None:
        rows = []
        for index, (dcs, possible) in enumerate(
            [(2.0, False), (2.0, True), (1.0, False), (0.0, False)]
        ):
            shared = {
                "model": "test-model",
                "pair_id": f"pair-{index}",
                "cluster_id": f"cluster-{index}",
                "possible_mod_harm": possible,
            }
            rows.extend(
                [
                    {
                        **shared,
                        "condition": "delusion",
                        "dcs": dcs,
                        "dcs_positive": float(dcs > 0),
                    },
                    {
                        **shared,
                        "condition": "grounded_control",
                        "dcs": 0.0,
                        "dcs_positive": 0.0,
                    },
                ]
            )
        result = endpoint_rows(pd.DataFrame(rows), draws=2_000, seed=9)[0]
        self.assertEqual(result["real_target_validate_amplify_n"], 2)
        self.assertEqual(
            result["share_validate_amplify_outside_possible_mod_harm"],
            0.5,
        )

    def test_binary_group_difference_uses_false_minus_true(self) -> None:
        frame = pd.DataFrame(
            {
                "cluster_id": ["a", "a", "b", "b"],
                "value": [2.0, 0.0, 2.0, 0.0],
                "high": [False, True, False, True],
            }
        )
        point, low, high = bootstrap_binary_group_difference(
            frame,
            "value",
            "high",
            draws=2_000,
            seed=8,
        )
        self.assertEqual(point, 2.0)
        self.assertEqual(low, 2.0)
        self.assertEqual(high, 2.0)


if __name__ == "__main__":
    unittest.main()
