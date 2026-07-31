#!/usr/bin/env python3
"""Tests for the fail-closed context-stability claim."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from analyze_context_ablation import (
    API_RESPONSE_ONLY_PROTOCOL,
    audit_generation_revision_attestation,
    audit_cross_arm_decoder_drift,
    audit_exact_judgment_manifest,
    audit_judgment_source_identity,
    audit_exact_judgments,
    audit_secondary_judgments,
    binary_stability,
    EXPECTED_RUNTIME_LOG_HASHES,
    secondary_retest_stability,
    sha256_file,
    validate_generation_artifacts,
    validate_runtime_logs,
)
from judge_responses_vllm import (
    DEFAULT_MODEL_ID,
    DEFAULT_REVISION,
    judge_protocol,
)


def judgment(generation_id: str) -> dict[str, object]:
    return {
        "generation_id": generation_id,
        "judge_protocol": judge_protocol(0),
        "judge_model_id": DEFAULT_MODEL_ID,
        "judge_model_revision": DEFAULT_REVISION,
        "judge_dtype": "bfloat16",
        "judge_previous_exchanges": 0,
        "judge_temperature": 0.6,
        "judge_top_p": 0.95,
        "judge_top_k": 20,
        "judge_finish_reason": "stop",
        "DCS": "2",
        "HES": "N/A",
        "SIS": 0,
    }


def secondary_judgment(
    generation_id: str,
    dcs: str,
    model: str,
) -> dict[str, object]:
    return {
        "generation_id": generation_id,
        "judge_protocol": API_RESPONSE_ONLY_PROTOCOL,
        "judge_model": model,
        "judge_reasoning_effort": "none",
        "judge_previous_exchanges": 0,
        "DCS": dcs,
    }


class ContextAblationClaimTests(unittest.TestCase):
    def test_identical_input_decoder_drift_fails_closed(self) -> None:
        base = {
            "model": "model",
            "pair_id": "pair",
            "condition": "delusion",
            "messages_used": [{"role": "user", "content": "Same."}],
            "system_prompt": "System.",
        }
        frame = pd.DataFrame(
            [
                {
                    **base,
                    "scope": "bounded_history",
                    "response": "First response.",
                },
                {
                    **base,
                    "scope": "last_user_only",
                    "response": "Second response.",
                },
            ]
        )
        audit = audit_cross_arm_decoder_drift(frame)
        self.assertFalse(audit["eligible"])
        self.assertEqual(audit["identical_model_input_units"], 1)
        self.assertEqual(
            audit["identical_input_divergent_response_units"], 1
        )

    def test_frozen_generations_match_frozen_inputs(self) -> None:
        here = Path(__file__).resolve().parent
        history, history_audit = validate_generation_artifacts(
            "bounded_history",
            [
                here / "results" / "generations_llama31_8b.jsonl",
                here / "results" / "generations_olmo.jsonl",
            ],
            here / "artifacts" / "model_inputs.jsonl",
        )
        last, last_audit = validate_generation_artifacts(
            "last_user_only",
            [
                here
                / "results"
                / "generations_llama31_8b_last_user_only.jsonl",
                here
                / "results"
                / "generations_olmo_last_user_only.jsonl",
            ],
            here / "artifacts" / "last_user_only_inputs.jsonl",
        )
        log_root = here / "results" / "provenance_logs"
        if not log_root.exists():
            log_root = here / "results"
        runtime = validate_runtime_logs(
            [
                log_root / name
                for name in sorted(EXPECTED_RUNTIME_LOG_HASHES)
            ]
        )
        self.assertEqual(len(history), 1246)
        self.assertEqual(len(last), 1246)
        self.assertTrue(history_audit["eligible"])
        self.assertTrue(last_audit["eligible"])
        self.assertTrue(runtime["eligible"])

    def test_temporal_attestation_binds_both_generation_arms(self) -> None:
        here = Path(__file__).resolve().parent
        paths = [
            here / "results" / "generations_llama31_8b.jsonl",
            here / "results" / "generations_llama31_8b_last_user_only.jsonl",
            here / "results" / "generations_olmo.jsonl",
            here / "results" / "generations_olmo_last_user_only.jsonl",
        ]
        attestation_path = (
            here.parent
            / "cross_domain_comparison"
            / "generation_revision_attestation.json"
        )
        audit = audit_generation_revision_attestation(
            attestation_path, paths
        )
        self.assertTrue(audit["eligible"])

        with tempfile.TemporaryDirectory() as directory:
            tampered = json.loads(attestation_path.read_text())
            first_model = next(iter(tampered["models"].values()))
            first_model["generation_artifacts"][0][
                "generation_sha256"
            ] = "0" * 64
            tampered_path = Path(directory) / "attestation.json"
            tampered_path.write_text(json.dumps(tampered))
            with self.assertRaisesRegex(ValueError, "attestation failed"):
                audit_generation_revision_attestation(
                    tampered_path, paths
                )

    def test_partial_judgments_fail_closed(self) -> None:
        frame = pd.DataFrame([judgment("a")])
        with self.assertRaisesRegex(ValueError, "integrity gate failed"):
            audit_exact_judgments(frame, {"a", "b"})

    def test_exact_judgments_pass(self) -> None:
        frame = pd.DataFrame([judgment("a"), judgment("b")])
        audit = audit_exact_judgments(frame, {"a", "b"})
        self.assertTrue(audit["eligible"])

    def test_exact_manifest_binds_inputs_and_output(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            inputs = [root / "first.jsonl", root / "second.jsonl"]
            for index, path in enumerate(inputs):
                path.write_text(
                    json.dumps({"generation_id": f"g-{index}"}) + "\n"
                )
            output = root / "judgments.jsonl"
            output.write_text("{}\n{}\n")
            manifest = {
                "model_id": DEFAULT_MODEL_ID,
                "model_revision": DEFAULT_REVISION,
                "dtype": "bfloat16",
                "judge_protocol": judge_protocol(0),
                "previous_exchanges": 0,
                "input_roles": ["user"],
                "extra_system_prompt": None,
                "temperature": 0.6,
                "top_p": 0.95,
                "top_k": 20,
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
            manifest_path = output.with_suffix(
                output.suffix + ".manifest.json"
            )
            manifest_path.write_text(json.dumps(manifest))
            audit = audit_exact_judgment_manifest(output, inputs)
            self.assertTrue(audit["eligible"])
            output.write_text("{}\n")
            with self.assertRaisesRegex(ValueError, "manifest integrity"):
                audit_exact_judgment_manifest(output, inputs)

    def test_binary_stability_counts_cancelling_flips(self) -> None:
        rows = []
        model = "meta-llama/Llama-3.1-8B-Instruct"
        values = [(1, 0), (0, 1), (1, 1), (0, 0)]
        for index, (history, last) in enumerate(values):
            for scope, value in (
                ("bounded_history", history),
                ("last_user_only", last),
            ):
                rows.append(
                    {
                        "model": model,
                        "pair_id": f"pair-{index}",
                        "cluster_id": f"cluster-{index}",
                        "condition": "delusion",
                        "scope": scope,
                        "DCS > 0": value,
                    }
                )
        result = binary_stability(pd.DataFrame(rows), draws=1_000).iloc[0]
        self.assertEqual(result["marginal_difference"], 0)
        self.assertEqual(result["binary_flip_rate"], 0.5)
        self.assertEqual(result["history_only_flips"], 1)
        self.assertEqual(result["last_user_only_flips"], 1)

    def test_secondary_retest_separates_context_and_judge_flips(
        self,
    ) -> None:
        model = "meta-llama/Llama-3.1-8B-Instruct"
        generations = []
        first = []
        second = []
        values = [
            ("2", "N/A", "2", "N/A"),
            ("2", "N/A", "2", "2"),
            ("N/A", "N/A", "N/A", "N/A"),
            ("2", "2", "2", "2"),
        ]
        for index, (first_history, first_last, second_history, second_last) in enumerate(
            values
        ):
            for scope, first_dcs, second_dcs in (
                ("bounded_history", first_history, second_history),
                ("last_user_only", first_last, second_last),
            ):
                generation_id = f"generation-{index}-{scope}"
                generations.append(
                    {
                        "generation_id": generation_id,
                        "model": model,
                        "pair_id": f"pair-{index}",
                        "cluster_id": f"cluster-{index}",
                        "condition": "delusion",
                        "scope": scope,
                    }
                )
                first.append(
                    secondary_judgment(
                        generation_id, first_dcs, "gpt-5.4-mini"
                    )
                )
                second.append(
                    secondary_judgment(
                        generation_id,
                        second_dcs,
                        "gpt-5.4-mini-2026-03-17",
                    )
                )
        result = secondary_retest_stability(
            pd.DataFrame(generations),
            pd.DataFrame(first),
            pd.DataFrame(second),
            draws=1_000,
        ).iloc[0]
        self.assertEqual(result["first_flip_rate"], 0.5)
        self.assertEqual(result["second_flip_rate"], 0.25)
        self.assertEqual(
            result["mean_same_response_disagreement_rate"], 0.125
        )
        self.assertEqual(result["first_excess_flip_rate"], 0.375)
        self.assertEqual(result["second_excess_flip_rate"], 0.125)

    def test_secondary_audit_fails_on_wrong_model(self) -> None:
        frame = pd.DataFrame(
            [secondary_judgment("a", "2", "unexpected-model")]
        )
        with self.assertRaisesRegex(
            ValueError, "Secondary judgment integrity gate failed"
        ):
            audit_secondary_judgments(frame, {"a"}, "gpt-5.4-mini")

    def test_judgment_must_match_exact_generation_text(self) -> None:
        generation = {
            "generation_id": "a",
            "pair_id": "pair",
            "cluster_id": "cluster",
            "condition": "delusion",
            "model": "model",
            "response": "Candidate response.",
            "context_scope": "final_4_user_turns",
            "counterfactual_scope": "all messages",
            "messages": [{"role": "user", "content": "Target text."}],
            "messages_used": [
                {"role": "user", "content": "Target text."}
            ],
        }
        judged = {
            key: generation[key]
            for key in (
                "generation_id",
                "pair_id",
                "cluster_id",
                "condition",
                "model",
                "response",
                "context_scope",
                "counterfactual_scope",
            )
        }
        judged["target_text"] = "Target text."
        audit = audit_judgment_source_identity(
            pd.DataFrame([judged]),
            pd.DataFrame([generation]),
        )
        self.assertTrue(audit["eligible"])
        with self.assertRaisesRegex(ValueError, "identity gate failed"):
            audit_judgment_source_identity(
                pd.DataFrame(
                    [{**judged, "response": "Different response."}]
                ),
                pd.DataFrame([generation]),
            )


if __name__ == "__main__":
    unittest.main()
