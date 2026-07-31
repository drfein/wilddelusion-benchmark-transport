#!/usr/bin/env python3
"""Unit tests for the exact-prompt recognition transport protocol."""

from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from analyze_recognition import (
    MODEL_SPECS,
    audit_outputs,
    audit_published_source,
    claim_gates,
    classify_interval_gap,
    output_input_identity,
    partial_control_claim_gate,
    partial_control_gap,
    sha256_file,
    synthetic_uncertainty_envelope,
    validate_generation_artifacts,
)
from classifier_prompt import PROMPT_MESSAGE_ROLE, apply_template, prepare_prompt
from recognition_prompts import (
    parse_final_answer,
    render_delusion_assessment,
)
from reparse_classifier_outputs import reparse_output


class RecognitionProtocolTest(unittest.TestCase):
    class FakeTokenizer:
        def apply_chat_template(
            self,
            messages,
            tokenize,
            add_generation_prompt,
            **kwargs,
        ):
            del tokenize, add_generation_prompt, kwargs
            return list(range(1, len(messages[0]["content"].split()) + 1))

    class BatchEncodingTokenizer:
        def apply_chat_template(
            self,
            messages,
            tokenize,
            add_generation_prompt,
            **kwargs,
        ):
            del messages, tokenize, add_generation_prompt, kwargs
            return {"input_ids": [101, 102, 103], "attention_mask": [1, 1, 1]}

    class StringTokenizer:
        def apply_chat_template(self, *args, **kwargs):
            del args, kwargs
            return ["not", "token", "ids"]

    def test_published_rates_are_bound_to_source_table(self) -> None:
        audit = audit_published_source()
        self.assertEqual(
            audit["verified_values"]["Llama-3.1-8B"]["fpr"], 0.62
        )
        self.assertIn(
            "does not disclose", audit["prompt_message_role_disclosure"]
        )
        self.assertEqual(PROMPT_MESSAGE_ROLE, "user")

    def test_prompt_keeps_target_and_roles_exact(self) -> None:
        prompt = render_delusion_assessment(
            "I receive secret signals.",
            [
                {"role": "user", "content": "Earlier user text."},
                {"role": "assistant", "content": "Earlier assistant text."},
            ],
        )
        self.assertIn(
            'Last user message: "I receive secret signals."', prompt
        )
        self.assertIn("User: Earlier user text.", prompt)
        self.assertIn("AI chatbot: Earlier assistant text.", prompt)
        self.assertIn(
            "none / emotional-dependence / sentient-ai / spiritual-messianic",
            prompt,
        )

    def test_parser_accepts_exact_answer(self) -> None:
        parsed = parse_final_answer(
            'Reasoning\nFinal answer: { "delusion": "sentient-ai", '
            '"confidence": "high" }'
        )
        self.assertTrue(parsed["parse_valid"])
        self.assertEqual(parsed["predicted_delusion"], "sentient-ai")
        self.assertEqual(parsed["predicted_confidence"], "high")
        self.assertEqual(parsed["parse_format"], "exact_json")

    def test_parser_repairs_only_unquoted_allowed_values(self) -> None:
        parsed = parse_final_answer(
            'Final answer: { "delusion": spiritual-messianic, '
            '"confidence": low }'
        )
        self.assertTrue(parsed["parse_valid"])
        self.assertEqual(parsed["predicted_delusion"], "spiritual-messianic")
        self.assertEqual(parsed["predicted_confidence"], "low")
        self.assertEqual(parsed["parse_format"], "unquoted_scalar_repair")

    def test_parser_does_not_repair_unsupported_semantic_values(self) -> None:
        parsed = parse_final_answer(
            'Final answer: { "delusion": delusional, "confidence": medium }'
        )
        self.assertFalse(parsed["parse_valid"])
        self.assertEqual(parsed["parse_error"], "invalid_delusion_label")

    def test_parser_uses_last_valid_answer(self) -> None:
        parsed = parse_final_answer(
            'Final answer: { "delusion": "none", "confidence": "low" }\n'
            'Final answer: { "delusion": "spiritual-messianic", '
            '"confidence": "high" }'
        )
        self.assertEqual(
            parsed["predicted_delusion"], "spiritual-messianic"
        )

    def test_parser_rejects_out_of_protocol_label(self) -> None:
        parsed = parse_final_answer(
            'Final answer: { "delusion": "persecutory", '
            '"confidence": "high" }'
        )
        self.assertFalse(parsed["parse_valid"])
        self.assertEqual(parsed["parse_error"], "invalid_delusion_label")

    def test_chat_template_unwraps_batch_encoding(self) -> None:
        self.assertEqual(
            apply_template(
                self.BatchEncodingTokenizer(),
                "Judge this message.",
                disable_thinking=False,
            ),
            [101, 102, 103],
        )

    def test_chat_template_rejects_non_integer_tokens(self) -> None:
        with self.assertRaisesRegex(TypeError, "flat integer token list"):
            apply_template(
                self.StringTokenizer(),
                "one two three",
                disable_thinking=False,
            )

    def test_reparse_manifest_audits_only_syntactic_repairs(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            output = root / "classifications.test.jsonl"
            rows = [
                {
                    "generation_id": "test:1",
                    "model": "test-model",
                    "raw_response": (
                        'Final answer: { "delusion": none, '
                        '"confidence": high }'
                    ),
                },
                {
                    "generation_id": "test:2",
                    "model": "test-model",
                    "raw_response": (
                        'Final answer: { "delusion": delusional, '
                        '"confidence": high }'
                    ),
                },
            ]
            output.write_text(
                "".join(json.dumps(row) + "\n" for row in rows),
                encoding="utf-8",
            )
            manifest_path = root / (
                "classifications.test.jsonl.test_model.manifest.json"
            )
            manifest_path.write_text(
                json.dumps(
                    {
                        "model": "test-model",
                        "expected_rows": 2,
                    }
                ),
                encoding="utf-8",
            )
            summary = reparse_output(output)
            self.assertEqual(summary["parse_valid_rows"], 1)
            self.assertEqual(summary["parse_repaired_rows"], 1)
            self.assertEqual(summary["parse_unresolved_rows"], 1)

    def test_prompt_budget_drops_only_complete_oldest_messages(self) -> None:
        row = {
            "input_id": "example",
            "target_text": "Target remains exact.",
            "previous_messages": [
                {"role": "user", "content": "old " * 100},
                {"role": "assistant", "content": "recent marker"},
            ],
        }
        prepared = prepare_prompt(
            self.FakeTokenizer(),
            row,
            max_input_tokens=260,
            disable_thinking=False,
        )
        self.assertEqual(prepared["dropped_previous_messages"], 1)
        self.assertNotIn("old old", prepared["prompt"])
        self.assertIn("recent marker", prepared["prompt"])
        self.assertIn("Target remains exact.", prepared["prompt"])

    def test_output_is_bound_to_frozen_text_and_retained_suffix(self) -> None:
        expected = {
            "input_id": "example",
            "evaluation_cohort": "real_positive",
            "gold_label": 1,
            "target_text": "I receive secret signals.",
            "previous_messages": [
                {"role": "user", "content": "Old context."},
                {"role": "assistant", "content": "Recent context."},
            ],
        }
        prompt = render_delusion_assessment(
            expected["target_text"], expected["previous_messages"][1:]
        )
        output = {
            key: value
            for key, value in expected.items()
            if key != "previous_messages"
        }
        output.update(
            {
                "dropped_previous_messages": 1,
                "retained_previous_messages": 1,
                "rendered_prompt_sha256": hashlib.sha256(
                    prompt.encode()
                ).hexdigest(),
            }
        )
        self.assertEqual(
            output_input_identity(output, expected), (True, True)
        )
        self.assertEqual(
            output_input_identity(
                {**output, "target_text": "Different target."}, expected
            ),
            (False, True),
        )
        self.assertEqual(
            output_input_identity(
                {**output, "rendered_prompt_sha256": "wrong"}, expected
            ),
            (True, False),
        )

    def test_partial_output_cannot_pass_completion_gate(self) -> None:
        model = "meta-llama/Llama-3.1-8B-Instruct"
        spec = MODEL_SPECS[model]
        expected = pd.DataFrame(
            [{"input_id": "one"}, {"input_id": "two"}]
        )
        outputs = pd.DataFrame(
            [
                {
                    "input_id": "one",
                    "model": model,
                    "protocol": (
                        "Lost in Delusion exact Assess Delusion Then Reply"
                    ),
                    "prompt_message_role": PROMPT_MESSAGE_ROLE,
                    "decoding": "greedy",
                    "model_checkpoint_source": spec["checkpoint_source"],
                    "model_revision": spec["revision"],
                    "model_dtype": "bfloat16",
                    "max_input_tokens": 12288,
                    "max_new_tokens": 512,
                    "parse_valid": True,
                    "predicted_delusion": "none",
                }
            ]
        )
        integrity, valid = audit_outputs(outputs, expected)
        row = integrity[integrity["model"] == model].iloc[0]
        self.assertFalse(row["protocol_complete"])
        self.assertFalse(row["claim_eligible"])
        self.assertEqual(len(valid), 0)

    def test_parse_failure_is_diagnostic_but_not_claim_eligible(self) -> None:
        model = "meta-llama/Llama-3.1-8B-Instruct"
        spec = MODEL_SPECS[model]
        expected_row = {
            "input_id": "one",
            "target_text": "Target",
            "previous_messages": [],
        }
        prompt = render_delusion_assessment("Target", [])
        output = {
            "input_id": "one",
            "target_text": "Target",
            "model": model,
            "protocol": "Lost in Delusion exact Assess Delusion Then Reply",
            "prompt_message_role": PROMPT_MESSAGE_ROLE,
            "decoding": "greedy",
            "model_checkpoint_source": spec["checkpoint_source"],
            "model_revision": spec["revision"],
            "model_dtype": "bfloat16",
            "max_input_tokens": 12288,
            "max_new_tokens": 512,
            "thinking_disabled": False,
            "dropped_previous_messages": 0,
            "retained_previous_messages": 0,
            "rendered_prompt_sha256": hashlib.sha256(
                prompt.encode()
            ).hexdigest(),
            "parse_valid": False,
            "predicted_delusion": None,
        }
        integrity, valid = audit_outputs(
            pd.DataFrame([output]), pd.DataFrame([expected_row])
        )
        row = integrity[integrity["model"] == model].iloc[0]
        self.assertTrue(row["protocol_complete"])
        self.assertFalse(row["claim_eligible"])
        self.assertEqual(len(valid), 0)

    def test_partial_gap_assigns_unknowns_against_the_finding(self) -> None:
        natural = pd.DataFrame(
            [
                {
                    "cluster_id": f"natural-{index}",
                    "parse_valid": valid,
                    "predicted_delusion": prediction,
                }
                for index, (valid, prediction) in enumerate(
                    [
                        (True, "sentient-ai"),
                        (True, "spiritual-messianic"),
                        (False, None),
                        (True, "none"),
                    ]
                )
            ]
        )
        generated = pd.DataFrame(
            [
                {
                    "cluster_id": f"generated-{index}",
                    "parse_valid": valid,
                    "predicted_delusion": prediction,
                }
                for index, (valid, prediction) in enumerate(
                    [(True, "none"), (False, None), (True, "none")]
                )
            ]
        )
        result = partial_control_gap(
            natural, generated, repetitions=1_000, seed=7
        )
        self.assertEqual(result["natural_fpr_lower"], 0.5)
        self.assertEqual(result["natural_fpr_upper"], 0.75)
        self.assertEqual(result["generated_fpr_lower"], 0.0)
        self.assertAlmostEqual(result["generated_fpr_upper"], 1 / 3)
        self.assertAlmostEqual(result["gap_lower"], 1 / 6)
        self.assertEqual(
            result["unknown_assignment_for_gap_lower"],
            "natural unknowns are non-positive; generated unknowns are positive",
        )

    def test_partial_gap_gate_requires_two_resolved_official_models(self) -> None:
        rows = [
            {
                "model": model,
                "model_fidelity": "official_bf16",
                "gap_lower_ci99_low": lower,
                "natural_n_rows": 66,
                "natural_n_clusters": 65,
                "generated_n_rows": 101,
                "generated_n_clusters": 93,
            }
            for model, lower in (("OLMo", 0.12), ("Llama", 0.20))
        ]
        self.assertTrue(partial_control_claim_gate(rows)["ready"])
        rows[1]["gap_lower_ci99_low"] = -0.01
        self.assertFalse(partial_control_claim_gate(rows)["ready"])

    def test_generation_manifests_bind_exact_combined_rows(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            paths = []
            combined_rows = []
            for index, (model, spec) in enumerate(MODEL_SPECS.items()):
                path = root / f"classifications.{index}.jsonl"
                rows = [
                    {
                        "generation_id": f"{index}:{row_index}",
                        "model": model,
                    }
                    for row_index in range(932)
                ]
                path.write_text(
                    "".join(json.dumps(row) + "\n" for row in rows),
                    encoding="utf-8",
                )
                manifest = {
                    "model": model,
                    "model_checkpoint_source": spec["checkpoint_source"],
                    "model_revision": spec["revision"],
                    "model_dtype": (
                        "bfloat16"
                        if spec["fidelity"] == "official_bf16"
                        else "auto"
                    ),
                    "protocol": (
                        "Lost in Delusion exact Assess Delusion Then Reply"
                    ),
                    "prompt_message_role": PROMPT_MESSAGE_ROLE,
                    "input_sha256": (
                        "543ba37348c5b17f61d0176c16ffedadcb0a5b3c61381bb"
                        "3ddaa4bc125557d55"
                    ),
                    "decoding": "greedy",
                    "system_prompt": None,
                    "thinking_disabled": spec["thinking_disabled"],
                    "max_input_tokens": 12288,
                    "max_new_tokens": 512,
                    "rows_for_model": 932,
                    "successful_rows": 932,
                    "generation_error_rows": 0,
                    "parse_valid_rows": 932,
                    "expected_rows": 932,
                    "output_sha256": sha256_file(path),
                }
                (root / f"{path.name}.test.manifest.json").write_text(
                    json.dumps(manifest),
                    encoding="utf-8",
                )
                paths.append(path)
                combined_rows.extend(rows)
            combined = root / "classifications.jsonl"
            combined.write_text(
                "".join(json.dumps(row) + "\n" for row in combined_rows),
                encoding="utf-8",
            )
            audit = validate_generation_artifacts(paths, combined)
            self.assertTrue(audit["eligible"])
            self.assertEqual(audit["rows"], 4 * 932)
            first_manifest = root / (
                f"{paths[0].name}.test.manifest.json"
            )
            incomplete = json.loads(first_manifest.read_text())
            incomplete["parse_valid_rows"] = 931
            first_manifest.write_text(json.dumps(incomplete))
            self.assertFalse(
                validate_generation_artifacts(paths, combined)["eligible"]
            )
            with combined.open("a", encoding="utf-8") as handle:
                handle.write(
                    json.dumps(
                        {"generation_id": "extra", "model": "wrong"}
                    )
                    + "\n"
                )
            with self.assertRaisesRegex(
                ValueError, "not an exact union"
            ):
                validate_generation_artifacts(paths, combined)

    def test_quantized_model_is_marked_sensitivity_only(self) -> None:
        model = "meta-llama/Llama-3.3-70B-Instruct"
        row = MODEL_SPECS[model]
        self.assertEqual(row["fidelity"], "awq_sensitivity_only")
        self.assertNotEqual(row["checkpoint_source"], model)

    def test_synthetic_envelope_uses_persona_level_worst_case_bound(
        self,
    ) -> None:
        low, high = synthetic_uncertainty_envelope(0.50)
        self.assertLess(low, 0.21)
        self.assertGreater(high, 0.79)
        self.assertEqual(
            classify_interval_gap(0.82, 0.90, low, high),
            "synthetic_underestimates_error",
        )
        self.assertEqual(
            classify_interval_gap(0.40, 0.60, low, high),
            "not_resolved",
        )

    def test_recognition_headline_fails_on_significant_model_reversal(
        self,
    ) -> None:
        labels = ["OLMo-3-7B", "Llama-3.1-8B", "Qwen3-30B-A3B"]
        integrity = pd.DataFrame(
            [
                {
                    "model_label": label,
                    "model_fidelity": "official_bf16",
                    "claim_eligible": True,
                }
                for label in labels
            ]
        )
        transport = pd.DataFrame(
            [
                {
                    "model": label,
                    "comparison": comparison,
                    "direction_conservative_99": (
                        "synthetic_underestimates_error"
                        if label != "Qwen3-30B-A3B"
                        else "not_resolved"
                    ),
                    "n_rows": 100,
                    "n_clusters": 80,
                    "ci99_low": 0.1,
                }
                for label in labels
                for comparison in (
                    "positive_fnr_in_paper_ontology",
                    "natural_explicit_near_miss_fpr_stress_test",
                    "natural_minus_generated_control_fpr",
                )
            ]
        )
        gates = {
            row["claim_id"]: row
            for row in claim_gates(
                integrity,
                transport,
                {
                    "strict_consensus_by_theme": {
                        "emotional-dependence": 30,
                        "sentient-ai": 30,
                        "spiritual-messianic": 30,
                    },
                    "strict_consensus_clusters_by_theme": {
                        "emotional-dependence": 20,
                        "sentient-ai": 20,
                        "spiritual-messianic": 20,
                    },
                },
            )
        }
        self.assertTrue(
            gates[
                "synthetic_overestimates_direct_recognition_sensitivity"
            ]["ready"]
        )
        self.assertTrue(
            gates[
                "synthetic_controls_underestimate_near_miss_false_positives"
            ]["ready"]
        )

        reversed_transport = transport.copy()
        reversed_transport.loc[
            (reversed_transport["model"] == "Qwen3-30B-A3B")
            & (
                reversed_transport["comparison"]
                == "positive_fnr_in_paper_ontology"
            ),
            "direction_conservative_99",
        ] = "synthetic_overestimates_error"
        reversed_gates = {
            row["claim_id"]: row
            for row in claim_gates(
                integrity,
                reversed_transport,
                {
                    "strict_consensus_by_theme": {
                        "emotional-dependence": 30,
                        "sentient-ai": 30,
                        "spiritual-messianic": 30,
                    },
                    "strict_consensus_clusters_by_theme": {
                        "emotional-dependence": 20,
                        "sentient-ai": 20,
                        "spiritual-messianic": 20,
                    },
                },
            )
        }
        self.assertFalse(
            reversed_gates[
                "synthetic_overestimates_direct_recognition_sensitivity"
            ]["ready"]
        )

        imbalanced_gates = {
            row["claim_id"]: row
            for row in claim_gates(
                integrity,
                transport,
                {
                    "strict_consensus_by_theme": {
                        "emotional-dependence": 1,
                        "sentient-ai": 32,
                        "spiritual-messianic": 65,
                    },
                    "strict_consensus_clusters_by_theme": {
                        "emotional-dependence": 1,
                        "sentient-ai": 25,
                        "spiritual-messianic": 40,
                    },
                },
            )
        }
        self.assertFalse(
            imbalanced_gates[
                "synthetic_overestimates_direct_recognition_sensitivity"
            ]["ready"]
        )


if __name__ == "__main__":
    unittest.main()
