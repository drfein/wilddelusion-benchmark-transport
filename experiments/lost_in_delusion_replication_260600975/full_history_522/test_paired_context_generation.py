#!/usr/bin/env python3
"""Integrity tests for prompt-keyed paired context generation."""

from __future__ import annotations

import argparse
import ast
import json
import tempfile
import unittest
from pathlib import Path

import generate_paired_context_vllm as generator
from generate_paired_context_vllm import (
    PROTOCOL,
    completed_arm_artifact,
    load_cache,
    token_ids_sha256,
    validate_inputs,
    write_arm_outputs,
)


class PairedContextGenerationTests(unittest.TestCase):
    def test_every_cache_reload_is_bound_to_run_provenance(self) -> None:
        tree = ast.parse(Path(generator.__file__).read_text(encoding="utf-8"))
        calls = [
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "load_cache"
        ]
        self.assertGreaterEqual(len(calls), 2)
        required = {"model", "checkpoint_source", "revision", "seed"}
        for call in calls:
            self.assertTrue(required.issubset({item.arg for item in call.keywords}))

    @staticmethod
    def input_rows(last_only: bool) -> list[dict[str, object]]:
        rows = []
        for index in range(623):
            target = {"role": "user", "content": f"target {index}"}
            messages = [target]
            if not last_only:
                messages = [
                    {"role": "user", "content": f"prior {index}"},
                    {"role": "assistant", "content": "prior reply"},
                    target,
                ]
            rows.append(
                {
                    "input_id": f"{'last' if last_only else 'history'}-{index}",
                    "pair_id": f"pair-{index}",
                    "condition": "delusion",
                    "messages": messages,
                }
            )
        return rows

    def test_input_identity_passes_and_target_drift_fails(self) -> None:
        history = self.input_rows(last_only=False)
        last_only = self.input_rows(last_only=True)
        validate_inputs(history, last_only)
        last_only[10]["messages"][-1]["content"] = "changed"
        with self.assertRaises(ValueError):
            validate_inputs(history, last_only)

    def test_token_hash_binds_full_token_sequence(self) -> None:
        self.assertEqual(token_ids_sha256([1, 2, 3]), token_ids_sha256([1, 2, 3]))
        self.assertNotEqual(
            token_ids_sha256([1, 2, 3]), token_ids_sha256([1, 3, 2])
        )

    def test_cache_reuse_is_checkpoint_specific(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "cache.jsonl"
            path.write_text(
                json.dumps(
                    {
                        "prompt_token_ids_sha256": "prompt",
                        "response": "cached response",
                        "model": "model",
                        "model_checkpoint_source": "source",
                        "model_revision": "old-revision",
                        "seed": 1,
                        "decoding": "greedy",
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "provenance mismatch"):
                load_cache(
                    path,
                    model="model",
                    checkpoint_source="source",
                    revision="new-revision",
                    seed=1,
                )

    def test_identical_prompt_arms_reuse_exact_response(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            history_input = root / "history.jsonl"
            last_input = root / "last.jsonl"
            history_input.write_text("{}\n", encoding="utf-8")
            last_input.write_text("{}\n", encoding="utf-8")
            cache_path = root / "cache.jsonl"
            cache_path.write_text(
                json.dumps(
                    {
                        "prompt_token_ids_sha256": "same",
                        "response": "one deterministic response",
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            base = {
                "input_id": "input",
                "pair_id": "pair",
                "condition": "delusion",
                "messages": [{"role": "user", "content": "target"}],
            }
            prepared = [
                {
                    "scope": scope,
                    "row": {**base, "input_id": scope},
                    "messages_used": base["messages"],
                    "token_ids": [1, 2, 3],
                    "dropped_message_count": 0,
                    "prompt_token_ids_sha256": "same",
                }
                for scope in ("bounded_history", "last_user_only")
            ]
            args = argparse.Namespace(
                reported_model="test/model",
                model=root / "model",
                checkpoint_source="test/model",
                revision="revision",
                dtype="bfloat16",
                max_new_tokens=8,
                seed=1,
                max_input_tokens=32,
                chunk_size=2,
                tensor_parallel_size=1,
                history_input=history_input,
                last_only_input=last_input,
            )
            outputs = {
                "bounded_history": root / "history-output.jsonl",
                "last_user_only": root / "last-output.jsonl",
            }
            write_arm_outputs(
                prepared,
                {"same": {"response": "one deterministic response"}},
                outputs,
                args,
                {"eligible": True},
                cache_path,
            )
            rows = [
                json.loads(path.read_text(encoding="utf-8"))
                for path in outputs.values()
            ]
            self.assertEqual(rows[0]["response"], rows[1]["response"])
            self.assertTrue(rows[0]["cross_arm_prompt_identical"])
            self.assertTrue(rows[1]["cross_arm_prompt_identical"])
            self.assertEqual(rows[0]["paired_generation_protocol"], PROTOCOL)

            history_manifest_path = outputs["bounded_history"].with_suffix(
                outputs["bounded_history"].suffix + ".manifest.json"
            )
            history_manifest = json.loads(history_manifest_path.read_text())
            history_manifest["successful_rows"] = 1
            history_manifest["expected_rows"] = 1
            history_manifest_path.write_text(json.dumps(history_manifest))
            self.assertTrue(
                completed_arm_artifact(
                    outputs["bounded_history"],
                    history_input,
                    model="test/model",
                    checkpoint_source="test/model",
                    revision="revision",
                    expected_rows=1,
                )
            )


if __name__ == "__main__":
    unittest.main()
