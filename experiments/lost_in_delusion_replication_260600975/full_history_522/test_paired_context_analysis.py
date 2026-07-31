#!/usr/bin/env python3
"""Unit tests for the repaired context-ablation estimands."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from analyze_paired_context_ablation import (
    calculate_model_metrics,
    json_native,
    manifest_input_matches,
    ordinal_score,
    sha256_file,
)


class PairedContextAnalysisTests(unittest.TestCase):
    def test_numpy_scalars_are_json_serializable(self) -> None:
        payload = json_native(
            {"flag": np.bool_(True), "score": np.float64(0.5)}
        )
        self.assertEqual(json.loads(json.dumps(payload)), payload)

    def test_manifest_input_identity_survives_directory_relocation(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "inputs.jsonl"
            path.write_text("{}\n", encoding="utf-8")
            manifest = {
                "input_files": [
                    {
                        "path": "/different/machine/inputs.jsonl",
                        "sha256": sha256_file(path),
                        "rows": 1,
                    }
                ]
            }
            self.assertTrue(manifest_input_matches(manifest, path, 1))
            manifest["input_files"][0]["sha256"] = "forged"
            self.assertFalse(manifest_input_matches(manifest, path, 1))

    def test_ordinal_mapping_matches_paper(self) -> None:
        self.assertEqual(ordinal_score("N/A"), 0)
        self.assertEqual(ordinal_score("1"), 0)
        self.assertEqual(ordinal_score("2"), 1)
        self.assertEqual(ordinal_score("3"), 2)
        with self.assertRaises(ValueError):
            ordinal_score("0")

    def test_cancelling_case_flips_are_not_aggregate_change(self) -> None:
        generation_rows = []
        exact_rows = []
        secondary_rows = []
        for index in range(449):
            if index < 50:
                history, last_only = 1, 0
            elif index < 100:
                history, last_only = 0, 1
            else:
                history = last_only = 0
            for scope, value in (
                ("bounded_history", history),
                ("last_user_only", last_only),
            ):
                generation_id = f"generation-{index}-{scope}"
                generation_rows.append(
                    {
                        "generation_id": generation_id,
                        "model": "test/model",
                        "pair_id": f"pair-{index}",
                        "cluster_id": f"cluster-{index % 260}",
                        "condition": "delusion",
                        "paired_ablation_scope": scope,
                        "cross_arm_prompt_identical": False,
                    }
                )
                score = "2" if value else "N/A"
                exact_rows.append({"generation_id": generation_id, "DCS": score})
                secondary_rows.append(
                    {"generation_id": generation_id, "DCS": score}
                )
        metrics, pairs = calculate_model_metrics(
            pd.DataFrame(generation_rows),
            pd.DataFrame(exact_rows),
            pd.DataFrame(secondary_rows),
            draws=2_000,
        )
        row = metrics.iloc[0]
        self.assertEqual(len(pairs), 449)
        self.assertAlmostEqual(row["exact_difference_estimate"], 0.0)
        self.assertAlmostEqual(row["secondary_difference_estimate"], 0.0)
        self.assertAlmostEqual(row["exact_flip_estimate"], 100 / 449)
        self.assertAlmostEqual(row["secondary_flip_estimate"], 100 / 449)
        self.assertAlmostEqual(
            row["exact_ordinal_difference_estimate"], 0.0
        )
        self.assertAlmostEqual(
            row["secondary_ordinal_difference_estimate"], 0.0
        )
        self.assertAlmostEqual(row["same_direction_flip_estimate"], 100 / 449)
        self.assertAlmostEqual(
            row["mean_same_response_disagreement_estimate"], 0.0
        )


if __name__ == "__main__":
    unittest.main()
