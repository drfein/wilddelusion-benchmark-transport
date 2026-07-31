#!/usr/bin/env python3
"""Tests for retained-context control adjudication invariants."""

from __future__ import annotations

import unittest
from pathlib import Path

import pandas as pd

from adjudicate_retained_control_context import (
    indexed_messages,
    retained_primary_eligible,
    validate_evidence,
)
from analyze_recognition import natural_control_consensus
from prepare_inputs import read_jsonl


HERE = Path(__file__).resolve().parent


class RetainedControlContextTest(unittest.TestCase):
    def setUp(self) -> None:
        self.row = {
            "common_dropped_previous_messages": 3,
            "target_message_index": 4,
            "common_retained_messages": [
                {"role": "assistant", "content": "Continue the fictional scene."},
                {"role": "user", "content": "I can control the moon."},
            ],
        }

    def test_original_indices_are_preserved(self) -> None:
        messages = indexed_messages(self.row)
        self.assertEqual(set(messages), {3, 4})
        self.assertEqual(messages[4]["content"], "I can control the moon.")

    def test_verbatim_evidence_is_required(self) -> None:
        valid = validate_evidence(
            self.row,
            {
                "label": "nonsincere",
                "evidence_message_index": 3,
                "evidence_quote": "Continue the fictional scene.",
            },
        )
        invalid = validate_evidence(
            self.row,
            {
                "label": "nonsincere",
                "evidence_message_index": 3,
                "evidence_quote": "This is fictional.",
            },
        )
        self.assertTrue(valid["evidence_quote_exact"])
        self.assertFalse(invalid["evidence_quote_exact"])

    def test_primary_gate_fails_weak_or_unverified_evidence(self) -> None:
        base = {
            "label": "nonsincere",
            "confidence": "high",
            "evidence_strength": "explicit",
            "evidence_message_index_valid": True,
            "evidence_quote_exact": True,
        }
        self.assertTrue(retained_primary_eligible(base))
        self.assertFalse(
            retained_primary_eligible({**base, "evidence_strength": "weak"})
        )
        self.assertFalse(
            retained_primary_eligible({**base, "evidence_quote_exact": False})
        )

    def test_frozen_artifacts_yield_strict_100_row_consensus(self) -> None:
        expected = pd.DataFrame(
            read_jsonl(HERE / "artifacts" / "classifier_inputs.jsonl")
        )
        strict, explicit, full, audit, _ = natural_control_consensus(
            expected,
            HERE
            / "artifacts"
            / "natural_controls_independent_adjudication_mini.jsonl",
            HERE / "artifacts" / "natural_control_context_retention.jsonl",
            HERE
            / "artifacts"
            / "natural_controls_retained_context_adjudication.jsonl",
        )
        self.assertEqual(len(strict), 100)
        self.assertEqual(len(explicit), 66)
        self.assertEqual(len(full), 183)
        self.assertEqual(audit["strict_intersection_clusters"], 97)
        self.assertEqual(audit["explicit_intersection_clusters"], 65)


if __name__ == "__main__":
    unittest.main()
