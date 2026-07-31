#!/usr/bin/env python3
"""Tests for the final abstract-claim completion audit."""

from __future__ import annotations

import copy
import json
import unittest

from verify_abstract_claims import REGISTRY, verify_rows


class AbstractClaimCompletionTests(unittest.TestCase):
    def setUp(self) -> None:
        self.rows = json.loads(REGISTRY.read_text(encoding="utf-8"))

    def test_current_registry_passes_completion_audit(self) -> None:
        result = verify_rows(self.rows)
        self.assertTrue(result["complete"])
        self.assertEqual(result["selected_claims"], 5)
        self.assertEqual(result["distinct_evidence_families"], 5)

    def test_four_selected_claims_fail_closed(self) -> None:
        rows = copy.deepcopy(self.rows)
        selected = next(row for row in rows if row["selected_for_abstract"])
        selected["selected_for_abstract"] = False
        with self.assertRaisesRegex(ValueError, "Exactly five"):
            verify_rows(rows)

    def test_stale_embedded_gate_fails_closed(self) -> None:
        rows = copy.deepcopy(self.rows)
        selected = next(row for row in rows if row["selected_for_abstract"])
        selected["gate"]["ready"] = False
        with self.assertRaisesRegex(ValueError, "stale"):
            verify_rows(rows)


if __name__ == "__main__":
    unittest.main()
