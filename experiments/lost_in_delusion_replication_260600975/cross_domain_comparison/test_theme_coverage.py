#!/usr/bin/env python3
"""Tests for the synthetic-theme coverage audit."""

from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from analyze_theme_coverage import cluster_bootstrap_share


class ThemeCoverageTests(unittest.TestCase):
    def test_cluster_bootstrap_uses_row_weighted_share(self) -> None:
        frame = pd.DataFrame(
            {
                "cluster_id": ["a", "a", "a", "b"],
                "outside": [1, 1, 1, 0],
            }
        )
        result = cluster_bootstrap_share(frame, "outside", 5_000, 1)
        self.assertEqual(result["point"], 0.75)
        self.assertEqual(result["clusters"], 2)
        self.assertLess(result["ci99_low"], result["point"])
        self.assertGreater(result["ci99_high"], result["point"])

    def test_cluster_bootstrap_is_reproducible(self) -> None:
        frame = pd.DataFrame(
            {
                "cluster_id": np.repeat(np.arange(20), 2),
                "outside": np.tile([0, 1], 20),
            }
        )
        first = cluster_bootstrap_share(frame, "outside", 1_000, 7)
        second = cluster_bootstrap_share(frame, "outside", 1_000, 7)
        self.assertEqual(first, second)


if __name__ == "__main__":
    unittest.main()
