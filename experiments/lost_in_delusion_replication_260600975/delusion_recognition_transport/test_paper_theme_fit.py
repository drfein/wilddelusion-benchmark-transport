#!/usr/bin/env python3
"""Tests for narrow paper-theme adjudication inputs."""

from __future__ import annotations

import unittest

from adjudicate_paper_theme_fit import render_context


class PaperThemeFitTests(unittest.TestCase):
    def test_context_keeps_target_and_only_six_preceding_messages(self) -> None:
        row = {
            "previous_messages": [
                {"role": "user", "content": f"message-{index}"}
                for index in range(10)
            ],
            "target_text": "marked-target",
        }
        rendered = render_context(row, max_characters=40_000)
        self.assertNotIn("message-3", rendered)
        self.assertIn("message-4", rendered)
        self.assertIn("message-9", rendered)
        self.assertIn("USER [TARGET]:\nmarked-target", rendered)

    def test_character_cap_preserves_both_ends(self) -> None:
        row = {
            "previous_messages": [
                {"role": "assistant", "content": "prefix" * 10_000}
            ],
            "target_text": "target-start " + ("x" * 50_000) + " target-end",
        }
        rendered = render_context(row, max_characters=20_000)
        self.assertLessEqual(len(rendered), 20_100)
        self.assertIn("target-end", rendered)


if __name__ == "__main__":
    unittest.main()
