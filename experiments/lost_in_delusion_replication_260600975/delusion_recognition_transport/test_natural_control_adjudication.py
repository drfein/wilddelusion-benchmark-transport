#!/usr/bin/env python3
"""Tests for context-preserving natural-control adjudication."""

from __future__ import annotations

import unittest

from adjudicate_natural_controls import render_transcript


class TranscriptRenderingTests(unittest.TestCase):
    def test_short_transcript_is_unchanged(self) -> None:
        messages = [
            {"role": "user", "content": "setup"},
            {"role": "assistant", "content": "response"},
            {"role": "user", "content": "target"},
        ]
        transcript, metadata = render_transcript(messages, 2)
        self.assertFalse(metadata["transcript_truncated"])
        self.assertIn("0002 USER [TARGET TO CLASSIFY]", transcript)
        self.assertIn("setup", transcript)

    def test_long_transcript_retains_target_and_surrounding_context(self) -> None:
        messages = [
            {"role": "user", "content": "opening"},
            {"role": "assistant", "content": "x" * 200_000},
            {"role": "user", "content": "critical target"},
            {"role": "assistant", "content": "clarifies roleplay"},
        ]
        transcript, metadata = render_transcript(
            messages, 2, max_characters=2_000
        )
        self.assertTrue(metadata["transcript_truncated"])
        self.assertIn("critical target", transcript)
        self.assertIn("clarifies roleplay", transcript)
        self.assertIn("opening", transcript)
        self.assertIn("[TARGET TO CLASSIFY]", transcript)


if __name__ == "__main__":
    unittest.main()
