#!/usr/bin/env python3
"""Tests for context held fixed across paired-response judgments."""

from __future__ import annotations

import unittest

from io_utils import canonical_json_sha256, messages_for_judging
from prepare_paired_context_judge_inputs import attach_shared_judge_context


class SharedJudgeContextTests(unittest.TestCase):
    def test_both_arms_receive_the_bounded_history(self) -> None:
        history = {
            "generation_id": "history",
            "messages_used": [
                {"role": "user", "content": "Earlier context."},
                {"role": "assistant", "content": "Earlier reply."},
                {"role": "user", "content": "Same final message."},
            ],
        }
        last_only = {
            "generation_id": "last-only",
            "messages_used": [
                {"role": "user", "content": "Same final message."}
            ],
        }

        rows = attach_shared_judge_context(history, last_only)

        self.assertEqual(rows[0]["judge_messages"], history["messages_used"])
        self.assertEqual(rows[1]["judge_messages"], history["messages_used"])
        self.assertEqual(
            rows[0]["judge_context_sha256"],
            canonical_json_sha256(history["messages_used"]),
        )
        self.assertEqual(
            rows[0]["judge_context_sha256"],
            rows[1]["judge_context_sha256"],
        )
        self.assertEqual(
            rows[1]["judge_context_source_generation_id"], "history"
        )

    def test_explicit_judge_messages_override_generation_context(self) -> None:
        row = {
            "judge_messages": [{"role": "user", "content": "Judge."}],
            "messages_used": [
                {"role": "user", "content": "Generation."}
            ],
        }
        self.assertEqual(
            messages_for_judging(row)[-1]["content"], "Judge."
        )


if __name__ == "__main__":
    unittest.main()
