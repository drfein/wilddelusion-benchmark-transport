#!/usr/bin/env python3
"""Regression tests for the published safety-judge protocol."""

from __future__ import annotations

import unittest

from judge_responses_vllm import JUDGE_PROTOCOL, judge_protocol, parse_scores, row_seed
from wilddelusion_prompts import safety_judge_prompt


class JudgeProtocolTest(unittest.TestCase):
    def setUp(self) -> None:
        self.messages = [
            {"role": "user", "content": f"user-{index}"}
            if index % 2 == 0
            else {"role": "assistant", "content": f"assistant-{index}"}
            for index in range(9)
        ]

    def test_prompt_uses_exact_published_headings(self) -> None:
        prompt = safety_judge_prompt(self.messages, "candidate")
        self.assertIn("\n# Instructions\n", prompt)
        self.assertIn("\n# Prior conversation (context only)\n", prompt)
        self.assertIn("\n# Exchange to judge\n", prompt)
        self.assertNotIn("\n## Instructions\n", prompt)

    def test_prompt_keeps_three_prior_exchanges(self) -> None:
        prompt = safety_judge_prompt(self.messages, "candidate")
        self.assertNotIn("user-0", prompt)
        self.assertNotIn("assistant-1", prompt)
        for index in range(2, 8):
            self.assertIn(
                ("user" if index % 2 == 0 else "assistant") + f"-{index}",
                prompt,
            )
        self.assertIn('User: "user-8"', prompt)
        self.assertIn('AI: "candidate"', prompt)

    def test_zero_previous_exchanges_keeps_no_context(self) -> None:
        prompt = safety_judge_prompt(
            self.messages, "candidate", previous_exchanges=0
        )
        self.assertIn("(none)", prompt)
        self.assertNotIn("assistant-7", prompt)

    def test_parser_and_seed_are_stable(self) -> None:
        text = (
            "Reasoning.\n"
            'Final answer: { "DCS": "3", "HES": "N/A", "SIS": "1" }'
        )
        self.assertEqual(parse_scores(text), ("3", "N/A", 1))
        self.assertEqual(row_seed("generation", 7, 1), row_seed("generation", 7, 1))
        self.assertNotEqual(
            row_seed("generation", 7, 1), row_seed("generation", 7, 2)
        )
        self.assertEqual(judge_protocol(3), JUDGE_PROTOCOL)
        self.assertEqual(
            judge_protocol(0), f"{JUDGE_PROTOCOL}; previous_exchanges=0"
        )


if __name__ == "__main__":
    unittest.main()
