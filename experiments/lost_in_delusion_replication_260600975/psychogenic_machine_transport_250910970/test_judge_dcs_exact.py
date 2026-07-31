#!/usr/bin/env python3
"""Protocol tests for the exact Psychosis-Bench DCS judge."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
PAPER_REPO = ROOT / "paper" / "psychosis-bench"
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(PAPER_REPO))

from judge_dcs_exact import (  # noqa: E402
    DEFAULT_JUDGE_MODEL,
    PAPER_JUDGE_ALIAS,
    filter_and_validate,
    parse_dcs,
    render_conversation,
    render_prompt,
)
from psy_bench.core.models import Turn  # noqa: E402
from psy_bench.core.scoring import Scorer  # noqa: E402


class _Response:
    content = "0"


class _CaptureClient:
    def __init__(self) -> None:
        self.prompt: str | None = None

    def chat(self, model: str, messages: list[object]) -> _Response:
        del model
        self.prompt = messages[0].content
        return _Response()


def generation(
    identifier: str,
    model: str = "model-a",
    turn_number: int = 4,
) -> dict[str, object]:
    return {
        "generation_id": identifier,
        "model": model,
        "turn_number": turn_number,
        "response": "world",
        "messages_used": [{"role": "user", "content": "hello"}],
    }


class ExactDCSJudgeTests(unittest.TestCase):
    def test_judge_matches_paper_disclosure(self) -> None:
        self.assertEqual(PAPER_JUDGE_ALIAS, "openai/gpt-4o-mini")
        self.assertEqual(
            DEFAULT_JUDGE_MODEL,
            "gpt-4o-mini-2024-07-18",
        )

    def test_prompt_is_byte_identical_to_paper_repository(self) -> None:
        client = _CaptureClient()
        Scorer(client).score_dcs(
            [Turn(number=1, user_message="hello", bot_response="world")]
        )
        self.assertEqual(
            render_prompt(generation("a")),
            client.prompt,
        )

    def test_conversation_uses_paper_labels(self) -> None:
        row = generation("a")
        row["messages_used"] = [
            {"role": "user", "content": "one"},
            {"role": "assistant", "content": "two"},
            {"role": "user", "content": "three"},
        ]
        row["response"] = "four"
        self.assertEqual(
            render_conversation(row),
            "Human: one\n\nAssistant: two\n\nHuman: three\n\nAssistant: four",
        )

    def test_immediate_stop_is_retained_as_an_empty_assistant_turn(self) -> None:
        row = generation("a")
        row["response"] = ""
        self.assertEqual(
            render_conversation(row),
            "Human: hello\n\nAssistant: ",
        )
        self.assertEqual(
            filter_and_validate(
                [row], {"model-a"}, min_turn=4, expected_per_model=1
            ),
            [row],
        )

    def test_candidate_gate_is_fail_closed(self) -> None:
        rows = [
            generation("a", "model-a"),
            generation("b", "model-b"),
        ]
        with self.assertRaisesRegex(ValueError, "candidate-count mismatch"):
            filter_and_validate(
                rows,
                {"model-a", "model-b"},
                min_turn=4,
                expected_per_model=2,
            )

    def test_dcs_parser_rejects_out_of_range_score(self) -> None:
        self.assertEqual(parse_dcs("SCORE: 2"), 2)
        with self.assertRaisesRegex(ValueError, "outside"):
            parse_dcs("3")


if __name__ == "__main__":
    unittest.main()
