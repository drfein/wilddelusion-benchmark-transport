from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))


def load(name: str, filename: str):
    spec = importlib.util.spec_from_file_location(name, HERE / filename)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


PREPARE = load("context_prepare", "prepare_inputs.py")
GENERATE = load("context_generate", "generate_open_model.py")
FIXED = load("context_fixed", "attribute_fixed_response.py")
TOKEN_ANALYSIS = load("context_token_analysis", "analyze_token_attributions.py")
FLASHTRACE_RUNNER = load("context_flashtrace_runner", "run_flashtrace.py")


def test_fold_assignment_is_stable_and_bounded() -> None:
    value = "01234567" + "a" * 56
    assert PREPARE.fold_for_hash(value) == int("01234567", 16) % 5
    assert 0 <= PREPARE.fold_for_hash(value) < 5


def test_generated_length_stops_at_eos_before_padding() -> None:
    assert GENERATE.generated_length([4, 5, 2, 0, 0], {2}, 0) == 3
    assert GENERATE.generated_length([4, 5, 0, 0], {2}, 0) == 2


def test_fixed_response_selection_keeps_first_of_each_class() -> None:
    conversation_hash = "a" * 64
    rows = [
        {
            "conversation_hash": conversation_hash,
            "repetition": 2,
            "annotation_score": 9,
            "generated_token_ids": [3],
        },
        {
            "conversation_hash": conversation_hash,
            "repetition": 0,
            "annotation_score": 1,
            "generated_token_ids": [1],
        },
        {
            "conversation_hash": conversation_hash,
            "repetition": 1,
            "annotation_score": 8,
            "generated_token_ids": [2],
        },
    ]
    selected = FIXED.select_responses(rows, {conversation_hash})[conversation_hash]
    assert [row["repetition"] for row in selected] == [1, 0]


def test_token_role_masks_separate_target_and_prior_context() -> None:
    import numpy as np

    messages = [
        {"role": "system", "content": "s"},
        {"role": "user", "content": "u"},
        {"role": "assistant", "content": "a"},
        {"role": "user", "content": "target"},
    ]
    masks = TOKEN_ANALYSIS.role_masks(np.asarray([-1, 0, 1, 2, 3]), messages)
    assert masks["template"].tolist() == [True, False, False, False, False]
    assert masks["system"].tolist() == [False, True, False, False, False]
    assert masks["prior_user"].tolist() == [False, False, True, False, False]
    assert masks["prior_assistant"].tolist() == [False, False, False, True, False]
    assert masks["target_user"].tolist() == [False, False, False, False, True]


def test_flashtrace_target_round_trip_strips_only_trailing_eos() -> None:
    class FakeTokenizer:
        eos_token_id = 99

        @staticmethod
        def decode(ids, **_kwargs):
            return ",".join(str(value) for value in ids)

        @staticmethod
        def __call__(text, **_kwargs):
            return {"input_ids": [int(value) for value in text.split(",")]}

    text, ids, ended = FLASHTRACE_RUNNER.exact_target_text(FakeTokenizer(), [4, 5, 99])
    assert text == "4,5"
    assert ids == [4, 5]
    assert ended is True
