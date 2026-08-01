from __future__ import annotations

import importlib.util
from pathlib import Path

import pandas as pd


HERE = Path(__file__).parent


def load(name: str):
    spec = importlib.util.spec_from_file_location(name, HERE / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


prepare = load("prepare_prefix_chunks")
analyze = load("analyze")
complete = load("build_complete_history_cohort")
rehydrate = load("rehydrate_hf_release")


def test_chunk_text_is_bounded_and_lossless() -> None:
    text = "  alpha\n beta   gamma delta epsilon "
    chunks = prepare.chunk_text(text, max_words=2)
    assert chunks == ["alpha beta", "gamma delta", "epsilon"]
    assert " ".join(chunks) == "alpha beta gamma delta epsilon"


def test_clustered_effect_uses_paired_difference() -> None:
    frame = pd.DataFrame(
        {
            "conversation_hash": ["a", "a", "b"],
            "difference": [1, 0, -1],
            "target_only_endorse": [0, 1, 1],
            "full_context_endorse": [1, 1, 0],
            "original_row_idx": [0, 1, 2],
        }
    )
    result = analyze.clustered_effect(frame, draws=200, seed=7)
    assert result["difference"] == 0
    assert result["pairs"] == 3
    assert result["conversations"] == 2


def test_canonical_conversations_rejects_conflicts() -> None:
    frame = pd.DataFrame(
        [
            {"conversation_id": "x", "messages": [{"role": "user", "content": "a"}]},
            {"conversation_id": "x", "messages": [{"role": "user", "content": "b"}]},
        ]
    )
    try:
        prepare.canonical_conversations(frame)
    except ValueError as error:
        assert "Conflicting" in str(error)
    else:
        raise AssertionError("Conflicting reconstructions must fail closed.")


def test_complete_history_normalization_preserves_roles() -> None:
    messages = [
        {"role": "human", "content": " first "},
        {"role": "llm", "content": "second"},
        {"role": "user", "content": "third"},
        {"role": "assistant", "content": "not included"},
    ]
    assert complete.normalize_messages(messages, 2) == [
        {"role": "user", "content": "first"},
        {"role": "assistant", "content": "second"},
        {"role": "user", "content": "third"},
    ]


def test_sharechat_normalization_preserves_empty_source_nodes() -> None:
    messages = [
        {"role": "user", "plain_text": "hello"},
        {"role": "llm", "plain_text": None},
    ]
    assert rehydrate.normalize_sharechat_messages(messages) == [
        {"role": "user", "content": "hello"},
        {"role": "assistant", "content": rehydrate.UNAVAILABLE_SOURCE_MESSAGE},
    ]


def test_find_target_requires_unique_user_match() -> None:
    messages = [
        {"role": "user", "content": "target"},
        {"role": "assistant", "content": "target"},
    ]
    assert rehydrate.find_target(messages, "target", None) == 0
