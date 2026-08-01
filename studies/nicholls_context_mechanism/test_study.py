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
analyze_transitions = load("analyze_transition_contexts")
compare = load("compare_context_models")
complete = load("build_complete_history_cohort")
generation = load("generate_openai_responses")
label = load("label_prefix_chunks")
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


def test_model_comparison_preserves_pairing() -> None:
    common = {
        "original_row_idx": [0, 1],
        "conversation_hash": ["a", "b"],
        "message_hash": ["x", "y"],
        "target_only_score": [0, 0],
        "full_context_score": [0, 8],
        "target_only_endorse": [0, 0],
        "full_context_endorse": [0, 1],
        "difference": [0, 1],
    }
    reference = pd.DataFrame({**common, "model": ["reference", "reference"]})
    comparison = pd.DataFrame(
        {
            **common,
            "model": ["comparison", "comparison"],
            "full_context_score": [8, 8],
            "full_context_endorse": [1, 1],
            "difference": [1, 1],
        }
    )
    summary, rows = compare.compare(reference, comparison, draws=200, seed=7)
    assert summary["reference"]["paired_effect"] == 0.5
    assert summary["comparison"]["paired_effect"] == 1
    assert summary["comparison_minus_reference_paired_effect"] == 0.5
    assert len(rows) == 2


def test_generation_validates_requested_model() -> None:
    row = {"response": "answer", "model": "gpt-4.1-mini", "error": None}
    assert generation.valid(row, "gpt-4.1-mini")
    assert not generation.valid(row, "gpt-5.4-mini")


def test_transition_labels_distinguish_threshold_directions() -> None:
    frame = pd.DataFrame(
        {
            "target_only_score": [0, 8, 0, 8],
            "full_context_score": [8, 0, 0, 8],
            "target_only_endorse": [0, 1, 0, 1],
            "full_context_endorse": [1, 0, 0, 1],
        }
    )
    result = analyze_transitions.add_transition_labels(frame)
    assert result["transition"].tolist() == [
        "increase_0_to_1",
        "decrease_1_to_0",
        "stable_0_to_0",
        "stable_1_to_1",
    ]


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


def test_canonical_conversations_namespaces_source_ids() -> None:
    frame = pd.DataFrame(
        [
            {
                "source": "a",
                "conversation_id": "x",
                "messages": [{"role": "user", "content": "one"}],
            },
            {
                "source": "b",
                "conversation_id": "x",
                "messages": [{"role": "user", "content": "two"}],
            },
        ]
    )
    assert len(prepare.canonical_conversations(frame)) == 2


def test_label_schema_uses_batch_local_indices() -> None:
    schema = label.response_schema(3)
    item = schema["properties"]["labels"]["items"]
    assert item["properties"]["item_index"]["enum"] == [0, 1, 2]
    assert "item_id" not in item["properties"]


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
