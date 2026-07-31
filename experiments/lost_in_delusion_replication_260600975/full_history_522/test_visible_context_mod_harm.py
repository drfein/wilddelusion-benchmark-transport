#!/usr/bin/env python3
"""Tests for the model-visible Mod+Harm sensitivity screen."""

from __future__ import annotations

from pathlib import Path

import pytest

from label_mod_harm_visible_context import (
    RUBRIC_A,
    load_targets,
    messages_sha256,
    validate_label,
)


HERE = Path(__file__).resolve().parent


def test_frozen_inputs_are_exact_model_visible_histories() -> None:
    rows = load_targets(HERE / "artifacts" / "model_inputs.jsonl")
    assert len(rows) == 522
    assert len({row["pair_id"] for row in rows}) == 522
    assert all(row["messages"][-1]["role"] == "user" for row in rows)
    assert all(
        sum(message["role"] == "user" for message in row["messages"]) <= 4
        for row in rows
    )


def test_positive_evidence_must_be_user_authored_and_verbatim() -> None:
    messages = [
        {"role": "user", "content": "ordinary request"},
        {"role": "assistant", "content": "harmful phrase"},
        {"role": "user", "content": "I might hurt myself tonight"},
    ]
    valid = {
        "label": "yes",
        "evidence_message_index": 2,
        "evidence_quote": "hurt myself tonight",
    }
    validate_label(valid, messages)
    invalid = {**valid, "evidence_message_index": 1, "evidence_quote": "harmful phrase"}
    with pytest.raises(ValueError, match="not user-authored"):
        validate_label(invalid, messages)


def test_nonpositive_label_cannot_carry_evidence() -> None:
    messages = [{"role": "user", "content": "ordinary request"}]
    validate_label(
        {
            "label": "no",
            "evidence_message_index": -1,
            "evidence_quote": "",
        },
        messages,
    )
    with pytest.raises(ValueError, match="blank quote"):
        validate_label(
            {
                "label": "unclear",
                "evidence_message_index": 0,
                "evidence_quote": "ordinary request",
            },
            messages,
        )


def test_hash_and_rubric_are_stable() -> None:
    messages = [{"role": "user", "content": "same"}]
    assert messages_sha256(messages) == messages_sha256([dict(messages[0])])
    assert "Only how directly" not in RUBRIC_A
    assert "every USER message" in RUBRIC_A
