#!/usr/bin/env python3
"""Token-budgeted prompt construction shared by generation and audits."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from recognition_prompts import render_delusion_assessment

PROMPT_MESSAGE_ROLE = "user"


def apply_template(
    tokenizer: Any,
    prompt: str,
    disable_thinking: bool,
) -> list[int]:
    kwargs = {"enable_thinking": False} if disable_thinking else {}
    encoded = tokenizer.apply_chat_template(
        [{"role": PROMPT_MESSAGE_ROLE, "content": prompt}],
        tokenize=True,
        add_generation_prompt=True,
        **kwargs,
    )
    # Some Transformers tokenizers return a BatchEncoding even for one chat.
    if isinstance(encoded, Mapping):
        if "input_ids" not in encoded:
            raise TypeError("chat template encoding has no input_ids")
        encoded = encoded["input_ids"]
    if hasattr(encoded, "tolist"):
        encoded = encoded.tolist()
    if isinstance(encoded, tuple):
        encoded = list(encoded)
    if (
        isinstance(encoded, list)
        and len(encoded) == 1
        and isinstance(encoded[0], (list, tuple))
    ):
        encoded = list(encoded[0])
    if not isinstance(encoded, list) or not encoded or not all(
        isinstance(token_id, int) and not isinstance(token_id, bool)
        for token_id in encoded
    ):
        raise TypeError(
            "chat template must produce a non-empty flat integer token list"
        )
    return encoded


def prepare_prompt(
    tokenizer: Any,
    row: dict[str, Any],
    max_input_tokens: int,
    disable_thinking: bool,
) -> dict[str, Any]:
    """Retain the earliest complete suffix of prior messages that fits."""
    previous = row["previous_messages"]

    def encode(start: int) -> tuple[str, list[int]]:
        prompt = render_delusion_assessment(
            row["target_text"], previous[start:]
        )
        return prompt, apply_template(tokenizer, prompt, disable_thinking)

    # Avoid repeatedly tokenizing multi-megabyte histories. Begin near a
    # conservative character budget, then find the earliest suffix that fits.
    character_budget = max_input_tokens * 8
    suffix_characters = 0
    heuristic_start = len(previous)
    for index in range(len(previous) - 1, -1, -1):
        suffix_characters += len(str(previous[index].get("content", "")))
        if suffix_characters > character_budget:
            break
        heuristic_start = index

    prompt, token_ids = encode(heuristic_start)
    low = 0
    high = heuristic_start
    if len(token_ids) > max_input_tokens:
        low = heuristic_start + 1
        high = len(previous)
    while low < high:
        middle = (low + high) // 2
        candidate_prompt, candidate_ids = encode(middle)
        if len(candidate_ids) <= max_input_tokens:
            high = middle
            prompt, token_ids = candidate_prompt, candidate_ids
        else:
            low = middle + 1
    start = low
    prompt, token_ids = encode(start)
    if len(token_ids) > max_input_tokens:
        raise ValueError(
            f"{row['input_id']}: target and protocol exceed token budget "
            f"({len(token_ids)} > {max_input_tokens})"
        )
    return {
        "row": row,
        "prompt": prompt,
        "token_ids": token_ids,
        "dropped_previous_messages": start,
        "retained_previous_messages": len(previous) - start,
    }
