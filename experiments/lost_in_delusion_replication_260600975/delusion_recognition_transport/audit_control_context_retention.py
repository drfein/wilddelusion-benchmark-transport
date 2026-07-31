#!/usr/bin/env python3
"""Audit what natural-control context survives every classifier tokenizer."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from transformers import AutoTokenizer

from classifier_prompt import prepare_prompt
from prepare_inputs import read_jsonl, sha256_file, write_jsonl


HERE = Path(__file__).resolve().parent
MODEL_SPECS = {
    "olmo3_7b": {
        "model": "allenai/Olmo-3-7B-Instruct",
        "revision": "6e5971d9eba42665f5bd5a0fcf047f299ce1dccc",
        "disable_thinking": False,
    },
    "llama31_8b": {
        "model": "meta-llama/Llama-3.1-8B-Instruct",
        "revision": "0e9e39f249a16976918f6564b8830bc894c89659",
        "disable_thinking": False,
    },
    "qwen3_30b": {
        "model": "Qwen/Qwen3-30B-A3B",
        "revision": "ad44e777bcd18fa416d9da3bd8f70d33ebb85d39",
        "disable_thinking": True,
    },
}


def normalize_quote(value: Any) -> str:
    text = str(value or "").strip()
    if len(text) >= 2 and text[0] == text[-1] and text[0] in "\"'":
        text = text[1:-1]
    return " ".join(text.split()).casefold()


def quote_is_retained(quote: Any, retained_text: str) -> bool:
    normalized = normalize_quote(quote)
    return bool(normalized) and normalized in normalize_quote(retained_text)


def content_hash(messages: list[dict[str, Any]]) -> str:
    payload = json.dumps(
        messages, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--inputs",
        type=Path,
        default=HERE / "artifacts" / "classifier_inputs.jsonl",
    )
    parser.add_argument(
        "--source-controls",
        type=Path,
        default=HERE.parents[2]
        / "data"
        / "controls"
        / "natural_near_miss_controls_20260730"
        / "strict_nonsincere_controls.jsonl",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=HERE
        / "artifacts"
        / "natural_control_context_retention.jsonl",
    )
    parser.add_argument("--max-input-tokens", type=int, default=12_288)
    args = parser.parse_args()

    inputs = [
        row
        for row in read_jsonl(args.inputs)
        if row["evaluation_cohort"] == "natural_near_miss_negative"
    ]
    source_by_id = {
        row["control_id"]: row for row in read_jsonl(args.source_controls)
    }
    if len(inputs) != 309 or len(source_by_id) != 309:
        raise ValueError("Expected 309 frozen natural controls")

    tokenizers = {
        name: AutoTokenizer.from_pretrained(
            spec["model"], revision=spec["revision"]
        )
        for name, spec in MODEL_SPECS.items()
    }

    output = []
    for row in inputs:
        per_model = {}
        for name, spec in MODEL_SPECS.items():
            prepared = prepare_prompt(
                tokenizers[name],
                row,
                args.max_input_tokens,
                spec["disable_thinking"],
            )
            per_model[name] = {
                "model": spec["model"],
                "revision": spec["revision"],
                "input_token_count": len(prepared["token_ids"]),
                "dropped_previous_messages": prepared[
                    "dropped_previous_messages"
                ],
                "retained_previous_messages": prepared[
                    "retained_previous_messages"
                ],
            }

        common_start = max(
            details["dropped_previous_messages"]
            for details in per_model.values()
        )
        retained = row["previous_messages"][common_start:] + [
            {"role": "user", "content": row["target_text"]}
        ]
        retained_text = "\n".join(
            str(message.get("content", "")) for message in retained
        )
        source = source_by_id[row["control_id"]]
        quotes = source.get("judge_supporting_quotes") or []
        quote_retention = [
            {
                "quote": quote,
                "retained": quote_is_retained(quote, retained_text),
            }
            for quote in quotes
        ]
        output.append(
            {
                "input_id": row["input_id"],
                "control_id": row["control_id"],
                "cluster_id": row["cluster_id"],
                "negative_exclusion": row["negative_exclusion"],
                "is_primary_natural_negative": bool(
                    row.get("is_primary_natural_negative")
                ),
                "is_human_confirmed_negative": bool(
                    row.get("is_human_confirmed_negative")
                ),
                "source_message_count_through_target": row[
                    "source_message_count_through_target"
                ],
                "target_message_index": row["target_message_index"],
                "common_dropped_previous_messages": common_start,
                "common_retained_previous_messages": (
                    len(row["previous_messages"]) - common_start
                ),
                "common_retained_messages": retained,
                "common_retained_context_sha256": content_hash(retained),
                "original_supporting_quote_count": len(quote_retention),
                "original_supporting_quotes_retained": sum(
                    item["retained"] for item in quote_retention
                ),
                "all_original_supporting_quotes_retained": bool(quote_retention)
                and all(item["retained"] for item in quote_retention),
                "supporting_quote_retention": quote_retention,
                "per_model": per_model,
            }
        )

    write_jsonl(args.output, output)
    primary = [row for row in output if row["is_primary_natural_negative"]]
    human = [row for row in output if row["is_human_confirmed_negative"]]
    manifest = {
        "protocol": (
            "Exact complete-message suffix retained by all official-BF16 "
            "recognition tokenizers"
        ),
        "inputs": str(args.inputs),
        "inputs_sha256": sha256_file(args.inputs),
        "source_controls": str(args.source_controls),
        "source_controls_sha256": sha256_file(args.source_controls),
        "max_input_tokens": args.max_input_tokens,
        "models": MODEL_SPECS,
        "rows": len(output),
        "primary_rows": len(primary),
        "human_confirmed_rows": len(human),
        "rows_dropping_prior_messages": sum(
            row["common_dropped_previous_messages"] > 0 for row in output
        ),
        "primary_rows_dropping_prior_messages": sum(
            row["common_dropped_previous_messages"] > 0 for row in primary
        ),
        "human_rows_dropping_prior_messages": sum(
            row["common_dropped_previous_messages"] > 0 for row in human
        ),
        "primary_rows_with_all_original_quotes_retained": sum(
            row["all_original_supporting_quotes_retained"] for row in primary
        ),
        "output": str(args.output),
        "output_sha256": sha256_file(args.output),
    }
    args.output.with_suffix(
        args.output.suffix + ".manifest.json"
    ).write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
