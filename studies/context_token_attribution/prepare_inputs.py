from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from config import (
    MAX_INPUT_TOKENS,
    MODEL_ID,
    MODEL_REVISION,
    SYSTEM_PROMPT,
    stable_hash,
)
from io_utils import read_jsonl, sha256_file, write_jsonl
from transformers import AutoTokenizer


def chat_messages(row: dict[str, Any]) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        *[
            {"role": str(message["role"]), "content": str(message["content"])}
            for message in row["messages"]
        ],
    ]


def tokenize_chat(tokenizer: Any, messages: list[dict[str, str]]) -> list[int]:
    kwargs = {
        "tokenize": True,
        "add_generation_prompt": True,
        "enable_thinking": False,
    }
    return list(tokenizer.apply_chat_template(messages, **kwargs))


def fold_for_hash(conversation_hash: str) -> int:
    return int(conversation_hash[:8], 16) % 5


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--private-output", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--model", default=MODEL_ID)
    parser.add_argument("--revision", default=MODEL_REVISION)
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(
        args.model, revision=args.revision, trust_remote_code=False
    )
    source_rows = read_jsonl(args.input)
    eligible: list[dict[str, Any]] = []
    exclusions: list[dict[str, Any]] = []
    for row in source_rows:
        messages = chat_messages(row)
        if not messages or messages[-1]["role"] != "user":
            raise ValueError(f"Row {row.get('original_row_idx')} does not end in user")
        token_ids = tokenize_chat(tokenizer, messages)
        record = {
            "original_row_idx": int(row["original_row_idx"]),
            "source": row["source"],
            "conversation_hash": row["conversation_hash"],
            "message_hash": row["message_hash"],
            "target_text": row["target_text"],
            "messages": messages,
            "input_tokens": len(token_ids),
            "fold": fold_for_hash(row["conversation_hash"]),
            "prompt_content_sha256": stable_hash(
                {
                    "model": args.model,
                    "revision": args.revision,
                    "messages": messages,
                    "thinking": False,
                }
            ),
        }
        if len(token_ids) <= MAX_INPUT_TOKENS:
            eligible.append(record)
        else:
            exclusions.append(
                {
                    "original_row_idx": record["original_row_idx"],
                    "source": record["source"],
                    "conversation_hash": record["conversation_hash"],
                    "input_tokens": record["input_tokens"],
                    "reason": "exceeds_exact_token_limit",
                }
            )

    write_jsonl(args.private_output, eligible)
    manifest = {
        "source_path": str(args.input),
        "source_sha256": sha256_file(args.input),
        "model": args.model,
        "model_revision": args.revision,
        "max_input_tokens": MAX_INPUT_TOKENS,
        "truncation_policy": "none",
        "source_rows": len(source_rows),
        "eligible_rows": len(eligible),
        "excluded_rows": len(exclusions),
        "fold_counts": {
            str(fold): sum(row["fold"] == fold for row in eligible) for fold in range(5)
        },
        "input_token_summary": {
            "min": min(row["input_tokens"] for row in eligible),
            "max": max(row["input_tokens"] for row in eligible),
            "total": sum(row["input_tokens"] for row in eligible),
        },
        "excluded": exclusions,
    }
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
