from __future__ import annotations

import argparse
import json
from pathlib import Path

from config import MODEL_ID, MODEL_REVISION, stable_hash
from io_utils import read_jsonl, sha256_file, write_jsonl
from transformers import AutoTokenizer


def retained_messages(messages: list[dict], condition: str) -> list[dict]:
    if condition == "target_only":
        return [messages[0], messages[-1]]
    if condition != "last_exchange":
        raise ValueError(f"Unknown condition: {condition}")
    assistant_index = next(
        (
            index
            for index in range(len(messages) - 2, 0, -1)
            if messages[index]["role"] == "assistant"
        ),
        None,
    )
    if assistant_index is None:
        raise ValueError("No prior assistant message for last_exchange")
    user_index = next(
        (
            index
            for index in range(assistant_index - 1, 0, -1)
            if messages[index]["role"] == "user"
        ),
        None,
    )
    if user_index is None:
        raise ValueError("No prior user message for last_exchange")
    return [messages[0], messages[user_index], messages[assistant_index], messages[-1]]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cohort", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--model", default=MODEL_ID)
    parser.add_argument("--revision", default=MODEL_REVISION)
    parser.add_argument(
        "--condition", choices=("target_only", "last_exchange"), default="target_only"
    )
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(
        args.model, revision=args.revision, trust_remote_code=False
    )
    rows = read_jsonl(args.cohort)
    output = []
    for row in rows:
        if row["messages"][0]["role"] != "system":
            raise ValueError("Expected a system message at index zero")
        if row["messages"][-1]["role"] != "user":
            raise ValueError("Expected the target user message to be last")
        messages = retained_messages(row["messages"], args.condition)
        prompt_hash = stable_hash(
            {
                "parent_prompt": row["prompt_content_sha256"],
                "condition": args.condition,
                "messages": messages,
            }
        )
        input_tokens = len(
            tokenizer.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=True,
                enable_thinking=False,
            )
        )
        output.append(
            {
                **{key: value for key, value in row.items() if key != "messages"},
                "condition": args.condition,
                "messages": messages,
                "input_tokens": input_tokens,
                "parent_prompt_content_sha256": row["prompt_content_sha256"],
                "prompt_content_sha256": prompt_hash,
            }
        )

    write_jsonl(args.output, output)
    manifest = {
        "cohort_sha256": sha256_file(args.cohort),
        "rows": len(output),
        "condition": args.condition,
        "retained_messages": (
            "system message and final target user message"
            if args.condition == "target_only"
            else "system, most recent prior user-assistant exchange, and target user"
        ),
        "removed_messages": (
            "all user and assistant history before the target"
            if args.condition == "target_only"
            else "all history preceding the most recent prior user-assistant exchange"
        ),
        "common_random_numbers": (
            "conversation hashes are unchanged, so generation seeds match full context"
        ),
        "output_sha256": sha256_file(args.output),
    }
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
