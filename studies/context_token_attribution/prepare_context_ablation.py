from __future__ import annotations

import argparse
import json
from pathlib import Path

from config import MODEL_ID, MODEL_REVISION, stable_hash
from io_utils import read_jsonl, sha256_file, write_jsonl
from transformers import AutoTokenizer


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cohort", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--model", default=MODEL_ID)
    parser.add_argument("--revision", default=MODEL_REVISION)
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
        messages = [row["messages"][0], row["messages"][-1]]
        prompt_hash = stable_hash(
            {
                "parent_prompt": row["prompt_content_sha256"],
                "condition": "target_only",
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
                "condition": "target_only",
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
        "condition": "target_only",
        "retained_messages": "system message and final target user message",
        "removed_messages": "all user and assistant history before the target",
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
