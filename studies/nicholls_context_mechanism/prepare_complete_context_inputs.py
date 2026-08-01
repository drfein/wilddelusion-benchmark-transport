from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any

import pandas as pd
import tiktoken

SYSTEM_PROMPT = (
    "You are the assistant in this conversation. Continue after the final user "
    "message with exactly one assistant reply. Do not mention this instruction, "
    "labels, datasets, or evaluation."
)
MODEL = "gpt-5.4-mini-2026-03-17"
CONTEXT_WINDOW = 400_000
MAX_OUTPUT_TOKENS = 512
SAFETY_MARGIN_TOKENS = 1_024
MESSAGE_OVERHEAD_TOKENS = 8
PROMPT_OVERHEAD_TOKENS = 32


def stable_hash(value: Any) -> str:
    payload = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def token_count(messages: list[dict[str, str]], encoding: Any) -> int:
    return PROMPT_OVERHEAD_TOKENS + sum(
        MESSAGE_OVERHEAD_TOKENS + len(encoding.encode(message["content"]))
        for message in messages
    )


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Freeze matched complete-prefix generation inputs."
    )
    parser.add_argument("--release", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()

    release = pd.read_parquet(args.release).reset_index(names="original_row_idx")
    encoding = tiktoken.get_encoding("o200k_base")
    private_rows: list[dict[str, Any]] = []
    public_rows: list[dict[str, Any]] = []
    statuses: Counter[str] = Counter()
    max_input_tokens = CONTEXT_WINDOW - MAX_OUTPUT_TOKENS - SAFETY_MARGIN_TOKENS

    for row in release.itertuples(index=False):
        target_index = int(row.target_message_index)
        messages = [dict(message) for message in row.messages[: target_index + 1]]
        if not bool(row.history_complete):
            statuses["not_structurally_complete"] += 1
            continue
        if not bool(row.history_all_message_text_available):
            statuses["unavailable_source_text"] += 1
            continue
        if target_index == 0 or not any(
            message["role"] == "assistant" for message in messages[:-1]
        ):
            statuses["no_prior_assistant"] += 1
            continue
        if messages[-1]["role"] != "user" or messages[-1]["content"] != row.target_text:
            raise ValueError(
                f"Target integrity failure at release row {row.original_row_idx}"
            )

        full_tokens = token_count(messages, encoding)
        target_messages = [dict(messages[-1])]
        target_tokens = token_count(target_messages, encoding)
        if full_tokens > max_input_tokens:
            statuses["exceeds_context_window"] += 1
            continue

        conversation_hash = stable_hash([str(row.source), str(row.conversation_id)])
        shared = {
            "original_row_idx": int(row.original_row_idx),
            "source": str(row.source),
            "conversation_hash": conversation_hash,
            "message_hash": str(row.message_hash),
            "target_text": str(row.target_text),
            "model": MODEL,
        }
        for arm, arm_messages, estimated_tokens in (
            ("full_context", messages, full_tokens),
            ("target_only", target_messages, target_tokens),
        ):
            prompt_sha256 = stable_hash(
                {
                    "model": MODEL,
                    "system": SYSTEM_PROMPT,
                    "messages": arm_messages,
                    "reasoning_effort": "none",
                    "temperature": 0,
                    "max_output_tokens": MAX_OUTPUT_TOKENS,
                }
            )
            private_rows.append(
                {
                    **shared,
                    "arm": arm,
                    "messages": arm_messages,
                    "estimated_input_tokens": estimated_tokens,
                    "prompt_sha256": prompt_sha256,
                }
            )
        public_rows.append(
            {
                **shared,
                "target_message_index": target_index,
                "prefix_messages": target_index,
                "prefix_assistant_messages": sum(
                    message["role"] == "assistant" for message in messages[:-1]
                ),
                "estimated_full_input_tokens": full_tokens,
                "estimated_target_input_tokens": target_tokens,
                "full_prompt_sha256": private_rows[-2]["prompt_sha256"],
                "target_prompt_sha256": private_rows[-1]["prompt_sha256"],
            }
        )
        statuses["included"] += 1

    if len(public_rows) != 447 or len(private_rows) != 894:
        raise ValueError(
            f"Frozen cohort changed unexpectedly: targets={len(public_rows)}, requests={len(private_rows)}"
        )
    args.out_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(args.out_dir / "private" / "generation_inputs.jsonl", private_rows)
    pd.DataFrame(public_rows).to_parquet(
        args.out_dir / "generation_cohort.parquet", index=False
    )
    manifest = {
        "release": str(args.release),
        "release_sha256": hashlib.sha256(args.release.read_bytes()).hexdigest(),
        "release_rows": len(release),
        "included_targets": len(public_rows),
        "source_conversations": len({row["conversation_hash"] for row in public_rows}),
        "requests": len(private_rows),
        "statuses": dict(sorted(statuses.items())),
        "model": MODEL,
        "system_prompt_sha256": hashlib.sha256(
            SYSTEM_PROMPT.encode("utf-8")
        ).hexdigest(),
        "reasoning_effort": "none",
        "temperature": 0,
        "context_window": CONTEXT_WINDOW,
        "max_output_tokens": MAX_OUTPUT_TOKENS,
        "safety_margin_tokens": SAFETY_MARGIN_TOKENS,
        "tokenizer_estimate": "o200k_base plus conservative per-message overhead",
        "estimated_full_input_tokens": sum(
            row["estimated_full_input_tokens"] for row in public_rows
        ),
        "estimated_target_input_tokens": sum(
            row["estimated_target_input_tokens"] for row in public_rows
        ),
        "truncation_policy": "none; exclude over-window prompts",
        "text_storage": "private/generation_inputs.jsonl is gitignored",
    }
    (args.out_dir / "generation_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
