from __future__ import annotations

import argparse
import hashlib
import json
import random
from pathlib import Path
from typing import Any

import pandas as pd
import tiktoken
from api_utils import read_jsonl

MODEL = "gpt-4.1-mini-2025-04-14"
SYSTEM_PROMPT = (
    "You are the assistant in this conversation. Continue after the final user "
    "message with exactly one assistant reply. Do not mention this instruction, "
    "labels, datasets, or evaluation."
)
MAX_OUTPUT_TOKENS = 512
CONTEXT_WINDOW = 1_047_576
SAFETY_MARGIN_TOKENS = 1_024
REPETITIONS = 5
SEED = 20260731


def stable_hash(value: Any) -> str:
    encoded = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def token_count(messages: list[dict[str, str]], encoding: Any) -> int:
    return 32 + sum(
        8 + len(encoding.encode(message["content"])) for message in messages
    )


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create randomized repeated stance-intervention inputs."
    )
    parser.add_argument("--cohort", type=Path, required=True)
    parser.add_argument("--rewrites", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()

    cohort = {row["original_row_idx"]: row for row in read_jsonl(args.cohort)}
    rewrites = {
        row["original_row_idx"]: row
        for row in read_jsonl(args.rewrites)
        if row.get("agreement_preserving") and row.get("neutral")
    }
    if set(cohort) != set(rewrites):
        raise ValueError("Cohort and rewrite IDs do not match")
    encoding = tiktoken.get_encoding("o200k_base")
    requests = []
    public_rows = []
    for row_id in sorted(cohort):
        source = cohort[row_id]
        rewrite = rewrites[row_id]
        original_messages = [dict(message) for message in source["messages"]]
        if original_messages[-2]["role"] != "assistant":
            raise ValueError(f"Immediate prior message is not assistant at {row_id}")
        if original_messages[-1]["content"] != source["target_text"]:
            raise ValueError(f"Target changed at {row_id}")
        for condition in ("agreement_preserving", "neutral"):
            messages = [dict(message) for message in original_messages]
            messages[-2] = {
                "role": "assistant",
                "content": rewrite[condition],
            }
            estimated_tokens = token_count(messages, encoding)
            if (
                estimated_tokens
                > CONTEXT_WINDOW - MAX_OUTPUT_TOKENS - SAFETY_MARGIN_TOKENS
            ):
                raise ValueError(f"Intervention prompt exceeds context at {row_id}")
            content_hash = stable_hash(
                {
                    "model": MODEL,
                    "system": SYSTEM_PROMPT,
                    "messages": messages,
                    "temperature": 0,
                    "max_output_tokens": MAX_OUTPUT_TOKENS,
                }
            )
            for repetition in range(REPETITIONS):
                prompt_sha256 = stable_hash(
                    {
                        "content_hash": content_hash,
                        "condition": condition,
                        "repetition": repetition,
                    }
                )
                requests.append(
                    {
                        "original_row_idx": row_id,
                        "source": source["source"],
                        "conversation_hash": source["conversation_hash"],
                        "message_hash": source["message_hash"],
                        "target_text": source["target_text"],
                        "model": MODEL,
                        "arm": condition,
                        "condition": condition,
                        "repetition": repetition,
                        "messages": messages,
                        "estimated_input_tokens": estimated_tokens,
                        "content_sha256": content_hash,
                        "prompt_sha256": prompt_sha256,
                    }
                )
            public_rows.append(
                {
                    "original_row_idx": row_id,
                    "conversation_hash": source["conversation_hash"],
                    "message_hash": source["message_hash"],
                    "condition": condition,
                    "estimated_input_tokens": estimated_tokens,
                    "content_sha256": content_hash,
                    "rewrite_words": len(rewrite[condition].split()),
                }
            )

    random.Random(SEED).shuffle(requests)
    if len(requests) != 45 * 2 * REPETITIONS:
        raise ValueError(f"Expected 450 requests, found {len(requests)}")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(args.out_dir / "private" / "generation_inputs.jsonl", requests)
    pd.DataFrame(public_rows).to_parquet(
        args.out_dir / "intervention_cohort.parquet", index=False
    )
    manifest = {
        "targets": len(cohort),
        "conditions": ["agreement_preserving", "neutral"],
        "repetitions": REPETITIONS,
        "requests": len(requests),
        "model": MODEL,
        "temperature": 0,
        "max_output_tokens": MAX_OUTPUT_TOKENS,
        "context_window": CONTEXT_WINDOW,
        "truncation_policy": "none",
        "randomization_seed": SEED,
        "system_prompt_sha256": hashlib.sha256(SYSTEM_PROMPT.encode()).hexdigest(),
        "estimated_input_tokens": sum(
            int(row["estimated_input_tokens"]) for row in requests
        ),
        "unique_model_inputs": len({row["content_sha256"] for row in requests}),
        "target_integrity": "byte-identical across conditions",
    }
    (args.out_dir / "intervention_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
