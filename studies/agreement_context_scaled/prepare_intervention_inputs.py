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
REPETITIONS = 3
SEED = 20260802
CONDITIONS = ("agreement_preserving", "local_neutral", "cumulative_neutral")


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
    parser = argparse.ArgumentParser(description="Create scaled intervention inputs.")
    parser.add_argument("--cohort", type=Path, required=True)
    parser.add_argument("--local-rewrites", type=Path, required=True)
    parser.add_argument("--cumulative-rewrites", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()

    cohort = {row["original_row_idx"]: row for row in read_jsonl(args.cohort)}
    local = {
        row["original_row_idx"]: row
        for row in read_jsonl(args.local_rewrites)
        if row.get("agreement_preserving")
        and row.get("neutral")
        and not row.get("error")
    }
    cumulative = {
        row["original_row_idx"]: row
        for row in read_jsonl(args.cumulative_rewrites)
        if isinstance(row.get("rewrites"), list) and not row.get("error")
    }
    if set(cohort) != set(local) or set(cohort) != set(cumulative):
        raise ValueError("Cohort and rewrite IDs do not match")

    encoding = tiktoken.get_encoding("o200k_base")
    requests: list[dict[str, Any]] = []
    public_rows: list[dict[str, Any]] = []
    for row_id in sorted(cohort):
        source = cohort[row_id]
        original = [dict(message) for message in source["messages"]]
        immediate = int(source["immediate_assistant_index"])
        if immediate != len(original) - 2 or original[immediate]["role"] != "assistant":
            raise ValueError(f"Immediate assistant integrity failure at {row_id}")
        if original[-1]["content"] != source["target_text"]:
            raise ValueError(f"Target integrity failure at {row_id}")

        condition_messages: dict[str, list[dict[str, str]]] = {}
        agreement = [dict(message) for message in original]
        agreement[immediate]["content"] = local[row_id]["agreement_preserving"]
        condition_messages["agreement_preserving"] = agreement

        neutral = [dict(message) for message in original]
        neutral[immediate]["content"] = local[row_id]["neutral"]
        condition_messages["local_neutral"] = neutral

        cumulative_neutral = [dict(message) for message in neutral]
        for item in cumulative[row_id]["rewrites"]:
            index = int(item["message_index"])
            if index >= immediate or cumulative_neutral[index]["role"] != "assistant":
                raise ValueError(f"Invalid cumulative index {index} at {row_id}")
            cumulative_neutral[index]["content"] = item["rewrite"]
        condition_messages["cumulative_neutral"] = cumulative_neutral

        if neutral[immediate]["content"] != cumulative_neutral[immediate]["content"]:
            raise ValueError(f"Neutral arms differ at immediate message for {row_id}")
        for condition, messages in condition_messages.items():
            if [m["content"] for m in messages if m["role"] == "user"] != [
                m["content"] for m in original if m["role"] == "user"
            ]:
                raise ValueError(f"User text changed in {condition} at {row_id}")
            estimated_tokens = token_count(messages, encoding)
            if (
                estimated_tokens
                > CONTEXT_WINDOW - MAX_OUTPUT_TOKENS - SAFETY_MARGIN_TOKENS
            ):
                raise ValueError(f"Prompt exceeds context at {row_id}")
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
                prompt_hash = stable_hash(
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
                        "prompt_sha256": prompt_hash,
                    }
                )
            public_rows.append(
                {
                    "original_row_idx": row_id,
                    "source": source["source"],
                    "conversation_hash": source["conversation_hash"],
                    "message_hash": source["message_hash"],
                    "condition": condition,
                    "estimated_input_tokens": estimated_tokens,
                    "content_sha256": content_hash,
                    "cumulative_changed_messages": len(cumulative[row_id]["rewrites"]),
                }
            )

    random.Random(SEED).shuffle(requests)
    expected = len(cohort) * len(CONDITIONS) * REPETITIONS
    if len(requests) != expected:
        raise ValueError(f"Expected {expected} requests, found {len(requests)}")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(args.out_dir / "private" / "generation_inputs.jsonl", requests)
    pd.DataFrame(public_rows).to_parquet(
        args.out_dir / "intervention_cohort.parquet", index=False
    )
    manifest = {
        "targets": len(cohort),
        "conversations": len({row["conversation_hash"] for row in cohort.values()}),
        "conditions": list(CONDITIONS),
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
        "user_history_integrity": "byte-identical across conditions",
        "local_and_cumulative_immediate_rewrite_identical": True,
    }
    (args.out_dir / "intervention_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
