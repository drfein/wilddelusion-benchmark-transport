from __future__ import annotations

import argparse
import hashlib
import json
import random
from pathlib import Path
from typing import Any

import pandas as pd

SEED = 20260801
EXPECTED_CONVERSATIONS = 221


def stable_hash(value: Any) -> str:
    encoded = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Freeze outcome-independent cohort.")
    parser.add_argument("--generation-cohort", type=Path, required=True)
    parser.add_argument("--release", type=Path, required=True)
    parser.add_argument("--pilot-cohort", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()

    generation = pd.read_parquet(args.generation_cohort)
    release = pd.read_parquet(args.release).reset_index(names="original_row_idx")
    pilot = pd.read_parquet(args.pilot_cohort)
    pilot_conversations = set(pilot["conversation_hash"].astype(str))

    if generation["original_row_idx"].duplicated().any():
        raise ValueError("Generation cohort row IDs are not unique")
    candidates = generation[
        ~generation["conversation_hash"].astype(str).isin(pilot_conversations)
    ].merge(
        release[["original_row_idx", "messages"]],
        on="original_row_idx",
        validate="one_to_one",
    )

    eligible: list[dict[str, Any]] = []
    for row in candidates.itertuples(index=False):
        target_index = int(row.target_message_index)
        messages = [dict(message) for message in row.messages[: target_index + 1]]
        if not messages or messages[-1].get("role") != "user":
            continue
        if str(messages[-1].get("content", "")) != str(row.target_text):
            raise ValueError(f"Target integrity failure at row {row.original_row_idx}")
        prefix = messages[:-1]
        if (
            not prefix
            or prefix[-1].get("role") != "assistant"
            or not any(message.get("role") == "user" for message in prefix[:-1])
        ):
            continue
        eligible.append(
            {
                "original_row_idx": int(row.original_row_idx),
                "source": str(row.source),
                "conversation_hash": str(row.conversation_hash),
                "message_hash": str(row.message_hash),
                "target_message_index": target_index,
                "messages": messages,
                "target_text": str(row.target_text),
                "estimated_full_input_tokens": int(row.estimated_full_input_tokens),
            }
        )

    rng = random.Random(SEED)
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in eligible:
        grouped.setdefault(row["conversation_hash"], []).append(row)
    selected = [
        rng.choice(sorted(rows, key=lambda x: x["original_row_idx"]))
        for _, rows in sorted(grouped.items())
    ]
    selected.sort(key=lambda row: row["conversation_hash"])
    if len(selected) != EXPECTED_CONVERSATIONS:
        raise ValueError(
            f"Frozen cohort changed: expected {EXPECTED_CONVERSATIONS}, found {len(selected)}"
        )

    private_rows: list[dict[str, Any]] = []
    public_rows: list[dict[str, Any]] = []
    for row in selected:
        prefix = row["messages"][:-1]
        previous_user = next(
            message for message in reversed(prefix[:-1]) if message["role"] == "user"
        )
        assistant_indices = [
            index
            for index, message in enumerate(prefix)
            if message["role"] == "assistant"
        ]
        shared = {
            key: row[key]
            for key in (
                "original_row_idx",
                "source",
                "conversation_hash",
                "message_hash",
                "target_message_index",
                "estimated_full_input_tokens",
            )
        }
        private_rows.append(
            {
                **shared,
                "messages": row["messages"],
                "target_text": row["target_text"],
                "previous_user_text": str(previous_user["content"]),
                "original_assistant_text": str(prefix[-1]["content"]),
                "immediate_assistant_index": len(prefix) - 1,
                "earlier_assistant_indices": assistant_indices[:-1],
            }
        )
        public_rows.append(
            {
                **shared,
                "message_count": len(row["messages"]),
                "prior_assistant_messages": len(assistant_indices),
                "earlier_assistant_messages": len(assistant_indices) - 1,
                "previous_user_words": len(str(previous_user["content"]).split()),
                "immediate_assistant_words": len(str(prefix[-1]["content"]).split()),
                "target_words": len(row["target_text"].split()),
                "conversation_input_sha256": stable_hash(row["messages"]),
                "local_rewrite_input_sha256": stable_hash(
                    [previous_user["content"], prefix[-1]["content"]]
                ),
            }
        )

    args.out_dir.mkdir(parents=True, exist_ok=True)
    private_path = args.out_dir / "private" / "cohort.jsonl"
    write_jsonl(private_path, private_rows)
    public = pd.DataFrame(public_rows)
    public.to_parquet(args.out_dir / "cohort.parquet", index=False)
    manifest = {
        "selection_frozen_before_new_outcomes": True,
        "selection_uses_prior_model_outcomes": False,
        "source_rows": len(generation),
        "source_conversations": int(generation["conversation_hash"].nunique()),
        "excluded_pilot_conversations": len(pilot_conversations),
        "eligible_nonpilot_rows": len(eligible),
        "targets": len(selected),
        "conversations": len(selected),
        "one_target_per_conversation": True,
        "selection_seed": SEED,
        "source_counts": public["source"].value_counts().sort_index().to_dict(),
        "private_cohort_sha256": hashlib.sha256(private_path.read_bytes()).hexdigest(),
        "text_storage": "private/cohort.jsonl is gitignored",
    }
    (args.out_dir / "cohort_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
