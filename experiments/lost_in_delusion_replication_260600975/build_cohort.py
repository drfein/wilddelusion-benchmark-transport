#!/usr/bin/env python3
"""Build the fixed human-confirmed WildDelusion cohort."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any

from io_utils import write_jsonl, write_manifest


def normalized_history(row: dict[str, Any], max_user_turns: int) -> list[dict[str, str]]:
    target_index = int(row["flagged_msg_idx"])
    source = row["messages"][: target_index + 1]
    messages: list[dict[str, str]] = []
    for message in source:
        role = str(message.get("role", "")).strip().lower()
        content = str(message.get("content", "")).strip()
        if role not in {"user", "assistant"} or not content:
            continue
        if messages and messages[-1]["role"] == role:
            messages[-1]["content"] += "\n\n" + content
        else:
            messages.append({"role": role, "content": content})
    while messages and messages[0]["role"] != "user":
        messages.pop(0)
    if not messages or messages[-1]["role"] != "user":
        raise ValueError(f"{row['id']}: normalized history does not end in user")
    user_positions = [
        index for index, message in enumerate(messages) if message["role"] == "user"
    ]
    if len(user_positions) > max_user_turns:
        messages = messages[user_positions[-max_user_turns] :]
    return messages


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input", type=Path, default=Path("data_sources/wd2/wd_2.jsonl")
    )
    parser.add_argument(
        "--output", type=Path, default=Path("artifacts/cohort_original.jsonl")
    )
    parser.add_argument("--max-user-turns", type=int, default=4)
    args = parser.parse_args()

    rows: list[dict[str, Any]] = []
    with args.input.open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            if (row.get("review_saved_count") or 0) <= 0:
                continue
            if (row.get("review_rejected_count") or 0) > 0:
                continue
            target_index = int(row["flagged_msg_idx"])
            target = row["messages"][target_index]
            if target.get("role") != "user":
                raise ValueError(f"{row['id']}: flagged message is not user")
            target_content = target.get("content", "").strip()
            flagged_text = row.get("flagged_text", "").strip()
            if target_content == flagged_text:
                target_match_mode = "exact"
            elif len(flagged_text) == 500 and target_content.startswith(flagged_text):
                target_match_mode = "exported_500_character_prefix"
            else:
                raise ValueError(f"{row['id']}: flagged text mismatch")
            messages = normalized_history(row, args.max_user_turns)
            pair_id = str(row["id"])
            digest = hashlib.sha256(
                json.dumps(messages, ensure_ascii=False, sort_keys=True).encode()
            ).hexdigest()
            rows.append(
                {
                    "pair_id": pair_id,
                    "conversation_sha256": row.get("conversation_sha256"),
                    "source": row.get("source"),
                    "theme": row.get("delusion_theme_primary"),
                    "theme_confidence": row.get("delusion_theme_confidence"),
                    "escalated_observed": row.get("escalated"),
                    "history_sha256": digest,
                    "history_messages": messages,
                    "target_text": messages[-1]["content"],
                    "target_match_mode": target_match_mode,
                    "source_message_count_through_target": target_index + 1,
                    "retained_message_count": len(messages),
                    "retained_user_turns": sum(
                        message["role"] == "user" for message in messages
                    ),
                }
            )

    rows.sort(key=lambda row: row["pair_id"])
    duplicate_ids = len(rows) - len({row["pair_id"] for row in rows})
    duplicate_conversations = len(rows) - len(
        {row["conversation_sha256"] for row in rows}
    )
    if duplicate_ids or duplicate_conversations:
        raise ValueError(
            f"cohort is not deduplicated: ids={duplicate_ids}, "
            f"conversations={duplicate_conversations}"
        )
    write_jsonl(args.output, rows)
    write_manifest(
        args.output.with_suffix(".manifest.json"),
        {
            "rows": len(rows),
            "human_saved_only": True,
            "rejected_excluded": True,
            "unique_conversations": len(
                {row["conversation_sha256"] for row in rows}
            ),
            "max_user_turns": args.max_user_turns,
            "themes": dict(Counter(row["theme"] for row in rows)),
            "sources": dict(Counter(row["source"] for row in rows)),
            "max_retained_characters": max(
                sum(len(message["content"]) for message in row["history_messages"])
                for row in rows
            ),
        },
    )
    print(f"Wrote {len(rows)} rows to {args.output}")


if __name__ == "__main__":
    main()
