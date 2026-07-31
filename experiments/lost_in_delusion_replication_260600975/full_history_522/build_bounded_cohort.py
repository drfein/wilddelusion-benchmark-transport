#!/usr/bin/env python3
"""Build the full-cohort, bounded-history arm for the Lost in Delusion study."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def retain_recent_user_turns(
    messages: list[dict[str, str]], max_user_turns: int
) -> list[dict[str, str]]:
    user_positions = [
        index for index, message in enumerate(messages) if message["role"] == "user"
    ]
    if len(user_positions) <= max_user_turns:
        return messages
    return messages[user_positions[-max_user_turns] :]


def fit_character_budget(
    messages: list[dict[str, str]], max_characters: int
) -> list[dict[str, str]]:
    retained = messages
    while sum(len(message["content"]) for message in retained) > max_characters:
        user_positions = [
            index
            for index, message in enumerate(retained)
            if message["role"] == "user"
        ]
        if len(user_positions) <= 1:
            break
        retained = retained[user_positions[1] :]
    return retained


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("../full_dataset_522/artifacts/cohort_full522.jsonl"),
    )
    parser.add_argument(
        "--output", type=Path, default=Path("artifacts/cohort_original.jsonl")
    )
    parser.add_argument("--max-user-turns", type=int, default=4)
    parser.add_argument("--max-retained-characters", type=int, default=40_000)
    args = parser.parse_args()

    output_rows: list[dict[str, Any]] = []
    for source in read_jsonl(args.input):
        full_history = source["history_messages"]
        retained = retain_recent_user_turns(full_history, args.max_user_turns)
        retained = fit_character_budget(
            retained, args.max_retained_characters
        )
        if not retained or retained[0]["role"] != "user":
            raise ValueError(f"{source['pair_id']}: retained history must start with user")
        if retained[-1]["role"] != "user":
            raise ValueError(f"{source['pair_id']}: retained history must end with user")
        if retained[-1]["content"] != source["target_text"]:
            raise ValueError(f"{source['pair_id']}: target changed during truncation")

        history_json = json.dumps(
            retained, ensure_ascii=False, sort_keys=True
        ).encode("utf-8")
        output_rows.append(
            {
                **{key: value for key, value in source.items() if key != "history_messages"},
                "history_messages": retained,
                "history_sha256": hashlib.sha256(history_json).hexdigest(),
                "context_scope": f"final_{args.max_user_turns}_user_turns",
                "source_message_count_through_target": len(full_history),
                "source_user_turn_count_through_target": sum(
                    message["role"] == "user" for message in full_history
                ),
                "source_character_count_through_target": sum(
                    len(message["content"]) for message in full_history
                ),
                "retained_message_count": len(retained),
                "retained_user_turns": sum(
                    message["role"] == "user" for message in retained
                ),
                "retained_character_count": sum(
                    len(message["content"]) for message in retained
                ),
                "dropped_message_count_before_control": len(full_history)
                - len(retained),
            }
        )

    output_rows.sort(key=lambda row: row["pair_id"])
    if len(output_rows) != 522:
        raise ValueError(f"expected 522 targets, found {len(output_rows)}")
    if len({row["pair_id"] for row in output_rows}) != len(output_rows):
        raise ValueError("pair IDs are not unique")

    write_jsonl(args.output, output_rows)
    manifest = {
        "rows": len(output_rows),
        "conversation_clusters": len(
            {row["cluster_id"] for row in output_rows}
        ),
        "max_user_turns": args.max_user_turns,
        "max_retained_character_budget": args.max_retained_characters,
        "context_scope": f"final_{args.max_user_turns}_user_turns",
        "all_real_targets_retained": True,
        "targets_with_prior_history_truncated": sum(
            row["dropped_message_count_before_control"] > 0 for row in output_rows
        ),
        "sources": dict(Counter(row["source"] for row in output_rows)),
        "themes": dict(Counter(row["theme"] for row in output_rows)),
        "max_retained_messages": max(
            row["retained_message_count"] for row in output_rows
        ),
        "max_retained_characters": max(
            row["retained_character_count"] for row in output_rows
        ),
        "targets_still_over_character_budget": sum(
            row["retained_character_count"] > args.max_retained_characters
            for row in output_rows
        ),
        "sha256": hashlib.sha256(args.output.read_bytes()).hexdigest(),
    }
    manifest_path = args.output.with_suffix(args.output.suffix + ".manifest.json")
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
