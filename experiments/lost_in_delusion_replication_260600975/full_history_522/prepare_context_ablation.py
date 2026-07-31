#!/usr/bin/env python3
"""Strip frozen full-history inputs to their exact final user message."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from io_utils import read_jsonl, write_jsonl, write_manifest  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    here = Path(__file__).resolve().parent
    parser.add_argument(
        "--input",
        type=Path,
        default=here / "artifacts" / "model_inputs.jsonl",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=here / "artifacts" / "last_user_only_inputs.jsonl",
    )
    args = parser.parse_args()

    source = read_jsonl(args.input)
    rows: list[dict[str, object]] = []
    for row in source:
        messages = row["messages"]
        if not messages or messages[-1]["role"] != "user":
            raise ValueError(f"{row['input_id']}: input does not end in user")
        final_message = dict(messages[-1])
        digest = hashlib.sha256(
            (
                str(row["input_id"])
                + ":exact_last_user_only:"
                + json.dumps(final_message, ensure_ascii=False, sort_keys=True)
            ).encode("utf-8")
        ).hexdigest()
        rows.append(
            {
                **row,
                "source_full_history_input_id": row["input_id"],
                "input_id": digest[:24],
                "messages": [final_message],
                "context_scope": "exact_last_user_only",
                "retained_message_count": 1,
                "retained_user_turns": 1,
                "retained_character_count": len(final_message["content"]),
                "context_ablation": True,
            }
        )

    if len(rows) != len(source):
        raise AssertionError("context ablation changed row count")
    if len({row["input_id"] for row in rows}) != len(rows):
        raise AssertionError("context ablation produced duplicate input IDs")
    write_jsonl(args.output, rows)
    write_manifest(
        args.output.with_suffix(args.output.suffix + ".manifest.json"),
        {
            "source": str(args.input),
            "rows": len(rows),
            "pairs": len({row["pair_id"] for row in rows}),
            "conditions": sorted({row["condition"] for row in rows}),
            "transformation": (
                "delete every message except the exact final user message"
            ),
            "final_text_changed": False,
        },
    )
    print(f"Wrote {len(rows)} exact last-user-only ablations")


if __name__ == "__main__":
    main()
