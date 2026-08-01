#!/usr/bin/env python3
"""Sample source-matched broad controls from the original public corpora."""

from __future__ import annotations

import argparse
import json
import random
from collections import Counter, defaultdict
from pathlib import Path

from datasets import load_dataset

from design import SEED, SHARECHAT_REVISION, WILDCHAT_REVISION


def read_jsonl(path: Path) -> list[dict]:
    with path.open() as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def reservoir_sample(rows, n: int, blocked: set[str], source: str, max_eligible: int = 100_000):
    rng = random.Random(f"{SEED}:{source}")
    reservoir = []
    eligible = 0
    for conversation_id, text in rows:
        text = (text or "").strip()
        if conversation_id in blocked or len(text) < 20:
            continue
        eligible += 1
        item = {"conversation_id": conversation_id, "source": source, "text": text}
        if len(reservoir) < n:
            reservoir.append(item)
        else:
            replacement = rng.randrange(eligible)
            if replacement < n:
                reservoir[replacement] = item
        if eligible >= max_eligible:
            break
    if len(reservoir) != n:
        raise RuntimeError(f"Only found {len(reservoir)} of {n} controls for {source}")
    return reservoir


def sharechat_rows(config: str):
    dataset = load_dataset(
        "tucnguyen/ShareChat", config, split="train", streaming=True, revision=SHARECHAT_REVISION
    )
    for row in dataset:
        if row.get("role") != "user" or row.get("detected_language_final") != "English":
            continue
        yield row["url"], row["plain_text"]


def wildchat_rows():
    dataset = load_dataset(
        "yuntian-deng/WildChat-4.8M-Full",
        split="train",
        streaming=True,
        revision=WILDCHAT_REVISION,
    )
    for row in dataset:
        if row.get("language") != "English":
            continue
        user_messages = [m for m in row["conversation"] if m.get("role") == "user" and m.get("content")]
        if user_messages:
            yield row["conversation_hash"], user_messages[-1]["content"]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--endpoints", type=Path, required=True)
    parser.add_argument("--natural-controls", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    endpoints = read_jsonl(args.endpoints)
    positives = [r for r in endpoints if r["condition"] == "delusion" and r["language"] == "ENGLISH"]
    needed = Counter(r["source"] for r in positives)
    blocked = {r["conversation_id"] for r in endpoints}
    blocked.update(r["conversation_id"] for r in read_jsonl(args.natural_controls))

    streams = {
        "sharechat_chatgpt": lambda: sharechat_rows("chatgpt"),
        "sharechat_grok": lambda: sharechat_rows("grok"),
        "sharechat_gemini": lambda: sharechat_rows("gemini"),
        "sharechat_claude": lambda: sharechat_rows("claude"),
        "wildchat_full": wildchat_rows,
    }
    controls = []
    for source, n in sorted(needed.items()):
        controls.extend(
            reservoir_sample(
                streams[source](), n, blocked, source, max_eligible=max(10_000, n * 100)
            )
        )

    if Counter(r["source"] for r in controls) != needed:
        raise RuntimeError("Source matching failed")
    write_jsonl(args.output, controls)
    print(json.dumps({"n": len(controls), "by_source": needed}, indent=2))


if __name__ == "__main__":
    main()
