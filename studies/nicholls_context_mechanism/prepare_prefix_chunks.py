from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path
from typing import Any

import pandas as pd


ROLE_ALIASES = {"human": "user", "llm": "assistant"}


def stable_hash(*parts: object) -> str:
    payload = "\x1f".join(str(part) for part in parts)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def normalize_role(value: object) -> str:
    role = str(value or "").lower().strip()
    return ROLE_ALIASES.get(role, role)


def chunk_text(text: str, max_words: int) -> list[str]:
    words = re.findall(r"\S+", text)
    if not words:
        return [""]
    return [" ".join(words[start : start + max_words]) for start in range(0, len(words), max_words)]


def canonical_conversations(release: pd.DataFrame) -> dict[str, list[dict[str, Any]]]:
    canonical: dict[str, list[dict[str, Any]]] = {}
    for row in release.itertuples(index=False):
        conversation_id = str(row.conversation_id)
        messages = list(row.messages)
        if conversation_id not in canonical or len(messages) > len(canonical[conversation_id]):
            canonical[conversation_id] = messages

    for row in release.itertuples(index=False):
        messages = list(row.messages)
        reference = canonical[str(row.conversation_id)]
        for index, message in enumerate(messages):
            if index >= len(reference):
                raise ValueError("A release row is longer than its canonical conversation.")
            left = (normalize_role(message.get("role")), str(message.get("content", "")))
            right = (
                normalize_role(reference[index].get("role")),
                str(reference[index].get("content", "")),
            )
            if left != right:
                raise ValueError(f"Conflicting conversation reconstruction at {row.conversation_id}:{index}")
    return canonical


def main() -> None:
    parser = argparse.ArgumentParser(description="Deduplicate and chunk WildDelusion prefixes.")
    parser.add_argument("--release", type=Path, required=True)
    parser.add_argument("--paired-scores", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--max-words", type=int, default=700)
    args = parser.parse_args()

    release = pd.read_parquet(args.release).reset_index(names="original_row_idx")
    paired = pd.read_csv(args.paired_scores)
    retained = set(paired["original_row_idx"].astype(int))
    release = release[release["original_row_idx"].isin(retained)].copy()
    if set(release["original_row_idx"]) != retained:
        raise ValueError("Paired scores contain rows absent from the release.")

    canonical = canonical_conversations(release)
    needed: dict[tuple[str, int], dict[str, Any]] = {}
    membership: list[dict[str, Any]] = []
    for row in release.itertuples(index=False):
        conversation_id = str(row.conversation_id)
        target_index = int(row.target_message_index)
        for message_index, message in enumerate(canonical[conversation_id][:target_index]):
            key = (conversation_id, message_index)
            needed[key] = message
            membership.append(
                {
                    "original_row_idx": int(row.original_row_idx),
                    "conversation_hash": stable_hash(conversation_id),
                    "message_index": message_index,
                    "message_key": stable_hash(conversation_id, message_index),
                }
            )

    chunks: list[dict[str, Any]] = []
    for (conversation_id, message_index), message in sorted(needed.items()):
        role = normalize_role(message.get("role"))
        text = str(message.get("content", ""))
        for chunk_index, chunk in enumerate(chunk_text(text, args.max_words)):
            chunks.append(
                {
                    "item_id": stable_hash(conversation_id, message_index, chunk_index, chunk),
                    "conversation_hash": stable_hash(conversation_id),
                    "message_key": stable_hash(conversation_id, message_index),
                    "message_index": message_index,
                    "chunk_index": chunk_index,
                    "role": role,
                    "word_count": len(re.findall(r"\S+", chunk)),
                    "text": chunk,
                }
            )

    args.out_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(args.out_dir / "private" / "blind_chunks.jsonl", chunks)
    pd.DataFrame(membership).drop_duplicates().to_parquet(
        args.out_dir / "private" / "prefix_membership.parquet", index=False
    )
    public_chunks = pd.DataFrame(chunks).drop(columns="text")
    public_chunks.to_parquet(args.out_dir / "chunk_index.parquet", index=False)
    manifest = {
        "release_sha256": hashlib.sha256(args.release.read_bytes()).hexdigest(),
        "paired_scores_sha256": hashlib.sha256(args.paired_scores.read_bytes()).hexdigest(),
        "release_rows": len(release),
        "source_conversations": int(release["conversation_id"].nunique()),
        "unique_prefix_messages": len(needed),
        "chunks": len(chunks),
        "prefix_memberships": len(membership),
        "max_words_per_chunk": args.max_words,
        "total_chunk_words": sum(row["word_count"] for row in chunks),
        "role_counts": public_chunks["role"].value_counts().to_dict(),
        "empty_prefix_targets": int((release["target_message_index"].astype(int) == 0).sum()),
        "text_storage": "private/blind_chunks.jsonl is gitignored; public index contains no text or source IDs",
    }
    (args.out_dir / "prepare_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
