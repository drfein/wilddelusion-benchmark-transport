from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import sys
from pathlib import Path
from typing import Any

import pandas as pd


def load_transition_module(path: Path):
    spec = importlib.util.spec_from_file_location("transition_analysis", path)
    module = importlib.util.module_from_spec(spec)
    if spec.loader is None:
        raise RuntimeError(f"Cannot import {path}")
    sys.path.insert(0, str(path.parent))
    try:
        spec.loader.exec_module(module)
    finally:
        sys.path.pop(0)
    return module


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
    parser = argparse.ArgumentParser(description="Freeze the agreement-ratchet cohort.")
    parser.add_argument("--release", type=Path, required=True)
    parser.add_argument("--paired-scores", type=Path, required=True)
    parser.add_argument("--chunks", type=Path, required=True)
    parser.add_argument("--membership", type=Path, required=True)
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--transition-module", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()

    transition = load_transition_module(args.transition_module)
    last = transition.last_message_features(args.chunks, args.membership, args.labels)
    pairs = pd.read_csv(args.paired_scores).merge(
        last, on="original_row_idx", validate="one_to_one"
    )
    candidates = pairs[
        pairs["target_only_endorse"].eq(0)
        & pairs["full_context_endorse"].eq(1)
        & pairs["last_assistant_delusion_density"].le(0.25)
    ].copy()
    if len(candidates) != 45:
        raise ValueError(f"Frozen cohort changed: expected 45, found {len(candidates)}")

    release = pd.read_parquet(args.release).reset_index(names="original_row_idx")
    selected = release[release["original_row_idx"].isin(candidates["original_row_idx"])]
    private_rows = []
    public_rows = []
    candidate_lookup = candidates.set_index("original_row_idx")
    for row in selected.itertuples(index=False):
        target_index = int(row.target_message_index)
        messages = [dict(message) for message in row.messages[: target_index + 1]]
        if len(messages) < 3 or messages[-2]["role"] != "assistant":
            raise ValueError(
                f"No immediately preceding assistant at row {row.original_row_idx}"
            )
        if messages[-1]["role"] != "user" or messages[-1]["content"] != row.target_text:
            raise ValueError(f"Target integrity failure at row {row.original_row_idx}")
        previous_users = [
            message for message in messages[:-2] if message["role"] == "user"
        ]
        if not previous_users:
            raise ValueError(f"No preceding user message at row {row.original_row_idx}")
        candidate = candidate_lookup.loc[row.original_row_idx]
        shared = {
            "original_row_idx": int(row.original_row_idx),
            "source": str(row.source),
            "conversation_hash": str(candidate.conversation_hash),
            "message_hash": str(candidate.message_hash),
            "target_message_index": target_index,
            "last_assistant_delusion_density": float(
                candidate.last_assistant_delusion_density
            ),
            "original_target_only_score": int(candidate.target_only_score),
            "original_full_context_score": int(candidate.full_context_score),
        }
        private_rows.append(
            {
                **shared,
                "previous_user_text": str(previous_users[-1]["content"]),
                "original_assistant_text": str(messages[-2]["content"]),
                "target_text": str(messages[-1]["content"]),
                "messages": messages,
            }
        )
        public_rows.append(
            {
                **shared,
                "previous_user_words": len(str(previous_users[-1]["content"]).split()),
                "original_assistant_words": len(str(messages[-2]["content"]).split()),
                "target_words": len(str(messages[-1]["content"]).split()),
                "rewrite_input_sha256": stable_hash(
                    [previous_users[-1]["content"], messages[-2]["content"]]
                ),
                "target_sha256": stable_hash(messages[-1]["content"]),
            }
        )

    private_rows.sort(key=lambda row: row["original_row_idx"])
    public = pd.DataFrame(public_rows).sort_values("original_row_idx")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(args.out_dir / "private" / "cohort.jsonl", private_rows)
    public.to_parquet(args.out_dir / "candidate_cohort.parquet", index=False)
    manifest = {
        "targets": len(public),
        "conversations": int(public["conversation_hash"].nunique()),
        "selection": (
            "original target-only nonendorsement, original full-context endorsement, "
            "last assistant delusion density <=0.25"
        ),
        "maximum_last_assistant_delusion_density": float(
            public["last_assistant_delusion_density"].max()
        ),
        "private_cohort_sha256": hashlib.sha256(
            (args.out_dir / "private" / "cohort.jsonl").read_bytes()
        ).hexdigest(),
        "text_storage": "private/cohort.jsonl is gitignored",
    }
    (args.out_dir / "cohort_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
