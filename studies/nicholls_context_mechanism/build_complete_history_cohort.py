from __future__ import annotations

import argparse
import hashlib
import json
import re
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import pandas as pd
import tiktoken


USER_ROLES = {"user", "human"}
ASSISTANT_ROLES = {"assistant", "llm"}


def canonical_text(value: object) -> str:
    text = unicodedata.normalize("NFKC", str(value or ""))
    return re.sub(r"\s+", " ", text).strip().casefold()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def normalize_messages(messages: list[dict[str, Any]], end: int) -> list[dict[str, str]]:
    output = []
    for message in messages[: end + 1]:
        raw_role = str(message.get("role", "")).lower()
        role = "user" if raw_role in USER_ROLES else "assistant" if raw_role in ASSISTANT_ROLES else ""
        content = str(message.get("content", "") or "").strip()
        if role and content:
            output.append({"role": role, "content": content})
    return output


def parse_provenance(row: dict[str, Any]) -> dict[str, Any]:
    value = row.get("hf_provenance") or {}
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except json.JSONDecodeError:
            return {}
    return value if isinstance(value, dict) else {}


def candidate_matches(release_row: dict[str, Any], candidate: dict[str, Any]) -> bool:
    release_id = str(release_row.get("conversation_id") or release_row.get("conversation_hash") or "")
    provenance = parse_provenance(candidate)
    candidate_ids = {
        str(candidate.get("conversation_id") or ""),
        str(provenance.get("url") or ""),
        str(provenance.get("conversation_id") or ""),
    }
    return bool(release_id and release_id in candidate_ids)


def token_count(messages: list[dict[str, str]], encoding_name: str) -> int:
    encoding = tiktoken.get_encoding(encoding_name)
    return 32 + sum(8 + len(encoding.encode(message["content"])) for message in messages)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a fail-closed complete-history cohort.")
    parser.add_argument("--release", type=Path, required=True)
    parser.add_argument("--canonical-conversations", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--reserved-output-tokens", type=int, default=768)
    parser.add_argument("--safety-margin-tokens", type=int, default=512)
    args = parser.parse_args()

    release = pd.read_parquet(args.release).reset_index(names="release_row_idx")
    canonical = read_jsonl(args.canonical_conversations)
    index: dict[str, list[tuple[int, int, dict[str, Any], str]]] = defaultdict(list)
    for canonical_row_idx, row in enumerate(canonical):
        messages = list(row["messages"])
        for message_index, message in enumerate(messages[:-1]):
            if str(message.get("role", "")).lower() not in USER_ROLES:
                continue
            reply = messages[message_index + 1]
            if str(reply.get("role", "")).lower() not in ASSISTANT_ROLES:
                continue
            if not str(reply.get("content", "") or "").strip():
                continue
            index[canonical_text(message.get("content"))].append(
                (canonical_row_idx, message_index, row, canonical_text(reply.get("content")))
            )

    statuses: Counter[str] = Counter()
    eligible: list[dict[str, Any]] = []
    seen_positions: set[tuple[int, int]] = set()
    for release_row in release.to_dict(orient="records"):
        candidates = index.get(canonical_text(release_row.get("target_text")), [])
        if not candidates:
            statuses["no_canonical_match"] += 1
            continue
        response_groups: dict[str, list[tuple[int, int, dict[str, Any], str]]] = defaultdict(list)
        for candidate in candidates:
            response_groups[candidate[3]].append(candidate)
        if len(response_groups) == 1:
            selected = next(iter(response_groups.values()))[0]
            match_status = "unique_response"
        else:
            provenance_matches = [candidate for candidate in candidates if candidate_matches(release_row, candidate[2])]
            matched_responses = {candidate[3] for candidate in provenance_matches}
            if len(matched_responses) != 1:
                statuses["ambiguous_response"] += 1
                continue
            selected = provenance_matches[0]
            match_status = "provenance_resolved"

        canonical_row_idx, target_index, canonical_row, _ = selected
        position = (canonical_row_idx, target_index)
        if position in seen_positions:
            statuses["duplicate_canonical_target"] += 1
            continue
        messages = normalize_messages(list(canonical_row["messages"]), target_index)
        roles = [message["role"] for message in messages]
        if not messages or roles[-1] != "user":
            statuses["target_not_final_user"] += 1
            continue
        if canonical_text(messages[-1]["content"]) != canonical_text(release_row.get("target_text")):
            statuses["target_text_mismatch"] += 1
            continue
        if "assistant" not in roles[:-1]:
            statuses["no_prior_assistant"] += 1
            continue
        if any(left == right for left, right in zip(roles, roles[1:])):
            statuses["nonalternating_prefix"] += 1
            continue

        seen_positions.add(position)
        cl100k_tokens = token_count(messages, "cl100k_base")
        o200k_tokens = token_count(messages, "o200k_base")
        reserve = args.reserved_output_tokens + args.safety_margin_tokens
        pair_id = hashlib.sha256(
            f"{canonical_row_idx}\x1f{target_index}\x1f{canonical_text(messages[-1]['content'])}".encode("utf-8")
        ).hexdigest()
        eligible.append(
            {
                "pair_id": pair_id,
                "release_row_idx": int(release_row["release_row_idx"]),
                "message_hash": release_row.get("message_hash"),
                "source": release_row.get("source"),
                "conversation_hash": hashlib.sha256(str(canonical_row.get("conversation_id", "")).encode("utf-8")).hexdigest(),
                "canonical_row_idx": canonical_row_idx,
                "canonical_target_message_index": target_index,
                "match_status": match_status,
                "messages": messages,
                "message_count": len(messages),
                "prior_user_messages": sum(role == "user" for role in roles[:-1]),
                "prior_assistant_messages": sum(role == "assistant" for role in roles[:-1]),
                "word_count": sum(len(message["content"].split()) for message in messages),
                "character_count": sum(len(message["content"]) for message in messages),
                "cl100k_tokens_conservative": cl100k_tokens,
                "o200k_tokens_conservative": o200k_tokens,
                "fits_16k_cl100k": cl100k_tokens + reserve <= 16_385,
                "fits_128k_cl100k": cl100k_tokens + reserve <= 128_000,
                "fits_128k_o200k": o200k_tokens + reserve <= 128_000,
                "fits_200k_o200k": o200k_tokens + reserve <= 200_000,
            }
        )
        statuses[f"included_{match_status}"] += 1

    args.out_dir.mkdir(parents=True, exist_ok=True)
    private = args.out_dir / "private"
    private.mkdir(parents=True, exist_ok=True)
    with (private / "complete_history_rows.jsonl").open("w", encoding="utf-8") as handle:
        for row in eligible:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    public = pd.DataFrame(eligible).drop(columns="messages")
    public.to_parquet(args.out_dir / "complete_history_index.parquet", index=False)
    fit_columns = ["fits_16k_cl100k", "fits_128k_cl100k", "fits_128k_o200k", "fits_200k_o200k"]
    manifest = {
        "release_rows": len(release),
        "canonical_transcripts": len(canonical),
        "eligible_complete_history_targets": len(eligible),
        "source_conversations": int(public["conversation_hash"].nunique()),
        "statuses": dict(sorted(statuses.items())),
        "role_requirement": "strict alternation through an exact user target with >=1 prior assistant turn",
        "truncation_policy": "none; model runs select only rows whose fit flag is true",
        "reserved_output_tokens": args.reserved_output_tokens,
        "safety_margin_tokens": args.safety_margin_tokens,
        "fit_counts": {column: int(public[column].sum()) for column in fit_columns},
        "message_count": public["message_count"].describe(percentiles=[0.5, 0.9, 0.95, 0.99]).to_dict(),
        "cl100k_tokens": public["cl100k_tokens_conservative"].describe(percentiles=[0.5, 0.9, 0.95, 0.99]).to_dict(),
        "release_sha256": hashlib.sha256(args.release.read_bytes()).hexdigest(),
        "canonical_sha256": hashlib.sha256(args.canonical_conversations.read_bytes()).hexdigest(),
    }
    (args.out_dir / "complete_history_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
