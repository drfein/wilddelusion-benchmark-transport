#!/usr/bin/env python3
"""Build natural near-miss controls from WildDelusion verifier exclusions."""

from __future__ import annotations

import argparse
import ast
import csv
import hashlib
import json
import math
import re
import sys
from collections import Counter, defaultdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

NONSINCERE_EXCLUSIONS = {
    "dreams",
    "fiction_or_story",
    "joke_or_absurd",
    "roleplay",
    "third_party_or_quoted",
    "translation_or_text_task",
}

INPUTS = (
    (
        "current_embedding_retrieval",
        Path("data/verification/whitened_top3k_verified_gpt54mini.jsonl"),
    ),
    (
        "historical_combined_verifier",
        Path("data/verification/wilddelusion_combined_judged_gpt54mini.jsonl"),
    ),
)

RELEASE_PATH = Path("data/releases/WildDelusionCombined/train.jsonl")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root",
        type=Path,
        default=Path(__file__).resolve().parents[1],
        help="Repository root.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("data/controls/natural_near_miss_controls_20260730"),
        help="Output path relative to --root unless absolute.",
    )
    parser.add_argument(
        "--human-reviews",
        type=Path,
        default=None,
        help=(
            "Optional earlier manual-review CSV. If omitted, the builder checks "
            "~/Desktop/review_annotations_with_conversations.csv."
        ),
    )
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def normalize_text(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value or "")).strip().casefold()


def normalize_role(role: Any) -> str:
    value = str(role or "").strip().lower()
    return {
        "bot": "assistant",
        "human": "user",
        "llm": "assistant",
    }.get(value, value)


def normalize_messages(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    normalized = []
    for message in messages:
        output = dict(message)
        output["role"] = normalize_role(message.get("role"))
        output["content"] = str(message.get("content") or "")
        normalized.append(output)
    return normalized


def conversation_hash(messages: list[dict[str, Any]]) -> str:
    canonical = [
        [normalize_role(message.get("role")), normalize_text(message.get("content"))]
        for message in messages
    ]
    payload = json.dumps(canonical, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_provenance(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
            return parsed if isinstance(parsed, dict) else {}
        except json.JSONDecodeError:
            return {}
    return {}


def canonical_source(row: dict[str, Any], cohort: str) -> str:
    if cohort == "current_embedding_retrieval":
        return str(row.get("source") or "unknown")

    provenance = parse_provenance(row.get("hf_provenance"))
    source_corpus = str(provenance.get("source_corpus") or "")
    if source_corpus == "sharechat":
        platform = provenance.get("platform") or provenance.get("config") or "unknown"
        return f"sharechat_{platform}"
    if source_corpus == "lmsys":
        return "lmsys_chat_1m"
    if source_corpus == "existing_wilddelusion_combined":
        return "legacy_existing_wilddelusion"
    return str(row.get("source") or source_corpus or "unknown")


def finite_or_none(value: Any) -> Any:
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def clean_json(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): clean_json(item) for key, item in value.items()}
    if isinstance(value, list):
        return [clean_json(item) for item in value]
    return finite_or_none(value)


def build_record(
    row: dict[str, Any],
    cohort: str,
    input_path: Path,
) -> dict[str, Any]:
    messages = normalize_messages(row.get("messages") or [])
    target_index = row.get("target_message_index")
    if not isinstance(target_index, int) or not 0 <= target_index < len(messages):
        raise ValueError(
            f"Invalid target index {target_index!r} for {row.get('conversation_id')!r}"
        )

    target_message = messages[target_index]
    if target_message["role"] != "user":
        raise ValueError(
            f"Target is {target_message['role']!r}, not user, for "
            f"{row.get('conversation_id')!r}:{target_index}"
        )

    conv_hash = conversation_hash(messages)
    candidate_target = row.get("target_text") or row.get("flagged_text")
    actual_target = target_message["content"]
    provenance = parse_provenance(row.get("hf_provenance"))
    if not provenance:
        provenance = parse_provenance(row.get("legacy_provenance"))

    origin = {
        "cohort": cohort,
        "input_path": str(input_path),
        "source_value": row.get("source"),
        "conversation_id": row.get("conversation_id"),
        "message_hash": row.get("message_hash"),
        "row_offset": row.get("row_offset"),
        "row_index": row.get("row_index"),
    }

    return clean_json(
        {
            "control_id": f"{conv_hash}:{target_index}",
            "conversation_hash": conv_hash,
            "conversation_id": row.get("conversation_id"),
            "source": canonical_source(row, cohort),
            "split": row.get("split") or "train",
            "target_message_index": target_index,
            "target_role": "user",
            "target_text": actual_target,
            "candidate_target_text": candidate_target,
            "candidate_target_was_exact": (
                candidate_target is None
                or normalize_text(candidate_target) == normalize_text(actual_target)
            ),
            "messages": messages,
            "judge_label": row.get("judge_label"),
            "judge_exclusion": row.get("judge_exclusion"),
            "judge_confidence": row.get("judge_confidence"),
            "judge_rationale": row.get("judge_rationale"),
            "judge_supporting_quotes": row.get("judge_supporting_quotes") or [],
            "judge_model": row.get("judge_model"),
            "annotation_score": row.get("annotation_score")
            if row.get("annotation_score") is not None
            else row.get("gpt_score"),
            "annotation_rationale": row.get("annotation_rationale"),
            "retrieval_score": row.get("retrieval_score"),
            "probe_score": row.get("probe_score"),
            "provenance": provenance,
            "origins": [origin],
        }
    )


def preference_key(record: dict[str, Any]) -> tuple[int, float, float]:
    cohort = record["origins"][0]["cohort"]
    cohort_priority = 1 if cohort == "current_embedding_retrieval" else 0
    confidence = float(record.get("judge_confidence") or 0)
    score = float(record.get("annotation_score") or 0)
    return cohort_priority, confidence, score


def deduplicate(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        grouped[record["control_id"]].append(record)

    output = []
    for control_id, group in grouped.items():
        primary = max(group, key=preference_key)
        primary = dict(primary)
        primary["origins"] = [
            origin for record in group for origin in record.get("origins", [])
        ]
        alternate_judgments = []
        for record in group:
            if record is primary:
                continue
            judgment = {
                "judge_label": record.get("judge_label"),
                "judge_exclusion": record.get("judge_exclusion"),
                "judge_confidence": record.get("judge_confidence"),
                "judge_rationale": record.get("judge_rationale"),
                "origin": record.get("origins", [{}])[0],
            }
            if (
                judgment["judge_label"] != primary.get("judge_label")
                or judgment["judge_exclusion"] != primary.get("judge_exclusion")
            ):
                alternate_judgments.append(judgment)
        primary["alternate_judgments"] = alternate_judgments
        primary["control_id"] = control_id
        output.append(primary)
    return output


def release_indexes(
    release_rows: list[dict[str, Any]],
) -> tuple[set[str], set[str], set[str]]:
    endpoints: set[str] = set()
    conversations: set[str] = set()
    target_texts: set[str] = set()
    for row in release_rows:
        messages = normalize_messages(row.get("messages") or [])
        target_index = row.get("target_message_index")
        if not isinstance(target_index, int) or not 0 <= target_index < len(messages):
            continue
        conv_hash = conversation_hash(messages)
        endpoints.add(f"{conv_hash}:{target_index}")
        conversations.add(conv_hash)
        target_texts.add(normalize_text(messages[target_index].get("content")))
    return endpoints, conversations, target_texts


def release_rows_by_target_text(
    release_rows: list[dict[str, Any]],
) -> dict[str, list[dict[str, Any]]]:
    output: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in release_rows:
        messages = normalize_messages(row.get("messages") or [])
        target_index = row.get("target_message_index")
        if not isinstance(target_index, int) or not 0 <= target_index < len(messages):
            continue
        target_text = messages[target_index]["content"]
        output[normalize_text(target_text)].append(
            clean_json(
                {
                    "conversation_hash": conversation_hash(messages),
                    "conversation_id": row.get("conversation_id"),
                    "source": row.get("source"),
                    "discovery_split": row.get("discovery_split"),
                    "target_message_index": target_index,
                    "target_text": target_text,
                    "messages": messages,
                    "judge_confidence": row.get("judge_confidence"),
                    "judge_rationale": row.get("judge_rationale"),
                }
            )
        )
    return output


def parse_tags(value: Any) -> list[str]:
    if isinstance(value, list):
        return [str(item) for item in value]
    if not isinstance(value, str) or not value.strip():
        return []
    try:
        parsed = ast.literal_eval(value)
    except (SyntaxError, ValueError):
        return []
    return [str(item) for item in parsed] if isinstance(parsed, list) else []


def attach_human_reviews(
    records: list[dict[str, Any]], human_review_path: Path | None
) -> dict[str, Any]:
    if human_review_path is None or not human_review_path.exists():
        for record in records:
            record["human_review"] = None
        return {
            "available": False,
            "path": str(human_review_path) if human_review_path else None,
            "rows": 0,
            "matched_rejection_rows": 0,
        }

    csv.field_size_limit(sys.maxsize)
    with human_review_path.open(newline="", encoding="utf-8-sig") as handle:
        reviews = list(csv.DictReader(handle))

    review_index: dict[tuple[str, str], dict[str, Any]] = {}
    for review in reviews:
        key = (
            str(review.get("conversation_id") or ""),
            normalize_text(review.get("flagged_text")),
        )
        review_index[key] = review

    matched = 0
    for record in records:
        candidate_texts = {
            normalize_text(record.get("target_text")),
            normalize_text(record.get("candidate_target_text")),
        }
        keys = [
            (str(origin.get("conversation_id") or ""), candidate_text)
            for origin in record.get("origins", [])
            for candidate_text in candidate_texts
            if candidate_text
        ]
        review = next((review_index[key] for key in keys if key in review_index), None)
        if review is None:
            record["human_review"] = None
            continue
        matched += 1
        record["human_review"] = clean_json(
            {
                "decision": review.get("decision"),
                "notes": review.get("notes") or None,
                "tags": parse_tags(review.get("annotation_tags")),
                "annotation_id": review.get("annotation_id"),
                "annotation_created_at": review.get("annotation_created_at"),
            }
        )

    return {
        "available": True,
        "path": str(human_review_path),
        "rows": len(reviews),
        "matched_rejection_rows": matched,
    }


def message_signatures(messages: list[dict[str, Any]]) -> set[tuple[str, str]]:
    return {
        (
            normalize_role(message.get("role")),
            normalize_text(message.get("content")),
        )
        for message in messages
        if normalize_text(message.get("content"))
    }


def likely_same_underlying_conversation(
    control: dict[str, Any], positive: dict[str, Any]
) -> bool:
    """Detect alternate reconstructions of the same source transcript."""
    shared_messages = message_signatures(control["messages"]) & message_signatures(
        positive["messages"]
    )
    # The shared target contributes one match; another exact turn is strong
    # evidence that the two differently keyed rows are transcript versions.
    return len(shared_messages) >= 2


def annotate_release_overlap(
    records: list[dict[str, Any]],
    release_conversations: set[str],
    release_target_texts: set[str],
) -> None:
    for record in records:
        record["same_conversation_as_release_positive"] = (
            record["conversation_hash"] in release_conversations
        )
        record["target_text_matches_release_positive_elsewhere"] = (
            normalize_text(record["target_text"]) in release_target_texts
        )


def record_sort_key(record: dict[str, Any]) -> tuple[Any, ...]:
    return (
        str(record.get("judge_exclusion") or ""),
        str(record.get("source") or ""),
        str(record.get("conversation_hash") or ""),
        int(record.get("target_message_index") or 0),
    )


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(clean_json(row), ensure_ascii=False) + "\n")


def format_conversation(messages: list[dict[str, Any]], target_index: int) -> str:
    chunks = []
    for index, message in enumerate(messages):
        marker = " [TARGET]" if index == target_index else ""
        chunks.append(
            f"{index:03d} {message.get('role', '').upper()}{marker}:\n"
            f"{message.get('content', '')}"
        )
    return "\n\n".join(chunks)


def write_review_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames = [
        "control_id",
        "judge_exclusion",
        "judge_confidence",
        "source",
        "conversation_id",
        "target_message_index",
        "target_text",
        "judge_rationale",
        "annotation_score",
        "retrieval_score",
        "same_conversation_as_release_positive",
        "target_text_matches_release_positive_elsewhere",
        "human_review_decision",
        "human_review_tags",
        "human_review_notes",
        "origin_cohorts",
        "full_conversation",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "control_id": row["control_id"],
                    "judge_exclusion": row["judge_exclusion"],
                    "judge_confidence": row["judge_confidence"],
                    "source": row["source"],
                    "conversation_id": row["conversation_id"],
                    "target_message_index": row["target_message_index"],
                    "target_text": row["target_text"],
                    "judge_rationale": row["judge_rationale"],
                    "annotation_score": row["annotation_score"],
                    "retrieval_score": row["retrieval_score"],
                    "same_conversation_as_release_positive": row[
                        "same_conversation_as_release_positive"
                    ],
                    "target_text_matches_release_positive_elsewhere": row[
                        "target_text_matches_release_positive_elsewhere"
                    ],
                    "human_review_decision": (
                        row.get("human_review") or {}
                    ).get("decision"),
                    "human_review_tags": ";".join(
                        (row.get("human_review") or {}).get("tags") or []
                    ),
                    "human_review_notes": (
                        row.get("human_review") or {}
                    ).get("notes"),
                    "origin_cohorts": ";".join(
                        sorted({origin["cohort"] for origin in row["origins"]})
                    ),
                    "full_conversation": format_conversation(
                        row["messages"], row["target_message_index"]
                    ),
                }
            )


def group_conversations(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[row["conversation_hash"]].append(row)

    conversations = []
    for conv_hash, group in grouped.items():
        representative = max(group, key=preference_key)
        conversations.append(
            {
                "conversation_hash": conv_hash,
                "conversation_ids": sorted(
                    {
                        str(origin.get("conversation_id"))
                        for row in group
                        for origin in row["origins"]
                        if origin.get("conversation_id") is not None
                    }
                ),
                "sources": sorted({row["source"] for row in group}),
                "messages": representative["messages"],
                "control_targets": [
                    {
                        "control_id": row["control_id"],
                        "target_message_index": row["target_message_index"],
                        "target_text": row["target_text"],
                        "judge_exclusion": row["judge_exclusion"],
                        "judge_confidence": row["judge_confidence"],
                        "judge_rationale": row["judge_rationale"],
                    }
                    for row in sorted(group, key=record_sort_key)
                ],
            }
        )
    return sorted(conversations, key=lambda row: row["conversation_hash"])


def representative_examples(rows: list[dict[str, Any]], per_category: int = 3) -> str:
    lines = [
        "# Natural near-miss control examples",
        "",
        "These examples were candidate delusion-like user turns in isolation but were",
        "rejected after the verifier read the full conversation.",
        "",
    ]
    for exclusion in sorted(NONSINCERE_EXCLUSIONS):
        candidates = sorted(
            [row for row in rows if row["judge_exclusion"] == exclusion],
            key=lambda row: (
                float(row.get("judge_confidence") or 0),
                float(row.get("annotation_score") or 0),
            ),
            reverse=True,
        )
        selected = []
        seen_conversations = set()
        for row in candidates:
            if row["conversation_hash"] in seen_conversations:
                continue
            selected.append(row)
            seen_conversations.add(row["conversation_hash"])
            if len(selected) == per_category:
                break

        lines.extend([f"## {exclusion}", ""])
        for row in selected:
            preview = re.sub(r"\s+", " ", row["target_text"]).strip()
            if len(preview) > 600:
                preview = preview[:597].rstrip() + "..."
            lines.extend(
                [
                    f"- `{row['control_id']}`",
                    f"  Target: {preview}",
                    f"  Why excluded: {row['judge_rationale']}",
                    "",
                ]
            )
    return "\n".join(lines)


def counts(rows: list[dict[str, Any]], field: str) -> dict[str, int]:
    return dict(sorted(Counter(str(row.get(field)) for row in rows).items()))


def main() -> None:
    args = parse_args()
    root = args.root.resolve()
    out_dir = args.out_dir if args.out_dir.is_absolute() else root / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    by_exclusion_dir = out_dir / "by_exclusion"
    by_exclusion_dir.mkdir(exist_ok=True)

    input_metadata = {}
    raw_records = []
    for cohort, relative_path in INPUTS:
        path = root / relative_path
        rows = read_jsonl(path)
        input_metadata[cohort] = {
            "path": str(relative_path),
            "rows": len(rows),
            "sha256": sha256_file(path),
        }
        for row in rows:
            if row.get("judge_label") == "positive":
                continue
            if not row.get("judge_exclusion") or row.get("judge_exclusion") == "none":
                continue
            raw_records.append(build_record(row, cohort, relative_path))

    release_path = root / RELEASE_PATH
    release_rows = read_jsonl(release_path)
    release_endpoints, release_conversations, release_target_texts = release_indexes(
        release_rows
    )
    positive_rows_by_text = release_rows_by_target_text(release_rows)

    all_rejections = deduplicate(raw_records)
    exact_release_conflicts = sorted(
        [row for row in all_rejections if row["control_id"] in release_endpoints],
        key=record_sort_key,
    )
    for row in exact_release_conflicts:
        row["release_conflict_type"] = "exact_conversation_and_target"
    all_rejections = sorted(
        [row for row in all_rejections if row["control_id"] not in release_endpoints],
        key=record_sort_key,
    )
    reconstruction_conflicts = []
    retained_rejections = []
    for row in all_rejections:
        positive_matches = positive_rows_by_text.get(
            normalize_text(row["target_text"]), []
        )
        matching_versions = [
            positive
            for positive in positive_matches
            if likely_same_underlying_conversation(row, positive)
        ]
        if matching_versions:
            row["release_conflict_type"] = (
                "alternate_reconstruction_of_positive_endpoint"
            )
            row["matching_positive_reconstructions"] = matching_versions
            reconstruction_conflicts.append(row)
        else:
            retained_rejections.append(row)
    all_rejections = retained_rejections
    release_conflicts = sorted(
        exact_release_conflicts + reconstruction_conflicts,
        key=record_sort_key,
    )
    annotate_release_overlap(
        all_rejections, release_conversations, release_target_texts
    )
    annotate_release_overlap(
        release_conflicts, release_conversations, release_target_texts
    )
    human_review_path = args.human_reviews
    if human_review_path is None:
        desktop_review_path = (
            Path.home() / "Desktop" / "review_annotations_with_conversations.csv"
        )
        human_review_path = (
            desktop_review_path if desktop_review_path.exists() else None
        )
    human_review_metadata = attach_human_reviews(
        all_rejections + release_conflicts, human_review_path
    )

    nonsincere = sorted(
        [
            row
            for row in all_rejections
            if row["judge_exclusion"] in NONSINCERE_EXCLUSIONS
        ],
        key=record_sort_key,
    )
    strict_nonsincere = [
        row for row in nonsincere if not row["same_conversation_as_release_positive"]
    ]
    mixed_conversation_controls = [
        row for row in nonsincere if row["same_conversation_as_release_positive"]
    ]
    matched_text_controls = [
        row
        for row in nonsincere
        if row["target_text_matches_release_positive_elsewhere"]
        and not row["same_conversation_as_release_positive"]
    ]
    high_confidence_strict_controls = [
        row
        for row in strict_nonsincere
        if float(row.get("judge_confidence") or 0) >= 0.95
        and (row.get("human_review") or {}).get("decision") != "saved"
    ]
    human_confirmed_controls = [
        row
        for row in nonsincere
        if (row.get("human_review") or {}).get("decision") == "rejected"
    ]
    human_verifier_disagreements = [
        row
        for row in nonsincere
        if (row.get("human_review") or {}).get("decision") == "saved"
    ]
    matched_text_pairs = [
        {
            "control": row,
            "positive_contexts": [
                positive
                for positive in positive_rows_by_text[
                    normalize_text(row["target_text"])
                ]
                if positive["conversation_hash"] != row["conversation_hash"]
            ],
        }
        for row in matched_text_controls
    ]
    conversations = group_conversations(nonsincere)

    write_jsonl(out_dir / "nonsincere_controls.jsonl", nonsincere)
    write_jsonl(
        out_dir / "strict_nonsincere_controls.jsonl", strict_nonsincere
    )
    write_jsonl(
        out_dir / "mixed_conversation_controls.jsonl",
        mixed_conversation_controls,
    )
    write_jsonl(
        out_dir / "matched_text_context_controls.jsonl", matched_text_controls
    )
    write_jsonl(
        out_dir / "matched_text_context_pairs.jsonl", matched_text_pairs
    )
    write_jsonl(
        out_dir / "high_confidence_strict_controls.jsonl",
        high_confidence_strict_controls,
    )
    write_jsonl(
        out_dir / "human_confirmed_controls.jsonl", human_confirmed_controls
    )
    write_jsonl(
        out_dir / "human_verifier_disagreements.jsonl",
        human_verifier_disagreements,
    )
    write_jsonl(out_dir / "all_contextual_rejections.jsonl", all_rejections)
    write_jsonl(out_dir / "adjudication_conflicts.jsonl", release_conflicts)
    write_jsonl(out_dir / "conversations.jsonl", conversations)
    write_review_csv(out_dir / "review.csv", nonsincere)

    for exclusion in sorted(NONSINCERE_EXCLUSIONS):
        write_jsonl(
            by_exclusion_dir / f"{exclusion}.jsonl",
            [row for row in nonsincere if row["judge_exclusion"] == exclusion],
        )

    (out_dir / "examples.md").write_text(
        representative_examples(nonsincere), encoding="utf-8"
    )

    summary = {
        "generated_at": datetime.now(UTC).isoformat(),
        "definition": (
            "Natural hard negatives that appeared delusion-like at the candidate "
            "message level but were rejected after full-conversation verification "
            "because the user was not sincerely endorsing the claim."
        ),
        "nonsincere_exclusions": sorted(NONSINCERE_EXCLUSIONS),
        "inputs": input_metadata,
        "release_reference": {
            "path": str(RELEASE_PATH),
            "rows": len(release_rows),
            "sha256": sha256_file(release_path),
        },
        "human_review_reference": human_review_metadata,
        "counts": {
            "raw_rejected_rows_across_inputs": len(raw_records),
            "deduplicated_contextual_rejections_before_release_conflict_removal": (
                len(all_rejections) + len(release_conflicts)
            ),
            "all_contextual_rejections": len(all_rejections),
            "nonsincere_control_endpoints": len(nonsincere),
            "nonsincere_control_conversations": len(conversations),
            "strict_nonsincere_control_endpoints": len(strict_nonsincere),
            "mixed_conversation_control_endpoints": len(
                mixed_conversation_controls
            ),
            "matched_target_text_in_distinct_positive_context": len(
                matched_text_controls
            ),
            "high_confidence_strict_controls": len(
                high_confidence_strict_controls
            ),
            "human_confirmed_controls": len(human_confirmed_controls),
            "human_verifier_disagreements": len(human_verifier_disagreements),
            "controls_without_human_review": sum(
                row.get("human_review") is None for row in nonsincere
            ),
            "exact_endpoint_adjudication_conflicts": len(exact_release_conflicts),
            "alternate_reconstruction_adjudication_conflicts": len(
                reconstruction_conflicts
            ),
            "all_release_adjudication_conflicts": len(release_conflicts),
            "same_conversation_as_release_positive": sum(
                bool(row["same_conversation_as_release_positive"])
                for row in nonsincere
            ),
            "same_target_text_as_release_positive_in_another_context": sum(
                bool(row["target_text_matches_release_positive_elsewhere"])
                and not bool(row["same_conversation_as_release_positive"])
                for row in nonsincere
            ),
        },
        "nonsincere_by_exclusion": counts(nonsincere, "judge_exclusion"),
        "nonsincere_by_source": counts(nonsincere, "source"),
        "all_rejections_by_exclusion": counts(
            all_rejections, "judge_exclusion"
        ),
        "validation": {
            "all_targets_are_user_messages": all(
                row["messages"][row["target_message_index"]]["role"] == "user"
                for row in nonsincere
            ),
            "all_target_texts_exactly_match_indexed_message": all(
                row["target_text"]
                == row["messages"][row["target_message_index"]]["content"]
                for row in nonsincere
            ),
            "unique_control_ids": len({row["control_id"] for row in nonsincere})
            == len(nonsincere),
            "no_exact_endpoint_in_current_positive_release": not any(
                row["control_id"] in release_endpoints for row in nonsincere
            ),
        },
        "files": {
            "primary": "nonsincere_controls.jsonl",
            "strict_conversation_negatives": "strict_nonsincere_controls.jsonl",
            "mixed_conversation_controls": "mixed_conversation_controls.jsonl",
            "matched_text_controls": "matched_text_context_controls.jsonl",
            "matched_text_pairs": "matched_text_context_pairs.jsonl",
            "conservative_subset": "high_confidence_strict_controls.jsonl",
            "human_confirmed": "human_confirmed_controls.jsonl",
            "human_disagreements": "human_verifier_disagreements.jsonl",
            "review": "review.csv",
            "grouped_conversations": "conversations.jsonl",
            "all_rejections": "all_contextual_rejections.jsonl",
            "judge_conflicts": "adjudication_conflicts.jsonl",
            "examples": "examples.md",
            "category_subsets": "by_exclusion/*.jsonl",
        },
    }
    (out_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    readme = """# Natural near-miss controls

The primary file, `nonsincere_controls.jsonl`, contains naturally occurring
candidate turns that looked delusion-like in isolation but were rejected after
full-conversation verification as role-play, fiction/story, joking/absurdity,
translation or rewriting, third-party quotation, or dream reporting.

Each row is one user target turn and includes the complete conversation,
the exact indexed target text, verifier rationale and confidence, retrieval
metadata, and provenance. `conversations.jsonl` groups multiple target turns
from the same transcript. `review.csv` is a spreadsheet-friendly rendering with
the complete conversation in one field.

`strict_nonsincere_controls.jsonl` excludes any control endpoint whose full
conversation also contains a target retained in the current positive release.
Use this for clean conversation-level negatives. The complementary
`mixed_conversation_controls.jsonl` is useful for within-conversation
attribution. `matched_text_context_controls.jsonl` contains the especially
useful cases where the exact same target wording occurs in a different positive
context.

`high_confidence_strict_controls.jsonl` is the conservative machine-labeled
subset: verifier confidence is at least 0.95, the conversation contains no
released positive target, and no earlier human reviewer saved the row.
`human_confirmed_controls.jsonl` contains controls manually rejected in the
earlier single-reviewer annotation pass. The complementary
`human_verifier_disagreements.jsonl` must not be treated as clean negatives.

`all_contextual_rejections.jsonl` additionally includes ordinary plausible
content and insufficient-context exclusions. Rows that conflict with the
current positive release at the exact same conversation and target index are
not controls; they are isolated in `adjudication_conflicts.jsonl`.

Identical target wording in different conversations is intentionally retained:
those pairs are useful for measuring whether context changes classification.
"""
    (out_dir / "README.md").write_text(readme, encoding="utf-8")

    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
