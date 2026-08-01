from __future__ import annotations

import argparse
import hashlib
import json
import re
import time
import unicodedata
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import requests
from huggingface_hub import HfApi, get_token
from tqdm import tqdm


PRIMARY_SPLIT = "openai_embedding_whitened"
LEGACY_SPLIT = "legacy_probe_gpt52"
SOURCE_SPECS = {
    "sharechat_chatgpt": ("anoynsharechat/sharechat", "chatgpt"),
    "sharechat_grok": ("anoynsharechat/sharechat", "grok"),
    "wildchat_full": ("yuntian-deng/WildChat-4.8M-Full", "default"),
}
USER_ROLES = {"user", "human"}
ASSISTANT_ROLES = {"assistant", "llm", "gpt", "bot", "model", "ai"}
UNAVAILABLE_SOURCE_MESSAGE = "[Non-text response unavailable in source export]"


def canonical_text(value: object) -> str:
    text = unicodedata.normalize("NFKC", str(value or ""))
    return re.sub(r"\s+", " ", text).strip().casefold()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def normalize_role(value: object) -> str | None:
    role = str(value or "").strip().lower()
    if role in USER_ROLES:
        return "user"
    if role in ASSISTANT_ROLES:
        return "assistant"
    return None


def normalize_messages(messages: list[dict[str, Any]]) -> list[dict[str, str]]:
    output = []
    for message in messages:
        role = normalize_role(message.get("role") or message.get("from"))
        content = str(
            message.get("content")
            or message.get("value")
            or message.get("plain_text")
            or ""
        ).strip()
        if role and content:
            output.append({"role": role, "content": content})
    return output


def normalize_sharechat_messages(messages: list[dict[str, Any]]) -> list[dict[str, str]]:
    output = []
    for message in messages:
        role = normalize_role(message.get("role") or message.get("from"))
        if not role:
            continue
        content = str(
            message.get("content")
            or message.get("value")
            or message.get("plain_text")
            or ""
        ).strip()
        output.append({"role": role, "content": content or UNAVAILABLE_SOURCE_MESSAGE})
    return output


def transcript_hash(messages: list[dict[str, str]]) -> str:
    normalized = [(message["role"], canonical_text(message["content"])) for message in messages]
    return hashlib.sha256(json.dumps(normalized, ensure_ascii=False).encode("utf-8")).hexdigest()


class RowsClient:
    def __init__(
        self,
        revisions: dict[str, str],
        timeout: float = 90.0,
        token: str | None = None,
    ) -> None:
        self.revisions = revisions
        self.timeout = timeout
        self.headers = {"Authorization": f"Bearer {token}"} if token else {}

    def get(self, dataset: str, config: str, offset: int, length: int) -> list[dict[str, Any]]:
        params = {
            "dataset": dataset,
            "config": config,
            "split": "train",
            "offset": int(offset),
            "length": min(int(length), 100),
            "revision": self.revisions[dataset],
        }
        last_error: Exception | None = None
        for attempt in range(1, 9):
            try:
                response = requests.get(
                    "https://datasets-server.huggingface.co/rows",
                    params=params,
                    headers=self.headers,
                    timeout=self.timeout,
                )
                if response.status_code in {429, 500, 502, 503, 504}:
                    raise requests.HTTPError(f"retryable status {response.status_code}")
                response.raise_for_status()
                rows = response.json().get("rows") or []
                return [item for item in rows if isinstance(item.get("row"), dict)]
            except (requests.RequestException, ValueError) as error:
                last_error = error
                time.sleep(min(2**attempt, 30))
        raise RuntimeError(f"Rows API failed for {dataset}/{config}@{offset}: {last_error}")


def find_target(messages: list[dict[str, str]], target_text: object, preferred: int | None) -> int:
    target = canonical_text(target_text)
    if preferred is not None and 0 <= preferred < len(messages):
        if messages[preferred]["role"] == "user" and canonical_text(messages[preferred]["content"]) == target:
            return preferred
    matches = [
        index
        for index, message in enumerate(messages)
        if message["role"] == "user" and canonical_text(message["content"]) == target
    ]
    if len(matches) != 1:
        raise ValueError(f"Expected one target match, found {len(matches)}")
    return matches[0]


def fetch_target_row(client: RowsClient, source: str, offset: int) -> dict[str, Any]:
    dataset, config = SOURCE_SPECS[source]
    items = client.get(dataset, config, offset, 1)
    if len(items) != 1 or int(items[0]["row_idx"]) != offset:
        raise ValueError(f"Did not fetch exact source row {source}:{offset}")
    return items[0]["row"]


def fetch_sharechat_conversation(
    client: RowsClient,
    source: str,
    target_offset: int,
    target_row: dict[str, Any],
) -> tuple[list[dict[str, str]], list[int]]:
    dataset, config = SOURCE_SPECS[source]
    url = str(target_row["url"])
    # turns_count is inconsistent for some conversations with branches or
    # omitted nodes. Discover the actual boundaries using the stable URL and
    # physically contiguous source rows instead of assuming two rows per turn.
    chunk_size = 100
    first_chunk = (target_offset // chunk_size) * chunk_size
    chunks: dict[int, list[dict[str, Any]]] = {
        first_chunk: client.get(dataset, config, first_chunk, chunk_size)
    }
    cursor = first_chunk
    while cursor > 0 and chunks[cursor] and str(chunks[cursor][0]["row"].get("url") or "") == url:
        cursor = max(0, cursor - chunk_size)
        chunks[cursor] = client.get(dataset, config, cursor, chunk_size)
    cursor = first_chunk
    while chunks[cursor] and str(chunks[cursor][-1]["row"].get("url") or "") == url:
        cursor += chunk_size
        chunks[cursor] = client.get(dataset, config, cursor, chunk_size)
    items = [item for chunk in chunks.values() for item in chunk]
    selected = [
        (int(item["row_idx"]), item["row"])
        for item in items
        if str(item["row"].get("url") or "") == url
    ]
    selected.sort(key=lambda item: item[0])
    by_offset = {offset: row for offset, row in selected}
    ordered = [(offset, by_offset[offset]) for offset in sorted(by_offset)]
    offsets = [offset for offset, _ in ordered]
    if not offsets or target_offset not in offsets:
        raise ValueError(f"Target offset missing from ShareChat conversation {url}")
    if offsets != list(range(offsets[0], offsets[-1] + 1)):
        raise ValueError(f"Non-contiguous ShareChat source rows for {url}")
    messages = normalize_sharechat_messages([row for _, row in ordered])
    if len(messages) != len(ordered):
        raise ValueError(f"Role/content normalization dropped messages for {url}")
    return messages, offsets


def extract_wildchat_messages(row: dict[str, Any]) -> list[dict[str, str]]:
    for key in ("conversation", "messages", "conversations", "turns"):
        value = row.get(key)
        if isinstance(value, list):
            messages = normalize_messages(value)
            if messages:
                return messages
    raise ValueError("WildChat source row contains no recognized conversation field")


def resolve_legacy(
    release_row: dict[str, Any],
    canonical_by_id: dict[str, dict[str, Any]],
) -> tuple[list[dict[str, str]], int]:
    conversation_id = str(release_row.get("conversation_id") or "")
    candidate = canonical_by_id.get(conversation_id)
    if candidate is None:
        raise ValueError(f"No archived canonical transcript for legacy row {conversation_id}")
    messages = normalize_messages(list(candidate["messages"]))
    target_index = find_target(messages, release_row.get("target_text"), int(release_row["target_message_index"]))
    return messages, target_index


def main() -> None:
    parser = argparse.ArgumentParser(description="Rehydrate every WildDelusionCombined transcript.")
    parser.add_argument("--release", type=Path, required=True)
    parser.add_argument("--canonical-conversations", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()

    release = pd.read_parquet(args.release).reset_index(names="release_row_idx")
    canonical = read_jsonl(args.canonical_conversations)
    canonical_by_id = {str(row.get("conversation_id") or ""): row for row in canonical}
    api = HfApi()
    revisions = {
        dataset: api.dataset_info(dataset).sha
        for dataset, _ in sorted(set(SOURCE_SPECS.values()))
    }
    client = RowsClient(revisions, token=get_token())
    args.out_dir.mkdir(parents=True, exist_ok=True)
    private_dir = args.out_dir / "private"
    private_dir.mkdir(parents=True, exist_ok=True)

    primary = release[release["discovery_split"] == PRIMARY_SPLIT]
    legacy = release[release["discovery_split"] == LEGACY_SPLIT]
    if len(primary) != 433 or len(legacy) != 89:
        raise ValueError(f"Unexpected discovery split counts: primary={len(primary)}, legacy={len(legacy)}")
    if primary["row_offset"].isna().any():
        raise ValueError("Every primary row must retain an upstream row offset.")

    target_cache_path = private_dir / "source_target_rows.jsonl"
    target_rows: dict[int, dict[str, Any]] = {}
    if target_cache_path.exists():
        for cached in read_jsonl(target_cache_path):
            target_rows[int(cached["release_row_idx"])] = cached["row"]
    pending_primary = [
        row for row in primary.itertuples(index=False)
        if int(row.release_row_idx) not in target_rows
    ]
    target_errors: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(fetch_target_row, client, str(row.source), int(row.row_offset)): int(row.release_row_idx)
            for row in pending_primary
        }
        with target_cache_path.open("a", encoding="utf-8") as cache_handle:
            for future in tqdm(as_completed(futures), total=len(futures), desc="Fetching source target rows"):
                release_row_idx = futures[future]
                try:
                    target_rows[release_row_idx] = future.result()
                    cache_handle.write(json.dumps({"release_row_idx": release_row_idx, "row": target_rows[release_row_idx]}, ensure_ascii=False) + "\n")
                    cache_handle.flush()
                except Exception as error:
                    target_errors.append({"release_row_idx": release_row_idx, "error": repr(error)})
    if len(target_rows) != len(primary):
        (args.out_dir / "target_fetch_errors.json").write_text(json.dumps(target_errors, indent=2) + "\n", encoding="utf-8")
        raise RuntimeError(f"Recovered {len(target_rows)}/{len(primary)} source target rows; rerun resumes the remainder")

    # Rehydrate each unique ShareChat URL once. WildChat already stores a full
    # conversation in the target source row.
    sharechat_jobs: dict[tuple[str, str], tuple[int, dict[str, Any]]] = {}
    for row in primary.itertuples(index=False):
        if str(row.source).startswith("sharechat_"):
            raw = target_rows[int(row.release_row_idx)]
            key = (str(row.source), str(raw["url"]))
            sharechat_jobs.setdefault(key, (int(row.row_offset), raw))
    conversation_cache_path = private_dir / "sharechat_conversations.jsonl"
    sharechat_conversations: dict[tuple[str, str], tuple[list[dict[str, str]], list[int]]] = {}
    if conversation_cache_path.exists():
        for cached in read_jsonl(conversation_cache_path):
            sharechat_conversations[(cached["source"], cached["url"])] = (cached["messages"], cached["offsets"])
    pending_jobs = {key: value for key, value in sharechat_jobs.items() if key not in sharechat_conversations}
    conversation_errors: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(fetch_sharechat_conversation, client, source, offset, raw): key
            for key, (offset, raw) in pending_jobs.items()
            for source in [key[0]]
        }
        with conversation_cache_path.open("a", encoding="utf-8") as cache_handle:
            for future in tqdm(as_completed(futures), total=len(futures), desc="Fetching full ShareChat transcripts"):
                key = futures[future]
                try:
                    sharechat_conversations[key] = future.result()
                    messages, offsets = sharechat_conversations[key]
                    cache_handle.write(json.dumps({"source": key[0], "url": key[1], "messages": messages, "offsets": offsets}, ensure_ascii=False) + "\n")
                    cache_handle.flush()
                except Exception as error:
                    conversation_errors.append({"source": key[0], "url": key[1], "error": repr(error)})
    if len(sharechat_conversations) != len(sharechat_jobs):
        (args.out_dir / "conversation_fetch_errors.json").write_text(json.dumps(conversation_errors, indent=2) + "\n", encoding="utf-8")
        raise RuntimeError(f"Recovered {len(sharechat_conversations)}/{len(sharechat_jobs)} ShareChat transcripts; rerun resumes the remainder")

    output_rows: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    for release_row in release.to_dict(orient="records"):
        try:
            original_target_index = int(release_row["target_message_index"])
            if release_row["discovery_split"] == LEGACY_SPLIT:
                messages, target_index = resolve_legacy(release_row, canonical_by_id)
                method = "archived_canonical_transcript"
                source_revision = "WildDelusion canonical sha256:" + hashlib.sha256(
                    args.canonical_conversations.read_bytes()
                ).hexdigest()
            else:
                source = str(release_row["source"])
                raw = target_rows[int(release_row["release_row_idx"])]
                if canonical_text(raw.get("plain_text")) and canonical_text(raw.get("plain_text")) != canonical_text(release_row["target_text"]):
                    raise ValueError("Upstream target row text does not match release target")
                if source.startswith("sharechat_"):
                    key = (source, str(raw["url"]))
                    messages, offsets = sharechat_conversations[key]
                    target_offset = int(release_row["row_offset"])
                    if target_offset not in offsets:
                        raise ValueError("Target source offset absent from rehydrated ShareChat transcript")
                    target_index = offsets.index(target_offset)
                    method = "pinned_sharechat_rows"
                elif source == "wildchat_full":
                    messages = extract_wildchat_messages(raw)
                    target_index = find_target(messages, release_row["target_text"], original_target_index)
                    method = "pinned_wildchat_row"
                else:
                    raise ValueError(f"Unsupported primary source: {source}")
                source_revision = revisions[SOURCE_SPECS[source][0]]

            if messages[target_index]["role"] != "user":
                raise ValueError("Resolved target is not a user message")
            if canonical_text(messages[target_index]["content"]) != canonical_text(release_row["target_text"]):
                raise ValueError("Resolved target text mismatch")
            # Keep the release's stable annotated target representation. Some
            # upstream rows retain paragraph breaks that the release collapsed.
            messages[target_index]["content"] = str(release_row["target_text"])
            if not any(message["role"] == "assistant" for message in messages):
                raise ValueError("Complete transcript has no assistant messages")
            updated = dict(release_row)
            updated.pop("release_row_idx", None)
            updated["messages"] = messages
            updated["target_message_index"] = target_index
            updated["history_complete"] = True
            updated["history_rehydration_method"] = method
            updated["history_source_revision"] = source_revision
            updated["history_message_count"] = len(messages)
            updated["history_sha256"] = transcript_hash(messages)
            updated["history_original_target_message_index"] = original_target_index
            updated["history_prefix_has_assistant"] = any(
                message["role"] == "assistant" for message in messages[:target_index]
            )
            unavailable_count = sum(
                message["content"] == UNAVAILABLE_SOURCE_MESSAGE for message in messages
            )
            updated["history_unavailable_message_count"] = unavailable_count
            updated["history_all_message_text_available"] = unavailable_count == 0
            reported_count = (
                int(raw["turns_count"]) * 2
                if release_row["discovery_split"] == PRIMARY_SPLIT
                and str(release_row["source"]).startswith("sharechat_")
                else None
            )
            updated["history_source_reported_message_count"] = reported_count
            updated["history_source_message_count_matches_metadata"] = (
                len(messages) == reported_count if reported_count is not None else None
            )
            output_rows.append(updated)
        except Exception as error:
            failures.append(
                {
                    "release_row_idx": int(release_row["release_row_idx"]),
                    "source": release_row.get("source"),
                    "message_hash": release_row.get("message_hash"),
                    "error": repr(error),
                }
            )

    if failures:
        (args.out_dir / "failures.json").write_text(json.dumps(failures, indent=2) + "\n", encoding="utf-8")
        raise RuntimeError(f"Failed to rehydrate {len(failures)}/{len(release)} rows")
    output = pd.DataFrame(output_rows)
    if len(output) != 522 or not output["history_complete"].all():
        raise RuntimeError("Rehydrated release is incomplete.")
    parquet = args.out_dir / "train-00000-of-00001.parquet"
    # Preserve every untouched nested Arrow field exactly. Round-tripping the
    # full frame through pandas can coerce nullable integers inside structs.
    original_table = pq.read_table(args.release)
    table_columns: dict[str, pa.Array | pa.ChunkedArray] = {}
    for field in original_table.schema:
        if field.name == "messages":
            table_columns[field.name] = pa.array(output["messages"].tolist(), type=field.type)
        elif field.name == "target_message_index":
            table_columns[field.name] = pa.array(
                output["target_message_index"].tolist(), type=field.type
            )
        else:
            table_columns[field.name] = original_table[field.name]
    for column in output.columns:
        if column not in table_columns:
            table_columns[column] = pa.array(output[column].tolist())
    pq.write_table(pa.table(table_columns), parquet)
    manifest = {
        "rows": len(output),
        "source_conversations": int(output[["source", "conversation_id"]].drop_duplicates().shape[0]),
        "all_targets_exact": True,
        "all_transcripts_have_assistant": True,
        "rows_with_all_message_text_available": int(output["history_all_message_text_available"].sum()),
        "rows_with_unavailable_message_text": int((~output["history_all_message_text_available"]).sum()),
        "unavailable_source_message_nodes": int(output["history_unavailable_message_count"].sum()),
        "rows_with_prior_assistant": int(output["history_prefix_has_assistant"].sum()),
        "rehydration_method_counts": output["history_rehydration_method"].value_counts().to_dict(),
        "source_target_counts": output["source"].value_counts().to_dict(),
        "message_count": output["history_message_count"].describe(percentiles=[0.5, 0.9, 0.95, 0.99]).to_dict(),
        "upstream_revisions": revisions,
        "input_release_sha256": hashlib.sha256(args.release.read_bytes()).hexdigest(),
        "canonical_archive_sha256": hashlib.sha256(args.canonical_conversations.read_bytes()).hexdigest(),
        "output_parquet_sha256": hashlib.sha256(parquet.read_bytes()).hexdigest(),
    }
    (args.out_dir / "rehydration_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
