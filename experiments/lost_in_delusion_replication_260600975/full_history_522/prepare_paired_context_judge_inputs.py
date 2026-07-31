#!/usr/bin/env python3
"""Freeze genuinely context-different paired generations for judging."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from io_utils import (  # noqa: E402
    canonical_json_sha256,
    read_jsonl,
    write_jsonl,
    write_manifest,
)


EXPECTED_PROTOCOL = "single_engine_unique_tokenized_prompt_cache_v1"
EXPECTED_MODELS = {
    "allenai/Olmo-3-7B-Instruct",
    "meta-llama/Llama-3.1-8B-Instruct",
}
OUTPUT_PROTOCOL = (
    "genuinely_context_different_delusion_pairs_shared_judge_context_v2"
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def pair_key(row: dict[str, Any]) -> tuple[str, str, str]:
    return str(row["model"]), str(row["pair_id"]), str(row["condition"])


def attach_shared_judge_context(
    history: dict[str, Any], last_only: dict[str, Any]
) -> list[dict[str, Any]]:
    shared_context = history["messages_used"]
    shared_context_sha256 = canonical_json_sha256(shared_context)
    selected = []
    for generation in (history, last_only):
        judged = dict(generation)
        judged.update(
            {
                "judge_messages": shared_context,
                "judge_context_source_generation_id": history[
                    "generation_id"
                ],
                "judge_context_sha256": shared_context_sha256,
            }
        )
        selected.append(judged)
    return selected


def audit_generation_file(path: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows = read_jsonl(path)
    manifest_path = path.with_suffix(path.suffix + ".manifest.json")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    models = {row.get("model") for row in rows}
    scopes = {row.get("paired_ablation_scope") for row in rows}
    checks = {
        "rows": len(rows) == 623,
        "model": len(models) == 1 and next(iter(models), None) in EXPECTED_MODELS,
        "scope": len(scopes) == 1
        and next(iter(scopes), None)
        in {"bounded_history", "last_user_only"},
        "protocol": all(
            row.get("paired_generation_protocol") == EXPECTED_PROTOCOL
            for row in rows
        ),
        "responses": all(
            row.get("response") and not row.get("generation_error")
            for row in rows
        ),
        "generation_ids": len({row.get("generation_id") for row in rows})
        == 623,
        "manifest_protocol": manifest.get("protocol") == EXPECTED_PROTOCOL,
        "manifest_rows": manifest.get("successful_rows") == 623,
        "manifest_output_hash": manifest.get("output_sha256")
        == sha256_file(path),
    }
    if not all(checks.values()):
        raise ValueError(
            f"{path}: paired-generation audit failed "
            + json.dumps(checks, sort_keys=True)
        )
    return rows, {
        "path": str(path),
        "sha256": sha256_file(path),
        "manifest_path": str(manifest_path),
        "manifest_sha256": sha256_file(manifest_path),
        "model": next(iter(models)),
        "scope": next(iter(scopes)),
        "checks": checks,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--generations", type=Path, nargs=4, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    rows: list[dict[str, Any]] = []
    audits = []
    file_keys: set[tuple[str, str]] = set()
    for path in args.generations:
        file_rows, audit = audit_generation_file(path)
        key = audit["model"], audit["scope"]
        if key in file_keys:
            raise ValueError(f"Duplicate model/scope artifact: {key}")
        file_keys.add(key)
        rows.extend(file_rows)
        audits.append(audit)
    expected_file_keys = {
        (model, scope)
        for model in EXPECTED_MODELS
        for scope in ("bounded_history", "last_user_only")
    }
    if file_keys != expected_file_keys:
        raise ValueError("The four paired model/scope artifacts are incomplete")

    grouped: dict[tuple[str, str, str], dict[str, dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(pair_key(row), {})[
            str(row["paired_ablation_scope"])
        ] = row
    if len(grouped) != 1246 or any(len(scopes) != 2 for scopes in grouped.values()):
        raise ValueError("Paired generation coverage is incomplete or duplicated")

    selected = []
    selected_pairs_by_model: dict[str, set[str]] = {
        model: set() for model in EXPECTED_MODELS
    }
    identical_prompt_pairs_by_model: dict[str, int] = {
        model: 0 for model in EXPECTED_MODELS
    }
    for (model, pair_id, condition), scopes in grouped.items():
        history = scopes["bounded_history"]
        last_only = scopes["last_user_only"]
        if history["messages_used"][-1] != last_only["messages_used"][-1]:
            raise ValueError(f"{model}/{pair_id}: final user message changed")
        prompt_identical = (
            history["prompt_token_ids_sha256"]
            == last_only["prompt_token_ids_sha256"]
        )
        if prompt_identical:
            identical_prompt_pairs_by_model[model] += 1
            if history["response"] != last_only["response"]:
                raise ValueError(
                    f"{model}/{pair_id}: cached identical prompts diverged"
                )
        if condition == "delusion" and not prompt_identical:
            if len(history["messages_used"]) <= 1:
                raise ValueError(
                    f"{model}/{pair_id}: prompt differs without retained history"
                )
            selected.extend(attach_shared_judge_context(history, last_only))
            selected_pairs_by_model[model].add(pair_id)

    pair_sets = list(selected_pairs_by_model.values())
    if len(pair_sets[0]) != 449 or any(pair_set != pair_sets[0] for pair_set in pair_sets[1:]):
        raise ValueError(
            "Expected the same 449 genuinely context-different delusion pairs "
            "for both models"
        )
    if len(selected) != 1796:
        raise ValueError(f"Expected 1,796 judge rows, found {len(selected)}")
    selected.sort(
        key=lambda row: (
            row["model"],
            row["pair_id"],
            row["paired_ablation_scope"],
        )
    )
    write_jsonl(args.output, selected)
    write_manifest(
        args.output.with_suffix(args.output.suffix + ".manifest.json"),
        {
            "protocol": OUTPUT_PROTOCOL,
            "source_generation_protocol": EXPECTED_PROTOCOL,
            "source_artifacts": audits,
            "models": sorted(EXPECTED_MODELS),
            "condition": "delusion",
            "scopes": ["bounded_history", "last_user_only"],
            "pairs_per_model": 449,
            "rows_per_model": 898,
            "rows": len(selected),
            "judge_context": (
                "bounded-history messages_used shared identically across "
                "both generation arms"
            ),
            "judge_previous_exchanges": 3,
            "judge_context_hash": "canonical JSON SHA-256",
            "identical_prompt_pairs_excluded_per_model": (
                identical_prompt_pairs_by_model
            ),
            "shared_pair_ids_sha256": hashlib.sha256(
                "\n".join(sorted(pair_sets[0])).encode("utf-8")
            ).hexdigest(),
            "output_sha256": sha256_file(args.output),
        },
    )
    print(
        json.dumps(
            {
                "rows": len(selected),
                "pairs_per_model": 449,
                "identical_prompt_pairs_excluded_per_model": (
                    identical_prompt_pairs_by_model
                ),
                "output": str(args.output),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
