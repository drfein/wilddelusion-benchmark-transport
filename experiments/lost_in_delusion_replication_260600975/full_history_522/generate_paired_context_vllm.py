#!/usr/bin/env python3
"""Generate both context-ablation arms through one prompt-keyed vLLM run.

Greedy GPU decoding is not reliably batch invariant.  This runner renders both
arms together and generates each unique tokenized prompt exactly once, so an
identical model input cannot produce a spurious context difference.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import torch
from tqdm import tqdm

import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from io_utils import append_jsonl, read_jsonl, write_jsonl, write_manifest  # noqa: E402
from wilddelusion_prompts import BASELINE_SYSTEM_PROMPT  # noqa: E402
from run_local_models import model_slug, trim_to_limit  # noqa: E402


PROTOCOL = "single_engine_unique_tokenized_prompt_cache_v1"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def token_ids_sha256(token_ids: list[int]) -> str:
    payload = json.dumps(token_ids, separators=(",", ":")).encode("ascii")
    return hashlib.sha256(payload).hexdigest()


def successful_cache_row(row: dict[str, Any]) -> bool:
    return not row.get("generation_error") and bool(row.get("response"))


def completed_arm_artifact(
    output_path: Path,
    input_path: Path,
    *,
    model: str,
    checkpoint_source: str,
    revision: str,
    expected_rows: int = 623,
) -> bool:
    manifest_path = output_path.with_suffix(
        output_path.suffix + ".manifest.json"
    )
    if not output_path.exists() or not manifest_path.exists():
        return False
    try:
        rows = read_jsonl(output_path)
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, ValueError, json.JSONDecodeError):
        return False
    return (
        len(rows) == expected_rows
        and len({row.get("generation_id") for row in rows}) == expected_rows
        and all(row.get("response") for row in rows)
        and manifest.get("protocol") == PROTOCOL
        and manifest.get("model") == model
        and manifest.get("model_checkpoint_source") == checkpoint_source
        and manifest.get("model_revision") == revision
        and manifest.get("input_sha256") == sha256_file(input_path)
        and manifest.get("successful_rows") == expected_rows
        and manifest.get("expected_rows") == expected_rows
        and manifest.get("output_sha256") == sha256_file(output_path)
    )


def load_cache(
    path: Path,
    *,
    model: str,
    checkpoint_source: str,
    revision: str,
    seed: int,
) -> dict[str, dict[str, Any]]:
    selected: dict[str, dict[str, Any]] = {}
    for row in read_jsonl(path):
        provenance = {
            "model": row.get("model") == model,
            "checkpoint_source": row.get("model_checkpoint_source")
            == checkpoint_source,
            "revision": row.get("model_revision") == revision,
            "seed": row.get("seed") == seed,
            "decoding": row.get("decoding") == "greedy",
        }
        if not all(provenance.values()):
            raise ValueError(
                "Prompt cache provenance mismatch: "
                + json.dumps(provenance, sort_keys=True)
            )
        key = str(row["prompt_token_ids_sha256"])
        if key not in selected or successful_cache_row(row):
            selected[key] = row
    return selected


def validate_staged_checkpoint(
    model_path: Path,
    checkpoint_source: str,
    revision: str,
) -> dict[str, Any]:
    manifest_path = model_path / "staging_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    checks = {
        "model": manifest.get("model_id") == checkpoint_source,
        "revision": manifest.get("revision") == revision,
        "resolved_revision": manifest.get("resolved_revision") == revision,
    }
    if not all(checks.values()):
        raise ValueError(
            "Staged checkpoint does not match the pinned source: "
            + json.dumps(checks, sort_keys=True)
        )
    return {
        "path": str(manifest_path),
        "sha256": sha256_file(manifest_path),
        "checks": checks,
    }


def pair_key(row: dict[str, Any]) -> tuple[str, str]:
    return str(row["pair_id"]), str(row["condition"])


def validate_inputs(
    history_rows: list[dict[str, Any]],
    last_only_rows: list[dict[str, Any]],
) -> None:
    if len(history_rows) != 623 or len(last_only_rows) != 623:
        raise ValueError("Both context arms must contain exactly 623 rows")
    history = {pair_key(row): row for row in history_rows}
    last_only = {pair_key(row): row for row in last_only_rows}
    if len(history) != 623 or len(last_only) != 623 or set(history) != set(last_only):
        raise ValueError("Context-arm pair coverage is incomplete or duplicated")
    for key in history:
        history_target = history[key]["messages"][-1]
        last_target = last_only[key]["messages"][-1]
        if (
            history_target != last_target
            or history_target.get("role") != "user"
            or len(last_only[key]["messages"]) != 1
        ):
            raise ValueError(f"{key}: final-user-message identity failed")


def prepare_rows(
    tokenizer: Any,
    rows_by_scope: dict[str, list[dict[str, Any]]],
    max_input_tokens: int,
) -> list[dict[str, Any]]:
    prepared: list[dict[str, Any]] = []
    for scope, rows in rows_by_scope.items():
        for row in rows:
            messages_used, token_ids, dropped = trim_to_limit(
                tokenizer,
                row["messages"],
                max_input_tokens,
                system_prompt=BASELINE_SYSTEM_PROMPT,
            )
            prepared.append(
                {
                    "scope": scope,
                    "row": row,
                    "messages_used": messages_used,
                    "token_ids": token_ids,
                    "dropped_message_count": dropped,
                    "prompt_token_ids_sha256": token_ids_sha256(token_ids),
                }
            )
    return prepared


def write_arm_outputs(
    prepared: list[dict[str, Any]],
    cache: dict[str, dict[str, Any]],
    outputs: dict[str, Path],
    args: argparse.Namespace,
    checkpoint_audit: dict[str, Any],
    cache_path: Path,
) -> None:
    slug = model_slug(args.reported_model)
    inputs = {
        "bounded_history": args.history_input,
        "last_user_only": args.last_only_input,
    }
    prompt_hashes_by_pair: dict[tuple[str, str], dict[str, str]] = {}
    for item in prepared:
        prompt_hashes_by_pair.setdefault(pair_key(item["row"]), {})[
            item["scope"]
        ] = item["prompt_token_ids_sha256"]
    identical_pairs = {
        key
        for key, scopes in prompt_hashes_by_pair.items()
        if scopes["bounded_history"] == scopes["last_user_only"]
    }

    for scope, output_path in outputs.items():
        result_rows = []
        for item in prepared:
            if item["scope"] != scope:
                continue
            prompt_hash = item["prompt_token_ids_sha256"]
            cached = cache.get(prompt_hash)
            if cached is None or not successful_cache_row(cached):
                raise ValueError(f"Missing successful prompt cache row {prompt_hash}")
            source = item["row"]
            result_rows.append(
                {
                    **source,
                    "generation_id": (
                        f"{slug}:paired_context:{scope}:{source['input_id']}"
                    ),
                    "model": args.reported_model,
                    "model_checkpoint": str(args.model),
                    "model_checkpoint_source": args.checkpoint_source,
                    "model_revision": args.revision,
                    "model_dtype": args.dtype,
                    "model_slug": slug,
                    "system_prompt": BASELINE_SYSTEM_PROMPT,
                    "decoding": "greedy",
                    "generation_backend": "vllm",
                    "paired_generation_protocol": PROTOCOL,
                    "paired_ablation_scope": scope,
                    "max_new_tokens": args.max_new_tokens,
                    "input_token_count": len(item["token_ids"]),
                    "dropped_message_count": item["dropped_message_count"],
                    "messages_used": item["messages_used"],
                    "prompt_token_ids_sha256": prompt_hash,
                    "cross_arm_prompt_identical": (
                        pair_key(source) in identical_pairs
                    ),
                    "response": cached["response"],
                    "finish_reason": cached.get("finish_reason"),
                }
            )
        write_jsonl(output_path, result_rows)
        manifest_path = output_path.with_suffix(
            output_path.suffix + ".manifest.json"
        )
        write_manifest(
            manifest_path,
            {
                "protocol": PROTOCOL,
                "scope": scope,
                "model": args.reported_model,
                "model_checkpoint": str(args.model),
                "model_checkpoint_source": args.checkpoint_source,
                "model_revision": args.revision,
                "model_dtype": args.dtype,
                "checkpoint_audit": checkpoint_audit,
                "backend": "vllm",
                "decoding": "greedy",
                "seed": args.seed,
                "system_prompt": BASELINE_SYSTEM_PROMPT,
                "max_input_tokens": args.max_input_tokens,
                "max_new_tokens": args.max_new_tokens,
                "chunk_size": args.chunk_size,
                "tensor_parallel_size": args.tensor_parallel_size,
                "input": str(inputs[scope]),
                "input_sha256": sha256_file(inputs[scope]),
                "paired_input": str(
                    inputs[
                        "last_user_only"
                        if scope == "bounded_history"
                        else "bounded_history"
                    ]
                ),
                "paired_input_sha256": sha256_file(
                    inputs[
                        "last_user_only"
                        if scope == "bounded_history"
                        else "bounded_history"
                    ]
                ),
                "prompt_cache": str(cache_path),
                "prompt_cache_sha256": sha256_file(cache_path),
                "total_rendered_prompts": len(prepared),
                "unique_rendered_prompts": len(
                    {item["prompt_token_ids_sha256"] for item in prepared}
                ),
                "cross_arm_identical_prompt_pairs": len(identical_pairs),
                "cross_arm_different_prompt_pairs": (
                    len(prompt_hashes_by_pair) - len(identical_pairs)
                ),
                "successful_rows": len(result_rows),
                "expected_rows": 623,
                "output_sha256": sha256_file(output_path),
                "cuda": torch.version.cuda,
                "torch": torch.__version__,
            },
        )


def main() -> None:
    from vllm import LLM, SamplingParams

    parser = argparse.ArgumentParser()
    parser.add_argument("--history-input", type=Path, required=True)
    parser.add_argument("--last-only-input", type=Path, required=True)
    parser.add_argument("--history-output", type=Path, required=True)
    parser.add_argument("--last-only-output", type=Path, required=True)
    parser.add_argument("--prompt-cache", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--reported-model", required=True)
    parser.add_argument("--checkpoint-source", required=True)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--max-input-tokens", type=int, default=12288)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--chunk-size", type=int, default=128)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.92)
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument("--seed", type=int, default=260600975)
    parser.add_argument("--enforce-eager", action="store_true")
    args = parser.parse_args()

    history_rows = read_jsonl(args.history_input)
    last_only_rows = read_jsonl(args.last_only_input)
    validate_inputs(history_rows, last_only_rows)
    complete = all(
        completed_arm_artifact(
            output_path,
            input_path,
            model=args.reported_model,
            checkpoint_source=args.checkpoint_source,
            revision=args.revision,
        )
        for output_path, input_path in (
            (args.history_output, args.history_input),
            (args.last_only_output, args.last_only_input),
        )
    )
    if complete:
        print(
            json.dumps(
                {
                    "model": args.reported_model,
                    "already_complete": True,
                    "history_output": str(args.history_output),
                    "last_only_output": str(args.last_only_output),
                },
                indent=2,
            )
        )
        return
    checkpoint_audit = validate_staged_checkpoint(
        args.model,
        args.checkpoint_source,
        args.revision,
    )

    llm = LLM(
        model=str(args.model),
        dtype=args.dtype,
        max_model_len=args.max_input_tokens + args.max_new_tokens,
        gpu_memory_utilization=args.gpu_memory_utilization,
        enable_prefix_caching=True,
        tensor_parallel_size=args.tensor_parallel_size,
        enforce_eager=args.enforce_eager,
        seed=args.seed,
    )
    prepared = prepare_rows(
        llm.get_tokenizer(),
        {
            "bounded_history": history_rows,
            "last_user_only": last_only_rows,
        },
        args.max_input_tokens,
    )
    unique: dict[str, dict[str, Any]] = {}
    for item in prepared:
        unique.setdefault(item["prompt_token_ids_sha256"], item)

    cache = load_cache(
        args.prompt_cache,
        model=args.reported_model,
        checkpoint_source=args.checkpoint_source,
        revision=args.revision,
        seed=args.seed,
    )
    pending = [
        item
        for prompt_hash, item in unique.items()
        if not successful_cache_row(cache.get(prompt_hash, {}))
    ]
    pending.sort(key=lambda item: len(item["token_ids"]))
    sampling = SamplingParams(
        temperature=0.0,
        max_tokens=args.max_new_tokens,
        seed=args.seed,
    )
    for start in tqdm(
        range(0, len(pending), args.chunk_size),
        desc=f"Paired context {model_slug(args.reported_model)}",
    ):
        chunk = pending[start : start + args.chunk_size]
        try:
            outputs = llm.generate(
                [
                    {"prompt_token_ids": item["token_ids"]}
                    for item in chunk
                ],
                sampling_params=sampling,
                use_tqdm=True,
            )
            for item, output in zip(chunk, outputs, strict=True):
                candidate = output.outputs[0]
                append_jsonl(
                    args.prompt_cache,
                    {
                        "prompt_token_ids_sha256": item[
                            "prompt_token_ids_sha256"
                        ],
                        "input_token_count": len(item["token_ids"]),
                        "representative_pair_id": item["row"]["pair_id"],
                        "representative_condition": item["row"]["condition"],
                        "representative_scope": item["scope"],
                        "model": args.reported_model,
                        "model_checkpoint_source": args.checkpoint_source,
                        "model_revision": args.revision,
                        "seed": args.seed,
                        "decoding": "greedy",
                        "response": candidate.text.strip(),
                        "finish_reason": candidate.finish_reason,
                    },
                )
        except Exception as error:  # noqa: BLE001 - persist batch failures
            for item in chunk:
                append_jsonl(
                    args.prompt_cache,
                    {
                        "prompt_token_ids_sha256": item[
                            "prompt_token_ids_sha256"
                        ],
                        "model": args.reported_model,
                        "model_revision": args.revision,
                        "generation_error": repr(error),
                    },
                )

    cache = load_cache(
        args.prompt_cache,
        model=args.reported_model,
        checkpoint_source=args.checkpoint_source,
        revision=args.revision,
        seed=args.seed,
    )
    missing = [
        prompt_hash
        for prompt_hash in unique
        if not successful_cache_row(cache.get(prompt_hash, {}))
    ]
    if missing:
        raise RuntimeError(f"{len(missing)} unique prompts failed generation")
    write_arm_outputs(
        prepared,
        cache,
        {
            "bounded_history": args.history_output,
            "last_user_only": args.last_only_output,
        },
        args,
        checkpoint_audit,
        args.prompt_cache,
    )
    print(
        json.dumps(
            {
                "model": args.reported_model,
                "rendered_prompts": len(prepared),
                "unique_prompts": len(unique),
                "deduplicated_prompts": len(prepared) - len(unique),
                "new_prompts_generated": len(pending),
                "history_output": str(args.history_output),
                "last_only_output": str(args.last_only_output),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
