#!/usr/bin/env python3
"""Generate paired intent/control responses through one pinned vLLM model."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
from collections import defaultdict
from pathlib import Path
from typing import Any

from tqdm import tqdm
from vllm import LLM, SamplingParams, __version__ as vllm_version

from design import CUE_PAIRS, REPLICATE_SEEDS, SELECTED_CONVERSATIONS, STUDY_ID, stable_hash
from io_utils import append_jsonl, canonicalize, read_jsonl, sha256_file


def token_hash(token_ids: list[int]) -> str:
    payload = json.dumps(token_ids, separators=(",", ":")).encode("ascii")
    return hashlib.sha256(payload).hexdigest()


def successful(row: dict[str, Any], model: str, revision: str) -> bool:
    return (
        row.get("model") == model
        and row.get("model_revision") == revision
        and isinstance(row.get("response"), str)
        and not row.get("generation_error")
    )


def render(tokenizer: Any, messages: list[dict[str, str]]) -> list[int]:
    encoded = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    # Transformers 5 returns BatchEncoding; older releases returned a list.
    # Normalize here so vLLM never interprets mapping keys as token IDs.
    if isinstance(encoded, dict) or hasattr(encoded, "keys"):
        encoded = encoded["input_ids"]
    token_ids = [int(token_id) for token_id in encoded]
    if not token_ids:
        raise ValueError("Chat template produced no input tokens")
    return token_ids


def paired_trim(
    tokenizer: Any,
    target: list[dict[str, str]],
    control: list[dict[str, str]],
    max_input_tokens: int,
) -> tuple[list[dict[str, str]], list[int], list[dict[str, str]], list[int], int]:
    if target[:-1] != control[:-1]:
        raise ValueError("Paired conditions do not share their original history")
    if target[-1]["role"] != "user" or control[-1]["role"] != "user":
        raise ValueError("Paired conditions must end with user messages")
    # Input length decreases monotonically as messages are removed from the
    # front. Binary search avoids quadratic tokenization on very long chats.
    low = 0
    high = len(target) - 1
    best: tuple[
        list[dict[str, str]], list[int], list[dict[str, str]], list[int], int
    ] | None = None
    while low <= high:
        dropped = (low + high) // 2
        target_used = target[dropped:]
        control_used = control[dropped:]
        target_tokens = render(tokenizer, target_used)
        control_tokens = render(tokenizer, control_used)
        if max(len(target_tokens), len(control_tokens)) <= max_input_tokens:
            best = (
                target_used,
                target_tokens,
                control_used,
                control_tokens,
                dropped,
            )
            high = dropped - 1
        else:
            low = dropped + 1
    if best is None:
        raise ValueError("Final user message alone exceeds max_input_tokens")
    return best


def validate_arms(rows: list[dict[str, Any]]) -> None:
    expected = SELECTED_CONVERSATIONS * len(CUE_PAIRS) * 2
    if len(rows) != expected or len({row["arm_id"] for row in rows}) != expected:
        raise ValueError(f"Expected {expected} unique arms, found {len(rows)}")
    pairs: dict[tuple[str, str], dict[str, dict[str, Any]]] = defaultdict(dict)
    for row in rows:
        if row.get("study_id") != STUDY_ID:
            raise ValueError("Study ID mismatch")
        pairs[(row["candidate_id"], row["intent"])][row["condition"]] = row
    if len(pairs) != SELECTED_CONVERSATIONS * len(CUE_PAIRS):
        raise ValueError("Incomplete candidate-intent pairs")
    for key, pair in pairs.items():
        if set(pair) != {"target_intent", "matched_control"}:
            raise ValueError(f"{key}: incomplete condition pair")
        if pair["target_intent"]["messages"][:-1] != pair["matched_control"]["messages"][:-1]:
            raise ValueError(f"{key}: history differs across conditions")
        if pair["target_intent"]["cue_word_count"] != pair["matched_control"]["cue_word_count"]:
            raise ValueError(f"{key}: cue word counts differ")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--max-input-tokens", type=int, default=7680)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--chunk-size", type=int, default=96)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.90)
    parser.add_argument("--dtype", default="half")
    parser.add_argument("--limit-pairs", type=int)
    parser.add_argument("--candidate-ids-file", type=Path)
    parser.add_argument("--replicate-seed", type=int, action="append")
    args = parser.parse_args()

    rows = read_jsonl(args.input)
    validate_arms(rows)
    if args.candidate_ids_file:
        candidate_ids = {
            line.strip()
            for line in args.candidate_ids_file.read_text(encoding="utf-8").splitlines()
            if line.strip()
        }
        rows = [row for row in rows if row["candidate_id"] in candidate_ids]
        if {row["candidate_id"] for row in rows} != candidate_ids:
            raise ValueError("Candidate ID filter contains unknown or missing candidates")
    if args.limit_pairs:
        keys = sorted({(row["candidate_id"], row["intent"]) for row in rows})[
            : args.limit_pairs
        ]
        retained = set(keys)
        rows = [row for row in rows if (row["candidate_id"], row["intent"]) in retained]

    prior = canonicalize(
        args.output,
        lambda row: str(row["generation_id"]),
        lambda row: successful(row, args.model_id, args.revision),
    ) if args.output.exists() else []
    completed = {
        row["generation_id"]
        for row in prior
        if successful(row, args.model_id, args.revision)
    }

    llm = LLM(
        model=args.model_path,
        dtype=args.dtype,
        max_model_len=args.max_input_tokens + args.max_new_tokens,
        gpu_memory_utilization=args.gpu_memory_utilization,
        enable_prefix_caching=True,
        trust_remote_code=False,
    )
    tokenizer = llm.get_tokenizer()
    grouped: dict[tuple[str, str], dict[str, dict[str, Any]]] = defaultdict(dict)
    for row in rows:
        grouped[(row["candidate_id"], row["intent"])][row["condition"]] = row

    encoded: dict[str, dict[str, Any]] = {}
    dropped_counts = []
    for key, pair in tqdm(grouped.items(), desc=f"Tokenizing {args.model_id}"):
        target = pair["target_intent"]
        control = pair["matched_control"]
        target_used, target_tokens, control_used, control_tokens, dropped = paired_trim(
            tokenizer,
            target["messages"],
            control["messages"],
            args.max_input_tokens,
        )
        dropped_counts.append(dropped)
        for source, used, token_ids in (
            (target, target_used, target_tokens),
            (control, control_used, control_tokens),
        ):
            encoded[source["arm_id"]] = {
                "source": source,
                "messages_used": used,
                "token_ids": token_ids,
                "dropped_message_count": dropped,
            }

    seeds = tuple(args.replicate_seed or REPLICATE_SEEDS)
    if not seeds or not set(seeds).issubset(REPLICATE_SEEDS):
        raise ValueError(f"Replicate seeds must be selected from {REPLICATE_SEEDS}")
    pending = []
    model_slug = args.model_id.replace("/", "__")
    for seed in seeds:
        for item in encoded.values():
            source = item["source"]
            generation_id = stable_hash(
                STUDY_ID, args.model_id, args.revision, source["arm_id"], seed
            )[:32]
            if generation_id in completed:
                continue
            # The arm is deliberately omitted, giving paired conditions common
            # random numbers for the same candidate, intent, and replicate.
            request_seed = int(
                stable_hash(
                    STUDY_ID,
                    args.model_id,
                    source["candidate_id"],
                    source["intent"],
                    seed,
                )[:8],
                16,
            ) & 0x7FFFFFFF
            pending.append(
                {
                    **item,
                    "generation_id": generation_id,
                    "replicate_seed": seed,
                    "request_seed": request_seed,
                }
            )
    pending.sort(key=lambda item: (len(item["token_ids"]), item["generation_id"]))

    for start in tqdm(
        range(0, len(pending), args.chunk_size),
        desc=f"Generating {args.model_id}",
    ):
        chunk = pending[start : start + args.chunk_size]
        outputs = llm.generate(
            [{"prompt_token_ids": item["token_ids"]} for item in chunk],
            sampling_params=[
                SamplingParams(
                    temperature=args.temperature,
                    top_p=args.top_p,
                    max_tokens=args.max_new_tokens,
                    seed=item["request_seed"],
                )
                for item in chunk
            ],
            use_tqdm=True,
        )
        for item, output in zip(chunk, outputs, strict=True):
            source = item["source"]
            response = output.outputs[0].text.strip()
            result = {
                **source,
                "generation_id": item["generation_id"],
                "model": args.model_id,
                "model_path": args.model_path,
                "model_revision": args.revision,
                "model_dtype": args.dtype,
                "generation_backend": "vllm",
                "vllm_version": vllm_version,
                "python_version": platform.python_version(),
                "thinking_disabled": True,
                "system_prompt": None,
                "temperature": args.temperature,
                "top_p": args.top_p,
                "max_input_tokens": args.max_input_tokens,
                "max_new_tokens": args.max_new_tokens,
                "replicate_seed": item["replicate_seed"],
                "request_seed": item["request_seed"],
                "input_token_count": len(item["token_ids"]),
                "prompt_token_ids_sha256": token_hash(item["token_ids"]),
                "dropped_message_count": item["dropped_message_count"],
                "messages_used": item["messages_used"],
                "response": response,
                "empty_response": not bool(response),
                "finish_reason": output.outputs[0].finish_reason,
            }
            append_jsonl(args.output, result)

    final = canonicalize(
        args.output,
        lambda row: str(row["generation_id"]),
        lambda row: successful(row, args.model_id, args.revision),
    )
    model_rows = [
        row for row in final if successful(row, args.model_id, args.revision)
    ]
    expected = len(rows) * len(seeds)
    manifest = {
        "study_id": STUDY_ID,
        "input": str(args.input),
        "input_sha256": sha256_file(args.input),
        "output": str(args.output),
        "model": args.model_id,
        "model_path": args.model_path,
        "model_revision": args.revision,
        "model_dtype": args.dtype,
        "vllm_version": vllm_version,
        "thinking_disabled": True,
        "system_prompt": None,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "replicate_seeds": seeds,
        "max_input_tokens": args.max_input_tokens,
        "max_new_tokens": args.max_new_tokens,
        "arms": len(rows),
        "expected_rows": expected,
        "successful_rows": len(model_rows),
        "empty_responses": sum(not row.get("response") for row in model_rows),
        "dropped_message_count_min": min(dropped_counts),
        "dropped_message_count_max": max(dropped_counts),
        "output_sha256": sha256_file(args.output),
        "smoke_limit_pairs": args.limit_pairs,
        "candidate_ids_file": (
            str(args.candidate_ids_file) if args.candidate_ids_file else None
        ),
        "model_slug": model_slug,
    }
    manifest_path = args.output.with_suffix(args.output.suffix + ".manifest.json")
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2))
    if len(model_rows) != expected:
        raise RuntimeError(f"Only {len(model_rows)}/{expected} rows generated")


if __name__ == "__main__":
    main()
