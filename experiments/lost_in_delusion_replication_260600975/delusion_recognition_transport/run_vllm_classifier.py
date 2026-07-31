#!/usr/bin/env python3
"""Run the paper's exact delusion classifier with resumable vLLM batches."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch
from tqdm import tqdm
from vllm import LLM, SamplingParams

from classifier_prompt import PROMPT_MESSAGE_ROLE, prepare_prompt
from prepare_inputs import read_jsonl, sha256_file
from recognition_prompts import parse_final_answer, prompt_sha256


def model_slug(model: str) -> str:
    return (
        model.lower()
        .replace("/", "_")
        .replace("-", "_")
        .replace(".", "_")
    )


def append_jsonl(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        handle.flush()


def canonicalize(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    latest: dict[str, dict[str, Any]] = {}
    for row in read_jsonl(path):
        latest[str(row["generation_id"])] = row
    rows = list(latest.values())
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    temporary.replace(path)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--reported-model")
    parser.add_argument("--checkpoint-source")
    parser.add_argument("--revision")
    parser.add_argument("--max-input-tokens", type=int, default=12288)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--chunk-size", type=int, default=128)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.90)
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument("--cpu-offload-gb", type=float, default=0.0)
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--disable-thinking", action="store_true")
    parser.add_argument("--enforce-eager", action="store_true")
    args = parser.parse_args()

    reported_model = args.reported_model or args.model
    checkpoint_source = args.checkpoint_source or args.model
    slug = model_slug(reported_model)
    prior = canonicalize(args.output)
    completed = {
        row["generation_id"]
        for row in prior
        if row.get("raw_response") is not None
    }
    source_rows = read_jsonl(args.input)
    pending = [
        row
        for row in source_rows
        if f"{slug}:{row['input_id']}" not in completed
    ]
    manifest_path = args.output.with_suffix(
        args.output.suffix + f".{slug}.manifest.json"
    )
    if not pending and manifest_path.exists():
        existing_manifest = json.loads(
            manifest_path.read_text(encoding="utf-8")
        )
        operationally_complete = (
            existing_manifest.get("model") == reported_model
            and existing_manifest.get("input_sha256")
            == sha256_file(args.input)
            and existing_manifest.get("rows_for_model") == len(source_rows)
            and existing_manifest.get("successful_rows") == len(source_rows)
            and existing_manifest.get("generation_error_rows") == 0
            and existing_manifest.get("output_sha256")
            == sha256_file(args.output)
        )
        if operationally_complete:
            print(json.dumps(existing_manifest, indent=2))
            return

    llm = LLM(
        model=args.model,
        revision=args.revision,
        dtype=args.dtype,
        max_model_len=args.max_input_tokens + args.max_new_tokens,
        gpu_memory_utilization=args.gpu_memory_utilization,
        trust_remote_code=args.trust_remote_code,
        enable_prefix_caching=True,
        tensor_parallel_size=args.tensor_parallel_size,
        cpu_offload_gb=args.cpu_offload_gb,
        enforce_eager=args.enforce_eager,
    )
    tokenizer = llm.get_tokenizer()
    prepared = [
        prepare_prompt(
            tokenizer,
            row,
            args.max_input_tokens,
            args.disable_thinking,
        )
        for row in tqdm(pending, desc="Preparing exact classifier prompts")
    ]
    prepared.sort(key=lambda item: len(item["token_ids"]))

    sampling = SamplingParams(
        temperature=0.0,
        max_tokens=args.max_new_tokens,
    )
    generated = 0
    failures = 0
    for start in tqdm(
        range(0, len(prepared), args.chunk_size),
        desc=f"Classifying with {slug}",
    ):
        chunk = prepared[start : start + args.chunk_size]
        prompts = [
            {"prompt_token_ids": item["token_ids"]} for item in chunk
        ]
        try:
            outputs = llm.generate(
                prompts,
                sampling_params=sampling,
                use_tqdm=True,
            )
            for item, output in zip(chunk, outputs, strict=True):
                source = item["row"]
                raw = output.outputs[0].text.strip()
                parsed = parse_final_answer(raw)
                metadata = {
                    key: value
                    for key, value in source.items()
                    if key != "previous_messages"
                }
                append_jsonl(
                    args.output,
                    {
                        **metadata,
                        "generation_id": f"{slug}:{source['input_id']}",
                        "model": reported_model,
                        "model_checkpoint": args.model,
                        "model_checkpoint_source": checkpoint_source,
                        "model_revision": args.revision,
                        "model_dtype": args.dtype,
                        "protocol": (
                            "Lost in Delusion exact Assess Delusion Then Reply"
                        ),
                        "prompt_message_role": PROMPT_MESSAGE_ROLE,
                        "decoding": "greedy",
                        "thinking_disabled": args.disable_thinking,
                        "max_input_tokens": args.max_input_tokens,
                        "max_new_tokens": args.max_new_tokens,
                        "input_token_count": len(item["token_ids"]),
                        "dropped_previous_messages": item[
                            "dropped_previous_messages"
                        ],
                        "retained_previous_messages": item[
                            "retained_previous_messages"
                        ],
                        "rendered_prompt_sha256": prompt_sha256(
                            item["prompt"]
                        ),
                        "raw_response": raw,
                        "finish_reason": output.outputs[0].finish_reason,
                        **parsed,
                    },
                )
                generated += 1
        except Exception as error:  # noqa: BLE001 - persist batch failures
            for item in chunk:
                source = item["row"]
                metadata = {
                    key: value
                    for key, value in source.items()
                    if key != "previous_messages"
                }
                append_jsonl(
                    args.output,
                    {
                        **metadata,
                        "generation_id": f"{slug}:{source['input_id']}",
                        "model": reported_model,
                        "model_checkpoint": args.model,
                        "model_checkpoint_source": checkpoint_source,
                        "model_revision": args.revision,
                        "model_dtype": args.dtype,
                        "generation_error": repr(error),
                    },
                )
                failures += 1

    final = canonicalize(args.output)
    model_rows = [row for row in final if row.get("model") == reported_model]
    manifest = {
        "model": reported_model,
        "model_checkpoint": args.model,
        "model_checkpoint_source": checkpoint_source,
        "model_revision": args.revision,
        "model_dtype": args.dtype,
        "protocol": "Lost in Delusion exact Assess Delusion Then Reply",
        "prompt_message_role": PROMPT_MESSAGE_ROLE,
        "input": str(args.input),
        "input_sha256": sha256_file(args.input),
        "output": str(args.output),
        "decoding": "greedy",
        "system_prompt": None,
        "thinking_disabled": args.disable_thinking,
        "max_input_tokens": args.max_input_tokens,
        "max_new_tokens": args.max_new_tokens,
        "chunk_size": args.chunk_size,
        "tensor_parallel_size": args.tensor_parallel_size,
        "cpu_offload_gb": args.cpu_offload_gb,
        "new_generated": generated,
        "new_failures": failures,
        "rows_for_model": len(model_rows),
        "successful_rows": sum(
            row.get("raw_response") is not None
            and not row.get("generation_error")
            for row in model_rows
        ),
        "generation_error_rows": sum(
            bool(row.get("generation_error")) for row in model_rows
        ),
        "parse_valid_rows": sum(
            bool(row.get("parse_valid")) for row in model_rows
        ),
        "expected_rows": len(source_rows),
        "output_sha256": sha256_file(args.output),
        "cuda": torch.version.cuda,
        "torch": torch.__version__,
    }
    manifest["parse_protocol_complete"] = (
        manifest["parse_valid_rows"] == manifest["expected_rows"]
    )
    manifest_path.write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2))
    incomplete = {
        key: manifest[key]
        for key in (
            "rows_for_model",
            "successful_rows",
            "generation_error_rows",
            "expected_rows",
        )
        if (
            key in {"rows_for_model", "successful_rows"}
            and manifest[key] != manifest["expected_rows"]
        )
        or (key == "generation_error_rows" and manifest[key] != 0)
    }
    if incomplete:
        raise RuntimeError(
            "Classifier run failed its operational completion gate: "
            + json.dumps(incomplete, sort_keys=True)
        )


if __name__ == "__main__":
    main()
