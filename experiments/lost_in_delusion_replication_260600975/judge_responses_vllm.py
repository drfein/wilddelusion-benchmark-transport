#!/usr/bin/env python3
"""Judge responses with the paper's native Qwen safety judge."""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import importlib.metadata
import json
import os
import re
import time
from pathlib import Path
from typing import Any

from io_utils import (
    append_jsonl,
    canonical_json_sha256,
    canonicalize,
    messages_for_judging,
    read_jsonl,
    write_manifest,
)
from wilddelusion_prompts import SAFETY_JUDGE_TEMPLATE, safety_judge_prompt


FINAL_PATTERN = re.compile(
    r'Final answer:\s*\{\s*"DCS"\s*:\s*"?(N/A|[123])"?\s*,\s*'
    r'"HES"\s*:\s*"?(N/A|[123])"?\s*,\s*'
    r'"SIS"\s*:\s*"?([01])"?\s*\}\s*\Z',
    re.IGNORECASE | re.DOTALL,
)
PAPER_MODEL_NAME = "Qwen3-30B-A3B-Thinking"
DEFAULT_MODEL_ID = "Qwen/Qwen3-30B-A3B-Thinking-2507"
DEFAULT_REVISION = "144afc2f379b542fdd4e85a1fcd5e1f79112d95d"
JUDGE_PROTOCOL = (
    "Lost in Delusion published combined DCS/HES/SIS rubric; "
    "native Qwen3-30B-A3B-Thinking"
)


def judge_protocol(previous_exchanges: int) -> str:
    if previous_exchanges == 3:
        return JUDGE_PROTOCOL
    return f"{JUDGE_PROTOCOL}; previous_exchanges={previous_exchanges}"


def parse_scores(text: str) -> tuple[str, str, int]:
    match = FINAL_PATTERN.search(text)
    if not match:
        raise ValueError("missing exact final DCS/HES/SIS line")
    return match.group(1).upper(), match.group(2).upper(), int(match.group(3))


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def row_seed(generation_id: str, base_seed: int, attempt: int) -> int:
    digest = hashlib.sha256(generation_id.encode("utf-8")).digest()
    offset = int.from_bytes(digest[:4], "big")
    return (base_seed + offset + attempt - 1) % (2**31 - 1)


def successful(row: dict[str, Any], protocol: str) -> bool:
    return (
        not row.get("judge_error")
        and all(metric in row for metric in ("DCS", "HES", "SIS"))
        and row.get("judge_protocol") == protocol
    )


def unique_responses(paths: list[Path]) -> list[dict[str, Any]]:
    selected: dict[str, dict[str, Any]] = {}
    for path in paths:
        for row in read_jsonl(path):
            if not row.get("response") or row.get("generation_error"):
                continue
            generation_id = row["generation_id"]
            prior = selected.get(generation_id)
            if prior is not None and (
                prior.get("response") != row.get("response")
                or messages_for_judging(prior)
                != messages_for_judging(row)
            ):
                raise ValueError(
                    f"conflicting rows for generation_id={generation_id}"
                )
            selected[generation_id] = row
    return list(selected.values())


def build_judgment(
    source: dict[str, Any],
    raw: str,
    dcs: str,
    hes: str,
    sis: int,
    *,
    args: argparse.Namespace,
    protocol: str,
    prompt_tokens: int,
    output_tokens: int,
    finish_reason: str | None,
    attempt: int,
    seed: int,
) -> dict[str, Any]:
    fields = (
        "pair_id",
        "cluster_id",
        "condition",
        "theme",
        "source",
        "model",
        "response",
        "original_harm",
        "control_harm",
        "high_stakes_action",
        "harm_pair_valid",
        "harm_type",
        "control_package_score",
        "selection_round",
        "speech_act_match",
        "distress_match",
        "harm_match",
        "semantic_closeness",
        "style_length_match",
        "validation_usable",
        "original_delusion",
        "control_delusion",
        "distress_preservation",
        "harm_preservation",
        "request_preservation",
        "coherent",
        "context_scope",
        "counterfactual_scope",
    )
    row = {field: source.get(field) for field in fields}
    row.update(
        {
            "generation_id": source["generation_id"],
            "cluster_id": source.get("cluster_id", source.get("pair_id")),
            "target_text": messages_for_judging(source)[-1]["content"],
            "judge_context_sha256": canonical_json_sha256(
                messages_for_judging(source)
            ),
            "judge_context_source_generation_id": source.get(
                "judge_context_source_generation_id"
            ),
            "DCS": dcs,
            "HES": hes,
            "SIS": sis,
            "judge_raw": raw,
            "judge_model": PAPER_MODEL_NAME,
            "judge_model_id": args.model_id,
            "judge_model_path": args.model,
            "judge_model_revision": args.revision,
            "judge_dtype": args.dtype,
            "judge_protocol": protocol,
            "judge_previous_exchanges": args.previous_exchanges,
            "judge_input_messages": ["user"],
            "judge_chat_template": "model tokenizer default",
            "judge_temperature": args.temperature,
            "judge_top_p": args.top_p,
            "judge_top_k": args.top_k,
            "judge_seed": seed,
            "judge_input_tokens": prompt_tokens,
            "judge_output_tokens": output_tokens,
            "judge_finish_reason": finish_reason,
            "judge_attempt": attempt,
        }
    )
    return row


def run(args: argparse.Namespace) -> None:
    # Imported lazily so prompt/parser tests do not require a GPU environment.
    import torch
    from vllm import LLM, SamplingParams
    from tqdm import tqdm

    protocol = judge_protocol(args.previous_exchanges)
    responses = unique_responses(args.inputs)
    if args.limit:
        responses = responses[: args.limit]

    args.output.parent.mkdir(parents=True, exist_ok=True)
    lock_path = args.output.with_suffix(args.output.suffix + ".lock")
    lock_handle = lock_path.open("w")
    try:
        fcntl.flock(lock_handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as error:
        raise RuntimeError(f"Another judge owns {lock_path}") from error

    prior = canonicalize(
        args.output,
        lambda row: row["generation_id"],
        lambda row: successful(row, protocol),
    )
    completed = {
        row["generation_id"] for row in prior if successful(row, protocol)
    }
    pending = [
        response
        for response in responses
        if response["generation_id"] not in completed
    ]
    manifest_path = args.output.with_suffix(
        args.output.suffix + ".manifest.json"
    )
    if not pending and manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if (
            manifest.get("model_id") == args.model_id
            and manifest.get("model_revision") == args.revision
            and manifest.get("judge_protocol") == protocol
            and manifest.get("successful_rows") == len(responses)
            and manifest.get("output_sha256") == file_sha256(args.output)
        ):
            print(json.dumps(manifest, indent=2))
            return

    local_model = Path(args.model).exists()
    llm = LLM(
        model=args.model,
        revision=None if local_model else args.revision,
        tokenizer=args.tokenizer or args.model,
        tokenizer_revision=(
            args.revision if not local_model and not args.tokenizer else None
        ),
        tensor_parallel_size=args.tensor_parallel_size,
        pipeline_parallel_size=args.pipeline_parallel_size,
        dtype=args.dtype,
        seed=args.base_seed,
        gpu_memory_utilization=args.gpu_memory_utilization,
        cpu_offload_gb=args.cpu_offload_gb,
        enforce_eager=args.enforce_eager,
        max_model_len=args.max_model_len,
        max_num_seqs=args.max_num_seqs,
        max_num_batched_tokens=args.max_num_batched_tokens,
        moe_backend=args.moe_backend,
        disable_custom_all_reduce=True,
        trust_remote_code=False,
    )

    started = time.time()
    counters = {
        "new_successes": 0,
        "new_errors": 0,
        "input_tokens": 0,
        "output_tokens": 0,
    }

    progress = tqdm(total=len(pending), desc="Judging responses", unit="row")
    for start in range(0, len(pending), args.batch_size):
        batch = pending[start : start + args.batch_size]
        unresolved = batch
        errors = {row["generation_id"]: "" for row in batch}
        failure_details: dict[str, dict[str, Any]] = {}
        for attempt in range(1, args.attempts + 1):
            if not unresolved:
                break
            conversations = []
            sampling_params = []
            for row in unresolved:
                prompt = safety_judge_prompt(
                    messages_for_judging(row),
                    row["response"],
                    previous_exchanges=args.previous_exchanges,
                )
                conversations.append([{"role": "user", "content": prompt}])
                seed = row_seed(row["generation_id"], args.base_seed, attempt)
                sampling_params.append(
                    SamplingParams(
                        temperature=args.temperature,
                        top_p=args.top_p,
                        top_k=args.top_k,
                        max_tokens=args.max_output_tokens
                        * (2 ** (attempt - 1)),
                        seed=seed,
                    )
                )

            outputs = llm.chat(
                conversations,
                sampling_params=sampling_params,
                use_tqdm=False,
            )
            next_unresolved = []
            for row, output in zip(unresolved, outputs, strict=True):
                candidate = output.outputs[0]
                raw = candidate.text
                try:
                    dcs, hes, sis = parse_scores(raw)
                    if candidate.finish_reason == "length":
                        raise ValueError(
                            "generation hit max_tokens instead of terminating"
                        )
                except ValueError as error:
                    errors[row["generation_id"]] = (
                        f"attempt {attempt}: {error}; "
                        f"finish_reason={candidate.finish_reason}; "
                        f"output_tokens={len(candidate.token_ids)}"
                    )
                    failure_details[row["generation_id"]] = {
                        "judge_raw_last": raw,
                        "judge_finish_reason_last": candidate.finish_reason,
                        "judge_output_tokens_last": len(candidate.token_ids),
                        "judge_attempt_last": attempt,
                    }
                    next_unresolved.append(row)
                    continue

                seed = row_seed(row["generation_id"], args.base_seed, attempt)
                judgment = build_judgment(
                    row,
                    raw,
                    dcs,
                    hes,
                    sis,
                    args=args,
                    protocol=protocol,
                    prompt_tokens=len(output.prompt_token_ids),
                    output_tokens=len(candidate.token_ids),
                    finish_reason=candidate.finish_reason,
                    attempt=attempt,
                    seed=seed,
                )
                append_jsonl(args.output, judgment)
                counters["new_successes"] += 1
                counters["input_tokens"] += len(output.prompt_token_ids)
                counters["output_tokens"] += len(candidate.token_ids)
                progress.update(1)
            unresolved = next_unresolved

        for row in unresolved:
            append_jsonl(
                args.output,
                {
                    "generation_id": row["generation_id"],
                    "pair_id": row.get("pair_id"),
                    "cluster_id": row.get("cluster_id", row.get("pair_id")),
                    "condition": row.get("condition"),
                    "model": row.get("model"),
                    "judge_model": PAPER_MODEL_NAME,
                    "judge_model_id": args.model_id,
                    "judge_model_path": args.model,
                    "judge_model_revision": args.revision,
                    "judge_protocol": protocol,
                    "judge_error": errors[row["generation_id"]],
                    **failure_details.get(row["generation_id"], {}),
                },
            )
            counters["new_errors"] += 1
            progress.update(1)
        progress.set_postfix(
            success=counters["new_successes"], errors=counters["new_errors"]
        )
    progress.close()

    final = canonicalize(
        args.output,
        lambda row: row["generation_id"],
        lambda row: successful(row, protocol),
    )
    successful_rows = sum(successful(row, protocol) for row in final)
    model_path = Path(args.model)
    generation_config_path = model_path / "generation_config.json"
    tokenizer_config_path = model_path / "tokenizer_config.json"
    staging_manifest_path = model_path / "staging_manifest.json"
    tokenizer_config = (
        json.loads(tokenizer_config_path.read_text())
        if tokenizer_config_path.exists()
        else {}
    )
    manifest = {
        "paper": "arXiv:2606.00975",
        "paper_judge_name": PAPER_MODEL_NAME,
        "model_id": args.model_id,
        "model_path": args.model,
        "model_revision": args.revision,
        "dtype": args.dtype,
        "quantization": None,
        "rubric": "verbatim published combined DCS/HES/SIS prompt",
        "rubric_template_sha256": hashlib.sha256(
            SAFETY_JUDGE_TEMPLATE.encode("utf-8")
        ).hexdigest(),
        "judge_protocol": protocol,
        "input_roles": ["user"],
        "extra_system_prompt": None,
        "judge_context_resolution": [
            "judge_messages",
            "messages_used",
            "messages",
        ],
        "chat_template": "model tokenizer default",
        "chat_template_sha256": (
            hashlib.sha256(
                tokenizer_config.get("chat_template", "").encode("utf-8")
            ).hexdigest()
            if tokenizer_config.get("chat_template")
            else None
        ),
        "tokenizer_config_sha256": (
            file_sha256(tokenizer_config_path)
            if tokenizer_config_path.exists()
            else None
        ),
        "generation_config": (
            json.loads(generation_config_path.read_text())
            if generation_config_path.exists()
            else None
        ),
        "generation_config_sha256": (
            file_sha256(generation_config_path)
            if generation_config_path.exists()
            else None
        ),
        "staging_manifest_sha256": (
            file_sha256(staging_manifest_path)
            if staging_manifest_path.exists()
            else None
        ),
        "previous_exchanges": args.previous_exchanges,
        "paper_reports_exact_model_revision": False,
        "paper_reports_decoding_parameters": False,
        "decoding_source": (
            "official model generation_config.json defaults for "
            "Qwen3-30B-A3B-Thinking-2507"
        ),
        "temperature": args.temperature,
        "top_p": args.top_p,
        "top_k": args.top_k,
        "base_seed": args.base_seed,
        "max_output_tokens_first_attempt": args.max_output_tokens,
        "attempts": args.attempts,
        "tensor_parallel_size": args.tensor_parallel_size,
        "pipeline_parallel_size": args.pipeline_parallel_size,
        "gpu_memory_utilization": args.gpu_memory_utilization,
        "cpu_offload_gb_per_gpu": args.cpu_offload_gb,
        "max_model_len": args.max_model_len,
        "max_num_seqs": args.max_num_seqs,
        "max_num_batched_tokens": args.max_num_batched_tokens,
        "moe_backend": args.moe_backend,
        "enforce_eager": args.enforce_eager,
        "input_files": [
            {
                "path": str(path),
                "sha256": file_sha256(path),
                "rows": len(read_jsonl(path)),
            }
            for path in args.inputs
        ],
        "candidate_responses": len(responses),
        "already_complete": len(completed),
        **counters,
        "successful_rows": successful_rows,
        "output_sha256": file_sha256(args.output),
        "elapsed_seconds": time.time() - started,
        "hostname": os.uname().nodename,
        "software": {
            package: importlib.metadata.version(package)
            for package in ("torch", "transformers", "vllm")
        },
        "cuda_version": torch.version.cuda,
        "gpu_names": [
            torch.cuda.get_device_name(index)
            for index in range(torch.cuda.device_count())
        ],
    }
    write_manifest(manifest_path, manifest)
    print(json.dumps(manifest, indent=2))
    if successful_rows != len(responses):
        raise RuntimeError(
            f"Only {successful_rows}/{len(responses)} judgments succeeded"
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--inputs", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", default=DEFAULT_MODEL_ID)
    parser.add_argument("--model-id", default=DEFAULT_MODEL_ID)
    parser.add_argument("--tokenizer")
    parser.add_argument("--revision", default=DEFAULT_REVISION)
    parser.add_argument("--previous-exchanges", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--attempts", type=int, default=3)
    parser.add_argument("--max-output-tokens", type=int, default=4096)
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--base-seed", type=int, default=260600975)
    parser.add_argument("--tensor-parallel-size", type=int, default=2)
    parser.add_argument("--pipeline-parallel-size", type=int, default=1)
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.95)
    parser.add_argument("--cpu-offload-gb", type=float, default=6.0)
    parser.add_argument("--max-model-len", type=int, default=32768)
    parser.add_argument("--max-num-seqs", type=int, default=32)
    parser.add_argument("--max-num-batched-tokens", type=int, default=4096)
    parser.add_argument(
        "--moe-backend",
        default="triton",
        choices=("triton", "flashinfer_cutlass", "flashinfer_trtllm"),
    )
    parser.add_argument(
        "--enforce-eager",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--limit", type=int)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
