from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from attribute_fixed_response import select_responses
from attribute_probe_tokens import content_message_indices
from config import ENDORSEMENT_THRESHOLD, MODEL_ID, MODEL_REVISION
from io_utils import append_jsonl, read_jsonl, sha256_file
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

FLASHTRACE_COMMIT = "e15e117a5fbbe6e8ad6ea6d0f1314fc7835e5784"


def exact_target_text(
    tokenizer, generated_token_ids: list[int]
) -> tuple[str, list[int], bool]:
    eos = tokenizer.eos_token_id
    ended_with_eos = bool(generated_token_ids and generated_token_ids[-1] == eos)
    core_ids = generated_token_ids[:-1] if ended_with_eos else generated_token_ids
    if not core_ids:
        raise ValueError("FlashTrace target is empty after stripping EOS")
    target = tokenizer.decode(
        core_ids, skip_special_tokens=False, clean_up_tokenization_spaces=False
    )
    round_trip = tokenizer(target, add_special_tokens=False)["input_ids"]
    if round_trip != core_ids:
        raise ValueError("Decoded target does not round-trip to saved generated IDs")
    return target, core_ids, ended_with_eos


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cohort", type=Path, required=True)
    parser.add_argument("--judgments", type=Path, required=True)
    parser.add_argument("--flashtrace-source", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--index", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--model", default=MODEL_ID)
    parser.add_argument("--revision", default=MODEL_REVISION)
    parser.add_argument("--count", type=int, default=40)
    parser.add_argument("--hops", type=int, default=1)
    args = parser.parse_args()

    sys.path.insert(0, str(args.flashtrace_source.resolve()))
    from flashtrace import FlashTrace

    cohort_rows = sorted(
        read_jsonl(args.cohort), key=lambda row: row["conversation_hash"]
    )[: args.count]
    cohort = {row["conversation_hash"]: row for row in cohort_rows}
    selected = select_responses(read_jsonl(args.judgments), set(cohort))
    targets = [
        (conversation_hash, response)
        for conversation_hash in sorted(selected)
        for response in selected[conversation_hash]
    ]

    tokenizer = AutoTokenizer.from_pretrained(
        args.model, revision=args.revision, trust_remote_code=False
    )
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        revision=args.revision,
        torch_dtype=torch.bfloat16,
        device_map="cuda",
        attn_implementation="sdpa",
        trust_remote_code=False,
    ).eval()
    model.requires_grad_(False)
    tracer = FlashTrace(
        model,
        tokenizer,
        use_chat_template=False,
        recompute_attention=True,
        chunk_tokens=128,
        sink_chunk_tokens=32,
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    completed = {
        (row["conversation_hash"], row["generation_sha256"])
        for row in read_jsonl(args.index)
        if row.get("complete") and not row.get("error")
    }
    processed = 0
    for conversation_hash, response in tqdm(targets, desc="FlashTrace"):
        identity = (conversation_hash, response["generation_sha256"])
        output_path = args.output_dir / f"{response['generation_sha256']}.npz"
        if identity in completed and output_path.exists():
            continue
        source = cohort[conversation_hash]
        try:
            prompt = tokenizer.apply_chat_template(
                source["messages"],
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )
            expected_ids, message_indices = content_message_indices(
                tokenizer, source["messages"]
            )
            prompt_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
            if prompt_ids != expected_ids:
                raise ValueError("Rendered prompt does not match chat-template IDs")
            target, target_ids, ended_with_eos = exact_target_text(
                tokenizer, [int(token) for token in response["generated_token_ids"]]
            )
            trace = tracer.trace(
                prompt=prompt,
                target=target,
                output_span=(0, len(target_ids) - 1),
                hops=args.hops,
                method="flashtrace",
            )
            scores = np.asarray(trace.scores, dtype=np.float32)
            if len(scores) != len(expected_ids):
                raise ValueError(
                    f"FlashTrace returned {len(scores)} prompt scores for "
                    f"{len(expected_ids)} prompt tokens"
                )
            np.savez_compressed(
                output_path,
                token_ids=np.asarray(expected_ids, dtype=np.int32),
                message_indices=np.asarray(message_indices, dtype=np.int32),
                flashtrace_scores=scores,
                target_token_ids=np.asarray(target_ids, dtype=np.int32),
            )
            record = {
                "original_row_idx": source["original_row_idx"],
                "conversation_hash": conversation_hash,
                "generation_sha256": response["generation_sha256"],
                "repetition": int(response["repetition"]),
                "annotation_score": int(response["annotation_score"]),
                "endorsement": int(
                    response["annotation_score"] >= ENDORSEMENT_THRESHOLD
                ),
                "prompt_tokens": len(expected_ids),
                "target_tokens": len(target_ids),
                "saved_response_ended_with_eos": ended_with_eos,
                "attribution_file": output_path.name,
                "complete": True,
            }
            processed += 1
        except Exception as error:  # noqa: BLE001 - checkpoint long GPU traces
            record = {
                "original_row_idx": source["original_row_idx"],
                "conversation_hash": conversation_hash,
                "generation_sha256": response["generation_sha256"],
                "error": repr(error),
            }
        append_jsonl(args.index, record)
        torch.cuda.empty_cache()

    final = read_jsonl(args.index)
    successful = {
        (row["conversation_hash"], row["generation_sha256"])
        for row in final
        if row.get("complete") and not row.get("error")
    }
    expected = {
        (conversation_hash, response["generation_sha256"])
        for conversation_hash, response in targets
    }
    manifest = {
        "cohort_sha256": sha256_file(args.cohort),
        "judgments_sha256": sha256_file(args.judgments),
        "model": args.model,
        "model_revision": args.revision,
        "flashtrace_commit": FLASHTRACE_COMMIT,
        "method": "FlashTrace span-wise recursive information-flow attribution",
        "attention_mode": "SDPA forward plus FlashTrace chunked recomputation",
        "hops": args.hops,
        "subset_rule": "smallest conversation SHA-256 hashes",
        "subset_conversations": len(cohort),
        "expected_fixed_responses": len(expected),
        "successful_fixed_responses": len(successful & expected),
        "missing_fixed_responses": len(expected - successful),
        "processed_this_run": processed,
        "target_note": (
            "Saved response IDs are round-trip checked; a method-required trailing "
            "EOS is excluded from the sink span when it was not originally sampled."
        ),
        "selection_note": (
            "First endorsing and first non-endorsing response per conversation when "
            "available; endpoint-conditioned and interpreted descriptively."
        ),
    }
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))
    if expected - successful:
        raise RuntimeError(f"Missing {len(expected - successful)} FlashTrace rows")


if __name__ == "__main__":
    main()
