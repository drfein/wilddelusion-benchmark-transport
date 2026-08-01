from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import torch
from config import ENDORSEMENT_THRESHOLD, MODEL_ID, MODEL_REVISION
from io_utils import append_jsonl, read_jsonl, sha256_file
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer


def response_log_probability(
    model: Any,
    tokenizer: Any,
    messages: list[dict[str, str]],
    response_token_ids: list[int],
) -> tuple[float, float, int]:
    prompt_ids = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    if not response_token_ids:
        raise ValueError("Cannot score an empty response")
    full_ids = torch.tensor(
        [prompt_ids + response_token_ids], dtype=torch.long, device=model.device
    )
    keep = len(response_token_ids) + 1
    with torch.inference_mode():
        output = model(
            input_ids=full_ids,
            attention_mask=torch.ones_like(full_ids),
            use_cache=False,
            return_dict=True,
            logits_to_keep=keep,
        )
    prediction_logits = output.logits[:, :-1, :].float()
    targets = torch.tensor([response_token_ids], dtype=torch.long, device=model.device)
    token_log_probs = torch.log_softmax(prediction_logits, dim=-1).gather(
        -1, targets[:, :, None]
    )[:, :, 0]
    total = float(token_log_probs.sum().cpu())
    return total, total / len(response_token_ids), len(prompt_ids)


def select_responses(
    rows: list[dict[str, Any]], eligible_hashes: set[str]
) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        if (
            row.get("conversation_hash") in eligible_hashes
            and isinstance(row.get("annotation_score"), int)
            and row.get("generated_token_ids")
            and not row.get("judge_error")
        ):
            grouped[row["conversation_hash"]].append(row)
    selected = {}
    for conversation_hash, candidates in grouped.items():
        candidates.sort(key=lambda row: int(row["repetition"]))
        endorsing = [
            row
            for row in candidates
            if row["annotation_score"] >= ENDORSEMENT_THRESHOLD
        ]
        nonendorsing = [
            row for row in candidates if row["annotation_score"] < ENDORSEMENT_THRESHOLD
        ]
        chosen = []
        if endorsing:
            chosen.append(endorsing[0])
        if nonendorsing:
            chosen.append(nonendorsing[0])
        selected[conversation_hash] = chosen
    return selected


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cohort", type=Path, required=True)
    parser.add_argument("--judgments", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--model", default=MODEL_ID)
    parser.add_argument("--revision", default=MODEL_REVISION)
    parser.add_argument("--count", type=int, default=40)
    parser.add_argument(
        "--subset-rule", choices=("smallest_hash", "shortest"), default="smallest_hash"
    )
    parser.add_argument(
        "--response-class-filter", choices=("any", "both"), default="any"
    )
    args = parser.parse_args()

    judgment_rows = read_jsonl(args.judgments)
    response_classes: dict[str, set[bool]] = defaultdict(set)
    for row in judgment_rows:
        if isinstance(row.get("annotation_score"), int) and not row.get("judge_error"):
            response_classes[row["conversation_hash"]].add(
                row["annotation_score"] >= ENDORSEMENT_THRESHOLD
            )
    cohort_rows = read_jsonl(args.cohort)
    if args.response_class_filter == "both":
        cohort_rows = [
            row
            for row in cohort_rows
            if response_classes.get(row["conversation_hash"]) == {False, True}
        ]
    if args.subset_rule == "shortest":
        cohort_rows.sort(key=lambda row: (row["input_tokens"], row["conversation_hash"]))
    else:
        cohort_rows.sort(key=lambda row: row["conversation_hash"])
    cohort_rows = cohort_rows[: args.count]
    cohort = {row["conversation_hash"]: row for row in cohort_rows}
    selected = select_responses(judgment_rows, set(cohort))
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
    completed = {
        (row["conversation_hash"], row["generation_sha256"])
        for row in read_jsonl(args.output)
        if row.get("complete") and not row.get("error")
    }
    processed = 0

    targets = [
        (conversation_hash, response)
        for conversation_hash in sorted(selected)
        for response in selected[conversation_hash]
    ]
    for conversation_hash, response in tqdm(targets, desc="Fixed-response AttriCoT"):
        identity = (conversation_hash, response["generation_sha256"])
        if identity in completed:
            continue
        source = cohort[conversation_hash]
        response_ids = [int(token) for token in response["generated_token_ids"]]
        try:
            baseline_sum, baseline_mean, baseline_prompt_tokens = (
                response_log_probability(
                    model, tokenizer, source["messages"], response_ids
                )
            )
            units = []
            for message_index in range(1, len(source["messages"]) - 1):
                perturbed = [
                    message
                    for index, message in enumerate(source["messages"])
                    if index != message_index
                ]
                deleted_sum, deleted_mean, deleted_prompt_tokens = (
                    response_log_probability(model, tokenizer, perturbed, response_ids)
                )
                units.append(
                    {
                        "message_index": message_index,
                        "role": source["messages"][message_index]["role"],
                        "relative_position": message_index
                        / (len(source["messages"]) - 1),
                        "deleted_prompt_tokens": deleted_prompt_tokens,
                        "deleted_log_probability_sum": deleted_sum,
                        "deleted_log_probability_mean": deleted_mean,
                        "attricot_sum_contribution": baseline_sum - deleted_sum,
                        "attricot_mean_contribution": baseline_mean - deleted_mean,
                    }
                )
            result = {
                "original_row_idx": source["original_row_idx"],
                "conversation_hash": conversation_hash,
                "generation_sha256": response["generation_sha256"],
                "repetition": response["repetition"],
                "annotation_score": response["annotation_score"],
                "endorsement": int(
                    response["annotation_score"] >= ENDORSEMENT_THRESHOLD
                ),
                "response_tokens": len(response_ids),
                "baseline_prompt_tokens": baseline_prompt_tokens,
                "baseline_log_probability_sum": baseline_sum,
                "baseline_log_probability_mean": baseline_mean,
                "units": units,
                "complete": True,
            }
            processed += 1
        except Exception as error:  # noqa: BLE001 - checkpoint GPU failures
            result = {
                "original_row_idx": source["original_row_idx"],
                "conversation_hash": conversation_hash,
                "generation_sha256": response["generation_sha256"],
                "error": repr(error),
            }
        append_jsonl(args.output, result)
        torch.cuda.empty_cache()

    final = read_jsonl(args.output)
    successful = {
        (row["conversation_hash"], row["generation_sha256"])
        for row in final
        if row.get("complete") and not row.get("error")
    }
    expected = {
        (conversation_hash, response["generation_sha256"])
        for conversation_hash, response in targets
    }
    paired = sum(len(responses) == 2 for responses in selected.values())
    manifest = {
        "cohort_sha256": sha256_file(args.cohort),
        "judgments_sha256": sha256_file(args.judgments),
        "model": args.model,
        "model_revision": args.revision,
        "subset_rule": args.subset_rule,
        "response_class_filter": args.response_class_filter,
        "subset_conversations": len(cohort),
        "conversations_with_both_response_classes": paired,
        "expected_fixed_responses": len(expected),
        "successful_fixed_responses": len(successful & expected),
        "missing_fixed_responses": len(expected - successful),
        "processed_this_run": processed,
        "method": (
            "AttriCoT default leave-one-unit-out intervention; with baseline plus "
            "one deletion per unit, the saturated linear-SCM coefficient equals "
            "baseline log probability minus deletion log probability."
        ),
        "unit": "complete prior user or assistant message",
        "target_unit": "one fixed generated assistant response",
        "selection_note": (
            "First endorsing and first non-endorsing sample per conversation when "
            "available; endpoint-conditioned and interpreted descriptively."
        ),
    }
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))
    if expected - successful:
        raise RuntimeError(f"Missing {len(expected - successful)} fixed responses")


if __name__ == "__main__":
    main()
