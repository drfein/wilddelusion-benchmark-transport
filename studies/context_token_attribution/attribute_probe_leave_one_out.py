from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from config import MODEL_ID, MODEL_REVISION
from io_utils import append_jsonl, read_jsonl, sha256_file
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer


def backbone(model: Any) -> Any:
    candidate = model.model
    if hasattr(candidate, "language_model"):
        candidate = candidate.language_model
    if hasattr(candidate, "model"):
        candidate = candidate.model
    if not hasattr(candidate, "layers"):
        raise TypeError("Could not locate transformer layers")
    return candidate


def prompt_hidden(model: Any, input_ids: torch.Tensor) -> np.ndarray:
    captured: list[torch.Tensor] = []

    def hook(_module: Any, _inputs: Any, output: Any) -> None:
        hidden = output[0] if isinstance(output, tuple) else output
        captured.append(hidden[:, -1, :])

    handle = backbone(model).layers[-1].register_forward_hook(hook)
    try:
        with torch.inference_mode():
            model(
                input_ids=input_ids,
                attention_mask=torch.ones_like(input_ids),
                use_cache=False,
                return_dict=True,
                logits_to_keep=1,
            )
    finally:
        handle.remove()
    return captured[0][0].float().cpu().numpy()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cohort", type=Path, required=True)
    parser.add_argument("--activation-dir", type=Path, required=True)
    parser.add_argument("--probe-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--model", default=MODEL_ID)
    parser.add_argument("--revision", default=MODEL_REVISION)
    parser.add_argument("--count", type=int, default=40)
    args = parser.parse_args()

    rows = sorted(read_jsonl(args.cohort), key=lambda row: row["conversation_hash"])[
        : args.count
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
    model_backbone = backbone(model)
    selected_layer_numbers = {
        int(str(np.load(path)["layer"].item()).rsplit("_", 1)[1])
        for path in args.probe_dir.glob("outer_fold_*.npz")
    }
    if not selected_layer_numbers:
        raise ValueError("No outer-fold probe checkpoints found")
    max_probe_layer = max(selected_layer_numbers)
    model_backbone.layers = torch.nn.ModuleList(
        list(model_backbone.layers[:max_probe_layer])
    )
    original_layers = model_backbone.layers
    torch.cuda.empty_cache()
    completed = {
        row["conversation_hash"]
        for row in read_jsonl(args.output)
        if row.get("complete") and not row.get("error")
    }
    processed = 0
    for row in tqdm(rows, desc="Probe message LOO"):
        conversation_hash = row["conversation_hash"]
        if conversation_hash in completed:
            continue
        checkpoint = np.load(args.probe_dir / f"outer_fold_{row['fold']}.npz")
        layer_key = str(checkpoint["layer"].item())
        layer_number = int(layer_key.rsplit("_", 1)[1])
        direction = checkpoint["raw_direction"].astype(np.float32)
        intercept = float(checkpoint["raw_intercept"])
        try:
            model_backbone.layers = torch.nn.ModuleList(
                list(original_layers[:layer_number])
            )
            baseline_activation = np.load(
                args.activation_dir / f"{conversation_hash}.npz"
            )[layer_key].astype(np.float32)
            baseline_logit = float(np.dot(direction, baseline_activation) + intercept)
            unit_results = []
            for message_index in range(1, len(row["messages"]) - 1):
                perturbed = [
                    message
                    for index, message in enumerate(row["messages"])
                    if index != message_index
                ]
                input_ids = tokenizer.apply_chat_template(
                    perturbed,
                    tokenize=True,
                    add_generation_prompt=True,
                    enable_thinking=False,
                    return_tensors="pt",
                ).to(model.device)
                hidden = prompt_hidden(model, input_ids)
                deleted_logit = float(np.dot(direction, hidden) + intercept)
                unit_results.append(
                    {
                        "message_index": message_index,
                        "role": row["messages"][message_index]["role"],
                        "relative_position": message_index / (len(row["messages"]) - 1),
                        "deleted_input_tokens": int(input_ids.shape[1]),
                        "deleted_probe_logit": deleted_logit,
                        "loo_contribution": baseline_logit - deleted_logit,
                    }
                )
            result = {
                "original_row_idx": row["original_row_idx"],
                "conversation_hash": conversation_hash,
                "fold": row["fold"],
                "probe_layer": layer_key,
                "baseline_probe_logit": baseline_logit,
                "units": unit_results,
                "complete": True,
            }
            processed += 1
        except Exception as error:  # noqa: BLE001 - checkpoint GPU failures
            result = {
                "original_row_idx": row["original_row_idx"],
                "conversation_hash": conversation_hash,
                "error": repr(error),
            }
        finally:
            model_backbone.layers = original_layers
        append_jsonl(args.output, result)
        torch.cuda.empty_cache()

    final = read_jsonl(args.output)
    successful = {
        row["conversation_hash"]
        for row in final
        if row.get("complete") and not row.get("error")
    }
    expected = {row["conversation_hash"] for row in rows}
    manifest = {
        "cohort_sha256": sha256_file(args.cohort),
        "model": args.model,
        "model_revision": args.revision,
        "subset_rule": "smallest conversation SHA-256 hashes",
        "expected_conversations": len(expected),
        "successful_conversations": len(successful & expected),
        "missing_conversations": len(expected - successful),
        "processed_this_run": processed,
        "unit": "complete prior user or assistant message",
        "score": "held-out-fold probe logit",
        "contribution": "baseline logit minus logit after unit deletion",
        "memory_optimization": (
            f"model permanently truncated to maximum selected layer {max_probe_layer}; "
            "each row is further truncated to its held-out-fold layer"
        ),
    }
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))
    if expected - successful:
        raise RuntimeError(f"Missing {len(expected - successful)} LOO rows")


if __name__ == "__main__":
    main()
