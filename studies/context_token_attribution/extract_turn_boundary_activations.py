from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from config import MODEL_ID, MODEL_REVISION
from io_utils import append_jsonl, read_jsonl, sha256_file
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer


def transformer_backbone(model: Any) -> Any:
    candidate = model
    for attribute in ("model", "language_model", "model"):
        if hasattr(candidate, attribute):
            candidate = getattr(candidate, attribute)
    if hasattr(candidate, "layers"):
        return candidate
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        return model.model
    raise TypeError("Could not locate transformer layers")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--index", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--model", default=MODEL_ID)
    parser.add_argument("--revision", default=MODEL_REVISION)
    parser.add_argument("--layers", type=int, nargs="+", default=[18, 27])
    args = parser.parse_args()

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
    backbone = transformer_backbone(model)
    number_of_layers = len(backbone.layers)
    layer_indices = sorted({layer - 1 for layer in args.layers})
    if min(layer_indices) < 0 or max(layer_indices) >= number_of_layers:
        raise ValueError(f"Invalid layers {args.layers} for {number_of_layers} layers")
    max_layer = max(layer_indices)
    backbone.layers = torch.nn.ModuleList(list(backbone.layers[: max_layer + 1]))

    endpoint_positions: list[int] = []
    captured: dict[int, np.ndarray] = {}
    hooks = []
    for layer_index in layer_indices:

        def capture(
            _module: Any, _inputs: Any, output: Any, *, index: int = layer_index
        ) -> None:
            hidden = output[0] if isinstance(output, tuple) else output
            captured[index] = (
                hidden[0, endpoint_positions, :].detach().float().cpu().numpy()
            )

        hooks.append(backbone.layers[layer_index].register_forward_hook(capture))

    rows = sorted(read_jsonl(args.input), key=lambda row: row["input_tokens"])
    completed = {
        row["conversation_hash"]
        for row in read_jsonl(args.index)
        if row.get("activation_file") and not row.get("error")
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    processed = 0
    try:
        for row in tqdm(rows, desc="Turn-boundary activations"):
            conversation_hash = row["conversation_hash"]
            output_path = args.output_dir / f"{conversation_hash}.npz"
            if conversation_hash in completed and output_path.exists():
                continue
            started = time.time()
            captured.clear()
            try:
                full_ids = tokenizer.apply_chat_template(
                    row["messages"],
                    tokenize=True,
                    add_generation_prompt=True,
                    enable_thinking=False,
                )
                if len(full_ids) != row["input_tokens"]:
                    raise ValueError("Full token count changed since cohort preparation")
                message_indices = []
                endpoint_positions.clear()
                for message_index, message in enumerate(row["messages"]):
                    if message["role"] != "user":
                        continue
                    prefix_ids = tokenizer.apply_chat_template(
                        row["messages"][: message_index + 1],
                        tokenize=True,
                        add_generation_prompt=True,
                        enable_thinking=True,
                    )
                    if full_ids[: len(prefix_ids)] != prefix_ids:
                        raise ValueError(
                            f"User prefix {message_index} is not an exact full-prompt prefix"
                        )
                    message_indices.append(message_index)
                    endpoint_positions.append(len(prefix_ids) - 1)
                input_ids = torch.as_tensor(
                    full_ids, dtype=torch.long, device=model.device
                ).unsqueeze(0)
                with torch.inference_mode():
                    model(
                        input_ids=input_ids,
                        attention_mask=torch.ones_like(input_ids),
                        use_cache=False,
                        return_dict=True,
                    )
                if set(captured) != set(layer_indices):
                    raise RuntimeError(f"Captured layers {sorted(captured)}")
                np.savez_compressed(
                    output_path,
                    message_indices=np.asarray(message_indices, dtype=np.int32),
                    token_positions=np.asarray(endpoint_positions, dtype=np.int32),
                    **{
                        f"layer_{index + 1:02d}": captured[index].astype(np.float16)
                        for index in layer_indices
                    },
                )
                record = {
                    "original_row_idx": row["original_row_idx"],
                    "conversation_hash": conversation_hash,
                    "fold": int(row["fold"]),
                    "input_tokens": int(row["input_tokens"]),
                    "user_turn_boundaries": len(endpoint_positions),
                    "activation_file": output_path.name,
                    "layers": [index + 1 for index in layer_indices],
                    "elapsed_seconds": time.time() - started,
                }
            except Exception as error:  # noqa: BLE001 - preserve resumable failures
                record = {
                    "original_row_idx": row["original_row_idx"],
                    "conversation_hash": conversation_hash,
                    "error": repr(error),
                }
            append_jsonl(args.index, record)
            processed += int(not record.get("error"))
            torch.cuda.empty_cache()
    finally:
        for hook in hooks:
            hook.remove()

    successful = {
        row["conversation_hash"]
        for row in read_jsonl(args.index)
        if row.get("activation_file") and not row.get("error")
    }
    expected = {row["conversation_hash"] for row in rows}
    manifest = {
        "input_sha256": sha256_file(args.input),
        "model": args.model,
        "model_revision": args.revision,
        "number_of_layers": number_of_layers,
        "captured_layers": [index + 1 for index in layer_indices],
        "expected_conversations": len(expected),
        "successful_conversations": len(successful & expected),
        "processed_this_run": processed,
        "missing_conversations": len(expected - successful),
        "endpoint": (
            "final token of header-only assistant generation prompt after every user turn"
        ),
        "exactness": (
            "each header-only user prefix was asserted to be an exact prefix of the "
            "complete false-thinking rendered prompt"
        ),
        "memory_optimization": f"model truncated after layer {max_layer + 1}",
    }
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))
    if expected - successful:
        raise RuntimeError(f"Missing {len(expected - successful)} conversations")


if __name__ == "__main__":
    main()
