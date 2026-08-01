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


def transformer_layers(model: Any) -> Any:
    candidate = model
    for attribute in ("model", "language_model", "model"):
        if hasattr(candidate, attribute):
            candidate = getattr(candidate, attribute)
    if hasattr(candidate, "layers"):
        return candidate.layers
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        return model.model.layers
    raise TypeError("Could not locate transformer layers")


def default_layer_indices(number_of_layers: int) -> list[int]:
    return sorted(
        {
            max(0, round(number_of_layers * fraction) - 1)
            for fraction in (0.25, 0.5, 0.75, 1.0)
        }
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--index", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--model", default=MODEL_ID)
    parser.add_argument("--revision", default=MODEL_REVISION)
    parser.add_argument("--layers", type=int, nargs="*")
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
    layers = transformer_layers(model)
    layer_indices = args.layers or default_layer_indices(len(layers))
    if not layer_indices or min(layer_indices) < 0 or max(layer_indices) >= len(layers):
        raise ValueError(
            f"Invalid layer indices {layer_indices} for {len(layers)} layers"
        )

    captured: dict[int, np.ndarray] = {}
    hooks = []
    for layer_index in layer_indices:

        def capture(
            _module: Any, _inputs: Any, output: Any, *, index: int = layer_index
        ) -> None:
            hidden = output[0] if isinstance(output, tuple) else output
            captured[index] = hidden[:, -1, :].detach().float().cpu().numpy()[0]

        hooks.append(layers[layer_index].register_forward_hook(capture))

    rows = sorted(read_jsonl(args.input), key=lambda row: row["input_tokens"])
    prior = {
        row["conversation_hash"]: row
        for row in read_jsonl(args.index)
        if row.get("activation_file") and not row.get("error")
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    processed = 0
    try:
        for row in tqdm(rows, desc="Prompt activations"):
            filename = f"{row['conversation_hash']}.npz"
            output_path = args.output_dir / filename
            if row["conversation_hash"] in prior and output_path.exists():
                continue
            captured.clear()
            started = time.time()
            try:
                input_ids = tokenizer.apply_chat_template(
                    row["messages"],
                    tokenize=True,
                    add_generation_prompt=True,
                    enable_thinking=False,
                    return_tensors="pt",
                ).to(model.device)
                if input_ids.shape[1] != row["input_tokens"]:
                    raise ValueError("Token count changed since cohort preparation")
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
                    **{
                        f"layer_{index + 1:02d}": captured[index].astype(np.float16)
                        for index in layer_indices
                    },
                )
                record = {
                    "original_row_idx": row["original_row_idx"],
                    "conversation_hash": row["conversation_hash"],
                    "fold": row["fold"],
                    "input_tokens": row["input_tokens"],
                    "activation_file": filename,
                    "layers": [index + 1 for index in layer_indices],
                    "hidden_size": int(next(iter(captured.values())).shape[0]),
                    "elapsed_seconds": time.time() - started,
                }
            except Exception as error:  # noqa: BLE001 - checkpoint GPU failures
                record = {
                    "original_row_idx": row["original_row_idx"],
                    "conversation_hash": row["conversation_hash"],
                    "error": repr(error),
                }
            append_jsonl(args.index, record)
            processed += int(not record.get("error"))
            torch.cuda.empty_cache()
    finally:
        for hook in hooks:
            hook.remove()

    final = read_jsonl(args.index)
    successful = {row["conversation_hash"] for row in final if not row.get("error")}
    expected = {row["conversation_hash"] for row in rows}
    manifest = {
        "input_sha256": sha256_file(args.input),
        "model": args.model,
        "model_revision": args.revision,
        "number_of_layers": len(layers),
        "captured_layers": [index + 1 for index in layer_indices],
        "expected_prompts": len(expected),
        "successful_prompts": len(successful & expected),
        "processed_this_run": processed,
        "missing_prompts": len(expected - successful),
        "pooling": "final prompt token before assistant generation",
    }
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))
    if expected - successful:
        raise RuntimeError(f"Missing {len(expected - successful)} activations")


if __name__ == "__main__":
    main()
