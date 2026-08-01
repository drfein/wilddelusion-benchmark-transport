from __future__ import annotations

import argparse
import hashlib
import json
import random
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from tqdm import tqdm
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoModelForImageTextToText,
    AutoTokenizer,
)


def read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def append_jsonl(path: Path, row: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        handle.flush()


def stable_hash(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, ensure_ascii=False, sort_keys=True).encode()
    ).hexdigest()


def transformer_layers(model: Any) -> Any:
    candidates = [
        model,
        getattr(model, "model", None),
        getattr(model, "language_model", None),
        getattr(getattr(model, "model", None), "language_model", None),
    ]
    for candidate in list(candidates):
        if candidate is not None:
            candidates.extend(
                [getattr(candidate, "model", None), getattr(candidate, "transformer", None)]
            )
    for candidate in candidates:
        if candidate is not None and hasattr(candidate, "layers"):
            return candidate.layers
    raise TypeError("Could not locate transformer layers")


def template_kwargs(tokenizer: Any) -> dict:
    name = f"{tokenizer.__class__.__name__} {tokenizer.name_or_path}".lower()
    return {"enable_thinking": False} if "qwen" in name else {}


def load_model(model_name: str, revision: str | None) -> Any:
    config = AutoConfig.from_pretrained(model_name, revision=revision)
    loader = (
        AutoModelForImageTextToText
        if "Gemma3ForConditionalGeneration" in (config.architectures or [])
        else AutoModelForCausalLM
    )
    return loader.from_pretrained(
        model_name,
        revision=revision,
        torch_dtype=torch.bfloat16,
        device_map="cuda",
        attn_implementation="sdpa",
        trust_remote_code=False,
    ).eval()


def generated_length(tokens: torch.Tensor, eos_ids: set[int], pad_id: int) -> int:
    count = 0
    for token in tokens.tolist():
        if token == pad_id and token not in eos_ids:
            break
        if token in eos_ids:
            break
        count += 1
    return count


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cohort", type=Path, required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--revision")
    parser.add_argument("--model-key", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--activation-dir", type=Path, required=True)
    parser.add_argument("--activation-index", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--repetitions", type=int, default=5)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--seed", type=int, default=20260801)
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()

    rows = read_jsonl(args.cohort)
    if args.limit is not None:
        rows = rows[: args.limit]
    tokenizer = AutoTokenizer.from_pretrained(
        args.model, revision=args.revision, trust_remote_code=False
    )
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    model = load_model(args.model, args.revision)
    layers = transformer_layers(model)
    layer_indices = sorted(
        {max(0, round(len(layers) * fraction) - 1) for fraction in (0.25, 0.5, 0.75, 1.0)}
    )
    captured: dict[int, np.ndarray] = {}
    hooks = []
    for layer_index in layer_indices:

        def capture(
            _module: Any, _inputs: Any, output: Any, *, index: int = layer_index
        ) -> None:
            if index in captured:
                return
            hidden = output[0] if isinstance(output, tuple) else output
            captured[index] = hidden[:, -1, :].detach().float().cpu().numpy()[0]

        hooks.append(layers[layer_index].register_forward_hook(capture))

    completed_rows = read_jsonl(args.output)
    completed = {
        (row["conversation_hash"], int(row["repetition"]))
        for row in completed_rows
        if row.get("response") and not row.get("error")
    }
    activation_done = {
        row["conversation_hash"]
        for row in read_jsonl(args.activation_index)
        if row.get("activation_file") and not row.get("error")
    }
    eos_config = model.generation_config.eos_token_id
    eos_ids = set(eos_config if isinstance(eos_config, list) else [eos_config])
    eos_ids.discard(None)
    args.activation_dir.mkdir(parents=True, exist_ok=True)
    started = time.time()
    try:
        for row in tqdm(rows, desc=f"{args.model_key} real histories"):
            pending = [
                repetition
                for repetition in range(args.repetitions)
                if (row["conversation_hash"], repetition) not in completed
            ]
            if not pending:
                continue
            rendered = tokenizer.apply_chat_template(
                row["messages"],
                tokenize=False,
                add_generation_prompt=True,
                **template_kwargs(tokenizer),
            )
            tokenized = tokenizer(rendered, return_tensors="pt", add_special_tokens=False)
            input_ids = tokenized["input_ids"].to(model.device)
            attention_mask = tokenized["attention_mask"].to(model.device)
            repeated_ids = input_ids.repeat(args.repetitions, 1)
            repeated_mask = attention_mask.repeat(args.repetitions, 1)
            conversation_seed = int(
                stable_hash([args.seed, row["conversation_hash"]])[:8], 16
            )
            random.seed(conversation_seed)
            np.random.seed(conversation_seed % (2**32 - 1))
            torch.manual_seed(conversation_seed)
            torch.cuda.manual_seed_all(conversation_seed)
            captured.clear()
            batch_started = time.time()
            with torch.inference_mode():
                output = model.generate(
                    input_ids=repeated_ids,
                    attention_mask=repeated_mask,
                    do_sample=True,
                    temperature=args.temperature,
                    top_p=args.top_p,
                    max_new_tokens=args.max_new_tokens,
                    pad_token_id=tokenizer.pad_token_id,
                    use_cache=True,
                )
            if set(captured) != set(layer_indices):
                raise RuntimeError(f"Captured layers {sorted(captured)}")
            activation_file = f"{row['conversation_hash']}.npz"
            activation_path = args.activation_dir / activation_file
            if row["conversation_hash"] not in activation_done:
                np.savez_compressed(
                    activation_path,
                    **{
                        f"layer_{index + 1:02d}": captured[index].astype(np.float16)
                        for index in layer_indices
                    },
                )
                append_jsonl(
                    args.activation_index,
                    {
                        "conversation_hash": row["conversation_hash"],
                        "original_row_idx": row["original_row_idx"],
                        "fold": row["fold"],
                        "activation_file": activation_file,
                        "layers": [index + 1 for index in layer_indices],
                        "hidden_size": int(next(iter(captured.values())).shape[0]),
                        "model_key": args.model_key,
                        "input_tokens": int(input_ids.shape[1]),
                    },
                )
                activation_done.add(row["conversation_hash"])
            generated = output[:, repeated_ids.shape[1] :]
            responses = tokenizer.batch_decode(generated, skip_special_tokens=True)
            for repetition in pending:
                response = responses[repetition].strip()
                token_count = generated_length(
                    generated[repetition], eos_ids, tokenizer.pad_token_id
                )
                generation_sha = stable_hash(
                    [args.model_key, row["conversation_hash"], repetition, response]
                )
                result = {
                    **{key: value for key, value in row.items() if key != "messages"},
                    "model": args.model,
                    "model_revision": args.revision,
                    "model_key": args.model_key,
                    "repetition": repetition,
                    "seed": conversation_seed,
                    "temperature": args.temperature,
                    "top_p": args.top_p,
                    "max_new_tokens": args.max_new_tokens,
                    "input_tokens_model": int(input_ids.shape[1]),
                    "generated_tokens": token_count,
                    "hit_token_cap": token_count >= args.max_new_tokens,
                    "response": response,
                    "generation_sha256": generation_sha,
                    "activation_file": activation_file,
                    "elapsed_seconds_shared_batch": time.time() - batch_started,
                }
                append_jsonl(args.output, result)
                completed.add((row["conversation_hash"], repetition))
    finally:
        for hook in hooks:
            hook.remove()

    final = [row for row in read_jsonl(args.output) if not row.get("error")]
    expected = len(rows) * args.repetitions
    manifest = {
        "model": args.model,
        "model_revision": args.revision,
        "model_key": args.model_key,
        "real_histories": len(rows),
        "repetitions": args.repetitions,
        "expected_generations": expected,
        "valid_generations": len(final),
        "activation_files": len(activation_done),
        "captured_layers": [index + 1 for index in layer_indices],
        "max_new_tokens": args.max_new_tokens,
        "hit_token_cap": sum(bool(row.get("hit_token_cap")) for row in final),
        "elapsed_seconds": time.time() - started,
    }
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))
    if len(final) != expected or len(activation_done) != len(rows):
        raise RuntimeError("Incomplete real-history generation or activations")


if __name__ == "__main__":
    main()
