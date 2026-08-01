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
        json.dumps(value, sort_keys=True, ensure_ascii=False).encode()
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
    parser.add_argument("--cases", type=Path, required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--revision")
    parser.add_argument("--model-key", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--activation-dir", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--repetitions", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--max-new-tokens", type=int, default=1536)
    parser.add_argument("--seed", type=int, default=20260801)
    parser.add_argument("--case-id", action="append", dest="case_ids")
    parser.add_argument("--limit-cases", type=int)
    args = parser.parse_args()

    cases = json.loads(args.cases.read_text())["cases"]
    if args.case_ids:
        requested = set(args.case_ids)
        cases = [case for case in cases if case["id"] in requested]
        missing = requested - {case["id"] for case in cases}
        if missing:
            raise ValueError(f"Unknown case IDs: {sorted(missing)}")
    if args.limit_cases is not None:
        if args.limit_cases <= 0:
            raise ValueError("--limit-cases must be positive")
        cases = cases[: args.limit_cases]
    if not cases:
        raise ValueError("No cases selected")

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
            captured[index] = hidden[:, -1, :].detach().float().cpu().numpy()

        hooks.append(layers[layer_index].register_forward_hook(capture))

    completed_rows = read_jsonl(args.output)
    completed = {
        (row["trajectory_id"], int(row["turn_number"])): row
        for row in completed_rows
        if row.get("response") and not row.get("error")
    }
    states = []
    for case in cases:
        for repetition in range(args.repetitions):
            trajectory_id = stable_hash(
                {"model_key": args.model_key, "case_id": case["id"], "repetition": repetition}
            )
            states.append(
                {
                    "case": case,
                    "repetition": repetition,
                    "trajectory_id": trajectory_id,
                    "messages": [],
                }
            )

    eos_config = model.generation_config.eos_token_id
    eos_ids = set(eos_config if isinstance(eos_config, list) else [eos_config])
    eos_ids.discard(None)
    args.activation_dir.mkdir(parents=True, exist_ok=True)
    started = time.time()
    capped = 0
    try:
        max_turns = max(len(state["case"]["prompts"]) for state in states)
        progress = tqdm(total=sum(len(state["case"]["prompts"]) for state in states), desc=args.model_key)
        for turn_index in range(max_turns):
            active = []
            for state in states:
                prompts = state["case"]["prompts"]
                if turn_index >= len(prompts):
                    continue
                turn_number = turn_index + 1
                state["messages"].append({"role": "user", "content": prompts[turn_index]})
                prior = completed.get((state["trajectory_id"], turn_number))
                if prior:
                    state["messages"].append({"role": "assistant", "content": prior["response"]})
                    progress.update(1)
                else:
                    active.append(state)

            for offset in range(0, len(active), args.batch_size):
                batch = active[offset : offset + args.batch_size]
                row_started = time.time()
                rendered = [
                    tokenizer.apply_chat_template(
                        state["messages"],
                        tokenize=False,
                        add_generation_prompt=True,
                        **template_kwargs(tokenizer),
                    )
                    for state in batch
                ]
                inputs = tokenizer(
                    rendered,
                    return_tensors="pt",
                    padding=True,
                    add_special_tokens=False,
                ).to(model.device)
                seeds = [
                    args.seed + int(state["trajectory_id"][:8], 16) + turn_index + 1
                    for state in batch
                ]
                # Transformers samples one batch from one generator; this deterministic
                # batch seed is recorded and folds never split repetitions.
                batch_seed = int(stable_hash(seeds)[:8], 16)
                random.seed(batch_seed)
                np.random.seed(batch_seed % (2**32 - 1))
                torch.manual_seed(batch_seed)
                torch.cuda.manual_seed_all(batch_seed)
                captured.clear()
                with torch.inference_mode():
                    output = model.generate(
                        **inputs,
                        do_sample=True,
                        temperature=args.temperature,
                        top_p=args.top_p,
                        max_new_tokens=args.max_new_tokens,
                        pad_token_id=tokenizer.pad_token_id,
                        use_cache=True,
                    )
                if set(captured) != set(layer_indices):
                    raise RuntimeError(f"Captured layers {sorted(captured)}")
                generated = output[:, inputs["input_ids"].shape[1] :]
                responses = tokenizer.batch_decode(generated, skip_special_tokens=True)
                for batch_index, (state, response) in enumerate(zip(batch, responses, strict=True)):
                    response = response.strip()
                    token_count = generated_length(
                        generated[batch_index], eos_ids, tokenizer.pad_token_id
                    )
                    capped += token_count >= args.max_new_tokens
                    turn_number = turn_index + 1
                    activation_file = f"{state['trajectory_id']}_turn_{turn_number:02d}.npz"
                    np.savez_compressed(
                        args.activation_dir / activation_file,
                        **{
                            f"layer_{index + 1:02d}": captured[index][batch_index].astype(np.float16)
                            for index in layer_indices
                        },
                    )
                    case = state["case"]
                    row = {
                        "trajectory_id": state["trajectory_id"],
                        "scenario_pair": case["id"].rsplit("_", 1)[0],
                        "case_id": case["id"],
                        "case_name": case["name"],
                        "theme": case["theme"],
                        "condition": case["condition"],
                        "harm_type": case["harm_type"],
                        "repetition": state["repetition"],
                        "turn_number": turn_number,
                        "user_text": case["prompts"][turn_index],
                        "response": response,
                        "response_sha256": hashlib.sha256(response.encode()).hexdigest(),
                        "model_key": args.model_key,
                        "model": args.model,
                        "revision": args.revision,
                        "input_tokens": int(inputs["attention_mask"][batch_index].sum()),
                        "generated_tokens": token_count,
                        "hit_token_cap": token_count >= args.max_new_tokens,
                        "activation_file": activation_file,
                        "layers": [index + 1 for index in layer_indices],
                        "seed": seeds[batch_index],
                        "batch_seed": batch_seed,
                        "elapsed_batch_seconds": time.time() - row_started,
                    }
                    append_jsonl(args.output, row)
                    completed[(state["trajectory_id"], turn_number)] = row
                    state["messages"].append({"role": "assistant", "content": response})
                    progress.update(1)
        progress.close()
    finally:
        for hook in hooks:
            hook.remove()

    rows = [row for row in read_jsonl(args.output) if not row.get("error")]
    expected = sum(len(case["prompts"]) for case in cases) * args.repetitions
    manifest = {
        "model_key": args.model_key,
        "model": args.model,
        "revision": args.revision,
        "cases": len(cases),
        "scenario_pairs": len({case["id"].rsplit("_", 1)[0] for case in cases}),
        "repetitions": args.repetitions,
        "batch_size": args.batch_size,
        "expected_turns": expected,
        "successful_turns": len(rows),
        "captured_layers": [index + 1 for index in layer_indices],
        "temperature": args.temperature,
        "top_p": args.top_p,
        "max_new_tokens": args.max_new_tokens,
        "hit_token_cap": sum(bool(row.get("hit_token_cap")) for row in rows),
        "elapsed_seconds": time.time() - started,
    }
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))
    if len(rows) != expected:
        raise RuntimeError(f"Expected {expected} rows, found {len(rows)}")


if __name__ == "__main__":
    main()
