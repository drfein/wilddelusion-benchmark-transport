from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
from config import (
    MAX_NEW_TOKENS,
    MODEL_ID,
    MODEL_REVISION,
    REPETITIONS,
    TEMPERATURE,
    TOP_LOGPROBS,
    TOP_P,
    repetition_seed,
    stable_hash,
)
from io_utils import append_jsonl, read_jsonl, sha256_file


def transformer_layers(model):
    candidate = model.model
    if hasattr(candidate, "language_model"):
        candidate = candidate.language_model
    if hasattr(candidate, "model"):
        candidate = candidate.model
    if not hasattr(candidate, "layers"):
        raise TypeError("Could not locate transformer layers")
    return candidate.layers


def selected_layer_indices(number_of_layers: int) -> list[int]:
    return sorted(
        {
            max(0, round(number_of_layers * fraction) - 1)
            for fraction in (0.25, 0.5, 0.75, 1.0)
        }
    )


from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer


def token_uncertainty(logits: torch.Tensor) -> dict[str, list[float | int]]:
    values = logits.float()
    log_z = torch.logsumexp(values, dim=-1, keepdim=True)
    log_probs = values - log_z
    probs = log_probs.exp()
    entropy = -(probs * log_probs).sum(dim=-1)
    top_values, _ = torch.topk(values, k=min(TOP_LOGPROBS, values.shape[-1]))
    top_probs = (top_values - log_z).exp()
    captured = top_probs.sum(dim=-1)
    margin = top_probs[:, 0] - top_probs[:, 1]
    top_nll = -(top_values[:, 0] - log_z[:, 0])
    cumulative = top_probs.cumsum(dim=-1)
    reached = cumulative.ge(0.9)
    nucleus = torch.where(
        reached.any(dim=-1),
        reached.float().argmax(dim=-1) + 1,
        torch.full((values.shape[0],), TOP_LOGPROBS + 1, device=values.device),
    )
    near_tie = top_probs.ge(0.9 * top_probs[:, :1]).float().mean(dim=-1)
    return {
        "entropy": entropy.cpu().tolist(),
        "margin": margin.cpu().tolist(),
        "top_nll": top_nll.cpu().tolist(),
        "nucleus_0p9": nucleus.cpu().tolist(),
        "near_tie": near_tie.cpu().tolist(),
        "top_mass": captured.cpu().tolist(),
    }


def generated_length(token_ids: list[int], eos_ids: set[int], pad_id: int) -> int:
    for index, token_id in enumerate(token_ids):
        if token_id in eos_ids:
            return index + 1
        if token_id == pad_id:
            return index
    return len(token_ids)


def sample_top_p(
    logits: torch.Tensor,
    *,
    temperature: float,
    top_p: float,
    generator: torch.Generator,
) -> torch.Tensor:
    scaled = logits.float() / temperature
    sorted_logits, sorted_indices = torch.sort(scaled, descending=True, dim=-1)
    sorted_probs = torch.softmax(sorted_logits, dim=-1)
    remove = sorted_probs.cumsum(dim=-1) - sorted_probs > top_p
    sorted_logits = sorted_logits.masked_fill(remove, -torch.inf)
    filtered = torch.full_like(scaled, -torch.inf).scatter(
        dim=-1, index=sorted_indices, src=sorted_logits
    )
    return torch.multinomial(
        torch.softmax(filtered, dim=-1), num_samples=1, generator=generator
    ).squeeze(-1)


def shared_prefix_generate(
    model,
    input_ids: torch.Tensor,
    *,
    repetitions: int,
    seed: int,
    eos_ids: set[int],
    pad_id: int,
    activation_path: Path | None,
) -> tuple[list[list[int]], list[dict[str, list[float | int]]], list[int]]:
    layers = transformer_layers(model)
    layer_indices = selected_layer_indices(len(layers))
    captured: dict[int, np.ndarray] = {}
    hooks = []
    if activation_path is not None and not activation_path.exists():
        for layer_index in layer_indices:

            def capture(_module, _inputs, output, *, index: int = layer_index) -> None:
                hidden = output[0] if isinstance(output, tuple) else output
                captured[index] = hidden[:, -1, :].detach().float().cpu().numpy()[0]

            hooks.append(layers[layer_index].register_forward_hook(capture))

    attention_mask = torch.ones_like(input_ids)
    try:
        with torch.inference_mode():
            output = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                use_cache=True,
                return_dict=True,
                logits_to_keep=1,
            )
    finally:
        for hook in hooks:
            hook.remove()
    if hooks:
        if set(captured) != set(layer_indices):
            raise RuntimeError(f"Captured layers {sorted(captured)}")
        activation_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            activation_path,
            **{
                f"layer_{index + 1:02d}": captured[index].astype(np.float16)
                for index in layer_indices
            },
        )

    logits = output.logits[:, -1, :].repeat(repetitions, 1)
    cache = output.past_key_values
    cache.batch_repeat_interleave(repetitions)
    attention_mask = attention_mask.repeat(repetitions, 1)
    generator = torch.Generator(device=model.device).manual_seed(seed)
    finished = torch.zeros(repetitions, dtype=torch.bool, device=model.device)
    token_steps: list[torch.Tensor] = []
    feature_steps: list[dict[str, list[float | int]]] = []
    sampled_nll_steps: list[list[float]] = []

    for _ in range(MAX_NEW_TOKENS):
        features = token_uncertainty(logits)
        tokens = sample_top_p(
            logits,
            temperature=TEMPERATURE,
            top_p=TOP_P,
            generator=generator,
        )
        tokens = torch.where(finished, torch.full_like(tokens, pad_id), tokens)
        sampled_nll = -torch.log_softmax(logits.float(), dim=-1).gather(
            -1, tokens[:, None]
        )[:, 0]
        token_steps.append(tokens.detach().cpu())
        feature_steps.append(features)
        sampled_nll_steps.append(sampled_nll.detach().cpu().tolist())
        finished |= torch.tensor(
            [int(token) in eos_ids for token in tokens.detach().cpu().tolist()],
            dtype=torch.bool,
            device=model.device,
        )
        if bool(finished.all()):
            break
        attention_mask = torch.cat(
            [
                attention_mask,
                torch.ones(
                    (repetitions, 1), dtype=attention_mask.dtype, device=model.device
                ),
            ],
            dim=1,
        )
        with torch.inference_mode():
            output = model(
                input_ids=tokens[:, None],
                attention_mask=attention_mask,
                past_key_values=cache,
                use_cache=True,
                return_dict=True,
                logits_to_keep=1,
            )
        logits = output.logits[:, -1, :]
        cache = output.past_key_values

    token_matrix = torch.stack(token_steps, dim=1).tolist()
    generated: list[list[int]] = []
    traces: list[dict[str, list[float | int]]] = []
    for sample_index, token_ids in enumerate(token_matrix):
        length = generated_length(token_ids, eos_ids=eos_ids, pad_id=pad_id)
        generated.append(token_ids[:length])
        trace = {
            key: [step[key][sample_index] for step in feature_steps[:length]]
            for key in feature_steps[0]
        }
        trace["sampled_nll"] = [
            step[sample_index] for step in sampled_nll_steps[:length]
        ]
        traces.append(trace)
    del cache, output
    return generated, traces, [index + 1 for index in layer_indices]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--activation-dir", type=Path)
    parser.add_argument("--activation-index", type=Path)
    parser.add_argument("--model", default=MODEL_ID)
    parser.add_argument("--revision", default=MODEL_REVISION)
    parser.add_argument("--repetitions", type=int, default=REPETITIONS)
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(
        args.model, revision=args.revision, trust_remote_code=False
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        revision=args.revision,
        torch_dtype=torch.bfloat16,
        device_map="cuda",
        attn_implementation="sdpa",
        trust_remote_code=False,
    ).eval()
    rows = sorted(read_jsonl(args.input), key=lambda row: row["input_tokens"])
    completed = {
        (row["prompt_content_sha256"], int(row["repetition"]))
        for row in read_jsonl(args.output)
        if row.get("response") and not row.get("error")
    }
    configured_eos = model.generation_config.eos_token_id
    if isinstance(configured_eos, int):
        eos_ids = {configured_eos}
    else:
        eos_ids = set(configured_eos or [])
    eos_ids.add(tokenizer.eos_token_id)
    generated_now = 0
    started_all = time.time()

    for row in tqdm(rows, desc="Conversations"):
        pending = [
            repetition
            for repetition in range(args.repetitions)
            if (row["prompt_content_sha256"], repetition) not in completed
        ]
        if not pending:
            continue
        input_ids = tokenizer.apply_chat_template(
            row["messages"],
            tokenize=True,
            add_generation_prompt=True,
            enable_thinking=False,
            return_tensors="pt",
        ).to(model.device)
        if input_ids.shape[1] != row["input_tokens"]:
            raise ValueError(
                f"Token drift for {row['original_row_idx']}: "
                f"{input_ids.shape[1]} != {row['input_tokens']}"
            )
        seed = repetition_seed(row["conversation_hash"], -1)
        started = time.time()
        activation_path = (
            args.activation_dir / f"{row['conversation_hash']}.npz"
            if args.activation_dir
            else None
        )
        try:
            generated_batch, trace_batch, captured_layers = shared_prefix_generate(
                model,
                input_ids,
                repetitions=args.repetitions,
                seed=seed,
                eos_ids=eos_ids,
                pad_id=tokenizer.pad_token_id,
                activation_path=activation_path,
            )
            if args.activation_index and activation_path:
                append_jsonl(
                    args.activation_index,
                    {
                        "original_row_idx": row["original_row_idx"],
                        "conversation_hash": row["conversation_hash"],
                        "fold": row["fold"],
                        "input_tokens": row["input_tokens"],
                        "activation_file": activation_path.name,
                        "layers": captured_layers,
                    },
                )
            for repetition in pending:
                generated = generated_batch[repetition]
                response = tokenizer.decode(generated, skip_special_tokens=True).strip()
                result = {
                    **{key: row[key] for key in row if key != "messages"},
                    "model": args.model,
                    "model_revision": args.revision,
                    "repetition": repetition,
                    "sample_slot": repetition,
                    "seed": seed,
                    "temperature": TEMPERATURE,
                    "top_p": TOP_P,
                    "max_new_tokens": MAX_NEW_TOKENS,
                    "generated_tokens": len(generated),
                    "response": response,
                    "generated_token_ids": generated,
                    "uncertainty": trace_batch[repetition],
                    "elapsed_seconds_shared_batch": time.time() - started,
                    "generation_sha256": stable_hash(
                        {
                            "prompt": row["prompt_content_sha256"],
                            "sample_slot": repetition,
                            "seed": seed,
                            "response": response,
                        }
                    ),
                }
                append_jsonl(args.output, result)
                generated_now += 1
        except Exception as error:  # noqa: BLE001 - checkpoint GPU failures
            for repetition in pending:
                append_jsonl(
                    args.output,
                    {
                        "original_row_idx": row["original_row_idx"],
                        "conversation_hash": row["conversation_hash"],
                        "prompt_content_sha256": row["prompt_content_sha256"],
                        "repetition": repetition,
                        "sample_slot": repetition,
                        "seed": seed,
                        "error": repr(error),
                    },
                )
        torch.cuda.empty_cache()

    final = read_jsonl(args.output)
    valid = [row for row in final if row.get("response") and not row.get("error")]
    expected = len(rows) * args.repetitions
    manifest = {
        "input_sha256": sha256_file(args.input),
        "model": args.model,
        "model_revision": args.revision,
        "eligible_prompts": len(rows),
        "repetitions": args.repetitions,
        "expected_generations": expected,
        "valid_generations": len(valid),
        "errors": sum(bool(row.get("error")) for row in final),
        "generated_this_run": generated_now,
        "elapsed_seconds_this_run": time.time() - started_all,
        "mean_generated_tokens": (
            sum(row["generated_tokens"] for row in valid) / len(valid)
            if valid
            else math.nan
        ),
        "checkpoint_policy": "append and flush after every response",
    }
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))
    if len(valid) != expected:
        raise RuntimeError(f"Expected {expected} valid generations, found {len(valid)}")


if __name__ == "__main__":
    main()
