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


def backbone_layers(model: Any) -> Any:
    candidate = model.model
    if hasattr(candidate, "language_model"):
        candidate = candidate.language_model
    if hasattr(candidate, "model"):
        candidate = candidate.model
    if not hasattr(candidate, "layers"):
        raise TypeError("Could not locate transformer layers")
    return candidate


def content_message_indices(
    tokenizer: Any, messages: list[dict[str, str]]
) -> tuple[list[int], list[int]]:
    rendered = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    encoded = tokenizer(
        rendered,
        add_special_tokens=False,
        return_offsets_mapping=True,
    )
    message_char_spans = []
    cursor = 0
    for message_index, message in enumerate(messages):
        content = message["content"]
        start = rendered.find(content, cursor)
        if start < 0:
            raise ValueError(f"Could not align message {message_index}")
        end = start + len(content)
        message_char_spans.append((start, end, message_index))
        cursor = end
    token_messages = np.full(len(encoded["input_ids"]), -1, dtype=np.int32)
    span_index = 0
    for token_index, (start, end) in enumerate(encoded["offset_mapping"]):
        while (
            span_index < len(message_char_spans)
            and start >= message_char_spans[span_index][1]
        ):
            span_index += 1
        if span_index >= len(message_char_spans):
            break
        message_start, message_end, message_index = message_char_spans[span_index]
        if end > message_start and start < message_end:
            token_messages[token_index] = message_index
    return list(encoded["input_ids"]), token_messages.tolist()


def probe_attribution(
    model: Any,
    input_ids: torch.Tensor,
    direction: torch.Tensor,
    *,
    integrated_gradient_steps: int,
) -> tuple[np.ndarray, np.ndarray | None, float]:
    embeddings = model.get_input_embeddings()(input_ids).detach()
    attention_mask = torch.ones_like(input_ids)

    def gradient_at(alpha: float) -> tuple[torch.Tensor, float]:
        scaled = (embeddings * alpha).detach().requires_grad_(True)
        captured: list[torch.Tensor] = []

        def hook(_module: Any, _inputs: Any, output: Any) -> None:
            hidden = output[0] if isinstance(output, tuple) else output
            captured.append(hidden[:, -1, :])

        handle = backbone_layers(model).layers[-1].register_forward_hook(hook)
        try:
            model(
                inputs_embeds=scaled,
                attention_mask=attention_mask,
                use_cache=False,
                return_dict=True,
                logits_to_keep=1,
            )
            score = torch.dot(captured[0][0].float(), direction)
            gradient = torch.autograd.grad(score, scaled)[0]
        finally:
            handle.remove()
        return gradient.detach(), float(score.detach().cpu())

    gradient, score = gradient_at(1.0)
    gradient_times_input = (gradient * embeddings).sum(dim=-1)[0].float().cpu().numpy()
    integrated = None
    if integrated_gradient_steps > 0:
        accumulated = torch.zeros_like(embeddings)
        for step in range(1, integrated_gradient_steps + 1):
            step_gradient, _ = gradient_at(step / integrated_gradient_steps)
            accumulated += step_gradient
        integrated = (
            (embeddings * accumulated / integrated_gradient_steps)
            .sum(dim=-1)[0]
            .float()
            .cpu()
            .numpy()
        )
    return gradient_times_input, integrated, score


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cohort", type=Path, required=True)
    parser.add_argument("--probe-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--index", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--model", default=MODEL_ID)
    parser.add_argument("--revision", default=MODEL_REVISION)
    parser.add_argument("--integrated-gradient-count", type=int, default=40)
    parser.add_argument("--integrated-gradient-steps", type=int, default=16)
    args = parser.parse_args()

    rows = sorted(read_jsonl(args.cohort), key=lambda row: row["conversation_hash"])
    ig_hashes = {
        row["conversation_hash"] for row in rows[: args.integrated_gradient_count]
    }
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
    model.gradient_checkpointing_enable(
        gradient_checkpointing_kwargs={"use_reentrant": False}
    )
    backbone = backbone_layers(model)
    original_layers = backbone.layers
    prior = {
        row["conversation_hash"]
        for row in read_jsonl(args.index)
        if row.get("attribution_file") and not row.get("error")
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    processed = 0

    for row in tqdm(rows, desc="Probe token attribution"):
        conversation_hash = row["conversation_hash"]
        output_path = args.output_dir / f"{conversation_hash}.npz"
        if conversation_hash in prior and output_path.exists():
            continue
        checkpoint = np.load(args.probe_dir / f"outer_fold_{row['fold']}.npz")
        layer_key = str(checkpoint["layer"].item())
        layer_number = int(layer_key.rsplit("_", 1)[1])
        direction = torch.tensor(
            checkpoint["raw_direction"], dtype=torch.float32, device=model.device
        )
        try:
            backbone.layers = torch.nn.ModuleList(list(original_layers[:layer_number]))
            expected_ids, message_indices = content_message_indices(
                tokenizer, row["messages"]
            )
            input_ids = tokenizer.apply_chat_template(
                row["messages"],
                tokenize=True,
                add_generation_prompt=True,
                enable_thinking=False,
                return_tensors="pt",
            ).to(model.device)
            if expected_ids != input_ids[0].tolist():
                raise ValueError("Offset-token alignment disagrees with chat template")
            use_ig = conversation_hash in ig_hashes
            gradient_x_input, integrated, score = probe_attribution(
                model,
                input_ids,
                direction,
                integrated_gradient_steps=(
                    args.integrated_gradient_steps if use_ig else 0
                ),
            )
            payload = {
                "token_ids": np.asarray(expected_ids, dtype=np.int32),
                "message_indices": np.asarray(message_indices, dtype=np.int32),
                "gradient_times_input": gradient_x_input.astype(np.float32),
            }
            if integrated is not None:
                payload["integrated_gradients"] = integrated.astype(np.float32)
            np.savez_compressed(output_path, **payload)
            record = {
                "original_row_idx": row["original_row_idx"],
                "conversation_hash": conversation_hash,
                "fold": row["fold"],
                "input_tokens": row["input_tokens"],
                "probe_layer": layer_key,
                "probe_score_without_intercept": score,
                "integrated_gradients": use_ig,
                "attribution_file": output_path.name,
            }
            processed += 1
        except Exception as error:  # noqa: BLE001 - checkpoint long-context failures
            record = {
                "original_row_idx": row["original_row_idx"],
                "conversation_hash": conversation_hash,
                "error": repr(error),
            }
        finally:
            backbone.layers = original_layers
        append_jsonl(args.index, record)
        torch.cuda.empty_cache()

    final = read_jsonl(args.index)
    successful = {row["conversation_hash"] for row in final if not row.get("error")}
    expected = {row["conversation_hash"] for row in rows}
    manifest = {
        "cohort_sha256": sha256_file(args.cohort),
        "model": args.model,
        "model_revision": args.revision,
        "expected_conversations": len(expected),
        "successful_conversations": len(successful & expected),
        "missing_conversations": len(expected - successful),
        "processed_this_run": processed,
        "gradient_method": "signed gradient times input embedding",
        "integrated_gradient_steps": args.integrated_gradient_steps,
        "integrated_gradient_subset": len(ig_hashes),
        "integrated_gradient_subset_rule": "smallest conversation SHA-256 hashes",
        "probe_leakage_control": "each row uses the probe trained without its outer fold",
    }
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))
    if expected - successful:
        raise RuntimeError(f"Missing {len(expected - successful)} attributions")


if __name__ == "__main__":
    main()
