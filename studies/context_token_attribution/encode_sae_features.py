from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from io_utils import read_jsonl, sha256_file
from safetensors import safe_open


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cohort", type=Path, required=True)
    parser.add_argument("--activation-dir", type=Path, required=True)
    parser.add_argument("--sae-checkpoint", type=Path, required=True)
    parser.add_argument("--sae-metadata", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--layer-key", default="layer_20")
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    rows = sorted(read_jsonl(args.cohort), key=lambda row: row["conversation_hash"])
    activations = np.stack(
        [
            np.load(args.activation_dir / f"{row['conversation_hash']}.npz")[
                args.layer_key
            ].astype(np.float32)
            for row in rows
        ]
    )
    metadata = json.loads(args.sae_metadata.read_text())
    required = {"W_enc", "W_dec", "b_enc", "b_dec"}
    with safe_open(
        args.sae_checkpoint, framework="pt", device=args.device
    ) as checkpoint:
        if set(checkpoint.keys()) != required:
            raise ValueError(f"Unexpected SAE tensors: {sorted(checkpoint.keys())}")
        w_enc = checkpoint.get_tensor("W_enc")
        b_enc = checkpoint.get_tensor("b_enc")
        b_dec = checkpoint.get_tensor("b_dec")
    if metadata["hookpoint"] != "layers.19" or metadata["layer_name"] != "l20":
        raise ValueError("SAE metadata does not identify Qwen layer 20")
    if metadata["d_in"] != activations.shape[1]:
        raise ValueError("Activation width does not match SAE input width")

    device = torch.device(args.device)
    x = torch.from_numpy(activations).to(device)
    pre_activations = (x - b_dec) @ w_enc + b_enc
    values, indices = torch.topk(pre_activations, k=int(metadata["k"]), dim=-1)
    values = torch.relu(values)
    nonzero = int((values > 0).sum().item())

    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output,
        conversation_hashes=np.asarray(
            [row["conversation_hash"] for row in rows], dtype="U64"
        ),
        feature_indices=indices.to(torch.int32).cpu().numpy(),
        feature_values=values.to(torch.float32).cpu().numpy(),
    )
    manifest = {
        "cohort_sha256": sha256_file(args.cohort),
        "sae_checkpoint_sha256": sha256_file(args.sae_checkpoint),
        "sae_metadata_sha256": sha256_file(args.sae_metadata),
        "activation_layer_key": args.layer_key,
        "conversations": len(rows),
        "d_in": int(metadata["d_in"]),
        "d_sae": int(metadata["d_sae"]),
        "top_k": int(metadata["k"]),
        "positive_features": nonzero,
        "possible_sparse_slots": int(values.numel()),
        "encoder_formula": (
            "relu(topk((x - b_dec) @ W_enc + b_enc, k=64)); matches the "
            "checkpoint's exported ONNX graph"
        ),
        "normalizer_note": (
            "input_mean and input_std metadata are descriptive training statistics; "
            "the exported ONNX encoder subtracts b_dec and does not use them"
        ),
        "output_sha256": sha256_file(args.output),
    }
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
