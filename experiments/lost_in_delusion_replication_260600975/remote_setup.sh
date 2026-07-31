#!/usr/bin/env bash
set -euo pipefail

ROOT=${1:-/workspace/lost_in_delusion_replication_260600975}
mkdir -p "$ROOT" /workspace/.hf_home
export HF_HOME=/workspace/.hf_home

if [[ ! -x "$ROOT/.venv/bin/python" ]]; then
  uv venv --python 3.12 "$ROOT/.venv"
fi

source "$ROOT/.venv/bin/activate"
uv pip install \
  --index-url https://download.pytorch.org/whl/cu128 \
  torch==2.8.0
uv pip install \
  "transformers>=4.56,<5" \
  accelerate \
  safetensors \
  sentencepiece \
  protobuf \
  tqdm

python - <<'PY'
import torch
import transformers

print("torch", torch.__version__)
print("transformers", transformers.__version__)
print("cuda", torch.cuda.is_available(), torch.version.cuda)
print("gpus", torch.cuda.device_count())
for index in range(torch.cuda.device_count()):
    print(index, torch.cuda.get_device_name(index))
PY
