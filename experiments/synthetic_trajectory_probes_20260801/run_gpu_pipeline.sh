#!/usr/bin/env bash
set -euo pipefail

ROOT=${ROOT:-/workspace/synthetic_trajectory_probes}
CODE="$ROOT/code"
CASES="$ROOT/input/test_cases.json"
QWEN=${QWEN:-/workspace/models/Qwen3-8B}
LLAMA=${LLAMA:-/dev/shm/model-cache/Llama-3.1-8B-Instruct}
GEMMA=${GEMMA:-/dev/shm/model-cache/gemma-3-12b-it}

wait_for_supervisor_exit() {
  local program=$1
  while supervisorctl status "$program" 2>/dev/null | grep -q RUNNING; do
    sleep 30
  done
}

validate_manifest() {
  local manifest=$1
  /venv/main/bin/python - "$manifest" <<'PY'
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
data = json.loads(path.read_text())
if data["successful_turns"] != data["expected_turns"]:
    raise SystemExit(f"Incomplete rollout: {data}")
PY
}

rollout() {
  local model_key=$1
  local model_path=$2
  local batch_size=$3
  local run_dir="$ROOT/runs/$model_key"
  mkdir -p "$run_dir/activations"
  /venv/main/bin/python "$CODE/rollout_batched_trajectories.py" \
    --cases "$CASES" \
    --model "$model_path" \
    --model-key "$model_key" \
    --output "$run_dir/responses.jsonl" \
    --activation-dir "$run_dir/activations" \
    --manifest "$run_dir/manifest.json" \
    --repetitions 3 \
    --temperature 0.7 \
    --top-p 0.9 \
    --batch-size "$batch_size" \
    --max-new-tokens 1536
  validate_manifest "$run_dir/manifest.json"
}

judge_and_fit() {
  local model_key=$1
  local run_dir="$ROOT/runs/$model_key"
  /venv/main/bin/python "$CODE/judge_trajectories.py" \
    --responses "$run_dir/responses.jsonl" \
    --output "$run_dir/judgments_qwen3_8b.jsonl" \
    --manifest "$run_dir/judgments_qwen3_8b_manifest.json" \
    --judge-model "$QWEN" \
    --judge-key qwen3_8b_dcs \
    --batch-size 8
  /venv/main/bin/python "$CODE/fit_trajectory_probes.py" \
    --responses "$run_dir/responses.jsonl" \
    --judgments "$run_dir/judgments_qwen3_8b.jsonl" \
    --activation-dir "$run_dir/activations" \
    --output-dir "$run_dir/probe_results" \
    --model-key "$model_key"
}

secondary_judge_and_fit() {
  local model_key=$1
  local run_dir="$ROOT/runs/$model_key"
  /venv/main/bin/python "$CODE/judge_trajectories.py" \
    --responses "$run_dir/responses.jsonl" \
    --output "$run_dir/judgments_gemma3_12b.jsonl" \
    --manifest "$run_dir/judgments_gemma3_12b_manifest.json" \
    --judge-model "$GEMMA" \
    --judge-key gemma3_12b_dcs \
    --batch-size 4
  /venv/main/bin/python "$CODE/fit_trajectory_probes.py" \
    --responses "$run_dir/responses.jsonl" \
    --judgments "$run_dir/judgments_gemma3_12b.jsonl" \
    --activation-dir "$run_dir/activations" \
    --output-dir "$run_dir/probe_results_gemma_judge" \
    --model-key "$model_key"
  /venv/main/bin/python "$CODE/compare_judges.py" \
    --primary "$run_dir/judgments_qwen3_8b.jsonl" \
    --secondary "$run_dir/judgments_gemma3_12b.jsonl" \
    --output "$run_dir/judge_agreement.json"
}

if [[ "${QWEN_EXTERNAL:-0}" == "1" ]]; then
  # Vast deployment starts Qwen under Supervisor for separately visible progress.
  wait_for_supervisor_exit qwen_trajectory_rollout
  validate_manifest "$ROOT/runs/qwen3_8b/manifest.json"
  wait_for_supervisor_exit download_llama
  wait_for_supervisor_exit download_gemma
else
  rollout qwen3_8b "$QWEN" 12
fi

judge_and_fit qwen3_8b
rollout llama31_8b "$LLAMA" 12
judge_and_fit llama31_8b
rollout gemma3_12b "$GEMMA" 8
judge_and_fit gemma3_12b

for model_key in qwen3_8b llama31_8b gemma3_12b; do
  secondary_judge_and_fit "$model_key"
done
