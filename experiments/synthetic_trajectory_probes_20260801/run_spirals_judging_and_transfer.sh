#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT=$(cd "$(dirname "$0")/../.." && pwd)
EXPERIMENT="$REPO_ROOT/experiments/synthetic_trajectory_probes_20260801"
ARTIFACTS=${ARTIFACTS:-$EXPERIMENT/artifacts/private/transfer}
GPU_EXPORT=${GPU_EXPORT:-$ARTIFACTS/gpu_export}
ENV_FILE=${ENV_FILE:?Set ENV_FILE to a file containing OPENAI_API_KEY}
REAL_COHORT=${REAL_COHORT:-$REPO_ROOT/studies/context_token_attribution/artifacts/private/cohort.jsonl}
JUDGE="$REPO_ROOT/studies/context_token_attribution/judge_endorsement.py"
FIT="$REPO_ROOT/studies/context_token_attribution/fit_activation_probe.py"

mkdir -p "$ARTIFACTS"

for model_key in qwen3_8b llama31_8b gemma3_12b; do
  synthetic_dir="$GPU_EXPORT/runs/$model_key"
  python "$EXPERIMENT/prepare_spirals_judge_inputs.py" \
    --input "$synthetic_dir/responses.jsonl" \
    --output "$ARTIFACTS/${model_key}_synthetic_judge_inputs.jsonl"
  PYTHONPATH="$REPO_ROOT/studies/context_token_attribution:$REPO_ROOT" \
    python "$JUDGE" \
      --input "$ARTIFACTS/${model_key}_synthetic_judge_inputs.jsonl" \
      --output "$ARTIFACTS/${model_key}_synthetic_judgments.jsonl" \
      --manifest "$ARTIFACTS/${model_key}_synthetic_judgment_manifest.json" \
      --env-file "$ENV_FILE" \
      --concurrency 24
done

for model_key in llama31_8b gemma3_12b; do
  real_dir="$GPU_EXPORT/real_runs/$model_key"
  PYTHONPATH="$REPO_ROOT/studies/context_token_attribution:$REPO_ROOT" \
    python "$JUDGE" \
      --input "$real_dir/generations.jsonl" \
      --output "$ARTIFACTS/${model_key}_real_judgments.jsonl" \
      --manifest "$ARTIFACTS/${model_key}_real_judgment_manifest.json" \
      --env-file "$ENV_FILE" \
      --concurrency 24
  PYTHONPATH="$REPO_ROOT/studies/context_token_attribution:$REPO_ROOT" \
    python "$FIT" \
      --cohort "$REAL_COHORT" \
      --judgments "$ARTIFACTS/${model_key}_real_judgments.jsonl" \
      --activation-dir "$real_dir/activations" \
      --checkpoint-dir "$ARTIFACTS/${model_key}_real_checkpoints" \
      --predictions "$ARTIFACTS/${model_key}_real_oof_predictions.jsonl" \
      --summary "$ARTIFACTS/${model_key}_real_probe_summary.json"
done

for model_key in qwen3_8b llama31_8b gemma3_12b; do
  if [[ "$model_key" == "qwen3_8b" ]]; then
    checkpoints="$REPO_ROOT/studies/context_token_attribution/artifacts/final/checkpoints"
  else
    checkpoints="$ARTIFACTS/${model_key}_real_checkpoints"
  fi
  synthetic_dir="$GPU_EXPORT/runs/$model_key"
  python "$EXPERIMENT/apply_real_probe_to_synthetic.py" \
    --responses "$synthetic_dir/responses.jsonl" \
    --judgments "$ARTIFACTS/${model_key}_synthetic_judgments.jsonl" \
    --activation-dir "$synthetic_dir/activations" \
    --checkpoint-dir "$checkpoints" \
    --predictions "$ARTIFACTS/${model_key}_real_to_synthetic_predictions.jsonl" \
    --summary "$ARTIFACTS/${model_key}_real_to_synthetic_summary.json" \
    --model-key "$model_key"
done
