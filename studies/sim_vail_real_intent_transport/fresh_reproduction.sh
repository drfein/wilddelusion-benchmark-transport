#!/usr/bin/env bash
set -euo pipefail

if [[ "${ALLOW_GPU:-0}" != "1" || "${ALLOW_API:-0}" != "1" ]]; then
  echo "Refusing costly rerun. Set ALLOW_GPU=1 and ALLOW_API=1 explicitly." >&2
  exit 2
fi
: "${SIM_VAIL_QWEN4B_PATH:?Set SIM_VAIL_QWEN4B_PATH to the pinned local checkpoint}"
: "${SIM_VAIL_QWEN14B_PATH:?Set SIM_VAIL_QWEN14B_PATH to the pinned local checkpoint}"
: "${OPENAI_API_KEY:?Set OPENAI_API_KEY}"

HERE=$(cd "$(dirname "$0")" && pwd)
WORK=${SIM_VAIL_WORK_DIR:-"$HERE/../../.repro/sim_vail_real_intent_transport_fresh"}
DATA="$WORK/data"
OUT="$WORK/outputs"
RESULTS="$WORK/results"
mkdir -p "$OUT" "$RESULTS"

python "$HERE/reproduce.py" materialize --work-dir "$WORK"

python "$HERE/generate_vllm.py" \
  --input "$DATA/arms.jsonl" \
  --output "$OUT/generations_qwen3_4b.jsonl" \
  --model-path "$SIM_VAIL_QWEN4B_PATH" \
  --model-id Qwen/Qwen3-4B-AWQ \
  --revision 74d4bd2bd4bff9cafc9345221320bffb08b406a3 \
  --max-new-tokens 1024 --temperature 1.0 --top-p 1.0 \
  --candidate-ids-file "$HERE/provenance/primary_candidate_ids_80.txt" \
  --replicate-seed 17011

python "$HERE/generate_vllm.py" \
  --input "$DATA/arms.jsonl" \
  --output "$OUT/generations_qwen3_14b.jsonl" \
  --model-path "$SIM_VAIL_QWEN14B_PATH" \
  --model-id Qwen/Qwen3-14B-AWQ \
  --revision 31c69efc29464b6bb0aee1398b5a7b50a99340c3 \
  --max-new-tokens 1024 --temperature 1.0 --top-p 1.0 \
  --candidate-ids-file "$HERE/provenance/primary_candidate_ids_80.txt" \
  --replicate-seed 17011

python "$HERE/prepare_primary_judge_input.py" \
  --generations "$OUT/generations_qwen3_4b.jsonl" "$OUT/generations_qwen3_14b.jsonl" \
  --output "$DATA/primary_judge_input_fresh.jsonl" \
  --replicate-seed 17011 --conversation-count 80

python "$HERE/score_exact_openai.py" \
  --input "$DATA/primary_judge_input_fresh.jsonl" \
  --output "$DATA/primary_scores_fresh.jsonl" \
  --released-prompt-file "$HERE/upstream/sim_vail_prompts.py" \
  --model gpt-5.2-2025-12-11 --reasoning-effort high \
  --max-output-tokens 4000 --concurrency 32

python "$HERE/analyze_outcomes.py" \
  --generations "$DATA/primary_judge_input_fresh.jsonl" \
  --scores "$DATA/primary_scores_fresh.jsonl" \
  --output-dir "$RESULTS" --manual-audit-status pending

echo "Fresh stochastic replication written to $WORK"
