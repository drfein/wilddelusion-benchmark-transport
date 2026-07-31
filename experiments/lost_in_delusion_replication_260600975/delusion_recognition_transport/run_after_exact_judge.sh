#!/usr/bin/env bash
set -euo pipefail

ROOT=/workspace/lost_in_delusion_replication_260600975
HERE="$ROOT/delusion_recognition_transport"
PY=/workspace/vllm_env/bin/python
INPUT="$HERE/artifacts/classifier_inputs.jsonl"
RESULTS="$HERE/results"
MODELS=/workspace/models
RAM=/dev/shm
PSY_CASES="$ROOT/paper/psychosis-bench/data/test_cases.json"
PSY_RESULTS="$ROOT/cross_domain_comparison/synthetic_psychosis_bench/stochastic_results"
PSY_LOCAL_RESULTS="$ROOT/cross_domain_comparison/synthetic_psychosis_bench/local_results"
PSY_REAL_INPUT="$ROOT/psychogenic_machine_transport_250910970/full522/artifacts/model_inputs.jsonl"
PSY_REAL_RESULTS="$ROOT/psychogenic_machine_transport_250910970/full522/stochastic_results"
PAIRED_CONTEXT="$ROOT/full_history_522/results/context_ablation_paired"

mkdir -p "$RESULTS/logs"
mkdir -p "$PSY_RESULTS/logs"
mkdir -p "$PSY_LOCAL_RESULTS/logs"
mkdir -p "$PSY_REAL_RESULTS/logs"
mkdir -p "$PAIRED_CONTEXT/logs"
rm -f "$HERE/recognition_queue.exit"
trap 'code=$?; echo "$code" > "$HERE/recognition_queue.exit"; trap - EXIT; exit "$code"' EXIT
export PYTHONUNBUFFERED=1
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export VLLM_MOE_USE_DEEP_GEMM=0
export MAX_JOBS=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export HF_HUB_ENABLE_HF_TRANSFER=0

rows_complete() {
  local path=$1
  local expected=$2
  [[ -f "$path" ]] && [[ $(wc -l < "$path") == "$expected" ]]
}

echo "$(date -Is) Waiting for exact Qwen safety judge"
JUDGE_PID=$(cat "$ROOT/qwen_full_run.pid")
while kill -0 "$JUDGE_PID" 2>/dev/null; do
  sleep 60
done

JUDGE_EXIT=$(cat "$ROOT/qwen_full_run.exit" 2>/dev/null || echo missing)
MAIN_ROWS=$(wc -l < "$ROOT/full_history_522/results/judgments.qwen3-30b-thinking.jsonl")
CONTEXT_ROWS=$(wc -l < "$ROOT/full_history_522/results/context_ablation_judgments.qwen3-30b-thinking.jsonl")
if [[ "$JUDGE_EXIT" != "0" || "$MAIN_ROWS" != "3738" || "$CONTEXT_ROWS" != "2492" ]]; then
  echo "Exact judge gate failed: exit=$JUDGE_EXIT main=$MAIN_ROWS context=$CONTEXT_ROWS"
  exit 2
fi

echo "$(date -Is) Binding exact-judge manifests to input and output hashes"
"$PY" - "$ROOT" <<'PY'
import hashlib
import json
import os
import sys
from pathlib import Path

root = Path(sys.argv[1])
outputs = [
    root / "full_history_522/results/judgments.qwen3-30b-thinking.jsonl",
    root
    / "full_history_522/results/context_ablation_judgments.qwen3-30b-thinking.jsonl",
]

def sha256_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()

for output in outputs:
    manifest_path = output.with_suffix(output.suffix + ".manifest.json")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    normalized = []
    for item in manifest.get("input_files", []):
        path = Path(item["path"] if isinstance(item, dict) else item)
        normalized.append(
            {
                "path": str(path),
                "sha256": sha256_file(path),
                "rows": sum(
                    1
                    for line in path.open(encoding="utf-8")
                    if line.strip()
                ),
            }
        )
    manifest["input_files"] = normalized
    manifest["output_sha256"] = sha256_file(output)
    temporary = manifest_path.with_suffix(manifest_path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(manifest, indent=2) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, manifest_path)
PY

echo "$(date -Is) Running exact Lost analysis; separately generated context arms remain exploratory only"
"$PY" "$ROOT/cross_domain_comparison/analyze_lost_exact_claims.py" \
  > "$ROOT/cross_domain_comparison/lost_exact_claim_audit.log" 2>&1

echo "$(date -Is) Exact judge complete; preserving RAM judge shards for the repaired context pass"
rm -rf "$MODELS/Qwen3-30B-A3B-Thinking-2507-144afc2"

stage_model() {
  local repo=$1
  local revision=$2
  local name=$3
  local disk_shards=$4
  "$PY" "$HERE/stage_hf_model.py" \
    --model "$repo" \
    --revision "$revision" \
    --model-dir "$MODELS/$name" \
    --ram-dir "$RAM/$name" \
    --disk-shards "$disk_shards" \
    --concurrency 6
}

run_classifier() {
  local visible_gpus=$1
  local checkpoint=$2
  local reported=$3
  local source_checkpoint=$4
  local revision=$5
  local output=$6
  local tensor_parallel=$7
  shift 7
  CUDA_VISIBLE_DEVICES="$visible_gpus" "$PY" "$HERE/run_vllm_classifier.py" \
    --input "$INPUT" \
    --output "$RESULTS/$output" \
    --model "$checkpoint" \
    --reported-model "$reported" \
    --checkpoint-source "$source_checkpoint" \
    --revision "$revision" \
    --max-input-tokens 12288 \
    --max-new-tokens 512 \
    --chunk-size 128 \
    --gpu-memory-utilization 0.92 \
    --tensor-parallel-size "$tensor_parallel" \
    "$@"
}

run_psychosis_stochastic() {
  local visible_gpus=$1
  local checkpoint=$2
  local reported=$3
  local source_checkpoint=$4
  local revision=$5
  local output=$6
  local tensor_parallel=$7
  shift 7
  CUDA_VISIBLE_DEVICES="$visible_gpus" "$PY" \
    "$HERE/run_psychosis_bench_stochastic.py" \
    --cases "$PSY_CASES" \
    --output "$PSY_RESULTS/$output" \
    --model "$checkpoint" \
    --reported-model "$reported" \
    --checkpoint-source "$source_checkpoint" \
    --revision "$revision" \
    --seeds 1101,2202,3303 \
    --temperature 1.0 \
    --top-p 1.0 \
    --max-input-tokens 11776 \
    --max-new-tokens 512 \
    --gpu-memory-utilization 0.92 \
    --tensor-parallel-size "$tensor_parallel" \
    "$@"
}

run_psychosis_real() {
  local visible_gpus=$1
  local checkpoint=$2
  local reported=$3
  local source_checkpoint=$4
  local revision=$5
  local output=$6
  local tensor_parallel=$7
  shift 7
  CUDA_VISIBLE_DEVICES="$visible_gpus" "$PY" \
    "$ROOT/psychogenic_machine_transport_250910970/run_real_pairs_stochastic.py" \
    --input "$PSY_REAL_INPUT" \
    --output "$PSY_REAL_RESULTS/$output" \
    --model "$checkpoint" \
    --reported-model "$reported" \
    --checkpoint-source "$source_checkpoint" \
    --revision "$revision" \
    --seeds 1101,2202,3303 \
    --temperature 1.0 \
    --top-p 1.0 \
    --max-input-tokens 11776 \
    --max-new-tokens 512 \
    --gpu-memory-utilization 0.92 \
    --tensor-parallel-size "$tensor_parallel" \
    "$@"
}

run_psychosis_local() {
  local visible_gpus=$1
  local checkpoint=$2
  local reported=$3
  local source_checkpoint=$4
  local revision=$5
  local output=$6
  local tensor_parallel=$7
  shift 7
  CUDA_VISIBLE_DEVICES="$visible_gpus" "$PY" \
    "$HERE/run_psychosis_bench_local_contrast.py" \
    --cases "$PSY_CASES" \
    --source-generations "$PSY_RESULTS/$output" \
    --output "$PSY_LOCAL_RESULTS/$output" \
    --model "$checkpoint" \
    --reported-model "$reported" \
    --checkpoint-source "$source_checkpoint" \
    --revision "$revision" \
    --temperature 1.0 \
    --top-p 1.0 \
    --max-input-tokens 11776 \
    --max-new-tokens 512 \
    --gpu-memory-utilization 0.92 \
    --tensor-parallel-size "$tensor_parallel" \
    "$@"
}

if rows_complete "$PAIRED_CONTEXT/generations_olmo3_7b_history.jsonl" 623 \
  && rows_complete "$PAIRED_CONTEXT/generations_olmo3_7b_last_user_only.jsonl" 623 \
  && rows_complete "$PAIRED_CONTEXT/generations_llama31_8b_history.jsonl" 623 \
  && rows_complete "$PAIRED_CONTEXT/generations_llama31_8b_last_user_only.jsonl" 623 \
  && rows_complete "$RESULTS/classifications.olmo3_7b.jsonl" 932 \
  && rows_complete "$RESULTS/classifications.llama31_8b.jsonl" 932 \
  && rows_complete "$PSY_RESULTS/generations_olmo3_7b_t1.jsonl" 576 \
  && rows_complete "$PSY_RESULTS/generations_llama31_8b_t1.jsonl" 576 \
  && rows_complete "$PSY_LOCAL_RESULTS/generations_olmo3_7b_t1.jsonl" 864 \
  && rows_complete "$PSY_LOCAL_RESULTS/generations_llama31_8b_t1.jsonl" 864 \
  && rows_complete "$PSY_REAL_RESULTS/generations_olmo3_7b_t1.jsonl" 1962 \
  && rows_complete "$PSY_REAL_RESULTS/generations_llama31_8b_t1.jsonl" 1962; then
  echo "$(date -Is) Reusing complete OLMo-7B and Llama-8B artifacts"
  PAIR_GENERATIONS=(
    "$PAIRED_CONTEXT/generations_olmo3_7b_history.jsonl"
    "$PAIRED_CONTEXT/generations_olmo3_7b_last_user_only.jsonl"
    "$PAIRED_CONTEXT/generations_llama31_8b_history.jsonl"
    "$PAIRED_CONTEXT/generations_llama31_8b_last_user_only.jsonl"
  )
  "$PY" "$ROOT/full_history_522/prepare_paired_context_judge_inputs.py" \
    --generations "${PAIR_GENERATIONS[@]}" \
    --output "$PAIRED_CONTEXT/judge_inputs.shared_context.jsonl" \
    > "$PAIRED_CONTEXT/logs/prepare_shared_context_judge_inputs.log" 2>&1
else
echo "$(date -Is) Staging OLMo-7B and Llama-8B"
stage_model \
  allenai/Olmo-3-7B-Instruct \
  6e5971d9eba42665f5bd5a0fcf047f299ce1dccc \
  recognition_olmo3_7b 2 \
  > "$RESULTS/logs/stage_olmo3_7b.log" 2>&1 &
STAGE_OLMO_PID=$!
stage_model \
  meta-llama/Llama-3.1-8B-Instruct \
  0e9e39f249a16976918f6564b8830bc894c89659 \
  recognition_llama31_8b 2 \
  > "$RESULTS/logs/stage_llama31_8b.log" 2>&1 &
STAGE_LLAMA_PID=$!
wait "$STAGE_OLMO_PID"
wait "$STAGE_LLAMA_PID"

echo "$(date -Is) Generating prompt-deduplicated paired context arms"
CUDA_VISIBLE_DEVICES=0 "$PY" \
  "$ROOT/full_history_522/generate_paired_context_vllm.py" \
  --history-input "$ROOT/full_history_522/artifacts/model_inputs.jsonl" \
  --last-only-input "$ROOT/full_history_522/artifacts/last_user_only_inputs.jsonl" \
  --history-output "$PAIRED_CONTEXT/generations_olmo3_7b_history.jsonl" \
  --last-only-output "$PAIRED_CONTEXT/generations_olmo3_7b_last_user_only.jsonl" \
  --prompt-cache "$PAIRED_CONTEXT/prompt_cache_olmo3_7b.jsonl" \
  --model "$MODELS/recognition_olmo3_7b" \
  --reported-model allenai/Olmo-3-7B-Instruct \
  --checkpoint-source allenai/Olmo-3-7B-Instruct \
  --revision 6e5971d9eba42665f5bd5a0fcf047f299ce1dccc \
  --enforce-eager \
  > "$PAIRED_CONTEXT/logs/generate_olmo3_7b.log" 2>&1 &
OLMO_CONTEXT_PID=$!
CUDA_VISIBLE_DEVICES=1 "$PY" \
  "$ROOT/full_history_522/generate_paired_context_vllm.py" \
  --history-input "$ROOT/full_history_522/artifacts/model_inputs.jsonl" \
  --last-only-input "$ROOT/full_history_522/artifacts/last_user_only_inputs.jsonl" \
  --history-output "$PAIRED_CONTEXT/generations_llama31_8b_history.jsonl" \
  --last-only-output "$PAIRED_CONTEXT/generations_llama31_8b_last_user_only.jsonl" \
  --prompt-cache "$PAIRED_CONTEXT/prompt_cache_llama31_8b.jsonl" \
  --model "$MODELS/recognition_llama31_8b" \
  --reported-model meta-llama/Llama-3.1-8B-Instruct \
  --checkpoint-source meta-llama/Llama-3.1-8B-Instruct \
  --revision 0e9e39f249a16976918f6564b8830bc894c89659 \
  --enforce-eager \
  > "$PAIRED_CONTEXT/logs/generate_llama31_8b.log" 2>&1 &
LLAMA_CONTEXT_PID=$!
wait "$OLMO_CONTEXT_PID"
wait "$LLAMA_CONTEXT_PID"

PAIR_GENERATIONS=(
  "$PAIRED_CONTEXT/generations_olmo3_7b_history.jsonl"
  "$PAIRED_CONTEXT/generations_olmo3_7b_last_user_only.jsonl"
  "$PAIRED_CONTEXT/generations_llama31_8b_history.jsonl"
  "$PAIRED_CONTEXT/generations_llama31_8b_last_user_only.jsonl"
)
"$PY" "$ROOT/full_history_522/prepare_paired_context_judge_inputs.py" \
  --generations "${PAIR_GENERATIONS[@]}" \
  --output "$PAIRED_CONTEXT/judge_inputs.shared_context.jsonl" \
  > "$PAIRED_CONTEXT/logs/prepare_shared_context_judge_inputs.log" 2>&1

echo "$(date -Is) Running OLMo-7B and Llama-8B concurrently"
run_classifier \
  0 "$MODELS/recognition_olmo3_7b" \
  allenai/Olmo-3-7B-Instruct \
  allenai/Olmo-3-7B-Instruct \
  6e5971d9eba42665f5bd5a0fcf047f299ce1dccc \
  classifications.olmo3_7b.jsonl 1 \
  > "$RESULTS/logs/classify_olmo3_7b.log" 2>&1 &
OLMO_PID=$!
run_classifier \
  1 "$MODELS/recognition_llama31_8b" \
  meta-llama/Llama-3.1-8B-Instruct \
  meta-llama/Llama-3.1-8B-Instruct \
  0e9e39f249a16976918f6564b8830bc894c89659 \
  classifications.llama31_8b.jsonl 1 \
  > "$RESULTS/logs/classify_llama31_8b.log" 2>&1 &
LLAMA_PID=$!
wait "$OLMO_PID"
wait "$LLAMA_PID"

echo "$(date -Is) Running stochastic Psychosis-Bench on OLMo-7B and Llama-8B"
run_psychosis_stochastic \
  0 "$MODELS/recognition_olmo3_7b" \
  allenai/Olmo-3-7B-Instruct \
  allenai/Olmo-3-7B-Instruct \
  6e5971d9eba42665f5bd5a0fcf047f299ce1dccc \
  generations_olmo3_7b_t1.jsonl 1 \
  > "$PSY_RESULTS/logs/generate_olmo3_7b_t1.log" 2>&1 &
OLMO_PSY_PID=$!
run_psychosis_stochastic \
  1 "$MODELS/recognition_llama31_8b" \
  meta-llama/Llama-3.1-8B-Instruct \
  meta-llama/Llama-3.1-8B-Instruct \
  0e9e39f249a16976918f6564b8830bc894c89659 \
  generations_llama31_8b_t1.jsonl 1 \
  > "$PSY_RESULTS/logs/generate_llama31_8b_t1.log" 2>&1 &
LLAMA_PSY_PID=$!
wait "$OLMO_PSY_PID"
wait "$LLAMA_PSY_PID"

echo "$(date -Is) Running matched-local synthetic contrasts on OLMo-7B and Llama-8B"
run_psychosis_local \
  0 "$MODELS/recognition_olmo3_7b" \
  allenai/Olmo-3-7B-Instruct \
  allenai/Olmo-3-7B-Instruct \
  6e5971d9eba42665f5bd5a0fcf047f299ce1dccc \
  generations_olmo3_7b_t1.jsonl 1 \
  > "$PSY_LOCAL_RESULTS/logs/generate_olmo3_7b_t1.log" 2>&1 &
OLMO_LOCAL_PID=$!
run_psychosis_local \
  1 "$MODELS/recognition_llama31_8b" \
  meta-llama/Llama-3.1-8B-Instruct \
  meta-llama/Llama-3.1-8B-Instruct \
  0e9e39f249a16976918f6564b8830bc894c89659 \
  generations_llama31_8b_t1.jsonl 1 \
  > "$PSY_LOCAL_RESULTS/logs/generate_llama31_8b_t1.log" 2>&1 &
LLAMA_LOCAL_PID=$!
wait "$OLMO_LOCAL_PID"
wait "$LLAMA_LOCAL_PID"

echo "$(date -Is) Running matched stochastic real pairs on OLMo-7B and Llama-8B"
run_psychosis_real \
  0 "$MODELS/recognition_olmo3_7b" \
  allenai/Olmo-3-7B-Instruct \
  allenai/Olmo-3-7B-Instruct \
  6e5971d9eba42665f5bd5a0fcf047f299ce1dccc \
  generations_olmo3_7b_t1.jsonl 1 \
  > "$PSY_REAL_RESULTS/logs/generate_olmo3_7b_t1.log" 2>&1 &
OLMO_REAL_PID=$!
run_psychosis_real \
  1 "$MODELS/recognition_llama31_8b" \
  meta-llama/Llama-3.1-8B-Instruct \
  meta-llama/Llama-3.1-8B-Instruct \
  0e9e39f249a16976918f6564b8830bc894c89659 \
  generations_llama31_8b_t1.jsonl 1 \
  > "$PSY_REAL_RESULTS/logs/generate_llama31_8b_t1.log" 2>&1 &
LLAMA_REAL_PID=$!
wait "$OLMO_REAL_PID"
wait "$LLAMA_REAL_PID"
rm -rf "$MODELS/recognition_olmo3_7b" "$RAM/recognition_olmo3_7b"
rm -rf "$MODELS/recognition_llama31_8b" "$RAM/recognition_llama31_8b"
fi

if rows_complete "$PAIRED_CONTEXT/judgments.shared_context.qwen3-30b-thinking.jsonl" 1796 \
  && rows_complete "$PAIRED_CONTEXT/judgments.shared_context.gpt-5.4-mini.jsonl" 1796 \
  && [[ -f "$PAIRED_CONTEXT/analysis_shared_context/claim_gate.json" ]]; then
  echo "$(date -Is) Reusing complete repaired context judgments and analysis"
else
echo "$(date -Is) Restaging native Qwen judge for repaired context pairs"
"$PY" "$HERE/stage_hf_model.py" \
  --model Qwen/Qwen3-30B-A3B-Thinking-2507 \
  --revision 144afc2f379b542fdd4e85a1fcd5e1f79112d95d \
  --model-dir "$MODELS/Qwen3-30B-A3B-Thinking-2507-144afc2" \
  --ram-dir "$RAM/qwen3-thinking-2507" \
  --disk-shards 5 \
  --concurrency 6 \
  > "$PAIRED_CONTEXT/logs/stage_qwen_judge.log" 2>&1

echo "$(date -Is) Judging repaired context pairs with native Qwen and pinned mini"
set -a
source "$ROOT/.env"
set +a
"$PY" "$ROOT/judge_responses.py" \
  --inputs "$PAIRED_CONTEXT/judge_inputs.shared_context.jsonl" \
  --output "$PAIRED_CONTEXT/judgments.shared_context.gpt-5.4-mini.jsonl" \
  --model gpt-5.4-mini-2026-03-17 \
  --previous-exchanges 3 \
  --concurrency 30 \
  > "$PAIRED_CONTEXT/logs/judge_shared_context_gpt54mini.log" 2>&1 &
PAIRED_MINI_PID=$!
"$PY" "$ROOT/judge_responses_vllm.py" \
  --inputs "$PAIRED_CONTEXT/judge_inputs.shared_context.jsonl" \
  --output "$PAIRED_CONTEXT/judgments.shared_context.qwen3-30b-thinking.jsonl" \
  --previous-exchanges 3 \
  --model "$MODELS/Qwen3-30B-A3B-Thinking-2507-144afc2" \
  --model-id Qwen/Qwen3-30B-A3B-Thinking-2507 \
  --revision 144afc2f379b542fdd4e85a1fcd5e1f79112d95d \
  --batch-size 64 \
  --max-num-seqs 16 \
  --max-num-batched-tokens 4096 \
  --tensor-parallel-size 1 \
  --pipeline-parallel-size 2 \
  --cpu-offload-gb 1 \
  --gpu-memory-utilization 0.94 \
  --max-model-len 16384 \
  --moe-backend triton \
  --dtype bfloat16 \
  --enforce-eager \
  > "$PAIRED_CONTEXT/logs/judge_shared_context_qwen3_thinking.log" 2>&1
wait "$PAIRED_MINI_PID"
if [[ $(wc -l < "$PAIRED_CONTEXT/judgments.shared_context.qwen3-30b-thinking.jsonl") != "1796" ]]; then
  echo "Repaired exact context judgment row-count gate failed"
  exit 8
fi
if [[ $(wc -l < "$PAIRED_CONTEXT/judgments.shared_context.gpt-5.4-mini.jsonl") != "1796" ]]; then
  echo "Repaired mini context judgment row-count gate failed"
  exit 8
fi
"$PY" "$ROOT/full_history_522/analyze_paired_context_ablation.py" \
  --generations "${PAIR_GENERATIONS[@]}" \
  --judge-inputs "$PAIRED_CONTEXT/judge_inputs.shared_context.jsonl" \
  --exact-judgments "$PAIRED_CONTEXT/judgments.shared_context.qwen3-30b-thinking.jsonl" \
  --secondary-judgments "$PAIRED_CONTEXT/judgments.shared_context.gpt-5.4-mini.jsonl" \
  --output-dir "$PAIRED_CONTEXT/analysis_shared_context" \
  > "$PAIRED_CONTEXT/logs/analyze_shared_context.log" 2>&1
rm -rf "$MODELS/Qwen3-30B-A3B-Thinking-2507-144afc2"
rm -rf "$RAM/qwen3-thinking-2507"
fi

if rows_complete "$RESULTS/classifications.qwen3_30b.jsonl" 932 \
  && rows_complete "$PSY_RESULTS/generations_qwen3_30b_t1.jsonl" 576; then
  echo "$(date -Is) Reusing complete Qwen3-30B recognition and stochastic runs"
else
echo "$(date -Is) Staging and running official Qwen3-30B-A3B BF16"
stage_model \
  Qwen/Qwen3-30B-A3B \
  ad44e777bcd18fa416d9da3bd8f70d33ebb85d39 \
  recognition_qwen3_30b_bf16 3 \
  > "$RESULTS/logs/stage_qwen3_30b.log" 2>&1
run_classifier \
  0,1 "$MODELS/recognition_qwen3_30b_bf16" \
  Qwen/Qwen3-30B-A3B \
  Qwen/Qwen3-30B-A3B \
  ad44e777bcd18fa416d9da3bd8f70d33ebb85d39 \
  classifications.qwen3_30b.jsonl 2 \
  --disable-thinking --enforce-eager --cpu-offload-gb 4 \
  > "$RESULTS/logs/classify_qwen3_30b.log" 2>&1
run_psychosis_stochastic \
  0,1 "$MODELS/recognition_qwen3_30b_bf16" \
  Qwen/Qwen3-30B-A3B \
  Qwen/Qwen3-30B-A3B \
  ad44e777bcd18fa416d9da3bd8f70d33ebb85d39 \
  generations_qwen3_30b_t1.jsonl 2 \
  --disable-thinking --enforce-eager --cpu-offload-gb 4 \
  > "$PSY_RESULTS/logs/generate_qwen3_30b_t1.log" 2>&1
fi
rm -rf "$MODELS/recognition_qwen3_30b_bf16" "$RAM/recognition_qwen3_30b_bf16"

if rows_complete "$RESULTS/classifications.llama33_70b.jsonl" 932 \
  && rows_complete "$PSY_RESULTS/generations_llama33_70b_t1.jsonl" 576; then
  echo "$(date -Is) Reusing complete Llama-3.3-70B recognition and stochastic runs"
else
echo "$(date -Is) Staging and running Llama-3.3-70B-AWQ"
stage_model \
  casperhansen/llama-3.3-70b-instruct-awq \
  64d255621f40b42adaf6d1f32a47e1d4534c0f14 \
  recognition_llama33_70b_awq 4 \
  > "$RESULTS/logs/stage_llama33_70b.log" 2>&1
run_classifier \
  0,1 "$MODELS/recognition_llama33_70b_awq" \
  meta-llama/Llama-3.3-70B-Instruct \
  casperhansen/llama-3.3-70b-instruct-awq \
  64d255621f40b42adaf6d1f32a47e1d4534c0f14 \
  classifications.llama33_70b.jsonl 2 \
  --dtype auto --enforce-eager \
  > "$RESULTS/logs/classify_llama33_70b.log" 2>&1
run_psychosis_stochastic \
  0,1 "$MODELS/recognition_llama33_70b_awq" \
  meta-llama/Llama-3.3-70B-Instruct \
  casperhansen/llama-3.3-70b-instruct-awq \
  64d255621f40b42adaf6d1f32a47e1d4534c0f14 \
  generations_llama33_70b_t1.jsonl 2 \
  --dtype auto --enforce-eager \
  > "$PSY_RESULTS/logs/generate_llama33_70b_t1.log" 2>&1
fi
rm -rf "$MODELS/recognition_llama33_70b_awq" "$RAM/recognition_llama33_70b_awq"

if rows_complete "$PSY_RESULTS/generations_qwen3_4b_t1.jsonl" 576 \
  && rows_complete "$PSY_RESULTS/generations_qwen3_14b_t1.jsonl" 576 \
  && rows_complete "$PSY_LOCAL_RESULTS/generations_qwen3_4b_t1.jsonl" 864 \
  && rows_complete "$PSY_REAL_RESULTS/generations_qwen3_4b_t1.jsonl" 1962; then
  echo "$(date -Is) Reusing complete Qwen3-4B/14B stochastic transport runs"
else
echo "$(date -Is) Staging Qwen3-4B and Qwen3-14B-AWQ"
stage_model \
  Qwen/Qwen3-4B \
  1cfa9a7208912126459214e8b04321603b3df60c \
  psych_qwen3_4b 3 \
  > "$PSY_RESULTS/logs/stage_qwen3_4b.log" 2>&1 &
STAGE_QWEN4_PID=$!
stage_model \
  Qwen/Qwen3-14B-AWQ \
  31c69efc29464b6bb0aee1398b5a7b50a99340c3 \
  psych_qwen3_14b_awq 2 \
  > "$PSY_RESULTS/logs/stage_qwen3_14b.log" 2>&1 &
STAGE_QWEN14_PID=$!
wait "$STAGE_QWEN4_PID"
wait "$STAGE_QWEN14_PID"

echo "$(date -Is) Running stochastic Psychosis-Bench on Qwen3-4B and Qwen3-14B"
run_psychosis_stochastic \
  0 "$MODELS/psych_qwen3_4b" \
  Qwen/Qwen3-4B \
  Qwen/Qwen3-4B \
  1cfa9a7208912126459214e8b04321603b3df60c \
  generations_qwen3_4b_t1.jsonl 1 \
  --disable-thinking \
  > "$PSY_RESULTS/logs/generate_qwen3_4b_t1.log" 2>&1 &
QWEN4_PSY_PID=$!
run_psychosis_stochastic \
  1 "$MODELS/psych_qwen3_14b_awq" \
  Qwen/Qwen3-14B \
  Qwen/Qwen3-14B-AWQ \
  31c69efc29464b6bb0aee1398b5a7b50a99340c3 \
  generations_qwen3_14b_t1.jsonl 1 \
  --disable-thinking --dtype auto --enforce-eager \
  > "$PSY_RESULTS/logs/generate_qwen3_14b_t1.log" 2>&1 &
QWEN14_PSY_PID=$!
wait "$QWEN4_PSY_PID"
wait "$QWEN14_PSY_PID"

echo "$(date -Is) Running matched-local synthetic contrasts on Qwen3-4B"
run_psychosis_local \
  0 "$MODELS/psych_qwen3_4b" \
  Qwen/Qwen3-4B \
  Qwen/Qwen3-4B \
  1cfa9a7208912126459214e8b04321603b3df60c \
  generations_qwen3_4b_t1.jsonl 1 \
  --disable-thinking \
  > "$PSY_LOCAL_RESULTS/logs/generate_qwen3_4b_t1.log" 2>&1

echo "$(date -Is) Running matched stochastic real pairs on Qwen3-4B"
run_psychosis_real \
  0 "$MODELS/psych_qwen3_4b" \
  Qwen/Qwen3-4B \
  Qwen/Qwen3-4B \
  1cfa9a7208912126459214e8b04321603b3df60c \
  generations_qwen3_4b_t1.jsonl 1 \
  --disable-thinking \
  > "$PSY_REAL_RESULTS/logs/generate_qwen3_4b_t1.log" 2>&1
fi
rm -rf "$MODELS/psych_qwen3_4b" "$RAM/psych_qwen3_4b"
rm -rf "$MODELS/psych_qwen3_14b_awq" "$RAM/psych_qwen3_14b_awq"

"$PY" "$HERE/reparse_classifier_outputs.py" \
  "$RESULTS/classifications.olmo3_7b.jsonl" \
  "$RESULTS/classifications.llama31_8b.jsonl" \
  "$RESULTS/classifications.qwen3_30b.jsonl" \
  "$RESULTS/classifications.llama33_70b.jsonl" \
  > "$RESULTS/logs/reparse_classifier_outputs.log" 2>&1

cat \
  "$RESULTS/classifications.olmo3_7b.jsonl" \
  "$RESULTS/classifications.llama31_8b.jsonl" \
  "$RESULTS/classifications.qwen3_30b.jsonl" \
  "$RESULTS/classifications.llama33_70b.jsonl" \
  > "$RESULTS/classifications.jsonl.tmp"
mv "$RESULTS/classifications.jsonl.tmp" "$RESULTS/classifications.jsonl"

for file in \
  "$RESULTS/classifications.olmo3_7b.jsonl" \
  "$RESULTS/classifications.llama31_8b.jsonl" \
  "$RESULTS/classifications.qwen3_30b.jsonl" \
  "$RESULTS/classifications.llama33_70b.jsonl"; do
  if [[ $(wc -l < "$file") != "932" ]]; then
    echo "Row-count gate failed for $file"
    exit 3
  fi
done

RESULTS_DIR="$RESULTS" "$PY" - <<'PY'
import json
import os
from pathlib import Path

root = Path(os.environ["RESULTS_DIR"])
files = sorted(root.glob("classifications.*.jsonl"))
audit = {}
failed = False
for path in files:
    rows = [json.loads(line) for line in path.open() if line.strip()]
    valid = sum(bool(row.get("parse_valid")) for row in rows)
    coverage = valid / len(rows) if rows else 0.0
    audit[path.name] = {
        "rows": len(rows),
        "parse_valid": valid,
        "parse_coverage": coverage,
    }
    failed |= len(rows) != 932 or coverage < 0.99
(root / "recognition_queue.audit.json").write_text(
    json.dumps(audit, indent=2) + "\n"
)
print(json.dumps(audit, indent=2))
if failed:
    print(
        "Recognition parse coverage is below the headline gate; "
        "continuing with complete-case diagnostics only."
    )
PY

echo "$(date -Is) Analyzing direct-recognition transport"
"$PY" "$HERE/analyze_recognition.py" \
  > "$RESULTS/analysis.log" 2>&1

PSY_FILES=(
  "$PSY_RESULTS/generations_olmo3_7b_t1.jsonl"
  "$PSY_RESULTS/generations_llama31_8b_t1.jsonl"
  "$PSY_RESULTS/generations_qwen3_4b_t1.jsonl"
  "$PSY_RESULTS/generations_qwen3_14b_t1.jsonl"
  "$PSY_RESULTS/generations_qwen3_30b_t1.jsonl"
  "$PSY_RESULTS/generations_llama33_70b_t1.jsonl"
)
for file in "${PSY_FILES[@]}"; do
  if [[ $(wc -l < "$file") != "576" ]]; then
    echo "Stochastic Psychosis-Bench row-count gate failed for $file"
    exit 4
  fi
done

PRIMARY_PSY_FILES=(
  "$PSY_RESULTS/generations_olmo3_7b_t1.jsonl"
  "$PSY_RESULTS/generations_llama31_8b_t1.jsonl"
  "$PSY_RESULTS/generations_qwen3_4b_t1.jsonl"
)
PRIMARY_LOCAL_FILES=(
  "$PSY_LOCAL_RESULTS/generations_olmo3_7b_t1.jsonl"
  "$PSY_LOCAL_RESULTS/generations_llama31_8b_t1.jsonl"
  "$PSY_LOCAL_RESULTS/generations_qwen3_4b_t1.jsonl"
)
PRIMARY_REAL_FILES=(
  "$PSY_REAL_RESULTS/generations_olmo3_7b_t1.jsonl"
  "$PSY_REAL_RESULTS/generations_llama31_8b_t1.jsonl"
  "$PSY_REAL_RESULTS/generations_qwen3_4b_t1.jsonl"
)
for file in "${PRIMARY_LOCAL_FILES[@]}"; do
  if [[ $(wc -l < "$file") != "864" ]]; then
    echo "Matched-local synthetic row-count gate failed for $file"
    exit 5
  fi
done
for file in "${PRIMARY_REAL_FILES[@]}"; do
  if [[ $(wc -l < "$file") != "1962" ]]; then
    echo "Stochastic real-pair row-count gate failed for $file"
    exit 5
  fi
done

echo "$(date -Is) Applying paper-matched Psychosis-Bench gpt-4o-mini DCS judge"
set -a
source "$ROOT/.env"
set +a
"$PY" "$ROOT/psychogenic_machine_transport_250910970/judge_dcs_exact.py" \
  --inputs "${PRIMARY_PSY_FILES[@]}" \
  --output "$PSY_RESULTS/judgments.dcs-gpt4o-mini.jsonl" \
  --models \
    allenai/Olmo-3-7B-Instruct \
    meta-llama/Llama-3.1-8B-Instruct \
    Qwen/Qwen3-4B \
  --expected-per-model 432 \
  --min-turn 4 \
  --model gpt-4o-mini-2024-07-18 \
  --concurrency 30 \
  > "$PSY_RESULTS/logs/judge_dcs_o4_mini.log" 2>&1
if [[ $(wc -l < "$PSY_RESULTS/judgments.dcs-gpt4o-mini.jsonl") != "1296" ]]; then
  echo "Stochastic Psychosis-Bench judgment row-count gate failed"
  exit 6
fi
"$PY" "$ROOT/psychogenic_machine_transport_250910970/judge_dcs_exact.py" \
  --inputs "${PRIMARY_LOCAL_FILES[@]}" \
  --output "$PSY_LOCAL_RESULTS/judgments.dcs-gpt4o-mini.jsonl" \
  --models \
    allenai/Olmo-3-7B-Instruct \
    meta-llama/Llama-3.1-8B-Instruct \
    Qwen/Qwen3-4B \
  --expected-per-model 864 \
  --model gpt-4o-mini-2024-07-18 \
  --concurrency 30 \
  --max-candidates 3000 \
  > "$PSY_LOCAL_RESULTS/logs/judge_dcs_o4_mini.log" 2>&1
if [[ $(wc -l < "$PSY_LOCAL_RESULTS/judgments.dcs-gpt4o-mini.jsonl") != "2592" ]]; then
  echo "Matched-local synthetic judgment row-count gate failed"
  exit 7
fi
"$PY" "$ROOT/psychogenic_machine_transport_250910970/judge_dcs_exact.py" \
  --inputs "${PRIMARY_REAL_FILES[@]}" \
  --output "$PSY_REAL_RESULTS/judgments.dcs-gpt4o-mini.jsonl" \
  --models \
    allenai/Olmo-3-7B-Instruct \
    meta-llama/Llama-3.1-8B-Instruct \
    Qwen/Qwen3-4B \
  --expected-per-model 1962 \
  --model gpt-4o-mini-2024-07-18 \
  --concurrency 30 \
  --max-candidates 6000 \
  > "$PSY_REAL_RESULTS/logs/judge_dcs_o4_mini.log" 2>&1
if [[ $(wc -l < "$PSY_REAL_RESULTS/judgments.dcs-gpt4o-mini.jsonl") != "5886" ]]; then
  echo "Stochastic real-pair judgment row-count gate failed"
  exit 7
fi

"$PY" - \
  "$PSY_RESULTS/judgments.dcs-gpt4o-mini.jsonl.manifest.json" 1296 \
  "$PSY_LOCAL_RESULTS/judgments.dcs-gpt4o-mini.jsonl.manifest.json" 2592 \
  "$PSY_REAL_RESULTS/judgments.dcs-gpt4o-mini.jsonl.manifest.json" 5886 <<'PY'
import json
import sys

for index in range(1, len(sys.argv), 2):
    path = sys.argv[index]
    expected = int(sys.argv[index + 1])
    manifest = json.load(open(path))
    if (
        manifest.get("successful_rows") != expected
        or manifest.get("new_errors") != 0
    ):
        raise SystemExit(
            "Exact Psychosis-Bench DCS judge success gate failed: "
            + json.dumps(manifest)
        )
PY

"$PY" "$ROOT/cross_domain_comparison/analyze_psychogenic_robustness.py" \
  > "$ROOT/cross_domain_comparison/psychogenic_robustness_exact.log" 2>&1

echo "$(date -Is) Auditing synthetic-theme ontology coverage"
"$PY" "$ROOT/cross_domain_comparison/analyze_theme_coverage.py" \
  > "$ROOT/cross_domain_comparison/theme_coverage.log" 2>&1

echo "$(date -Is) Building fail-closed abstract claim registry"
"$PY" "$ROOT/cross_domain_comparison/build_claim_registry.py" \
  > "$ROOT/cross_domain_comparison/abstract_claim_registry.log" 2>&1

echo "$(date -Is) Recognition transport complete: $(wc -l < "$RESULTS/classifications.jsonl") rows"
