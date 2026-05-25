#!/usr/bin/env bash
set -euo pipefail

REPO=${REPO:-/home/umair/TW/PolyAlign}
MODEL_ROOT=${MODEL_ROOT:-$REPO/models}
DATA_ROOT=${DATA_ROOT:-$REPO/data/english/hdpo}
RUNS_DIR=${RUNS_DIR:-$REPO/data/english/merged_sft_dedup/runs}
HF_DATASET_REPO=${HF_DATASET_REPO:-saiteja33/PolyAlign-All}
HF_REMOTE_RUNS_DIR=${HF_REMOTE_RUNS_DIR:-english/merged_sft_dedup/runs}
RUN_ID=${RUN_ID:-$(date +%Y%m%d_%H%M%S)}
LOG_DIR=${LOG_DIR:-$REPO/logs/hdpo_ref_conditioned_vllm_en/$RUN_ID}

HOST=${HOST:-127.0.0.1}
MAX_MODEL_LEN=${MAX_MODEL_LEN:-8192}
MAX_TOKENS=${MAX_TOKENS:-128}
BATCH_SIZE=${BATCH_SIZE:-4}
GPU_MEMORY_UTILIZATION=${GPU_MEMORY_UTILIZATION:-0.70}
SERVER_TIMEOUT_SECONDS=${SERVER_TIMEOUT_SECONDS:-1800}
INFERENCE_SCRIPT=${INFERENCE_SCRIPT:-experiments/hdpo/run_vllm_hdpo_ref_conditioned.py}
REFERENCE_CONDITIONING_INSTRUCTION=${REFERENCE_CONDITIONING_INSTRUCTION:-Use the reference answer only as semantic guidance. Write an answer that is similar in meaning and appropriate for the question, but strictly do not output the exact same text as the reference answer.}

cd "$REPO"
export PYTHONPATH="$REPO/src${PYTHONPATH:+:$PYTHONPATH}"

models=(gemma2_2b qwen25_1_5b)

declare -A model_dir test_path run_name gpu port
model_dir[gemma2_2b]=gemma2-2b-hdpo-en
model_dir[qwen25_1_5b]=qwen25-1-5b-hdpo-en

test_path[gemma2_2b]=${GEMMA2_2B_TEST_PATH:-$DATA_ROOT/hdpo_prepared/gemma2-2b/hdpo_test.json}
test_path[qwen25_1_5b]=${QWEN25_1_5B_TEST_PATH:-$DATA_ROOT/hdpo_prepared/qwen25-1-5b/hdpo_test.json}

run_name[gemma2_2b]=gemma2-2b-hdpo-en-ref-conditioned
run_name[qwen25_1_5b]=qwen25-1-5b-hdpo-en-ref-conditioned

gpu[gemma2_2b]=4
gpu[qwen25_1_5b]=5

port[gemma2_2b]=8064
port[qwen25_1_5b]=8065

require_file() {
  local path="$1"
  if [[ ! -f "$path" ]]; then
    echo "Missing file: $path" >&2
    exit 1
  fi
}

require_dir() {
  local path="$1"
  if [[ ! -d "$path" ]]; then
    echo "Missing directory: $path" >&2
    exit 1
  fi
}

require_ref_conditioned_inference_support() {
  local help_text
  if ! help_text="$(python "$INFERENCE_SCRIPT" --help 2>&1)"; then
    echo "$help_text" >&2
    echo "Could not inspect $INFERENCE_SCRIPT." >&2
    exit 1
  fi
  if [[ "$help_text" != *"--reference-conditioning-instruction"* ]]; then
    echo "$INFERENCE_SCRIPT does not expose --reference-conditioning-instruction." >&2
    echo "Update experiments/rl/vllm_pair_inference.py before launching vLLM." >&2
    exit 1
  fi
}

stop_server_pid() {
  local pid="${1:-}"
  if [[ -n "$pid" ]] && kill -0 "$pid" 2>/dev/null; then
    kill "$pid" 2>/dev/null || true
    wait "$pid" 2>/dev/null || true
  fi
}

wait_for_server() {
  local base_url="$1"
  local pid="$2"
  local log_path="$3"
  local waited=0

  while true; do
    if ! kill -0 "$pid" 2>/dev/null; then
      echo "vLLM server exited before it became ready. Last log lines:" >&2
      tail -n 80 "$log_path" >&2 || true
      return 1
    fi
    if curl -fsS "$base_url/v1/models" >/dev/null 2>&1; then
      return 0
    fi
    if (( waited >= SERVER_TIMEOUT_SECONDS )); then
      echo "Timed out waiting for vLLM at $base_url. Last log lines:" >&2
      tail -n 80 "$log_path" >&2 || true
      return 1
    fi
    sleep 10
    waited=$((waited + 10))
  done
}

start_vllm_server() {
  local alias="$1"
  local model_path="$2"
  local served_name="$3"
  local gpu_id="$4"
  local port_id="$5"
  local log_path="$6"

  CUDA_VISIBLE_DEVICES="$gpu_id" vllm serve "$model_path" \
    --served-model-name "$served_name" \
    --host "$HOST" \
    --port "$port_id" \
    --tensor-parallel-size 1 \
    --max-model-len "$MAX_MODEL_LEN" \
    --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION" \
    --dtype auto \
    --trust-remote-code \
    >"$log_path" 2>&1 &

  local pid=$!
  if ! wait_for_server "http://$HOST:$port_id" "$pid" "$log_path"; then
    stop_server_pid "$pid"
    return 1
  fi
  echo "vLLM ready for $alias on GPU $gpu_id at http://$HOST:$port_id" >&2
  echo "$pid"
}

upload_run_to_hf() {
  local run_dir="$1"
  local remote_dir="$2"

  if command -v hf >/dev/null 2>&1; then
    hf upload "$HF_DATASET_REPO" "$run_dir" "$remote_dir" --repo-type dataset
  elif command -v huggingface-cli >/dev/null 2>&1; then
    huggingface-cli upload --repo-type dataset "$HF_DATASET_REPO" "$run_dir" "$remote_dir"
  else
    echo "Neither 'hf' nor 'huggingface-cli' is available for Hugging Face upload." >&2
    exit 1
  fi
}

validate_model_inputs() {
  local alias="$1"
  local model_path="$MODEL_ROOT/${model_dir[$alias]}"
  local test_lf_path="${test_path[$alias]}"

  require_dir "$model_path"
  require_file "$test_lf_path"
}

run_model() {
  local alias="$1"
  local name="${run_name[$alias]}"
  local model_path="$MODEL_ROOT/${model_dir[$alias]}"
  local test_lf_path="${test_path[$alias]}"
  local gpu_id="${gpu[$alias]}"
  local port_id="${port[$alias]}"
  local run_dir="$RUNS_DIR/$name"

  local job_log="$LOG_DIR/$name.job.log"
  local server_log="$LOG_DIR/$name.vllm.log"
  local inference_log="$LOG_DIR/$name.inference.log"
  local upload_log="$LOG_DIR/$name.hf-upload.log"

  (
    set -euo pipefail
    local server_pid=""
    trap 'stop_server_pid "$server_pid"' EXIT

    mkdir -p "$run_dir"
    echo "=== Ref-conditioned HDPO inference: $name on CUDA device $gpu_id ==="
    echo "Test file: $test_lf_path"
    echo "Run dir: $run_dir"
    echo "vLLM log: $server_log"

    server_pid="$(start_vllm_server "$alias" "$model_path" "$name" "$gpu_id" "$port_id" "$server_log")"

    local inference_mode=(--overwrite)
    if [[ "${RESUME:-0}" == "1" ]]; then
      inference_mode=(--resume)
    fi

    CUDA_VISIBLE_DEVICES="$gpu_id" python "$INFERENCE_SCRIPT" \
      --input-path "$test_lf_path" \
      --output-dir "$run_dir" \
      --model-name "$name" \
      --tokenizer-name-or-path "$model_path" \
      --base-url "http://$HOST:$port_id" \
      --conditioning-mode profile \
      --sample-size 0 \
      --sample-mode first \
      --max-tokens "$MAX_TOKENS" \
      --max-model-len "$MAX_MODEL_LEN" \
      --batch-size "$BATCH_SIZE" \
      --temperature 0.0 \
      --top-p 1.0 \
      --trust-remote-code \
      --reference-conditioning-instruction "$REFERENCE_CONDITIONING_INSTRUCTION" \
      "${inference_mode[@]}" \
      2>&1 | tee "$inference_log"

    stop_server_pid "$server_pid"
    server_pid=""

    echo "Uploading $name predictions to Hugging Face"
    upload_run_to_hf "$run_dir" "$HF_REMOTE_RUNS_DIR/$name" 2>&1 | tee "$upload_log"

    echo "=== Completed $name ==="
  ) > >(tee "$job_log") 2>&1
}

mkdir -p "$RUNS_DIR" "$LOG_DIR"
echo "Logs directory: $LOG_DIR"

require_ref_conditioned_inference_support

for alias in "${models[@]}"; do
  validate_model_inputs "$alias"
done

declare -A job_pids

cleanup_jobs() {
  for pid in "${job_pids[@]:-}"; do
    if kill -0 "$pid" 2>/dev/null; then
      kill "$pid" 2>/dev/null || true
    fi
  done
}

trap cleanup_jobs INT TERM

for alias in "${models[@]}"; do
  run_model "$alias" &
  job_pids[$alias]=$!
  echo "Started ${run_name[$alias]} on CUDA device ${gpu[$alias]} with PID ${job_pids[$alias]}"
done

status=0
for alias in "${models[@]}"; do
  if wait "${job_pids[$alias]}"; then
    echo "Finished ${run_name[$alias]}"
  else
    echo "Failed ${run_name[$alias]} (see $LOG_DIR/${run_name[$alias]}.job.log)" >&2
    status=1
  fi
done

exit "$status"
