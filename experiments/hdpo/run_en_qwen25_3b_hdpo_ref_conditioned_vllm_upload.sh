#!/usr/bin/env bash
set -euo pipefail

REPO=${REPO:-/home/umair/TW/PolyAlign}
MODEL_ROOT=${MODEL_ROOT:-$REPO/models}
DATA_ROOT=${DATA_ROOT:-$REPO/data/english/hdpo}
RUNS_DIR=${RUNS_DIR:-$REPO/data/english/merged_sft_dedup/runs}
HF_DATASET_REPO=${HF_DATASET_REPO:-saiteja33/PolyAlign-All}
HF_REMOTE_RUNS_DIR=${HF_REMOTE_RUNS_DIR:-english/merged_sft_dedup/runs}
RUN_ID=${RUN_ID:-$(date +%Y%m%d_%H%M%S)}
LOG_DIR=${LOG_DIR:-$REPO/logs/hdpo_ref_conditioned_vllm_en_qwen25_3b/$RUN_ID}

HOST=${HOST:-127.0.0.1}
GPU_ID=${GPU_ID:-7}
PORT=${PORT:-8067}
MAX_MODEL_LEN=${MAX_MODEL_LEN:-8192}
MAX_TOKENS=${MAX_TOKENS:-128}
BATCH_SIZE=${BATCH_SIZE:-4}
GPU_MEMORY_UTILIZATION=${GPU_MEMORY_UTILIZATION:-0.70}
SERVER_TIMEOUT_SECONDS=${SERVER_TIMEOUT_SECONDS:-1800}
INFERENCE_SCRIPT=${INFERENCE_SCRIPT:-experiments/hdpo/run_vllm_hdpo_ref_conditioned.py}
REFERENCE_CONDITIONING_INSTRUCTION=${REFERENCE_CONDITIONING_INSTRUCTION:-Use the reference answer only as semantic guidance. Write an answer that is similar in meaning and appropriate for the question, but strictly do not output the exact same text as the reference answer.}

MODEL_DIR=${MODEL_DIR:-qwen25-3b-hdpo-en}
TEST_PATH=${TEST_PATH:-$DATA_ROOT/hdpo_prepared/qwen25-3b/hdpo_val.json}
RUN_NAME=${RUN_NAME:-qwen25-3b-hdpo-en-ref-conditioned}

cd "$REPO"
export PYTHONPATH="$REPO/src${PYTHONPATH:+:$PYTHONPATH}"

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
    echo "Update experiments/hdpo/run_vllm_hdpo_ref_conditioned.py before launching vLLM." >&2
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
  local model_path="$1"
  local log_path="$2"

  CUDA_VISIBLE_DEVICES="$GPU_ID" vllm serve "$model_path" \
    --served-model-name "$RUN_NAME" \
    --host "$HOST" \
    --port "$PORT" \
    --tensor-parallel-size 1 \
    --max-model-len "$MAX_MODEL_LEN" \
    --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION" \
    --dtype auto \
    --trust-remote-code \
    >"$log_path" 2>&1 &

  local pid=$!
  if ! wait_for_server "http://$HOST:$PORT" "$pid" "$log_path"; then
    stop_server_pid "$pid"
    return 1
  fi
  echo "vLLM ready for $RUN_NAME on GPU $GPU_ID at http://$HOST:$PORT" >&2
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

MODEL_PATH="$MODEL_ROOT/$MODEL_DIR"
RUN_DIR="$RUNS_DIR/$RUN_NAME"
JOB_LOG="$LOG_DIR/$RUN_NAME.job.log"
SERVER_LOG="$LOG_DIR/$RUN_NAME.vllm.log"
INFERENCE_LOG="$LOG_DIR/$RUN_NAME.inference.log"
UPLOAD_LOG="$LOG_DIR/$RUN_NAME.hf-upload.log"

mkdir -p "$RUNS_DIR" "$LOG_DIR"
echo "Logs directory: $LOG_DIR"

require_ref_conditioned_inference_support
require_dir "$MODEL_PATH"
require_file "$TEST_PATH"

(
  set -euo pipefail
  server_pid=""
  trap 'stop_server_pid "$server_pid"' EXIT

  mkdir -p "$RUN_DIR"
  echo "=== Ref-conditioned HDPO inference: $RUN_NAME on CUDA device $GPU_ID ==="
  echo "Model path: $MODEL_PATH"
  echo "Test file: $TEST_PATH"
  echo "Run dir: $RUN_DIR"
  echo "vLLM log: $SERVER_LOG"

  server_pid="$(start_vllm_server "$MODEL_PATH" "$SERVER_LOG")"

  inference_mode=(--overwrite)
  if [[ "${RESUME:-0}" == "1" ]]; then
    inference_mode=(--resume)
  fi

  CUDA_VISIBLE_DEVICES="$GPU_ID" python "$INFERENCE_SCRIPT" \
    --input-path "$TEST_PATH" \
    --output-dir "$RUN_DIR" \
    --model-name "$RUN_NAME" \
    --tokenizer-name-or-path "$MODEL_PATH" \
    --base-url "http://$HOST:$PORT" \
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
    2>&1 | tee "$INFERENCE_LOG"

  stop_server_pid "$server_pid"
  server_pid=""

  echo "Uploading $RUN_NAME predictions to Hugging Face"
  upload_run_to_hf "$RUN_DIR" "$HF_REMOTE_RUNS_DIR/$RUN_NAME" 2>&1 | tee "$UPLOAD_LOG"

  echo "=== Completed $RUN_NAME ==="
) > >(tee "$JOB_LOG") 2>&1
