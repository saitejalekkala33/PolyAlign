#!/usr/bin/env bash
set -euo pipefail

# Runs the remaining English Dist-SFT val prediction jobs through vLLM in parallel:
# - GPU 3: gemma2_2b_dist-sft-val-en
# - GPU 4: qwen25_1_5b_dist_sft_val-en
# - GPU 5: qwen25_3b_dist-sft-val-en
#
# Model context lengths checked from HF configs:
# - gemma2-2b:        max_position_embeddings=8192
# - qwen2.5-3b:       max_position_embeddings=32768
# - qwen2.5-1.5b:     max_position_embeddings=131072
# For A100 40GB runs, Qwen jobs are capped to 16384 server context to avoid
# wasting KV cache on the full theoretical config limits.
# Llama3.2 Dist-SFT is intentionally excluded for now because the target repo is
# not exposing usable model files yet.
#
# Inference generation max length is fixed by default to 3072 new tokens.

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

HF_DATASET_REPO="${HF_DATASET_REPO:-saiteja33/PolyAlign-All}"
DATA_ROOT="${DATA_ROOT:-data/hf}"
LLAMAFACTORY_DIR="$DATA_ROOT/english/merged_sft_dedup/llamafactory"
RUNS_DIR="$DATA_ROOT/english/merged_sft_dedup/runs"
LOG_ROOT="${LOG_ROOT:-logs/en-dist-sft-vllm}"

GPUS=(${GPUS:-3 4 5})
PORTS=(${PORTS:-8103 8104 8105})

MAX_TOKENS="${MAX_TOKENS:-3072}"
BATCH_SIZE="${BATCH_SIZE:-8}"
TEMPERATURE="${TEMPERATURE:-0.2}"
TOP_P="${TOP_P:-0.95}"
TIMEOUT="${TIMEOUT:-600}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.90}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-8}"
VLLM_READY_TIMEOUT_SECONDS="${VLLM_READY_TIMEOUT_SECONDS:-1800}"
UPLOAD_TO_HF="${UPLOAD_TO_HF:-1}"
SKIP_MISSING_MODELS="${SKIP_MISSING_MODELS:-0}"
TASK_SET="${TASK_SET:-missing}"

mkdir -p "$LLAMAFACTORY_DIR" "$RUNS_DIR" "$LOG_ROOT"

require_cmd() {
  command -v "$1" >/dev/null 2>&1 || {
    echo "Missing required command: $1" >&2
    exit 1
  }
}

require_cmd hf
require_cmd vllm
require_cmd curl
require_cmd python

echo "Downloading Dist-SFT split files into $DATA_ROOT"
hf download "$HF_DATASET_REPO" \
  --repo-type dataset \
  --local-dir "$DATA_ROOT" \
  --include "english/merged_sft_dedup/llamafactory/dist_sft_val-en.json"

load_tasks() {
  case "$TASK_SET" in
    missing)
      cat <<'TASKS'
gemma2_2b_dist-sft-val-en|sathiiiii/polyalign-gemma2-2b-en-dist-sft|val|8192
qwen25_1_5b_dist_sft_val-en|sathiiiii/polyalign-qwen2.5-1.5b-en-dist-sft|val|16384
qwen25_3b_dist-sft-val-en|sathiiiii/polyalign-qwen2.5-3b-en-dist-sft|val|16384
TASKS
      ;;
    all|val)
      cat <<'TASKS'
gemma2_2b_dist-sft-val-en|sathiiiii/polyalign-gemma2-2b-en-dist-sft|val|8192
qwen25_1_5b_dist_sft_val-en|sathiiiii/polyalign-qwen2.5-1.5b-en-dist-sft|val|16384
qwen25_3b_dist-sft-val-en|sathiiiii/polyalign-qwen2.5-3b-en-dist-sft|val|16384
TASKS
      ;;
    *)
      echo "Unsupported TASK_SET=$TASK_SET. Use TASK_SET=missing, TASK_SET=val, or TASK_SET=all." >&2
      exit 1
      ;;
  esac
}

mapfile -t TASKS < <(load_tasks)

validate_model() {
  local model_id="$1"
  if hf download "$model_id" config.json --repo-type model >/dev/null 2>&1; then
    return 0
  fi

  echo "Model repo is missing config.json or is not accessible: $model_id" >&2
  if [[ "$SKIP_MISSING_MODELS" == "1" ]]; then
    echo "Skipping because SKIP_MISSING_MODELS=1: $model_id" >&2
    return 1
  fi
  exit 1
}

wait_for_vllm() {
  local base_url="$1"
  local server_pid="$2"
  local server_log="$3"
  local waited=0

  until curl -fsS "$base_url/v1/models" >/dev/null 2>&1; do
    if ! kill -0 "$server_pid" >/dev/null 2>&1; then
      echo "vLLM process exited before becoming ready. Last log lines:" >&2
      tail -n 120 "$server_log" >&2 || true
      return 1
    fi
    if (( waited >= VLLM_READY_TIMEOUT_SECONDS )); then
      echo "Timed out waiting for vLLM at $base_url. Last log lines:" >&2
      tail -n 120 "$server_log" >&2 || true
      return 1
    fi
    sleep 5
    waited=$((waited + 5))
  done
}

stop_vllm() {
  local server_pid="$1"
  if kill -0 "$server_pid" >/dev/null 2>&1; then
    kill "$server_pid" >/dev/null 2>&1 || true
    wait "$server_pid" >/dev/null 2>&1 || true
  fi
}

is_completed() {
  local progress_path="$1"
  [[ -f "$progress_path" ]] && grep -q '"status": "completed"' "$progress_path"
}

run_one() {
  local gpu="$1"
  local port="$2"
  local run_name="$3"
  local model_id="$4"
  local split="$5"
  local max_model_len="$6"

  local input_path="$LLAMAFACTORY_DIR/dist_sft_${split}-en.json"
  local output_dir="$RUNS_DIR/$run_name"
  local base_url="http://127.0.0.1:$port"
  local server_log="$output_dir/vllm_server.gpu${gpu}.log"
  local inference_log="$output_dir/inference.log"
  local upload_log="$output_dir/upload.log"

  mkdir -p "$output_dir"

  if [[ ! -f "$input_path" ]]; then
    echo "Missing input split file: $input_path" >&2
    return 1
  fi

  if ! validate_model "$model_id"; then
    return 0
  fi

  if is_completed "$output_dir/progress.json"; then
    echo "Already completed locally: $run_name"
  else
    echo "Starting vLLM for $run_name on GPU $gpu, port $port"
    CUDA_VISIBLE_DEVICES="$gpu" vllm serve "$model_id" \
      --served-model-name "$run_name" \
      --host 127.0.0.1 \
      --port "$port" \
      --dtype auto \
      --max-model-len "$max_model_len" \
      --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION" \
      --max-num-seqs "$MAX_NUM_SEQS" \
      --trust-remote-code \
      >"$server_log" 2>&1 &

    local server_pid=$!
    if ! wait_for_vllm "$base_url" "$server_pid" "$server_log"; then
      stop_vllm "$server_pid"
      return 1
    fi

    echo "Running predictions for $run_name"
    set +e
    PYTHONPATH="$ROOT_DIR/src" python experiments/sft/run_vllm_dist_sft.py \
      --input-path "$input_path" \
      --output-dir "$output_dir" \
      --model-name "$run_name" \
      --tokenizer-name-or-path "$model_id" \
      --base-url "$base_url" \
      --conditioning-mode profile \
      --sample-size 0 \
      --max-tokens "$MAX_TOKENS" \
      --temperature "$TEMPERATURE" \
      --top-p "$TOP_P" \
      --batch-size "$BATCH_SIZE" \
      --timeout "$TIMEOUT" \
      --trust-remote-code \
      --resume \
      2>&1 | tee "$inference_log"
    local infer_status=${PIPESTATUS[0]}
    set -e

    stop_vllm "$server_pid"

    if (( infer_status != 0 )); then
      echo "Inference failed for $run_name; see $inference_log" >&2
      return "$infer_status"
    fi
  fi

  if [[ "$UPLOAD_TO_HF" == "1" ]]; then
    echo "Uploading $run_name to $HF_DATASET_REPO"
    hf upload "$HF_DATASET_REPO" \
      "$output_dir" \
      "english/merged_sft_dedup/runs/$run_name" \
      --repo-type dataset \
      --commit-message "Upload $run_name Dist-SFT predictions" \
      2>&1 | tee "$upload_log"
  fi
}

run_worker() {
  local worker_index="$1"
  local gpu="${GPUS[$worker_index]}"
  local port="${PORTS[$worker_index]}"
  local num_workers="${#GPUS[@]}"

  for task_index in "${!TASKS[@]}"; do
    if (( task_index % num_workers != worker_index )); then
      continue
    fi

    IFS='|' read -r run_name model_id split max_model_len <<<"${TASKS[$task_index]}"
    run_one "$gpu" "$port" "$run_name" "$model_id" "$split" "$max_model_len"
  done
}

if (( ${#GPUS[@]} != ${#PORTS[@]} )); then
  echo "GPUS and PORTS must have the same number of entries." >&2
  exit 1
fi

echo "Task set: $TASK_SET"
printf '  %s\n' "${TASKS[@]}"

pids=()
for worker_index in "${!GPUS[@]}"; do
  run_worker "$worker_index" >"$LOG_ROOT/worker-gpu${GPUS[$worker_index]}.log" 2>&1 &
  pids+=("$!")
done

set +e
status=0
for pid in "${pids[@]}"; do
  wait "$pid"
  worker_status=$?
  if (( worker_status != 0 )); then
    status=$worker_status
  fi
done
set -e

if (( status == 0 )); then
  echo "All scheduled Dist-SFT jobs completed."
else
  echo "At least one worker failed. Check $LOG_ROOT/worker-gpu*.log" >&2
fi

exit "$status"
