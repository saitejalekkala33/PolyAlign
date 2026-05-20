#!/usr/bin/env bash
set -euo pipefail

REPO="${REPO:-/home/umair/TW/PolyAlign}"
cd "$REPO"

HF_DATASET_REPO="${HF_DATASET_REPO:-saiteja33/PolyAlign-All}"
RAW_ROOT="${RAW_ROOT:-$REPO/data/llm_judge/raw}"
WORK_DIR="${WORK_DIR:-$REPO/data/llm_judge/work}"
MODELS_ROOT="${MODELS_ROOT:-$REPO/models/llm-judges}"
LOG_ROOT="${LOG_ROOT:-$REPO/logs/llm-judge}"

RUBRIC_YAML="${RUBRIC_YAML:-$REPO/scripts/llm_judge/rubric.yaml}"
PROMPTS_PY="${PROMPTS_PY:-$REPO/scripts/llm_judge/llm-judge-prompts.py}"

QWEN_JUDGE_ID="${QWEN_JUDGE_ID:-qwen3_30b_a3b_instruct_2507}"
QWEN_MODEL_ID="${QWEN_MODEL_ID:-Qwen/Qwen3-30B-A3B-Instruct-2507}"
QWEN_GPU_GROUPS="${QWEN_GPU_GROUPS:-0,1,2,3;4,5,6,7}"
QWEN_PORTS=(${QWEN_PORTS:-8100 8101})
QWEN_CHAT_TEMPLATE_KWARGS_JSON="${QWEN_CHAT_TEMPLATE_KWARGS_JSON:-{}}"
QWEN_EXTRA_BODY_JSON="${QWEN_EXTRA_BODY_JSON:-{}}"

GLM_JUDGE_ID="${GLM_JUDGE_ID:-glm45_air_fp8}"
GLM_MODEL_ID="${GLM_MODEL_ID:-zai-org/GLM-4.5-Air-FP8}"
GLM_GPU_GROUPS="${GLM_GPU_GROUPS:-0,1,2,3;4,5,6,7}"
GLM_PORTS=(${GLM_PORTS:-8100 8101})
GLM_CHAT_TEMPLATE_KWARGS_JSON="${GLM_CHAT_TEMPLATE_KWARGS_JSON:-{\"enable_thinking\":false}}"
GLM_EXTRA_BODY_JSON="${GLM_EXTRA_BODY_JSON:-{}}"

JUDGES=(${JUDGES:-$QWEN_JUDGE_ID $GLM_JUDGE_ID})

DOWNLOAD_DATA="${DOWNLOAD_DATA:-1}"
DOWNLOAD_JUDGE_MODELS="${DOWNLOAD_JUDGE_MODELS:-1}"
BUILD_INPUTS="${BUILD_INPUTS:-1}"
RUN_JUDGES="${RUN_JUDGES:-1}"
UPLOAD_TO_HF="${UPLOAD_TO_HF:-1}"
GIT_PUSH_SUMMARIES="${GIT_PUSH_SUMMARIES:-1}"
GIT_PUSH_CONTINUE_ON_ERROR="${GIT_PUSH_CONTINUE_ON_ERROR:-1}"
CLEANUP_JUDGE_MODEL_AFTER_RUN="${CLEANUP_JUDGE_MODEL_AFTER_RUN:-0}"
RESUME="${RESUME:-1}"

SAMPLE_SIZE="${SAMPLE_SIZE:-0}"
SAMPLE_MODE="${SAMPLE_MODE:-first}"
SEED="${SEED:-42}"
BATCH_SIZE="${BATCH_SIZE:-8}"
MAX_TOKENS="${MAX_TOKENS:-384}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-32768}"
PROMPT_SAFETY_MARGIN="${PROMPT_SAFETY_MARGIN:-64}"
OVERLENGTH_POLICY="${OVERLENGTH_POLICY:-fail}"
TEMPERATURE="${TEMPERATURE:-0}"
TOP_P="${TOP_P:-1}"
TIMEOUT="${TIMEOUT:-600}"
MAX_RETRIES="${MAX_RETRIES:-2}"
GUIDED_JSON="${GUIDED_JSON:-1}"

VLLM_DTYPE="${VLLM_DTYPE:-auto}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.90}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-32}"
VLLM_TRUST_REMOTE_CODE="${VLLM_TRUST_REMOTE_CODE:-1}"
ENABLE_EXPERT_PARALLEL="${ENABLE_EXPERT_PARALLEL:-0}"
GIT_LD_LIBRARY_PATH="${GIT_LD_LIBRARY_PATH:-/usr/lib/x86_64-linux-gnu:/lib/x86_64-linux-gnu}"

mkdir -p "$RAW_ROOT" "$WORK_DIR" "$MODELS_ROOT" "$LOG_ROOT"

SERVER_PIDS=()

log_step() {
  echo ""
  echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"
}

require_cmd() {
  command -v "$1" >/dev/null 2>&1 || {
    echo "Missing required command: $1" >&2
    exit 1
  }
}

run_python() {
  PYTHONPATH="$REPO/src:${PYTHONPATH:-}" python "$REPO/scripts/llm_judge/polyalign_llm_judge.py" "$@"
}

judge_model_id() {
  case "$1" in
    "$QWEN_JUDGE_ID") printf '%s\n' "$QWEN_MODEL_ID" ;;
    "$GLM_JUDGE_ID") printf '%s\n' "$GLM_MODEL_ID" ;;
    *) echo "Unknown judge id: $1" >&2; return 1 ;;
  esac
}

judge_gpu_groups() {
  case "$1" in
    "$QWEN_JUDGE_ID") printf '%s\n' "$QWEN_GPU_GROUPS" ;;
    "$GLM_JUDGE_ID") printf '%s\n' "$GLM_GPU_GROUPS" ;;
    *) echo "Unknown judge id: $1" >&2; return 1 ;;
  esac
}

judge_ports() {
  case "$1" in
    "$QWEN_JUDGE_ID") printf '%s\n' "${QWEN_PORTS[*]}" ;;
    "$GLM_JUDGE_ID") printf '%s\n' "${GLM_PORTS[*]}" ;;
    *) echo "Unknown judge id: $1" >&2; return 1 ;;
  esac
}

judge_chat_template_kwargs_json() {
  case "$1" in
    "$QWEN_JUDGE_ID") printf '%s\n' "$QWEN_CHAT_TEMPLATE_KWARGS_JSON" ;;
    "$GLM_JUDGE_ID") printf '%s\n' "$GLM_CHAT_TEMPLATE_KWARGS_JSON" ;;
    *) echo "Unknown judge id: $1" >&2; return 1 ;;
  esac
}

judge_extra_body_json() {
  case "$1" in
    "$QWEN_JUDGE_ID") printf '%s\n' "$QWEN_EXTRA_BODY_JSON" ;;
    "$GLM_JUDGE_ID") printf '%s\n' "$GLM_EXTRA_BODY_JSON" ;;
    *) echo "Unknown judge id: $1" >&2; return 1 ;;
  esac
}

tensor_parallel_size() {
  local gpus="$1"
  awk -F',' '{print NF}' <<<"$gpus"
}

local_model_dir() {
  printf '%s\n' "$MODELS_ROOT/$1"
}

resolve_model_ref() {
  local judge_id="$1"
  local model_id
  model_id="$(judge_model_id "$judge_id")"
  local local_dir
  local_dir="$(local_model_dir "$judge_id")"
  if [[ -f "$local_dir/config.json" ]]; then
    printf '%s\n' "$local_dir"
  else
    printf '%s\n' "$model_id"
  fi
}

download_dataset_file() {
  local rel_path="$1"
  if [[ -f "$RAW_ROOT/$rel_path" ]]; then
    echo "Dataset file already present: $rel_path"
    return 0
  fi
  echo "Downloading dataset file: $rel_path"
  hf download "$HF_DATASET_REPO" "$rel_path" --repo-type dataset --local-dir "$RAW_ROOT"
}

download_required_data() {
  if [[ "$DOWNLOAD_DATA" != "1" ]]; then
    log_step "Skipping dataset download because DOWNLOAD_DATA=$DOWNLOAD_DATA"
    return 0
  fi

  log_step "Downloading required HF dataset files into $RAW_ROOT"
  local downloads_file="$LOG_ROOT/required_hf_files.txt"
  run_python list-downloads --lang all --include-human > "$downloads_file"
  while IFS= read -r rel_path; do
    [[ -n "$rel_path" ]] || continue
    download_dataset_file "$rel_path"
  done < "$downloads_file"
}

download_judge_model() {
  local judge_id="$1"
  local model_id
  model_id="$(judge_model_id "$judge_id")"
  local local_dir
  local_dir="$(local_model_dir "$judge_id")"
  if [[ "$DOWNLOAD_JUDGE_MODELS" != "1" ]]; then
    return 0
  fi
  if [[ -f "$local_dir/config.json" ]]; then
    echo "Judge model already present: $judge_id -> $local_dir"
    return 0
  fi
  mkdir -p "$local_dir"
  echo "Downloading judge model: $model_id -> $local_dir"
  hf download "$model_id" --repo-type model --local-dir "$local_dir"
}

cleanup_judge_model() {
  local judge_id="$1"
  if [[ "$CLEANUP_JUDGE_MODEL_AFTER_RUN" != "1" ]]; then
    return 0
  fi
  local local_dir
  local_dir="$(local_model_dir "$judge_id")"
  if [[ -d "$local_dir" ]]; then
    log_step "Removing local judge model after run: $local_dir"
    rm -rf -- "$local_dir"
  fi
}

build_inputs() {
  if [[ "$BUILD_INPUTS" != "1" ]]; then
    log_step "Skipping input build because BUILD_INPUTS=$BUILD_INPUTS"
    return 0
  fi
  log_step "Building LLM-judge input files under $WORK_DIR"
  run_python build-inputs \
    --raw-root "$RAW_ROOT" \
    --output-root "$WORK_DIR" \
    --lang all \
    2>&1 | tee "$LOG_ROOT/build-inputs.log"
}

wait_for_vllm() {
  local base_url="$1"
  local server_pid="$2"
  local server_log="$3"
  local waited=0
  while (( waited < 1800 )); do
    if ! kill -0 "$server_pid" >/dev/null 2>&1; then
      echo "vLLM process exited before becoming ready. Log: $server_log" >&2
      tail -n 80 "$server_log" >&2 || true
      return 1
    fi
    if curl -fsS "$base_url/v1/models" >/dev/null 2>&1; then
      echo "vLLM ready at $base_url"
      return 0
    fi
    sleep 5
    waited=$((waited + 5))
  done
  echo "Timed out waiting for vLLM at $base_url. Log: $server_log" >&2
  tail -n 80 "$server_log" >&2 || true
  return 1
}

stop_vllm() {
  local server_pid="$1"
  if [[ -n "$server_pid" ]] && kill -0 "$server_pid" >/dev/null 2>&1; then
    kill "$server_pid" >/dev/null 2>&1 || true
    wait "$server_pid" >/dev/null 2>&1 || true
  fi
}

cleanup_servers() {
  local pid
  for pid in "${SERVER_PIDS[@]:-}"; do
    stop_vllm "$pid"
  done
}
trap cleanup_servers EXIT

start_vllm_server() {
  local judge_id="$1"
  local model_ref="$2"
  local gpus="$3"
  local port="$4"
  local server_label="${5:-main}"
  local tp_size
  tp_size="$(tensor_parallel_size "$gpus")"
  local base_url="http://127.0.0.1:$port"
  local server_log="$LOG_ROOT/${judge_id}.${server_label}.vllm.log"

  local serve_args=(
    "$model_ref"
    --served-model-name "$judge_id"
    --host 127.0.0.1
    --port "$port"
    --tensor-parallel-size "$tp_size"
    --dtype "$VLLM_DTYPE"
    --max-model-len "$MAX_MODEL_LEN"
    --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION"
    --max-num-seqs "$MAX_NUM_SEQS"
  )
  if [[ "$VLLM_TRUST_REMOTE_CODE" == "1" ]]; then
    serve_args+=(--trust-remote-code)
  fi
  if [[ "$judge_id" == "$GLM_JUDGE_ID" ]]; then
    serve_args+=(--reasoning-parser glm45)
  fi
  if [[ "$ENABLE_EXPERT_PARALLEL" == "1" ]]; then
    serve_args+=(--enable-expert-parallel)
  fi
  if [[ -n "${MAX_NUM_BATCHED_TOKENS:-}" ]]; then
    serve_args+=(--max-num-batched-tokens "$MAX_NUM_BATCHED_TOKENS")
  fi
  if [[ -n "${KV_CACHE_DTYPE:-}" ]]; then
    serve_args+=(--kv-cache-dtype "$KV_CACHE_DTYPE")
  fi
  if [[ -n "${CPU_OFFLOAD_GB:-}" ]]; then
    serve_args+=(--cpu-offload-gb "$CPU_OFFLOAD_GB")
  fi

  log_step "Starting vLLM judge=$judge_id GPUs=$gpus TP=$tp_size port=$port" >&2
  CUDA_VISIBLE_DEVICES="$gpus" vllm serve "${serve_args[@]}" >"$server_log" 2>&1 &
  local server_pid=$!
  SERVER_PIDS+=("$server_pid")
  if ! wait_for_vllm "$base_url" "$server_pid" "$server_log" >&2; then
    stop_vllm "$server_pid"
    return 1
  fi
  printf '%s\n' "$server_pid"
}

run_one_judge_shard() {
  local judge_id="$1"
  local shard_index="$2"
  local num_shards="$3"
  local gpus
  gpus="$4"
  local port
  port="$5"
  local model_ref
  model_ref="$6"
  local base_url="http://127.0.0.1:$port"
  local chat_template_kwargs_json
  chat_template_kwargs_json="$(judge_chat_template_kwargs_json "$judge_id")"
  local extra_body_json
  extra_body_json="$(judge_extra_body_json "$judge_id")"
  local server_pid=""
  local judge_log="$LOG_ROOT/${judge_id}.shard${shard_index}.judge.log"

  server_pid="$(start_vllm_server "$judge_id" "$model_ref" "$gpus" "$port" "shard${shard_index}")"

  local resume_args=()
  if [[ "$RESUME" == "1" ]]; then
    resume_args+=(--resume)
  fi

  local guided_args=()
  if [[ "$GUIDED_JSON" != "1" ]]; then
    guided_args+=(--disable-guided-json)
  fi

  log_step "Running LLM judge: $judge_id shard $shard_index/$num_shards"
  set +e
  run_python judge \
    --work-dir "$WORK_DIR" \
    --lang all \
    --judge-id "$judge_id" \
    --judge-model-name "$judge_id" \
    --tokenizer-name-or-path "$model_ref" \
    --base-url "$base_url" \
    --rubric-yaml "$RUBRIC_YAML" \
    --prompts-py "$PROMPTS_PY" \
    --sample-size "$SAMPLE_SIZE" \
    --sample-mode "$SAMPLE_MODE" \
    --seed "$SEED" \
    --source-shard-index "$shard_index" \
    --source-num-shards "$num_shards" \
    --batch-size "$BATCH_SIZE" \
    --max-tokens "$MAX_TOKENS" \
    --max-model-len "$MAX_MODEL_LEN" \
    --prompt-safety-margin "$PROMPT_SAFETY_MARGIN" \
    --overlength-policy "$OVERLENGTH_POLICY" \
    --temperature "$TEMPERATURE" \
    --top-p "$TOP_P" \
    --timeout "$TIMEOUT" \
    --max-retries "$MAX_RETRIES" \
    --extra-body-json "$extra_body_json" \
    --chat-template-kwargs-json "$chat_template_kwargs_json" \
    --trust-remote-code \
    "${guided_args[@]}" \
    "${resume_args[@]}" \
    2>&1 | tee "$judge_log"
  local judge_status=${PIPESTATUS[0]}
  set -e

  stop_vllm "$server_pid"
  server_pid=""

  if (( judge_status != 0 )); then
    echo "Judge failed: $judge_id shard $shard_index/$num_shards. Check $judge_log" >&2
    return "$judge_status"
  fi
}

run_one_judge_model() {
  local judge_id="$1"
  local model_ref
  model_ref="$(resolve_model_ref "$judge_id")"
  local groups_string
  groups_string="$(judge_gpu_groups "$judge_id")"
  local ports_string
  ports_string="$(judge_ports "$judge_id")"
  local gpu_groups=()
  IFS=';' read -r -a gpu_groups <<<"$groups_string"
  local ports=($ports_string)

  if (( ${#gpu_groups[@]} == 0 )); then
    echo "No GPU groups configured for judge $judge_id" >&2
    return 1
  fi
  if (( ${#gpu_groups[@]} != ${#ports[@]} )); then
    echo "GPU group count (${#gpu_groups[@]}) must match port count (${#ports[@]}) for $judge_id" >&2
    return 1
  fi

  log_step "Launching $judge_id on ${#gpu_groups[@]} parallel source-file shards"
  local pids=()
  local shard_index
  for shard_index in "${!gpu_groups[@]}"; do
    run_one_judge_shard \
      "$judge_id" \
      "$shard_index" \
      "${#gpu_groups[@]}" \
      "${gpu_groups[$shard_index]}" \
      "${ports[$shard_index]}" \
      "$model_ref" \
      >"$LOG_ROOT/${judge_id}.shard${shard_index}.worker.log" 2>&1 &
    pids+=("$!")
  done

  local status=0
  local pid
  for pid in "${pids[@]}"; do
    if ! wait "$pid"; then
      status=1
    fi
  done
  if (( status != 0 )); then
    echo "Judge $judge_id failed on at least one shard. Check $LOG_ROOT/${judge_id}.shard*.worker.log" >&2
    return 1
  fi

  run_python summarize \
    --work-dir "$WORK_DIR" \
    --lang all \
    --judge-id "$judge_id" \
    --rubric-yaml "$RUBRIC_YAML" \
    2>&1 | tee "$LOG_ROOT/${judge_id}.summary.log"
}

run_judges_sequential_by_model() {
  if [[ "$RUN_JUDGES" != "1" ]]; then
    log_step "Skipping judge runs because RUN_JUDGES=$RUN_JUDGES"
    return 0
  fi

  local judge_id
  for judge_id in "${JUDGES[@]}"; do
    log_step "Preparing judge model: $judge_id"
    download_judge_model "$judge_id"
    run_one_judge_model "$judge_id"
    cleanup_judge_model "$judge_id"
  done
}

combine_summaries() {
  log_step "Combining judge summaries"
  run_python summarize --work-dir "$WORK_DIR" --lang all --judge-id all --rubric-yaml "$RUBRIC_YAML"
  run_python combine-summaries --work-dir "$WORK_DIR" --lang all
}

upload_outputs() {
  if [[ "$UPLOAD_TO_HF" != "1" ]]; then
    log_step "Skipping HF upload because UPLOAD_TO_HF=$UPLOAD_TO_HF"
    return 0
  fi

  local lang hf_prefix judge_id
  for lang in en zh; do
    [[ "$lang" == "en" ]] && hf_prefix="english" || hf_prefix="chinese"
    for judge_id in "${JUDGES[@]}"; do
      if [[ -d "$WORK_DIR/$lang/scores/$judge_id" ]]; then
        log_step "Uploading $lang $judge_id judge score files to HF"
        hf upload "$HF_DATASET_REPO" "$WORK_DIR/$lang/scores/$judge_id" \
          "$hf_prefix/llm-judge/scores/$judge_id" \
          --repo-type dataset \
          --commit-message "Upload $lang $judge_id LLM judge scores"
      fi
    done
    if [[ -d "$WORK_DIR/$lang/metrics/llm_judge" ]]; then
      log_step "Uploading $lang LLM judge metric summaries to HF"
      hf upload "$HF_DATASET_REPO" "$WORK_DIR/$lang/metrics/llm_judge" \
        "$hf_prefix/llm-judge/metrics" \
        --repo-type dataset \
        --commit-message "Upload $lang LLM judge metrics"
    fi
  done

  if [[ -d "$WORK_DIR/metrics" ]]; then
    log_step "Uploading combined LLM judge summaries to HF"
    hf upload "$HF_DATASET_REPO" "$WORK_DIR/metrics" \
      "llm-judge/metrics" \
      --repo-type dataset \
      --commit-message "Upload combined LLM judge metrics"
  fi
}

git_commit_outputs() {
  if [[ "$GIT_PUSH_SUMMARIES" != "1" ]]; then
    log_step "Skipping git commit/push because GIT_PUSH_SUMMARIES=$GIT_PUSH_SUMMARIES"
    return 0
  fi

  log_step "Git add/commit/push LLM judge code and summaries"
  LD_LIBRARY_PATH="$GIT_LD_LIBRARY_PATH" git add \
    scripts/llm_judge \
    "$WORK_DIR/input_summary.json" \
    "$WORK_DIR/en/input_summary.json" \
    "$WORK_DIR/en/manifest.json" \
    "$WORK_DIR/en/metrics" \
    "$WORK_DIR/zh/input_summary.json" \
    "$WORK_DIR/zh/manifest.json" \
    "$WORK_DIR/zh/metrics" \
    "$WORK_DIR/metrics"

  if LD_LIBRARY_PATH="$GIT_LD_LIBRARY_PATH" git diff --cached --quiet; then
    echo "No LLM judge code or summary changes to commit."
    return 0
  fi

  LD_LIBRARY_PATH="$GIT_LD_LIBRARY_PATH" git commit -m "Add PolyAlign LLM judge evaluation"
  set +e
  LD_LIBRARY_PATH="$GIT_LD_LIBRARY_PATH" git push
  local push_status=$?
  set -e
  if (( push_status != 0 )); then
    log_step "Git push failed for LLM judge summaries"
    if [[ "$GIT_PUSH_CONTINUE_ON_ERROR" == "1" ]]; then
      log_step "Continuing because GIT_PUSH_CONTINUE_ON_ERROR=$GIT_PUSH_CONTINUE_ON_ERROR"
      return 0
    fi
    return "$push_status"
  fi
  log_step "Git push complete for LLM judge summaries"
}

require_cmd python
require_cmd hf
require_cmd curl
require_cmd tee
require_cmd vllm
LD_LIBRARY_PATH="$GIT_LD_LIBRARY_PATH" git --version >/dev/null 2>&1 || {
  echo "Missing or broken git even with LD_LIBRARY_PATH=$GIT_LD_LIBRARY_PATH" >&2
  exit 1
}

download_required_data
build_inputs
run_judges_sequential_by_model
combine_summaries
upload_outputs
git_commit_outputs

echo "PolyAlign LLM judge evaluation completed."
