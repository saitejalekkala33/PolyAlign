#!/usr/bin/env bash
set -euo pipefail

REPO="${REPO:-/home/umair/TW/PolyAlign}"
cd "$REPO"

HF_DATASET_REPO="${HF_DATASET_REPO:-saiteja33/PolyAlign-All}"
RAW_ROOT="${RAW_ROOT:-$REPO/data/llm_judge/raw}"
WORK_DIR="${WORK_DIR:-$REPO/data/llm_judge/work}"
MODELS_ROOT="${MODELS_ROOT:-$REPO/models/llm-judges}"
LOG_ROOT="${LOG_ROOT:-$REPO/logs/llm-judge-selected-missing}"

RUBRIC_YAML="${RUBRIC_YAML:-$REPO/scripts/llm_judge/rubric.yaml}"
PROMPTS_PY="${PROMPTS_PY:-$REPO/scripts/llm_judge/llm-judge-prompts.py}"

QWEN_JUDGE_ID="${QWEN_JUDGE_ID:-qwen3_8b}"
QWEN_MODEL_ID="${QWEN_MODEL_ID:-Qwen/Qwen3-8B}"
QWEN_GPU_GROUPS="${QWEN_GPU_GROUPS:-0;1;2;3;4;5;6;7}"
QWEN_PORTS=(${QWEN_PORTS:-8100 8101 8102 8103 8104 8105 8106 8107})
QWEN_CHAT_TEMPLATE_KWARGS_JSON="${QWEN_CHAT_TEMPLATE_KWARGS_JSON:-{\"enable_thinking\":false}}"
QWEN_GENERATION_CONFIG="${QWEN_GENERATION_CONFIG:-vllm}"
QWEN_VLLM_EXTRA_ARGS="${QWEN_VLLM_EXTRA_ARGS:-}"

QWEN25_7B_JUDGE_ID="${QWEN25_7B_JUDGE_ID:-qwen25_7b_instruct}"
QWEN25_7B_MODEL_ID="${QWEN25_7B_MODEL_ID:-Qwen/Qwen2.5-7B-Instruct}"
QWEN25_7B_GPU_GROUPS="${QWEN25_7B_GPU_GROUPS:-0;1;2;3;4;5;6;7}"
QWEN25_7B_PORTS=(${QWEN25_7B_PORTS:-8100 8101 8102 8103 8104 8105 8106 8107})
QWEN25_7B_CHAT_TEMPLATE_KWARGS_JSON="${QWEN25_7B_CHAT_TEMPLATE_KWARGS_JSON:-{}}"
QWEN25_7B_GENERATION_CONFIG="${QWEN25_7B_GENERATION_CONFIG:-vllm}"
QWEN25_7B_VLLM_EXTRA_ARGS="${QWEN25_7B_VLLM_EXTRA_ARGS:-}"

JUDGES=(${JUDGES:-$QWEN_JUDGE_ID $QWEN25_7B_JUDGE_ID})

DOWNLOAD_DATA="${DOWNLOAD_DATA:-1}"
DOWNLOAD_JUDGE_MODELS="${DOWNLOAD_JUDGE_MODELS:-1}"
BUILD_INPUTS="${BUILD_INPUTS:-1}"
RUN_JUDGES="${RUN_JUDGES:-1}"
UPLOAD_TO_HF="${UPLOAD_TO_HF:-1}"
GIT_PULL_BEFORE_RUN="${GIT_PULL_BEFORE_RUN:-1}"
GIT_PUSH_SUMMARIES="${GIT_PUSH_SUMMARIES:-1}"
GIT_PUSH_CONTINUE_ON_ERROR="${GIT_PUSH_CONTINUE_ON_ERROR:-1}"
CLEANUP_JUDGE_MODEL_AFTER_RUN="${CLEANUP_JUDGE_MODEL_AFTER_RUN:-1}"
RESUME="${RESUME:-1}"

SELECTED_ZH_SOURCE_IDS="${SELECTED_ZH_SOURCE_IDS:-qwen25_1_5b__hdpo gemma2_2b__hdpo qwen25_3b__hdpo llama32_3b__hdpo}"
QWEN_SELECTED_EN_SOURCE_IDS="${QWEN_SELECTED_EN_SOURCE_IDS:-qwen25_3b__dpo llama32_3b__dpo llama32_3b__dist_sft qwen25_1_5b__hdpo qwen25_3b__hdpo gemma2_2b__hdpo}"
QWEN25_7B_SELECTED_EN_SOURCE_IDS="${QWEN25_7B_SELECTED_EN_SOURCE_IDS:-llama32_3b__dpo qwen25_1_5b__hdpo qwen25_3b__hdpo gemma2_2b__hdpo}"

SAMPLE_SIZE="${SAMPLE_SIZE:-0}"
SAMPLE_MODE="${SAMPLE_MODE:-first}"
SEED="${SEED:-42}"
BATCH_SIZE="${BATCH_SIZE:-4}"
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
MAX_NUM_SEQS="${MAX_NUM_SEQS:-16}"
VLLM_TRUST_REMOTE_CODE="${VLLM_TRUST_REMOTE_CODE:-1}"
VLLM_ENFORCE_EAGER="${VLLM_ENFORCE_EAGER:-0}"
VLLM_EXTRA_ARGS="${VLLM_EXTRA_ARGS:-}"

mkdir -p "$RAW_ROOT" "$WORK_DIR" "$MODELS_ROOT" "$LOG_ROOT"

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
    "$QWEN25_7B_JUDGE_ID") printf '%s\n' "$QWEN25_7B_MODEL_ID" ;;
    *) echo "Unknown judge id: $1" >&2; return 1 ;;
  esac
}

judge_gpu_groups() {
  case "$1" in
    "$QWEN_JUDGE_ID") printf '%s\n' "$QWEN_GPU_GROUPS" ;;
    "$QWEN25_7B_JUDGE_ID") printf '%s\n' "$QWEN25_7B_GPU_GROUPS" ;;
    *) echo "Unknown judge id: $1" >&2; return 1 ;;
  esac
}

judge_ports() {
  case "$1" in
    "$QWEN_JUDGE_ID") printf '%s\n' "${QWEN_PORTS[*]}" ;;
    "$QWEN25_7B_JUDGE_ID") printf '%s\n' "${QWEN25_7B_PORTS[*]}" ;;
    *) echo "Unknown judge id: $1" >&2; return 1 ;;
  esac
}

judge_chat_template_kwargs_json() {
  case "$1" in
    "$QWEN_JUDGE_ID") printf '%s\n' "$QWEN_CHAT_TEMPLATE_KWARGS_JSON" ;;
    "$QWEN25_7B_JUDGE_ID") printf '%s\n' "$QWEN25_7B_CHAT_TEMPLATE_KWARGS_JSON" ;;
    *) echo "Unknown judge id: $1" >&2; return 1 ;;
  esac
}

judge_generation_config() {
  case "$1" in
    "$QWEN_JUDGE_ID") printf '%s\n' "$QWEN_GENERATION_CONFIG" ;;
    "$QWEN25_7B_JUDGE_ID") printf '%s\n' "$QWEN25_7B_GENERATION_CONFIG" ;;
    *) echo "Unknown judge id: $1" >&2; return 1 ;;
  esac
}

judge_vllm_extra_args() {
  case "$1" in
    "$QWEN_JUDGE_ID") printf '%s\n' "$QWEN_VLLM_EXTRA_ARGS" ;;
    "$QWEN25_7B_JUDGE_ID") printf '%s\n' "$QWEN25_7B_VLLM_EXTRA_ARGS" ;;
    *) echo "Unknown judge id: $1" >&2; return 1 ;;
  esac
}

judge_en_source_ids() {
  case "$1" in
    "$QWEN_JUDGE_ID") printf '%s\n' "$QWEN_SELECTED_EN_SOURCE_IDS" ;;
    "$QWEN25_7B_JUDGE_ID") printf '%s\n' "$QWEN25_7B_SELECTED_EN_SOURCE_IDS" ;;
    *) echo "Unknown judge id: $1" >&2; return 1 ;;
  esac
}

judge_zh_source_ids() {
  case "$1" in
    "$QWEN_JUDGE_ID"|"$QWEN25_7B_JUDGE_ID") printf '%s\n' "$SELECTED_ZH_SOURCE_IDS" ;;
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
  if [[ -d "$model_id" ]]; then
    printf '%s\n' "$model_id"
    return 0
  fi
  local local_dir
  local_dir="$(local_model_dir "$judge_id")"
  if [[ -f "$local_dir/config.json" ]]; then
    printf '%s\n' "$local_dir"
  else
    printf '%s\n' "$model_id"
  fi
}

git_pull_latest() {
  if [[ "$GIT_PULL_BEFORE_RUN" != "1" ]]; then
    log_step "Skipping git pull because GIT_PULL_BEFORE_RUN=$GIT_PULL_BEFORE_RUN"
    return 0
  fi
  log_step "Pulling latest git state"
  git pull --ff-only
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

  log_step "Downloading selected missing LLM-judge dataset files into $RAW_ROOT"
  local downloads_file="$LOG_ROOT/required_hf_files_selected_missing.txt"
  run_python list-downloads --source-set selected_missing --lang all --include-human > "$downloads_file"
  while IFS= read -r rel_path; do
    [[ -n "$rel_path" ]] || continue
    download_dataset_file "$rel_path"
  done < "$downloads_file"
}

download_judge_model() {
  local judge_id="$1"
  local model_id
  model_id="$(judge_model_id "$judge_id")"
  if [[ -d "$model_id" ]]; then
    echo "Using existing local judge model: $judge_id -> $model_id"
    return 0
  fi
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
  log_step "Building selected missing LLM-judge inputs under $WORK_DIR"
  run_python build-inputs \
    --raw-root "$RAW_ROOT" \
    --output-root "$WORK_DIR" \
    --lang all \
    --source-set selected_missing \
    --merge-existing-manifest \
    2>&1 | tee "$LOG_ROOT/build-inputs-selected-missing.log"
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

start_vllm_server() {
  local judge_id="$1"
  local model_ref="$2"
  local gpus="$3"
  local port="$4"
  local server_label="$5"
  local tp_size
  tp_size="$(tensor_parallel_size "$gpus")"
  local base_url="http://127.0.0.1:$port"
  local server_log="$LOG_ROOT/${judge_id}.${server_label}.vllm.log"
  local chat_template_kwargs_json
  chat_template_kwargs_json="$(judge_chat_template_kwargs_json "$judge_id")"
  local generation_config
  generation_config="$(judge_generation_config "$judge_id")"
  local per_judge_vllm_extra_args
  per_judge_vllm_extra_args="$(judge_vllm_extra_args "$judge_id")"

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
  if [[ -n "$chat_template_kwargs_json" && "$chat_template_kwargs_json" != "{}" ]]; then
    serve_args+=(--default-chat-template-kwargs "$chat_template_kwargs_json")
  fi
  if [[ -n "$generation_config" ]]; then
    serve_args+=(--generation-config "$generation_config")
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
  if [[ "$VLLM_ENFORCE_EAGER" == "1" ]]; then
    serve_args+=(--enforce-eager)
  fi
  if [[ -n "$per_judge_vllm_extra_args" ]]; then
    # shellcheck disable=SC2206
    local judge_extra_args=($per_judge_vllm_extra_args)
    serve_args+=("${judge_extra_args[@]}")
  fi
  if [[ -n "$VLLM_EXTRA_ARGS" ]]; then
    # shellcheck disable=SC2206
    local extra_args=($VLLM_EXTRA_ARGS)
    serve_args+=("${extra_args[@]}")
  fi

  log_step "Starting vLLM judge=$judge_id GPUs=$gpus TP=$tp_size port=$port" >&2
  CUDA_VISIBLE_DEVICES="$gpus" vllm serve "${serve_args[@]}" >"$server_log" 2>&1 &
  local server_pid=$!
  if ! wait_for_vllm "$base_url" "$server_pid" "$server_log" >&2; then
    stop_vllm "$server_pid"
    return 1
  fi
  printf '%s\n' "$server_pid"
}

run_judge_call() {
  local judge_id="$1"
  local shard_index="$2"
  local num_shards="$3"
  local port="$4"
  local model_ref="$5"
  local lang="$6"
  local include_ids="$7"

  [[ -n "$include_ids" ]] || return 0

  local base_url="http://127.0.0.1:$port"
  local chat_template_kwargs_json
  chat_template_kwargs_json="$(judge_chat_template_kwargs_json "$judge_id")"
  local judge_log="$LOG_ROOT/${judge_id}.${lang}.shard${shard_index}.judge.log"
  local include_args=()
  local source_id
  for source_id in $include_ids; do
    include_args+=(--include-source-id "$source_id")
  done

  local resume_args=()
  if [[ "$RESUME" == "1" ]]; then
    resume_args+=(--resume)
  fi

  local guided_args=()
  if [[ "$GUIDED_JSON" != "1" ]]; then
    guided_args+=(--disable-guided-json)
  fi

  log_step "Running LLM judge: $judge_id lang=$lang shard $shard_index/$num_shards sources=[$include_ids]"
  set +e
  run_python judge \
    --work-dir "$WORK_DIR" \
    --lang "$lang" \
    --judge-id "$judge_id" \
    --judge-model-name "$judge_id" \
    --tokenizer-name-or-path "$model_ref" \
    --base-url "$base_url" \
    --rubric-yaml "$RUBRIC_YAML" \
    --prompts-py "$PROMPTS_PY" \
    "${include_args[@]}" \
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
    --chat-template-kwargs-json "$chat_template_kwargs_json" \
    --trust-remote-code \
    "${guided_args[@]}" \
    "${resume_args[@]}" \
    2>&1 | tee "$judge_log"
  local judge_status=${PIPESTATUS[0]}
  set -e

  if (( judge_status != 0 )); then
    echo "Judge failed: $judge_id lang=$lang shard $shard_index/$num_shards. Check $judge_log" >&2
    return "$judge_status"
  fi
}

run_one_judge_worker() {
  local judge_id="$1"
  local shard_index="$2"
  local num_shards="$3"
  local gpus="$4"
  local port="$5"
  local model_ref="$6"
  local server_pid=""
  local status=0

  server_pid="$(start_vllm_server "$judge_id" "$model_ref" "$gpus" "$port" "shard${shard_index}")"
  run_judge_call "$judge_id" "$shard_index" "$num_shards" "$port" "$model_ref" "en" "$(judge_en_source_ids "$judge_id")" || status=1
  run_judge_call "$judge_id" "$shard_index" "$num_shards" "$port" "$model_ref" "zh" "$(judge_zh_source_ids "$judge_id")" || status=1
  stop_vllm "$server_pid"
  return "$status"
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
    run_one_judge_worker \
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
    --merge-existing \
    2>&1 | tee "$LOG_ROOT/${judge_id}.summary.log"
}

combine_summaries() {
  log_step "Combining selected LLM-judge summaries"
  local judge_ids_csv
  judge_ids_csv="$(IFS=,; echo "${JUDGES[*]}")"
  run_python combine-summaries --work-dir "$WORK_DIR" --lang all --judge-id "$judge_ids_csv"
}

upload_outputs_for_judge() {
  local judge_id="$1"
  if [[ "$UPLOAD_TO_HF" != "1" ]]; then
    log_step "Skipping HF upload because UPLOAD_TO_HF=$UPLOAD_TO_HF"
    return 0
  fi

  local lang hf_prefix
  for lang in en zh; do
    [[ "$lang" == "en" ]] && hf_prefix="english" || hf_prefix="chinese"
    if [[ -d "$WORK_DIR/$lang/scores/$judge_id" ]]; then
      log_step "Uploading $lang $judge_id judge score files to HF"
      hf upload "$HF_DATASET_REPO" "$WORK_DIR/$lang/scores/$judge_id" \
        "$hf_prefix/llm-judge/scores/$judge_id" \
        --repo-type dataset \
        --commit-message "Upload selected $lang $judge_id LLM judge scores"
    fi
    if [[ -d "$WORK_DIR/$lang/metrics/llm_judge/$judge_id" ]]; then
      log_step "Uploading $lang $judge_id judge metric summaries to HF"
      hf upload "$HF_DATASET_REPO" "$WORK_DIR/$lang/metrics/llm_judge/$judge_id" \
        "$hf_prefix/llm-judge/metrics/$judge_id" \
        --repo-type dataset \
        --commit-message "Upload selected $lang $judge_id LLM judge metrics"
    fi
  done

  if [[ -d "$WORK_DIR/metrics" ]]; then
    log_step "Uploading combined selected LLM judge summaries to HF"
    hf upload "$HF_DATASET_REPO" "$WORK_DIR/metrics" \
      "llm-judge/metrics" \
      --repo-type dataset \
      --commit-message "Upload selected combined LLM judge metrics"
  fi
}

git_commit_outputs() {
  local judge_id="$1"
  if [[ "$GIT_PUSH_SUMMARIES" != "1" ]]; then
    log_step "Skipping git commit/push because GIT_PUSH_SUMMARIES=$GIT_PUSH_SUMMARIES"
    return 0
  fi

  log_step "Git add/commit/push selected LLM judge code and summaries for $judge_id"
  git add \
    scripts/llm_judge \
    "$WORK_DIR/input_summary.json" \
    "$WORK_DIR/en/input_summary.json" \
    "$WORK_DIR/en/manifest.json" \
    "$WORK_DIR/en/metrics" \
    "$WORK_DIR/zh/input_summary.json" \
    "$WORK_DIR/zh/manifest.json" \
    "$WORK_DIR/zh/metrics" \
    "$WORK_DIR/metrics"

  if git diff --cached --quiet; then
    echo "No selected LLM judge code or summary changes to commit."
    return 0
  fi

  git commit -m "Add selected PolyAlign LLM judge results for $judge_id"
  set +e
  git push
  local push_status=$?
  set -e
  if (( push_status != 0 )); then
    log_step "Git push failed for selected LLM judge summaries"
    if [[ "$GIT_PUSH_CONTINUE_ON_ERROR" == "1" ]]; then
      log_step "Continuing because GIT_PUSH_CONTINUE_ON_ERROR=$GIT_PUSH_CONTINUE_ON_ERROR"
      return 0
    fi
    return "$push_status"
  fi
  log_step "Git push complete for selected LLM judge summaries"
}

require_cmd python
require_cmd hf
require_cmd curl
require_cmd tee
require_cmd vllm
git --version >/dev/null 2>&1 || {
  echo "Missing or broken git" >&2
  exit 1
}

git_pull_latest
download_required_data
build_inputs

if [[ "$RUN_JUDGES" == "1" ]]; then
  for judge_id in "${JUDGES[@]}"; do
    log_step "Preparing selected missing judge model: $judge_id"
    download_judge_model "$judge_id"
    run_one_judge_model "$judge_id"
    combine_summaries
    upload_outputs_for_judge "$judge_id"
    git_commit_outputs "$judge_id"
    cleanup_judge_model "$judge_id"
  done
else
  log_step "Skipping judge runs because RUN_JUDGES=$RUN_JUDGES"
  combine_summaries
  upload_outputs_for_judge "combined"
  git_commit_outputs "combined"
fi

echo "Selected missing PolyAlign LLM judge evaluation completed."
