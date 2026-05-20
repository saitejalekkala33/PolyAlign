#!/usr/bin/env bash
set -euo pipefail

# Run only the DPO metrics stage for completed prediction runs.
# This intentionally skips qwen25-3b-dpo-en, so it can keep running elsewhere.

ROOT_DIR="${ROOT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "$ROOT_DIR"

HF_DATASET_REPO="${HF_DATASET_REPO:-saiteja33/PolyAlign-All}"
DATA_ROOT="${DATA_ROOT:-$ROOT_DIR/data}"
METRICS_DIR="${METRICS_DIR:-$DATA_ROOT/metrics}"
LOG_ROOT="${LOG_ROOT:-$ROOT_DIR/logs/dpo-metrics-only}"

METRICS_GPUS=(${METRICS_GPUS:-${METRICS_GPU:-3}})
METRICS_DEVICE="${METRICS_DEVICE:-cuda}"
METRICS_DTYPE="${METRICS_DTYPE:-auto}"
METRICS_MAX_SEQ_LENGTH="${METRICS_MAX_SEQ_LENGTH:-4096}"
MAUVE_DEVICE_ID="${MAUVE_DEVICE_ID:-0}"
SKIP_MAUVE="${SKIP_MAUVE:-0}"

DOWNLOAD_DATA="${DOWNLOAD_DATA:-1}"
UPLOAD_TO_HF="${UPLOAD_TO_HF:-1}"
UPLOAD_METRIC_ARTIFACTS_TO_HF="${UPLOAD_METRIC_ARTIFACTS_TO_HF:-0}"
HF_UPLOAD_CONTINUE_ON_ERROR="${HF_UPLOAD_CONTINUE_ON_ERROR:-1}"
GIT_PUSH_OUTPUTS="${GIT_PUSH_OUTPUTS:-1}"
GIT_ADD_METRIC_ARTIFACTS="${GIT_ADD_METRIC_ARTIFACTS:-0}"
ALLOW_REFERENCE_FALLBACK="${ALLOW_REFERENCE_FALLBACK:-1}"
SKIP_INCOMPLETE_PREDICTIONS="${SKIP_INCOMPLETE_PREDICTIONS:-1}"
FORCE_METRICS="${FORCE_METRICS:-0}"
PUBLISH_EXISTING_METRICS="${PUBLISH_EXISTING_METRICS:-1}"
CONTINUE_ON_ERROR="${CONTINUE_ON_ERROR:-1}"
HF_HUB_DISABLE_XET="${HF_HUB_DISABLE_XET:-1}"
HF_XET_CACHE="${HF_XET_CACHE:-$LOG_ROOT/hf-xet-cache}"
export HF_HUB_DISABLE_XET HF_XET_CACHE

GIT_LD_LIBRARY_PATH="${GIT_LD_LIBRARY_PATH:-/usr/lib/x86_64-linux-gnu:/lib/x86_64-linux-gnu}"
QWEN15_METRIC_ALIAS="${QWEN15_METRIC_ALIAS:-qwen25_3b}"
QWEN15_REFERENCE_ALIAS="${QWEN15_REFERENCE_ALIAS:-$QWEN15_METRIC_ALIAS}"
PUBLISH_LOCK_FILE="$LOG_ROOT/publish.lock"
QUEUE_FILE="$LOG_ROOT/metrics.queue"
QUEUE_LOCK_FILE="$LOG_ROOT/metrics.queue.lock"
RESULTS_DIR="$LOG_ROOT/results"

mkdir -p "$DATA_ROOT" "$METRICS_DIR" "$LOG_ROOT" "$RESULTS_DIR" "$HF_XET_CACHE"

log_step() {
  printf '[%(%Y-%m-%d %H:%M:%S)T] %s\n' -1 "$*"
}

require_cmd() {
  command -v "$1" >/dev/null 2>&1 || {
    echo "Missing required command: $1" >&2
    exit 1
  }
}

git_cmd() {
  LD_LIBRARY_PATH="$GIT_LD_LIBRARY_PATH" git "$@"
}

hf_download_dataset_file() {
  local remote_path="$1"

  if [[ -f "$DATA_ROOT/$remote_path" ]]; then
    log_step "Dataset file already present: $remote_path"
    return 0
  fi

  log_step "Downloading dataset file: $remote_path"
  hf download "$HF_DATASET_REPO" "$remote_path" --repo-type dataset --local-dir "$DATA_ROOT"
}

download_required_data() {
  if [[ "$DOWNLOAD_DATA" != "1" ]]; then
    log_step "Skipping dataset download because DOWNLOAD_DATA=$DOWNLOAD_DATA"
    return 0
  fi

  log_step "Downloading required metrics inputs from $HF_DATASET_REPO"
  local remote_path
  while IFS= read -r remote_path; do
    [[ -n "$remote_path" ]] || continue
    hf_download_dataset_file "$remote_path"
  done <<'FILES'
chinese/merged_sft_dedup/llamafactory/test.json
chinese/merged_sft_dedup/current/test.jsonl
english/merged_sft_dedup/llamafactory/test.json
english/merged_sft_dedup/current/test.jsonl
chinese/reference_artifacts/gemma2_2b/bucket_references.json
chinese/reference_artifacts/llama32_3b/bucket_references.json
chinese/reference_artifacts/qwen25_3b/bucket_references.json
english/reference_artifacts/qwen25_1_5b/bucket_references.json
english/reference_artifacts/qwen25_3b/bucket_references.json
chinese/features/research_models/test/gemma_2_2b/test_answer_features_dedup-gemma.jsonl
chinese/features/research_models/test/llama32_3b/test_answer_features_dedup-llama32_3b.jsonl
chinese/features/research_models/test/qwen25_3b/test_answer_features_dedup-qwen25-3b.jsonl
english/features/research_models/test/gemma_2_2b/test_answer_features_dedup-gemma.jsonl
english/features/research_models/test/qwen25_1_5b/test_answer_features_dedup.jsonl
english/features/research_models/test/qwen25_3b/test_answer_features_dedup.jsonl
FILES
}

verify_required_data_files() {
  local missing=0
  local path
  for path in \
    "$DATA_ROOT/chinese/merged_sft_dedup/llamafactory/test.json" \
    "$DATA_ROOT/chinese/merged_sft_dedup/current/test.jsonl" \
    "$DATA_ROOT/english/merged_sft_dedup/llamafactory/test.json" \
    "$DATA_ROOT/english/merged_sft_dedup/current/test.jsonl" \
    "$DATA_ROOT/chinese/reference_artifacts/gemma2_2b/bucket_references.json" \
    "$DATA_ROOT/chinese/reference_artifacts/llama32_3b/bucket_references.json" \
    "$DATA_ROOT/chinese/reference_artifacts/qwen25_3b/bucket_references.json" \
    "$DATA_ROOT/english/reference_artifacts/qwen25_1_5b/bucket_references.json" \
    "$DATA_ROOT/english/reference_artifacts/qwen25_3b/bucket_references.json"; do
    if [[ ! -f "$path" ]]; then
      echo "Missing required metrics input: $path" >&2
      missing=1
    fi
  done

  if (( missing != 0 )); then
    echo "Required metrics inputs are missing after download." >&2
    return 1
  fi
}

task_lang_dir() {
  case "$1" in
    zh) printf '%s\n' "chinese" ;;
    en) printf '%s\n' "english" ;;
    *) echo "Unknown language key: $1" >&2; return 1 ;;
  esac
}

load_metric_tasks() {
  cat <<TASKS
zh|qwen25-15b-dpo-zh|qwen25_1_5b|qwen25_1_5b
zh|gemma2-2b-dpo-zh|gemma2_2b|gemma2_2b
zh|qwen25-3b-dpo-zh|qwen25_3b|qwen25_3b
zh|llama32-3b-dpo-zh|llama32_3b|llama32_3b
en|qwen25-1-5b-dpo-en|qwen25_1_5b|qwen25_1_5b
en|gemma2-2b-dpo-en|gemma2_2b|gemma2_2b
TASKS
}

is_completed() {
  local progress_path="$1"
  [[ -f "$progress_path" ]] && grep -q '"status": "completed"' "$progress_path"
}

resolve_bucket_references_path() {
  local lang_dir="$1"
  local reference_alias="$2"
  local metric_alias="$3"

  local primary="$DATA_ROOT/$lang_dir/reference_artifacts/$reference_alias/bucket_references.json"
  if [[ -f "$primary" ]]; then
    printf '%s\n' "$primary"
    return 0
  fi

  if [[ "$ALLOW_REFERENCE_FALLBACK" == "1" ]]; then
    local fallback="$DATA_ROOT/$lang_dir/reference_artifacts/qwen25_3b/bucket_references.json"
    if [[ -f "$fallback" ]]; then
      echo "Missing $primary; using qwen25_3b bucket references for $metric_alias." >&2
      printf '%s\n' "$fallback"
      return 0
    fi
  fi

  echo "Missing bucket references: $primary" >&2
  return 1
}

resolve_human_feature_path() {
  local lang_dir="$1"
  local metric_alias="$2"
  local alt_alias="$metric_alias"

  if [[ "$metric_alias" == "gemma2_2b" ]]; then
    alt_alias="gemma_2_2b"
  fi

  local candidates=(
    "$DATA_ROOT/$lang_dir/features/research_models/test/$metric_alias/test_answer_features_dedup.jsonl"
    "$DATA_ROOT/$lang_dir/features/research_models/test/$metric_alias/test_answer_features_dedup.csv"
    "$DATA_ROOT/$lang_dir/features/research_models/test/$alt_alias/test_answer_features_dedup.jsonl"
    "$DATA_ROOT/$lang_dir/features/research_models/test/$alt_alias/test_answer_features_dedup.csv"
    "$DATA_ROOT/$lang_dir/features/research_models/test/$alt_alias/test_answer_features_dedup-gemma.jsonl"
    "$DATA_ROOT/$lang_dir/features/research_models/test/$alt_alias/test_answer_features_dedup-gemma.csv"
  )

  local candidate
  for candidate in "${candidates[@]}"; do
    if [[ -f "$candidate" ]]; then
      printf '%s\n' "$candidate"
      return 0
    fi
  done

  local search_dir
  for search_dir in \
    "$DATA_ROOT/$lang_dir/features/research_models/test/$metric_alias" \
    "$DATA_ROOT/$lang_dir/features/research_models/test/$alt_alias"; do
    if [[ -d "$search_dir" ]]; then
      local discovered
      discovered="$(
        find "$search_dir" -maxdepth 1 -type f \
          \( -name 'test_answer_features_dedup*.jsonl' -o -name 'test_answer_features_dedup*.csv' \) \
          | sort \
          | head -n 1
      )"
      if [[ -n "$discovered" ]]; then
        printf '%s\n' "$discovered"
        return 0
      fi
    fi
  done

  return 1
}

publish_metric_outputs() {
  local run_name="$1"
  local output_json="$2"
  local work_dir="$3"
  local upload_log="$4"

  (
    flock 200

    if [[ "$UPLOAD_TO_HF" == "1" ]]; then
      log_step "Uploading $run_name metrics JSON to HF"
      set +e
      hf upload "$HF_DATASET_REPO" "$output_json" "metrics/${run_name}_eval.json" \
        --repo-type dataset \
        --commit-message "Upload $run_name DPO metrics" \
        2>&1 | tee -a "$upload_log"
      local hf_json_status=${PIPESTATUS[0]}
      set -e
      if (( hf_json_status != 0 )); then
        log_step "HF metrics JSON upload failed for $run_name with status $hf_json_status"
        if [[ "$HF_UPLOAD_CONTINUE_ON_ERROR" != "1" ]]; then
          exit "$hf_json_status"
        fi
      fi

      if [[ "$UPLOAD_METRIC_ARTIFACTS_TO_HF" == "1" ]]; then
        log_step "Uploading $run_name metric artifacts to HF"
        set +e
        hf upload "$HF_DATASET_REPO" "$work_dir" "metrics/${run_name}_eval_artifacts" \
          --repo-type dataset \
          --commit-message "Upload $run_name DPO metric artifacts" \
          2>&1 | tee -a "$upload_log"
        local hf_artifacts_status=${PIPESTATUS[0]}
        set -e
        if (( hf_artifacts_status != 0 )); then
          log_step "HF metric artifacts upload failed for $run_name with status $hf_artifacts_status"
          if [[ "$HF_UPLOAD_CONTINUE_ON_ERROR" != "1" ]]; then
            exit "$hf_artifacts_status"
          fi
        fi
      else
        log_step "Skipping HF artifact directory upload for $run_name because UPLOAD_METRIC_ARTIFACTS_TO_HF=$UPLOAD_METRIC_ARTIFACTS_TO_HF"
      fi
    fi

    if [[ "$GIT_PUSH_OUTPUTS" == "1" ]]; then
      if ! git_cmd rev-parse --is-inside-work-tree >/dev/null 2>&1; then
        log_step "Not inside a git work tree; skipping git push for $run_name"
        exit 0
      fi

      local git_paths=("$output_json")
      if [[ "$GIT_ADD_METRIC_ARTIFACTS" == "1" ]]; then
        git_paths+=("$work_dir")
      fi

      git_cmd add "${git_paths[@]}"
      if git_cmd diff --cached --quiet; then
        log_step "No git changes to commit for $run_name metrics"
        exit 0
      fi

      log_step "Committing and pushing $run_name metrics"
      git_cmd commit -m "Upload $run_name DPO metrics"
      git_cmd push
    fi
  ) 200>"$PUBLISH_LOCK_FILE"
}

run_one_metric_task() {
  local task_line="$1"
  local gpu="$2"
  local lang run_name metric_alias reference_alias
  IFS='|' read -r lang run_name metric_alias reference_alias <<<"$task_line"

  local lang_dir
  lang_dir="$(task_lang_dir "$lang")"

  local test_lf_path="$DATA_ROOT/$lang_dir/merged_sft_dedup/llamafactory/test.json"
  local current_test_path="$DATA_ROOT/$lang_dir/merged_sft_dedup/current/test.jsonl"
  local run_dir="$DATA_ROOT/$lang_dir/merged_sft_dedup/runs/$run_name"
  local predictions_path="$run_dir/predictions.jsonl"
  local progress_path="$run_dir/progress.json"
  local output_json="$METRICS_DIR/${run_name}_eval.json"
  local work_dir="$METRICS_DIR/${run_name}_eval_artifacts"
  local metric_log="$work_dir/metrics.log"
  local upload_log="$work_dir/upload.log"

  mkdir -p "$work_dir"

  if [[ ! -f "$predictions_path" ]]; then
    log_step "Skipping $run_name: missing predictions at $predictions_path"
    return 2
  fi

  if [[ -f "$progress_path" ]] && ! is_completed "$progress_path"; then
    if [[ "$SKIP_INCOMPLETE_PREDICTIONS" == "1" ]]; then
      log_step "Skipping $run_name: progress.json is not completed"
      return 2
    fi
    log_step "Running $run_name despite incomplete progress.json because SKIP_INCOMPLETE_PREDICTIONS=$SKIP_INCOMPLETE_PREDICTIONS"
  fi

  if [[ -f "$output_json" && "$FORCE_METRICS" != "1" ]]; then
    log_step "Metrics already exist for $run_name: $output_json"
    if [[ "$PUBLISH_EXISTING_METRICS" == "1" ]]; then
      publish_metric_outputs "$run_name" "$output_json" "$work_dir" "$upload_log"
    fi
    return 0
  fi

  local bucket_references_path
  bucket_references_path="$(resolve_bucket_references_path "$lang_dir" "$reference_alias" "$metric_alias")"

  local args=(
    -m metrics
    --test-lf-path "$test_lf_path"
    --predictions-path "$predictions_path"
    --output-json "$output_json"
    --current-test-path "$current_test_path"
    --bucket-references-path "$bucket_references_path"
    --work-dir "$work_dir"
    --model-alias "$metric_alias"
    --device "$METRICS_DEVICE"
    --dtype "$METRICS_DTYPE"
    --max-seq-length "$METRICS_MAX_SEQ_LENGTH"
    --mauve-device-id "$MAUVE_DEVICE_ID"
  )

  local human_feature_path=""
  if human_feature_path="$(resolve_human_feature_path "$lang_dir" "$metric_alias")"; then
    args+=(--human-feature-path "$human_feature_path")
  fi

  if [[ "$SKIP_MAUVE" == "1" ]]; then
    args+=(--skip-mauve)
  fi

  log_step "Running metrics for $run_name on CUDA_VISIBLE_DEVICES=$gpu"
  CUDA_VISIBLE_DEVICES="$gpu" PYTHONPATH="$ROOT_DIR/src:${PYTHONPATH:-}" \
    python "${args[@]}" 2>&1 | tee "$metric_log"

  publish_metric_outputs "$run_name" "$output_json" "$work_dir" "$upload_log"
  log_step "Finished metrics for $run_name"
}

prepare_metric_queue() {
  : > "$QUEUE_FILE"
  rm -f "$RESULTS_DIR"/task_*.status

  local index
  for index in "${!METRIC_TASKS[@]}"; do
    printf '%s|%s\n' "$index" "${METRIC_TASKS[$index]}" >> "$QUEUE_FILE"
  done
}

pop_next_metric_task() {
  local entry=""
  local result_file
  result_file="$(mktemp "$LOG_ROOT/metrics.queue.pop.XXXXXX")"

  (
    flock 200
    if [[ -s "$QUEUE_FILE" ]]; then
      IFS= read -r entry < "$QUEUE_FILE" || true
      tail -n +2 "$QUEUE_FILE" > "$QUEUE_FILE.tmp"
      mv "$QUEUE_FILE.tmp" "$QUEUE_FILE"
      printf '%s\n' "$entry" > "$result_file"
    else
      : > "$result_file"
    fi
  ) 200>"$QUEUE_LOCK_FILE"

  entry="$(cat "$result_file")"
  rm -f "$result_file"
  [[ -n "$entry" ]] && printf '%s\n' "$entry"
}

run_metric_worker() {
  local worker_index="$1"
  local gpu="$2"
  local worker_log="$LOG_ROOT/worker-gpu${gpu}.log"
  local entry task_index task task_status

  : > "$worker_log"
  log_step "Worker $worker_index starting on GPU $gpu"

  while entry="$(pop_next_metric_task)"; do
    [[ -n "$entry" ]] || break
    task_index="${entry%%|*}"
    task="${entry#*|}"

    log_step "GPU $gpu picked metrics task: $task"
    set +e
    run_one_metric_task "$task" "$gpu" 2>&1 | tee -a "$worker_log"
    task_status=${PIPESTATUS[0]}
    set -e

    printf '%s|%s\n' "$task_status" "$task" > "$RESULTS_DIR/task_${task_index}.status"

    if (( task_status != 0 && task_status != 2 )) && [[ "$CONTINUE_ON_ERROR" != "1" ]]; then
      log_step "Worker $worker_index stopping after failure on GPU $gpu"
      return "$task_status"
    fi
  done

  log_step "Worker $worker_index finished on GPU $gpu"
}

require_cmd python
require_cmd find
require_cmd flock
require_cmd mktemp
require_cmd tail
if [[ "$DOWNLOAD_DATA" == "1" || "$UPLOAD_TO_HF" == "1" ]]; then
  require_cmd hf
fi
if [[ "$GIT_PUSH_OUTPUTS" == "1" ]]; then
  require_cmd git
fi

mapfile -t METRIC_TASKS < <(load_metric_tasks)

if (( ${#METRICS_GPUS[@]} == 0 )); then
  echo "No metrics GPUs configured. Set METRICS_GPUS='3 4 5'." >&2
  exit 1
fi

download_required_data
verify_required_data_files

log_step "Metrics-only tasks:"
printf '  %s\n' "${METRIC_TASKS[@]}"
log_step "Explicitly skipped: en|qwen25-3b-dpo-en"
log_step "Metrics GPU workers: ${METRICS_GPUS[*]}"

completed=0
skipped=0
failed=0
failed_runs=()

prepare_metric_queue

worker_pids=()
for worker_index in "${!METRICS_GPUS[@]}"; do
  run_metric_worker "$worker_index" "${METRICS_GPUS[$worker_index]}" &
  worker_pids+=("$!")
done

worker_failed=0
for pid in "${worker_pids[@]}"; do
  set +e
  wait "$pid"
  worker_status=$?
  set -e
  if (( worker_status != 0 )); then
    worker_failed=1
  fi
done

for task_index in "${!METRIC_TASKS[@]}"; do
  status_file="$RESULTS_DIR/task_${task_index}.status"
  task="${METRIC_TASKS[$task_index]}"
  if [[ ! -f "$status_file" ]]; then
    failed=$((failed + 1))
    failed_runs+=("$task")
    continue
  fi

  IFS='|' read -r status recorded_task < "$status_file"
  if (( status == 0 )); then
    completed=$((completed + 1))
  elif (( status == 2 )); then
    skipped=$((skipped + 1))
  else
    failed=$((failed + 1))
    failed_runs+=("$recorded_task")
  fi
done

log_step "Metrics-only run complete: completed=$completed skipped=$skipped failed=$failed"
if (( failed > 0 || worker_failed != 0 )); then
  printf 'Failed tasks:\n' >&2
  printf '  %s\n' "${failed_runs[@]}" >&2
  exit 1
fi
