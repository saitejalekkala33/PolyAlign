#!/usr/bin/env bash
set -euo pipefail

REPO="${REPO:-/home/umair/TW/PolyAlign}"
cd "$REPO"

HF_DATASET_REPO="${HF_DATASET_REPO:-saiteja33/PolyAlign-All}"
RAW_ROOT="${RAW_ROOT:-$REPO/data/ai_detection/raw}"
WORK_DIR="${WORK_DIR:-$REPO/data/ai_detection/work}"
MODELS_ROOT="${MODELS_ROOT:-$REPO/models/ai-detectors}"
DETECTOR_ROOT="${DETECTOR_ROOT:-$REPO/vendor/ai-detectors}"
LOG_ROOT="${LOG_ROOT:-$REPO/logs/ai-detection}"

HUMAN_EN="${HUMAN_EN:-$REPO/vendor/LlamaFactory/data/test.json}"
HUMAN_ZH="${HUMAN_ZH:-$REPO/vendor/LlamaFactory/data/test-zh.json}"
DETECTORS=(${DETECTORS:-binoculars fast_detect_gpt ghostbuster radar detect_gpt})

GPUS=(${GPUS:-0 1 2})
NUM_SHARDS="${NUM_SHARDS:-${#GPUS[@]}}"

DOWNLOAD_DATA="${DOWNLOAD_DATA:-1}"
DOWNLOAD_MODELS="${DOWNLOAD_MODELS:-1}"
CLONE_DETECTORS="${CLONE_DETECTORS:-1}"
INSTALL_BINOCULARS="${INSTALL_BINOCULARS:-0}"
UPLOAD_TO_HF="${UPLOAD_TO_HF:-1}"
GIT_PUSH_SUMMARIES="${GIT_PUSH_SUMMARIES:-1}"
RESUME="${RESUME:-1}"
CLEANUP_DETECTOR_MODELS="${CLEANUP_DETECTOR_MODELS:-1}"
CHECK_COUNTS_ONLY="${CHECK_COUNTS_ONLY:-0}"
CHECK_COUNTS_BEFORE_RUN="${CHECK_COUNTS_BEFORE_RUN:-0}"

MAX_LENGTH="${MAX_LENGTH:-512}"
RADAR_BATCH_SIZE="${RADAR_BATCH_SIZE:-32}"
BINOCULARS_BATCH_SIZE="${BINOCULARS_BATCH_SIZE:-4}"
FAST_DETECT_GPT_BATCH_SIZE="${FAST_DETECT_GPT_BATCH_SIZE:-1}"
DETECT_GPT_BATCH_SIZE="${DETECT_GPT_BATCH_SIZE:-1}"
DETECTGPT_PERTURBATIONS="${DETECTGPT_PERTURBATIONS:-10}"

GIT_LD_LIBRARY_PATH="${GIT_LD_LIBRARY_PATH:-/usr/lib/x86_64-linux-gnu:/lib/x86_64-linux-gnu}"

mkdir -p "$RAW_ROOT" "$WORK_DIR" "$MODELS_ROOT" "$DETECTOR_ROOT" "$LOG_ROOT"

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

require_cmd python
require_cmd hf
require_cmd tee
require_cmd rm
LD_LIBRARY_PATH="$GIT_LD_LIBRARY_PATH" git --version >/dev/null 2>&1 || {
  echo "Missing or broken git even with LD_LIBRARY_PATH=$GIT_LD_LIBRARY_PATH" >&2
  exit 1
}

if (( NUM_SHARDS > ${#GPUS[@]} )); then
  echo "NUM_SHARDS=$NUM_SHARDS cannot exceed number of GPUS (${#GPUS[@]})." >&2
  exit 1
fi

run_python() {
  PYTHONPATH="$REPO/src:${PYTHONPATH:-}" python "$REPO/scripts/ai_detection/polyalign_ai_detection.py" "$@"
}

clone_detector_repo() {
  local url="$1"
  local dir_name="$2"
  local dest="$DETECTOR_ROOT/$dir_name"

  if [[ "$CLONE_DETECTORS" != "1" ]]; then
    return 0
  fi

  if [[ -d "$dest" ]]; then
    echo "Detector repo already present: $dest"
  else
    echo "Cloning detector repo: $url -> $dest"
    LD_LIBRARY_PATH="$GIT_LD_LIBRARY_PATH" git clone --depth 1 "$url" "$dest"
  fi

  rm -rf "$dest/.git"
}

prepare_detector_repos() {
  log_step "Preparing detector repos under $DETECTOR_ROOT"
  clone_detector_repo "https://github.com/ahans30/Binoculars.git" "Binoculars"
  clone_detector_repo "https://github.com/baoguangsheng/fast-detect-gpt.git" "fast-detect-gpt"
  clone_detector_repo "https://github.com/vivek3141/ghostbuster.git" "ghostbuster"
  clone_detector_repo "https://github.com/eric-mitchell/detect-gpt.git" "detect-gpt"
  clone_detector_repo "https://github.com/IBM/RADAR.git" "RADAR"

  if [[ "$INSTALL_BINOCULARS" == "1" && -d "$DETECTOR_ROOT/Binoculars" ]]; then
    pip install --no-deps -e "$DETECTOR_ROOT/Binoculars"
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

download_model() {
  local repo_id="$1"
  local local_dir="$2"
  shift 2
  local exclude_args=("$@")
  if [[ "$DOWNLOAD_MODELS" != "1" ]]; then
    return 0
  fi
  if [[ -f "$local_dir/config.json" ]]; then
    echo "Model already present: $repo_id -> $local_dir"
    return 0
  fi
  mkdir -p "$local_dir"
  echo "Downloading model: $repo_id -> $local_dir"
  hf download "$repo_id" --repo-type model --local-dir "$local_dir" "${exclude_args[@]}"
}

prepare_detector_models() {
  local detector="$1"

  case "$detector" in
    binoculars)
      log_step "Preparing Binoculars models"
      download_model "tiiuae/falcon-7b" "$MODELS_ROOT/binoculars/falcon-7b" \
        --exclude "coreml/*" --exclude "coreml/**" \
        --exclude "*.bin" --exclude "*.h5" --exclude "*.ot" --exclude "*.msgpack" \
        --exclude "flax_model.msgpack" --exclude "tf_model.h5" --exclude "rust_model.ot"
      download_model "tiiuae/falcon-7b-instruct" "$MODELS_ROOT/binoculars/falcon-7b-instruct" \
        --exclude "coreml/*" --exclude "coreml/**" \
        --exclude "*.bin" --exclude "*.h5" --exclude "*.ot" --exclude "*.msgpack" \
        --exclude "flax_model.msgpack" --exclude "tf_model.h5" --exclude "rust_model.ot"
      ;;
    fast_detect_gpt)
      log_step "Preparing Fast-DetectGPT models"
      download_model "meta-llama/Meta-Llama-3-8B" "$MODELS_ROOT/fast-detect-gpt/Meta-Llama-3-8B" \
        --exclude "original/*" --exclude "original/**" \
        --exclude "*.bin" --exclude "*.h5" --exclude "*.ot" --exclude "*.msgpack"
      download_model "meta-llama/Meta-Llama-3-8B-Instruct" "$MODELS_ROOT/fast-detect-gpt/Meta-Llama-3-8B-Instruct" \
        --exclude "original/*" --exclude "original/**" \
        --exclude "*.bin" --exclude "*.h5" --exclude "*.ot" --exclude "*.msgpack"
      ;;
    radar)
      log_step "Preparing RADAR model"
      download_model "TrustSafeAI/RADAR-Vicuna-7B" "$MODELS_ROOT/radar/RADAR-Vicuna-7B" \
        --exclude "*.bin" --exclude "*.h5" --exclude "*.ot" --exclude "*.msgpack"
      ;;
    detect_gpt)
      log_step "Preparing DetectGPT models"
      download_model "EleutherAI/gpt-neo-2.7B" "$MODELS_ROOT/detect-gpt/gpt-neo-2.7B" \
        --exclude "*.bin" --exclude "*.h5" --exclude "*.ot" --exclude "*.msgpack" \
        --exclude "flax_model.msgpack" --exclude "tf_model.h5" --exclude "rust_model.ot"
      download_model "google-t5/t5-large" "$MODELS_ROOT/detect-gpt/t5-mask" \
        --exclude "*.bin" --exclude "*.h5" --exclude "*.ot" --exclude "*.msgpack" \
        --exclude "flax_model.msgpack" --exclude "tf_model.h5" --exclude "rust_model.ot"
      ;;
    ghostbuster)
      log_step "Ghostbuster needs no local model download because it is skipped"
      ;;
    *)
      echo "Unknown detector for model prep: $detector" >&2
      exit 1
      ;;
  esac
}

cleanup_detector_models() {
  local detector="$1"
  if [[ "$CLEANUP_DETECTOR_MODELS" != "1" ]]; then
    return 0
  fi

  case "$detector" in
    binoculars|radar|detect_gpt)
      local model_dir="$MODELS_ROOT/$detector"
      if [[ -d "$model_dir" ]]; then
        echo "Removing detector models: $model_dir"
        rm -rf -- "$model_dir"
      fi
      ;;
    fast_detect_gpt)
      local model_dir="$MODELS_ROOT/fast-detect-gpt"
      if [[ -d "$model_dir" ]]; then
        echo "Removing detector models: $model_dir"
        rm -rf -- "$model_dir"
      fi
      ;;
    ghostbuster)
      ;;
  esac
}

check_row_counts() {
  log_step "Checking per-language row counts"
  run_python check-row-counts \
    --raw-root "$RAW_ROOT" \
    --lang all \
    --human-en "$HUMAN_EN" \
    --human-zh "$HUMAN_ZH" \
    --output-json "$WORK_DIR/row_count_summary.json" \
    2>&1 | tee "$LOG_ROOT/row-counts.log"
}

build_detector_inputs() {
  log_step "Building all-row detector inputs under $WORK_DIR"
  run_python build-inputs \
    --raw-root "$RAW_ROOT" \
    --output-root "$WORK_DIR" \
    --lang all \
    --human-en "$HUMAN_EN" \
    --human-zh "$HUMAN_ZH" \
    2>&1 | tee "$LOG_ROOT/build-inputs.log"
}

detector_batch_size() {
  case "$1" in
    radar) printf '%s\n' "$RADAR_BATCH_SIZE" ;;
    binoculars) printf '%s\n' "$BINOCULARS_BATCH_SIZE" ;;
    fast_detect_gpt) printf '%s\n' "$FAST_DETECT_GPT_BATCH_SIZE" ;;
    detect_gpt) printf '%s\n' "$DETECT_GPT_BATCH_SIZE" ;;
    *) printf '%s\n' "1" ;;
  esac
}

run_detector_language() {
  local detector="$1"
  local lang="$2"

  if [[ "$detector" == "ghostbuster" ]]; then
    log_step "Skipping Ghostbuster for $lang"
    run_python mark-skipped --work-dir "$WORK_DIR" --lang "$lang" --detector "$detector" \
      --reason "Ghostbuster standalone inference requires OpenAI API token logprobs; skipped for offline/local run."
    return 0
  fi

  local batch_size
  batch_size="$(detector_batch_size "$detector")"

  log_step "Running detector=$detector language=$lang on GPUs ${GPUS[*]}"

  pids=()
  local resume_args=()
  if [[ "$RESUME" == "1" ]]; then
    resume_args+=(--resume)
  fi
  for shard_index in $(seq 0 $((NUM_SHARDS - 1))); do
    local gpu="${GPUS[$shard_index]}"
    local log_file="$LOG_ROOT/${lang}-${detector}-gpu${gpu}.log"
    CUDA_VISIBLE_DEVICES="$gpu" run_python score-lang \
      --work-dir "$WORK_DIR" \
      --lang "$lang" \
      --detector "$detector" \
      --models-root "$MODELS_ROOT" \
      --detector-root "$DETECTOR_ROOT" \
      --device cuda:0 \
      --batch-size "$batch_size" \
      --max-length "$MAX_LENGTH" \
      --shard-index "$shard_index" \
      --num-shards "$NUM_SHARDS" \
      --detectgpt-perturbations "$DETECTGPT_PERTURBATIONS" \
      "${resume_args[@]}" \
      > "$log_file" 2>&1 &
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
    echo "Detector failed: $detector / $lang. Check $LOG_ROOT/${lang}-${detector}-gpu*.log" >&2
    exit 1
  fi

  run_python summarize --work-dir "$WORK_DIR" --lang "$lang" --detector "$detector" \
    2>&1 | tee "$LOG_ROOT/${lang}-${detector}-summary.log"
  log_step "Finished detector=$detector language=$lang"
}

upload_detector_outputs() {
  local detector="$1"
  local lang="$2"
  local hf_prefix
  case "$lang" in
    en) hf_prefix="english" ;;
    zh) hf_prefix="chinese" ;;
    *) echo "Unknown lang: $lang" >&2; exit 1 ;;
  esac

  if [[ "$UPLOAD_TO_HF" != "1" ]]; then
    log_step "Skipping HF upload for $lang/$detector because UPLOAD_TO_HF=$UPLOAD_TO_HF"
    return 0
  fi

  if [[ -d "$WORK_DIR/$lang/scores/$detector" ]]; then
    log_step "Uploading score files for $lang/$detector to HF"
    hf upload "$HF_DATASET_REPO" "$WORK_DIR/$lang/scores/$detector" \
      "$hf_prefix/ai-analysis/scores/$detector" \
      --repo-type dataset \
      --commit-message "Upload $lang $detector AI detector scores"
  fi

  if [[ -d "$WORK_DIR/$lang/metrics/$detector" ]]; then
    log_step "Uploading metric files for $lang/$detector to HF"
    hf upload "$HF_DATASET_REPO" "$WORK_DIR/$lang/metrics/$detector" \
      "$hf_prefix/ai-analysis/metrics/$detector" \
      --repo-type dataset \
      --commit-message "Upload $lang $detector AI detector metrics"
  fi
}

upload_input_summaries() {
  if [[ "$UPLOAD_TO_HF" != "1" ]]; then
    log_step "Skipping input summary HF upload because UPLOAD_TO_HF=$UPLOAD_TO_HF"
    return 0
  fi
  for lang in zh en; do
    local hf_prefix
    [[ "$lang" == "zh" ]] && hf_prefix="chinese" || hf_prefix="english"
    log_step "Uploading input summary outputs for $lang to HF"
    hf upload "$HF_DATASET_REPO" "$WORK_DIR/$lang/input_summary.json" \
      "$hf_prefix/ai-analysis/inputs/input_summary.json" \
      --repo-type dataset \
      --commit-message "Upload $lang AI detector input summary"
    hf upload "$HF_DATASET_REPO" "$WORK_DIR/$lang/manifest.json" \
      "$hf_prefix/ai-analysis/inputs/manifest.json" \
      --repo-type dataset \
      --commit-message "Upload $lang AI detector input manifest"
  done
}

git_commit_paths() {
  local message="$1"
  shift

  if [[ "$GIT_PUSH_SUMMARIES" != "1" ]]; then
    log_step "Skipping git commit/push for $message because GIT_PUSH_SUMMARIES=$GIT_PUSH_SUMMARIES"
    return 0
  fi

  log_step "Git add/commit/push: $message"
  LD_LIBRARY_PATH="$GIT_LD_LIBRARY_PATH" git add "$@"

  if LD_LIBRARY_PATH="$GIT_LD_LIBRARY_PATH" git diff --cached --quiet; then
    echo "No summary/metrics changes to commit for: $message"
    return 0
  fi

  LD_LIBRARY_PATH="$GIT_LD_LIBRARY_PATH" git commit -m "$message"
  LD_LIBRARY_PATH="$GIT_LD_LIBRARY_PATH" git push
  log_step "Git push complete: $message"
}

git_commit_input_summaries() {
  git_commit_paths "Add AI detector input summaries" \
    "$WORK_DIR/input_summary.json" \
    "$WORK_DIR/en/input_summary.json" \
    "$WORK_DIR/en/manifest.json" \
    "$WORK_DIR/zh/input_summary.json" \
    "$WORK_DIR/zh/manifest.json"
}

git_commit_detector_summary() {
  local detector="$1"
  local lang="$2"
  git_commit_paths "Add $lang $detector AI detector metrics" \
    "$WORK_DIR/$lang/metrics/$detector"
}

git_commit_combined_summary() {
  git_commit_paths "Add combined AI detector metrics summary" \
    "$WORK_DIR/metrics"
}

download_required_data

if [[ "$CHECK_COUNTS_ONLY" == "1" ]]; then
  check_row_counts
  exit 0
fi

prepare_detector_repos
if [[ "$CHECK_COUNTS_BEFORE_RUN" == "1" ]]; then
  check_row_counts || log_step "Row-count check reported mismatch/missing files; continuing because CHECK_COUNTS_BEFORE_RUN=$CHECK_COUNTS_BEFORE_RUN"
fi
build_detector_inputs
upload_input_summaries
git_commit_input_summaries

for detector in "${DETECTORS[@]}"; do
  prepare_detector_models "$detector"
  for lang in zh en; do
    run_detector_language "$detector" "$lang"
    upload_detector_outputs "$detector" "$lang"
    git_commit_detector_summary "$detector" "$lang"
  done
  cleanup_detector_models "$detector"
done

run_python combine-summaries --work-dir "$WORK_DIR"
git_commit_combined_summary

echo "AI detector analysis completed."
