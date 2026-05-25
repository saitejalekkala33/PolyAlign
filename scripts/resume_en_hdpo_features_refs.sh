#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/home/umair/TW/PolyAlign}"
DATA_ROOT="${DATA_ROOT:-$ROOT/data/english/hdpo}"
CURRENT_HDPO_ROOT="${CURRENT_HDPO_ROOT:-$DATA_ROOT/current-hdpo-en}"
FEATURE_ROOT="${FEATURE_ROOT:-$DATA_ROOT/features-hdpo}"
REF_ROOT="${REF_ROOT:-$DATA_ROOT/reference_artifacts-hdpo}"

GPU_LIST="${GPU_LIST:-0 1 2 3 4 5 6 7}"
read -r -a GPUS <<< "$GPU_LIST"

MODEL_DIR_LIST="${MODEL_DIRS:-gemma2-2b qwen25-1-5b qwen25-3b llama32_3b llama32-3b}"
read -r -a MODEL_DIR_CANDIDATES <<< "$MODEL_DIR_LIST"

SPLIT_LIST="${SPLITS:-train val test}"
read -r -a SPLITS_ARR <<< "$SPLIT_LIST"

LM_DTYPE="${LM_DTYPE:-auto}"
VALIDATE_IDS="${VALIDATE_IDS:-1}"
TRUST_REMOTE_CODE="${TRUST_REMOTE_CODE:-0}"

RUN_ID="$(date +%Y%m%d_%H%M%S)"
TMP_ROOT="${TMP_ROOT:-$ROOT/tmp/resume_en_hdpo_features_refs_$RUN_ID}"
LOG_DIR="${LOG_DIR:-$ROOT/logs/resume_en_hdpo_features_refs_$RUN_ID}"
mkdir -p "$TMP_ROOT" "$LOG_DIR"

declare -A LM_MODEL
LM_MODEL[gemma2-2b]=gemma2_2b
LM_MODEL[qwen25-1-5b]=qwen25_1_5b
LM_MODEL[qwen25-3b]=qwen25_3b
LM_MODEL[llama32_3b]=llama32_3b
LM_MODEL[llama32-3b]=llama32_3b

declare -A MAX_SEQ
MAX_SEQ[gemma2_2b]=8192
MAX_SEQ[qwen25_1_5b]=4096
MAX_SEQ[qwen25_3b]=8192
MAX_SEQ[llama32_3b]=4096

TRUST_ARGS=()
if [[ "$TRUST_REMOTE_CODE" == "1" ]]; then
  TRUST_ARGS+=(--lm-trust-remote-code)
fi

cd "$ROOT"

count_lines() {
  local path="$1"
  if [[ -f "$path" ]]; then
    wc -l < "$path"
  else
    printf '0\n'
  fi
}

validate_prefix_alignment() {
  local current_path="$1"
  local feature_path="$2"
  local done_rows="$3"
  local label="$4"

  if [[ "$VALIDATE_IDS" != "1" || "$done_rows" == "0" ]]; then
    return 0
  fi

  python - "$current_path" "$feature_path" "$done_rows" "$label" <<'PY'
import json
import sys
from pathlib import Path

current_path = Path(sys.argv[1])
feature_path = Path(sys.argv[2])
done_rows = int(sys.argv[3])
label = sys.argv[4]

with current_path.open("r", encoding="utf-8") as current_handle, feature_path.open("r", encoding="utf-8") as feature_handle:
    for row_index in range(1, done_rows + 1):
        try:
            current = json.loads(current_handle.readline())
            feature = json.loads(feature_handle.readline())
        except json.JSONDecodeError as exc:
            raise SystemExit(f"{label}: invalid JSON before resume at row {row_index}: {exc}") from exc
        if current.get("id") != feature.get("id"):
            raise SystemExit(
                f"{label}: id mismatch at row {row_index}: current={current.get('id')} feature={feature.get('id')}"
            )

print(f"{label}: existing {done_rows} rows are JSON-valid and id-aligned.")
PY
}

validate_full_alignment() {
  local current_path="$1"
  local feature_path="$2"
  local label="$3"

  if [[ "$VALIDATE_IDS" != "1" ]]; then
    return 0
  fi

  python - "$current_path" "$feature_path" "$label" <<'PY'
import json
import sys
from pathlib import Path

current_path = Path(sys.argv[1])
feature_path = Path(sys.argv[2])
label = sys.argv[3]

current_count = 0
feature_count = 0

with current_path.open("r", encoding="utf-8") as current_handle, feature_path.open("r", encoding="utf-8") as feature_handle:
    for row_index, (current_line, feature_line) in enumerate(zip(current_handle, feature_handle), start=1):
        current_count += 1
        feature_count += 1
        current = json.loads(current_line)
        feature = json.loads(feature_line)
        if current.get("id") != feature.get("id"):
            raise SystemExit(
                f"{label}: id mismatch at row {row_index}: current={current.get('id')} feature={feature.get('id')}"
            )

    remaining_current = sum(1 for _ in current_handle)
    remaining_feature = sum(1 for _ in feature_handle)

if remaining_current or remaining_feature:
    raise SystemExit(
        f"{label}: full alignment ended with extra rows: current_extra={remaining_current}, feature_extra={remaining_feature}"
    )

print(f"{label}: final file is JSON-valid and id-aligned for {current_count} rows.")
PY
}

write_tail_shards() {
  local current_path="$1"
  local done_rows="$2"
  local out_dir="$3"
  local num_shards="$4"
  local total_rows="$5"

  python - "$current_path" "$done_rows" "$out_dir" "$num_shards" "$total_rows" <<'PY'
import math
import sys
from pathlib import Path

current_path = Path(sys.argv[1])
done_rows = int(sys.argv[2])
out_dir = Path(sys.argv[3])
num_shards = int(sys.argv[4])
total_rows = int(sys.argv[5])
remaining = max(0, total_rows - done_rows)
chunk = math.ceil(remaining / num_shards) if remaining else 0

out_dir.mkdir(parents=True, exist_ok=True)
handles = []
try:
    for idx in range(num_shards):
        handles.append((out_dir / f"shard_{idx:03d}.jsonl").open("w", encoding="utf-8"))

    with current_path.open("r", encoding="utf-8") as source:
        for row_number, line in enumerate(source):
            if row_number < done_rows:
                continue
            tail_index = row_number - done_rows
            shard_index = min(num_shards - 1, tail_index // chunk) if chunk else 0
            handles[shard_index].write(line)
finally:
    for handle in handles:
        handle.close()

for idx in range(num_shards):
    path = out_dir / f"shard_{idx:03d}.jsonl"
    with path.open("r", encoding="utf-8") as handle:
        rows = sum(1 for line in handle if line.strip())
    print(f"{path}\t{rows}")
PY
}

append_csv_without_header() {
  local source_csv="$1"
  local target_csv="$2"

  if [[ ! -f "$source_csv" ]]; then
    return 0
  fi

  if [[ ! -f "$target_csv" || ! -s "$target_csv" ]]; then
    cp "$source_csv" "$target_csv"
  else
    tail -n +2 "$source_csv" >> "$target_csv"
  fi
}

resume_split() {
  local model_dir="$1"
  local split="$2"
  local lm_model="${LM_MODEL[$model_dir]}"
  local max_seq="${MAX_SEQ[$lm_model]}"

  local current_path="$CURRENT_HDPO_ROOT/$model_dir/current_hdpo_${split}.jsonl"
  if [[ ! -f "$current_path" && "$model_dir" == "llama32-3b" ]]; then
    current_path="$CURRENT_HDPO_ROOT/llama32_3b/current_hdpo_${split}.jsonl"
  fi

  if [[ ! -s "$current_path" ]]; then
    echo "[skip] missing current file for $model_dir/$split: $current_path"
    return 0
  fi

  local final_lm_dir="$FEATURE_ROOT/research_models/$split/$lm_model"
  local final_text_dir="$FEATURE_ROOT/text/$model_dir"
  local final_lm_jsonl="$final_lm_dir/${split}_answer_features_dedup.jsonl"
  local final_lm_csv="$final_lm_dir/${split}_answer_features_dedup.csv"
  local final_text_jsonl="$final_text_dir/${split}_answer_features_dedup.jsonl"
  local final_text_csv="$final_text_dir/${split}_answer_features_dedup.csv"

  mkdir -p "$final_lm_dir" "$final_text_dir"

  local total_rows
  total_rows="$(count_lines "$current_path")"
  local done_rows
  done_rows="$(count_lines "$final_lm_jsonl")"
  local text_rows
  text_rows="$(count_lines "$final_text_jsonl")"

  echo "[status] $model_dir/$split current_rows=$total_rows existing_lm_rows=$done_rows existing_text_rows=$text_rows"

  if (( done_rows == total_rows )); then
    echo "[skip] $model_dir/$split already complete"
    validate_full_alignment "$current_path" "$final_lm_jsonl" "$model_dir/$split"
    return 0
  fi

  if (( done_rows > total_rows )); then
    echo "[error] $model_dir/$split has more feature rows than current rows: feature=$done_rows current=$total_rows" >&2
    exit 1
  fi

  if (( text_rows > total_rows )); then
    echo "[error] $model_dir/$split has more text feature rows than current rows: text=$text_rows current=$total_rows" >&2
    exit 1
  fi

  local append_text_tail=0
  if (( text_rows == total_rows )); then
    echo "[skip-text] $model_dir/$split text features already complete; only LM tail will be appended"
  elif (( text_rows == done_rows )); then
    append_text_tail=1
    echo "[resume-text] $model_dir/$split text features match LM prefix; text tail will also be appended"
  else
    echo "[error] $model_dir/$split text rows do not match current or LM prefix: text=$text_rows lm=$done_rows current=$total_rows" >&2
    echo "[error] rebuild or fix text feature file before resuming this split: $final_text_jsonl" >&2
    exit 1
  fi

  validate_prefix_alignment "$current_path" "$final_lm_jsonl" "$done_rows" "$model_dir/$split"

  local split_tmp="$TMP_ROOT/$model_dir/$split"
  local shard_dir="$split_tmp/shards"
  local text_tmp="$split_tmp/text"
  local lm_tmp="$split_tmp/lm"
  mkdir -p "$shard_dir" "$text_tmp" "$lm_tmp"

  write_tail_shards "$current_path" "$done_rows" "$shard_dir" "${#GPUS[@]}" "$total_rows" \
    | tee "$LOG_DIR/shards_${model_dir}_${split}.txt"

  PIDS=()
  LABELS=()

  for idx in "${!GPUS[@]}"; do
    local gpu="${GPUS[$idx]}"
    local shard_path
    shard_path="$(printf "%s/shard_%03d.jsonl" "$shard_dir" "$idx")"
    local shard_rows
    shard_rows="$(count_lines "$shard_path")"

    if (( shard_rows == 0 )); then
      continue
    fi

    local shard_name
    shard_name="$(printf "shard_%03d" "$idx")"
    local output_jsonl="$text_tmp/${shard_name}_answer_features_dedup.jsonl"
    local output_csv="$text_tmp/${shard_name}_answer_features_dedup.csv"
    local log_path="$LOG_DIR/features_${model_dir}_${split}_${shard_name}_gpu${gpu}.log"

    echo "[launch] $model_dir/$split $shard_name rows=$shard_rows gpu=$gpu log=$log_path"
    CUDA_VISIBLE_DEVICES="$gpu" PYTHONPATH="$ROOT/src:${PYTHONPATH:-}" \
      python -m polyalign_data.extract_linguistic_features \
        --input-path "$shard_path" \
        --output-jsonl "$output_jsonl" \
        --output-csv "$output_csv" \
        --text-field human_answer \
        --include-text \
        --lm-model "$lm_model" \
        --lm-output-root "$lm_tmp" \
        --lm-write-csv \
        --lm-device cuda \
        --lm-dtype "$LM_DTYPE" \
        --lm-max-seq-length "$max_seq" \
        "${TRUST_ARGS[@]}" \
        > "$log_path" 2>&1 &

    PIDS+=("$!")
    LABELS+=("$model_dir/$split/$shard_name/gpu$gpu")
  done

  local failures=0
  for idx in "${!PIDS[@]}"; do
    if ! wait "${PIDS[$idx]}"; then
      echo "[failed] ${LABELS[$idx]}" >&2
      failures=1
    fi
  done

  if (( failures != 0 )); then
    echo "[error] not appending partial shard outputs for $model_dir/$split because at least one shard failed" >&2
    exit 1
  fi

  local backup_dir="$split_tmp/backup"
  mkdir -p "$backup_dir"
  [[ -f "$final_lm_jsonl" ]] && cp "$final_lm_jsonl" "$backup_dir/${split}_lm.jsonl.bak"
  [[ -f "$final_lm_csv" ]] && cp "$final_lm_csv" "$backup_dir/${split}_lm.csv.bak"
  [[ -f "$final_text_jsonl" ]] && cp "$final_text_jsonl" "$backup_dir/${split}_text.jsonl.bak"
  [[ -f "$final_text_csv" ]] && cp "$final_text_csv" "$backup_dir/${split}_text.csv.bak"

  for idx in "${!GPUS[@]}"; do
    local shard_path
    shard_path="$(printf "%s/shard_%03d.jsonl" "$shard_dir" "$idx")"
    local shard_rows
    shard_rows="$(count_lines "$shard_path")"
    if (( shard_rows == 0 )); then
      continue
    fi

    local shard_name
    shard_name="$(printf "shard_%03d" "$idx")"
    local shard_lm_jsonl="$lm_tmp/$lm_model/${shard_name}_answer_features_dedup.jsonl"
    local shard_lm_csv="$lm_tmp/$lm_model/${shard_name}_answer_features_dedup.csv"
    local shard_text_jsonl="$text_tmp/${shard_name}_answer_features_dedup.jsonl"
    local shard_text_csv="$text_tmp/${shard_name}_answer_features_dedup.csv"

    local shard_lm_rows
    shard_lm_rows="$(count_lines "$shard_lm_jsonl")"
    local shard_text_rows
    shard_text_rows="$(count_lines "$shard_text_jsonl")"

    if (( shard_lm_rows != shard_rows || shard_text_rows != shard_rows )); then
      echo "[error] shard row mismatch for $model_dir/$split/$shard_name: source=$shard_rows lm=$shard_lm_rows text=$shard_text_rows" >&2
      exit 1
    fi

    cat "$shard_lm_jsonl" >> "$final_lm_jsonl"
    if (( append_text_tail == 1 )); then
      cat "$shard_text_jsonl" >> "$final_text_jsonl"
    fi
    append_csv_without_header "$shard_lm_csv" "$final_lm_csv"
    if (( append_text_tail == 1 )); then
      append_csv_without_header "$shard_text_csv" "$final_text_csv"
    fi
  done

  local final_rows
  final_rows="$(count_lines "$final_lm_jsonl")"
  echo "[status] $model_dir/$split final_lm_rows=$final_rows"

  if (( final_rows != total_rows )); then
    echo "[error] final row mismatch for $model_dir/$split: final=$final_rows current=$total_rows backups=$backup_dir" >&2
    exit 1
  fi

  local final_text_rows
  final_text_rows="$(count_lines "$final_text_jsonl")"
  if (( final_text_rows != total_rows )); then
    echo "[error] final text row mismatch for $model_dir/$split: text=$final_text_rows current=$total_rows" >&2
    exit 1
  fi

  validate_full_alignment "$current_path" "$final_lm_jsonl" "$model_dir/$split"
}

build_reference() {
  local model_dir="$1"
  local lm_model="${LM_MODEL[$model_dir]}"
  local current_model_dir="$CURRENT_HDPO_ROOT/$model_dir"
  if [[ ! -d "$current_model_dir" && "$model_dir" == "llama32-3b" && -d "$CURRENT_HDPO_ROOT/llama32_3b" ]]; then
    current_model_dir="$CURRENT_HDPO_ROOT/llama32_3b"
  fi

  for split in train val test; do
    local current_path="$current_model_dir/current_hdpo_${split}.jsonl"
    local feature_path="$FEATURE_ROOT/research_models/$split/$lm_model/${split}_answer_features_dedup.jsonl"
    if [[ ! -s "$current_path" || ! -s "$feature_path" ]]; then
      echo "[skip] cannot build reference for $model_dir; missing $split current/features"
      return 0
    fi
    local current_rows
    current_rows="$(count_lines "$current_path")"
    local feature_rows
    feature_rows="$(count_lines "$feature_path")"
    if (( current_rows != feature_rows )); then
      echo "[skip] cannot build reference for $model_dir; $split row mismatch current=$current_rows features=$feature_rows"
      return 0
    fi
  done

  local ref_log="$LOG_DIR/reference_${model_dir}.log"
  echo "[reference] $model_dir log=$ref_log"
  PYTHONPATH="$ROOT/src:${PYTHONPATH:-}" \
    python -m polyalign_data reference-build \
      --records-path "$current_model_dir/current_hdpo_train.jsonl" \
      --features-path "$FEATURE_ROOT/research_models/train/$lm_model/train_answer_features_dedup.jsonl" \
      --records-path "$current_model_dir/current_hdpo_val.jsonl" \
      --features-path "$FEATURE_ROOT/research_models/val/$lm_model/val_answer_features_dedup.jsonl" \
      --records-path "$current_model_dir/current_hdpo_test.jsonl" \
      --features-path "$FEATURE_ROOT/research_models/test/$lm_model/test_answer_features_dedup.jsonl" \
      --output-root "$REF_ROOT/$model_dir" \
      --overwrite \
      2>&1 | tee "$ref_log"
}

echo "[info] logs: $LOG_DIR"
echo "[info] tmp: $TMP_ROOT"
echo "[info] GPUs: ${GPUS[*]}"
echo "[info] splits: ${SPLITS_ARR[*]}"

PROCESSED_MODELS=()
for model_dir in "${MODEL_DIR_CANDIDATES[@]}"; do
  if [[ -z "${LM_MODEL[$model_dir]+set}" ]]; then
    echo "[skip] unknown model dir alias: $model_dir"
    continue
  fi
  if [[ ! -d "$CURRENT_HDPO_ROOT/$model_dir" ]]; then
    echo "[skip] current model dir not present: $CURRENT_HDPO_ROOT/$model_dir"
    continue
  fi

  PROCESSED_MODELS+=("$model_dir")
  for split in "${SPLITS_ARR[@]}"; do
    resume_split "$model_dir" "$split"
  done
done

echo "[info] building reference artifacts where all splits are complete"
for model_dir in "${PROCESSED_MODELS[@]}"; do
  build_reference "$model_dir"
done

echo "[done] resume completed"
echo "[done] logs: $LOG_DIR"
echo "[done] tmp/backups: $TMP_ROOT"
