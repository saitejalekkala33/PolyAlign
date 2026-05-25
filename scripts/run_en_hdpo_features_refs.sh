#!/usr/bin/env bash
set -euo pipefail

ROOT="/home/umair/TW/PolyAlign"
DATA_ROOT="${ROOT}/data/english/hdpo"

CURRENT_HDPO_ROOT="${DATA_ROOT}/current-hdpo-en"
FEATURE_ROOT="${DATA_ROOT}/features-hdpo"
REF_ROOT="${DATA_ROOT}/reference_artifacts-hdpo"

LOG_DIR="${ROOT}/logs/en_hdpo_features_refs_$(date +%Y%m%d_%H%M%S)"
mkdir -p "${LOG_DIR}"

GPU_LIST="${GPU_LIST:-1 2 3 4 5 6 7}"
read -r -a GPUS <<< "${GPU_LIST}"

MODEL_DIRS=(gemma2-2b qwen25-1-5b qwen25-3b)
SPLITS=(train val test)

MAX_PARALLEL="${MAX_PARALLEL:-${#GPUS[@]}}"
LM_DTYPE="${LM_DTYPE:-auto}"

declare -A LM_MODEL
LM_MODEL[gemma2-2b]=gemma2_2b
LM_MODEL[qwen25-1-5b]=qwen25_1_5b
LM_MODEL[qwen25-3b]=qwen25_3b

declare -A MAX_SEQ
MAX_SEQ[gemma2_2b]=8192
MAX_SEQ[qwen25_1_5b]=4096
MAX_SEQ[qwen25_3b]=8192

TRUST_ARGS=()
if [[ "${TRUST_REMOTE_CODE:-0}" == "1" ]]; then
  TRUST_ARGS+=(--lm-trust-remote-code)
fi

cd "${ROOT}"

run_feature_job() {
  local gpu="$1"
  local model_dir="$2"
  local split="$3"
  local lm_model="${LM_MODEL[$model_dir]}"

  local input_path="${CURRENT_HDPO_ROOT}/${model_dir}/current_hdpo_${split}.jsonl"
  local text_dir="${FEATURE_ROOT}/text/${model_dir}"
  local lm_root="${FEATURE_ROOT}/research_models/${split}"
  local output_jsonl="${text_dir}/${split}_answer_features_dedup.jsonl"
  local output_csv="${text_dir}/${split}_answer_features_dedup.csv"

  if [[ ! -s "${input_path}" ]]; then
    echo "[error] missing input: ${input_path}" >&2
    exit 1
  fi

  mkdir -p "${text_dir}" "${lm_root}"

  echo "[start] model_dir=${model_dir} lm_model=${lm_model} split=${split} gpu=${gpu}"
  CUDA_VISIBLE_DEVICES="${gpu}" PYTHONPATH="${ROOT}/src:${PYTHONPATH:-}" \
    python -m polyalign_data.extract_linguistic_features \
      --input-path "${input_path}" \
      --output-jsonl "${output_jsonl}" \
      --output-csv "${output_csv}" \
      --text-field human_answer \
      --include-text \
      --lm-model "${lm_model}" \
      --lm-output-root "${lm_root}" \
      --lm-write-csv \
      --lm-device cuda \
      --lm-dtype "${LM_DTYPE}" \
      --lm-max-seq-length "${MAX_SEQ[$lm_model]}" \
      "${TRUST_ARGS[@]}"
  echo "[done] model_dir=${model_dir} lm_model=${lm_model} split=${split} gpu=${gpu}"
}

wait_for_batch() {
  local failures=0
  for idx in "${!PIDS[@]}"; do
    if ! wait "${PIDS[$idx]}"; then
      echo "[failed] ${LABELS[$idx]}" >&2
      failures=1
    fi
  done
  PIDS=()
  LABELS=()
  if (( failures != 0 )); then
    exit 1
  fi
}

PIDS=()
LABELS=()
job_index=0

echo "[info] logs: ${LOG_DIR}"
echo "[info] extracting HDPO-aligned English features"

for model_dir in "${MODEL_DIRS[@]}"; do
  for split in "${SPLITS[@]}"; do
    gpu_index=$((job_index % ${#GPUS[@]}))
    gpu="${GPUS[$gpu_index]}"
    log_path="${LOG_DIR}/features_${model_dir}_${split}_gpu${gpu}.log"

    run_feature_job "${gpu}" "${model_dir}" "${split}" > "${log_path}" 2>&1 &
    PIDS+=("$!")
    LABELS+=("${model_dir}/${split}/gpu${gpu}")

    echo "[launch] ${model_dir}/${split} on gpu ${gpu}; log=${log_path}"

    job_index=$((job_index + 1))

    if (( ${#PIDS[@]} >= MAX_PARALLEL )); then
      echo "[wait] feature batch"
      wait_for_batch
    fi
  done
done

if (( ${#PIDS[@]} > 0 )); then
  echo "[wait] final feature batch"
  wait_for_batch
fi

echo "[info] building reference artifacts"

for model_dir in "${MODEL_DIRS[@]}"; do
  lm_model="${LM_MODEL[$model_dir]}"
  ref_log="${LOG_DIR}/reference_${model_dir}.log"
  echo "[reference] model_dir=${model_dir} lm_model=${lm_model}; log=${ref_log}"

  PYTHONPATH="${ROOT}/src:${PYTHONPATH:-}" \
    python -m polyalign_data reference-build \
      --records-path "${CURRENT_HDPO_ROOT}/${model_dir}/current_hdpo_train.jsonl" \
      --features-path "${FEATURE_ROOT}/research_models/train/${lm_model}/train_answer_features_dedup.jsonl" \
      --records-path "${CURRENT_HDPO_ROOT}/${model_dir}/current_hdpo_val.jsonl" \
      --features-path "${FEATURE_ROOT}/research_models/val/${lm_model}/val_answer_features_dedup.jsonl" \
      --records-path "${CURRENT_HDPO_ROOT}/${model_dir}/current_hdpo_test.jsonl" \
      --features-path "${FEATURE_ROOT}/research_models/test/${lm_model}/test_answer_features_dedup.jsonl" \
      --output-root "${REF_ROOT}/${model_dir}" \
      --overwrite \
      2>&1 | tee "${ref_log}"
done

echo "[info] validating row counts"

MODEL_DIRS_STR="${MODEL_DIRS[*]}" CURRENT_HDPO_ROOT="${CURRENT_HDPO_ROOT}" FEATURE_ROOT="${FEATURE_ROOT}" REF_ROOT="${REF_ROOT}" python - <<'PY'
import json
import os
from pathlib import Path

model_dirs = os.environ["MODEL_DIRS_STR"].split()
splits = ["train", "val", "test"]
current_root = Path(os.environ["CURRENT_HDPO_ROOT"])
feature_root = Path(os.environ["FEATURE_ROOT"])
ref_root = Path(os.environ["REF_ROOT"])

lm_model = {
    "gemma2-2b": "gemma2_2b",
    "qwen25-1-5b": "qwen25_1_5b",
    "qwen25-3b": "qwen25_3b",
}

def count_jsonl(path: Path) -> int:
    with path.open("r", encoding="utf-8") as handle:
        return sum(1 for line in handle if line.strip())

report = []

for model_dir in model_dirs:
    alias = lm_model[model_dir]

    for split in splits:
        current_path = current_root / model_dir / f"current_hdpo_{split}.jsonl"
        feature_path = feature_root / "research_models" / split / alias / f"{split}_answer_features_dedup.jsonl"

        current_rows = count_jsonl(current_path)
        feature_rows = count_jsonl(feature_path)

        if current_rows != feature_rows:
            raise SystemExit(f"row mismatch: {model_dir}/{split}: current={current_rows}, features={feature_rows}")

        report.append({
            "model_dir": model_dir,
            "lm_model": alias,
            "split": split,
            "current_rows": current_rows,
            "feature_rows": feature_rows,
            "feature_path": str(feature_path),
        })

    for required in ["bucket_references.json", "feature_matrix.jsonl", "summary.json"]:
        path = ref_root / model_dir / required
        if not path.exists():
            raise SystemExit(f"missing reference artifact: {path}")

print(json.dumps(report, indent=2, ensure_ascii=False))
print("All English HDPO-aligned current files, LM feature files, and reference artifacts are present and row-count aligned.")
PY

echo "[done] all English HDPO feature extraction and reference builds completed"
echo "[done] logs: ${LOG_DIR}"
BASH
