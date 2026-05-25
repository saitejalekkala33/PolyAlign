#!/usr/bin/env bash
set -euo pipefail

ROOT="/home/umair/TW/PolyAlign"
DATA_ROOT="${ROOT}/data/english/hdpo"

MODEL_DIR="llama32_3b"
LM_MODEL="llama32_3b"

CURRENT_HDPO_ROOT="${DATA_ROOT}/current-hdpo-en"
FEATURE_ROOT="${DATA_ROOT}/features-hdpo"
REF_ROOT="${DATA_ROOT}/reference_artifacts-hdpo"

LOG_DIR="${ROOT}/logs/en_hdpo_features_refs_llama32_3b_$(date +%Y%m%d_%H%M%S)"
mkdir -p "${LOG_DIR}"

GPU_LIST="${GPU_LIST:-0 1 2 3 4 5 6 7}"
read -r -a GPUS <<< "${GPU_LIST}"

SPLITS=(train val test)

MAX_PARALLEL="${MAX_PARALLEL:-${#GPUS[@]}}"
LM_DTYPE="${LM_DTYPE:-auto}"
LM_MAX_SEQ_LENGTH="${LM_MAX_SEQ_LENGTH:-4096}"

TRUST_ARGS=()
if [[ "${TRUST_REMOTE_CODE:-0}" == "1" ]]; then
  TRUST_ARGS+=(--lm-trust-remote-code)
fi

CURRENT_MODEL_DIR="${CURRENT_HDPO_ROOT}/${MODEL_DIR}"

if [[ ! -d "${CURRENT_MODEL_DIR}" && -d "${CURRENT_HDPO_ROOT}/llama32-3b" ]]; then
  CURRENT_MODEL_DIR="${CURRENT_HDPO_ROOT}/llama32-3b"
fi

if [[ ! -d "${CURRENT_MODEL_DIR}" ]]; then
  echo "[error] missing current-aligned HDPO dir: ${CURRENT_HDPO_ROOT}/${MODEL_DIR}" >&2
  echo "[error] also checked: ${CURRENT_HDPO_ROOT}/llama32-3b" >&2
  echo "[error] hdpo_prepared/llama32_3b/hdpo_*.json is not enough for this feature/reference step." >&2
  exit 1
fi

cd "${ROOT}"

run_feature_job() {
  local gpu="$1"
  local split="$2"

  local input_path="${CURRENT_MODEL_DIR}/current_hdpo_${split}.jsonl"
  local text_dir="${FEATURE_ROOT}/text/${MODEL_DIR}"
  local lm_root="${FEATURE_ROOT}/research_models/${split}"
  local output_jsonl="${text_dir}/${split}_answer_features_dedup.jsonl"
  local output_csv="${text_dir}/${split}_answer_features_dedup.csv"

  if [[ ! -s "${input_path}" ]]; then
    echo "[error] missing input: ${input_path}" >&2
    exit 1
  fi

  mkdir -p "${text_dir}" "${lm_root}"

  echo "[start] model_dir=${MODEL_DIR} lm_model=${LM_MODEL} split=${split} gpu=${gpu}"
  CUDA_VISIBLE_DEVICES="${gpu}" PYTHONPATH="${ROOT}/src:${PYTHONPATH:-}" \
    python -m polyalign_data.extract_linguistic_features \
      --input-path "${input_path}" \
      --output-jsonl "${output_jsonl}" \
      --output-csv "${output_csv}" \
      --text-field human_answer \
      --include-text \
      --lm-model "${LM_MODEL}" \
      --lm-output-root "${lm_root}" \
      --lm-write-csv \
      --lm-device cuda \
      --lm-dtype "${LM_DTYPE}" \
      --lm-max-seq-length "${LM_MAX_SEQ_LENGTH}" \
      "${TRUST_ARGS[@]}"
  echo "[done] model_dir=${MODEL_DIR} lm_model=${LM_MODEL} split=${split} gpu=${gpu}"
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
echo "[info] extracting English HDPO features for ${MODEL_DIR}"
echo "[info] current dir: ${CURRENT_MODEL_DIR}"

for split in "${SPLITS[@]}"; do
  gpu_index=$((job_index % ${#GPUS[@]}))
  gpu="${GPUS[$gpu_index]}"
  log_path="${LOG_DIR}/features_${MODEL_DIR}_${split}_gpu${gpu}.log"

  run_feature_job "${gpu}" "${split}" > "${log_path}" 2>&1 &
  PIDS+=("$!")
  LABELS+=("${MODEL_DIR}/${split}/gpu${gpu}")

  echo "[launch] ${MODEL_DIR}/${split} on gpu ${gpu}; log=${log_path}"

  job_index=$((job_index + 1))

  if (( ${#PIDS[@]} >= MAX_PARALLEL )); then
    echo "[wait] feature batch"
    wait_for_batch
  fi
done

if (( ${#PIDS[@]} > 0 )); then
  echo "[wait] final feature batch"
  wait_for_batch
fi

echo "[info] building reference artifacts"

ref_log="${LOG_DIR}/reference_${MODEL_DIR}.log"

PYTHONPATH="${ROOT}/src:${PYTHONPATH:-}" \
  python -m polyalign_data reference-build \
    --records-path "${CURRENT_MODEL_DIR}/current_hdpo_train.jsonl" \
    --features-path "${FEATURE_ROOT}/research_models/train/${LM_MODEL}/train_answer_features_dedup.jsonl" \
    --records-path "${CURRENT_MODEL_DIR}/current_hdpo_val.jsonl" \
    --features-path "${FEATURE_ROOT}/research_models/val/${LM_MODEL}/val_answer_features_dedup.jsonl" \
    --records-path "${CURRENT_MODEL_DIR}/current_hdpo_test.jsonl" \
    --features-path "${FEATURE_ROOT}/research_models/test/${LM_MODEL}/test_answer_features_dedup.jsonl" \
    --output-root "${REF_ROOT}/${MODEL_DIR}" \
    --overwrite \
    2>&1 | tee "${ref_log}"

echo "[info] validating row counts"

CURRENT_MODEL_DIR="${CURRENT_MODEL_DIR}" FEATURE_ROOT="${FEATURE_ROOT}" REF_ROOT="${REF_ROOT}" MODEL_DIR="${MODEL_DIR}" LM_MODEL="${LM_MODEL}" python - <<'PY'
import json
import os
from pathlib import Path

splits = ["train", "val", "test"]
current_dir = Path(os.environ["CURRENT_MODEL_DIR"])
feature_root = Path(os.environ["FEATURE_ROOT"])
ref_root = Path(os.environ["REF_ROOT"])
model_dir = os.environ["MODEL_DIR"]
lm_model = os.environ["LM_MODEL"]

def count_jsonl(path: Path) -> int:
    with path.open("r", encoding="utf-8") as handle:
        return sum(1 for line in handle if line.strip())

report = []

for split in splits:
    current_path = current_dir / f"current_hdpo_{split}.jsonl"
    feature_path = feature_root / "research_models" / split / lm_model / f"{split}_answer_features_dedup.jsonl"

    current_rows = count_jsonl(current_path)
    feature_rows = count_jsonl(feature_path)

    if current_rows != feature_rows:
        raise SystemExit(f"row mismatch: {model_dir}/{split}: current={current_rows}, features={feature_rows}")

    report.append({
        "model_dir": model_dir,
        "lm_model": lm_model,
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
print("All llama32_3b English HDPO current files, LM feature files, and reference artifacts are row-count aligned.")
PY

echo "[done] llama32_3b English HDPO feature extraction and reference build completed"
echo "[done] logs: ${LOG_DIR}"
BASH