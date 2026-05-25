#!/usr/bin/env bash
set -euo pipefail

ROOT="/home/umair/TW/PolyAlign"
DATA_ROOT="${ROOT}/data/chinese/merged_sft_dedup"

CURRENT_HDPO_ROOT="${DATA_ROOT}/current-hdpo-zh"
FEATURE_ROOT="${DATA_ROOT}/features-hdpo"
REF_ROOT="${DATA_ROOT}/reference_artifacts-hdpo"

LOG_DIR="${ROOT}/logs/zh_hdpo_features_refs_$(date +%Y%m%d_%H%M%S)"
mkdir -p "${LOG_DIR}"

GPUS=(1 2 3 4 5 6 7)
MODELS=(gemma2_2b llama32_3b qwen25_1_5b qwen25_3b)
SPLITS=(train val test)

MAX_PARALLEL="${MAX_PARALLEL:-7}"
LM_DTYPE="${LM_DTYPE:-auto}"

declare -A MAX_SEQ
MAX_SEQ[gemma2_2b]=8192
MAX_SEQ[llama32_3b]=4096
MAX_SEQ[qwen25_1_5b]=4096
MAX_SEQ[qwen25_3b]=8192

TRUST_ARGS=()
if [[ "${TRUST_REMOTE_CODE:-0}" == "1" ]]; then
  TRUST_ARGS+=(--lm-trust-remote-code)
fi

cd "${ROOT}"

run_feature_job() {
  local gpu="$1"
  local model="$2"
  local split="$3"

  local input_path="${CURRENT_HDPO_ROOT}/${model}/current_hdpo_${split}.jsonl"
  local text_dir="${FEATURE_ROOT}/text/${model}"
  local lm_root="${FEATURE_ROOT}/research_models/${split}"
  local output_jsonl="${text_dir}/${split}_answer_features_dedup.jsonl"
  local output_csv="${text_dir}/${split}_answer_features_dedup.csv"

  if [[ ! -s "${input_path}" ]]; then
    echo "[error] missing input: ${input_path}" >&2
    exit 1
  fi

  mkdir -p "${text_dir}" "${lm_root}"

  echo "[start] model=${model} split=${split} gpu=${gpu}"
  CUDA_VISIBLE_DEVICES="${gpu}" PYTHONPATH="${ROOT}/src:${PYTHONPATH:-}" \
    python -m polyalign_data.extract_linguistic_features \
      --input-path "${input_path}" \
      --output-jsonl "${output_jsonl}" \
      --output-csv "${output_csv}" \
      --text-field human_answer \
      --include-text \
      --lm-model "${model}" \
      --lm-output-root "${lm_root}" \
      --lm-write-csv \
      --lm-device cuda \
      --lm-dtype "${LM_DTYPE}" \
      --lm-max-seq-length "${MAX_SEQ[$model]}" \
      "${TRUST_ARGS[@]}"
  echo "[done] model=${model} split=${split} gpu=${gpu}"
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
echo "[info] extracting HDPO-aligned zh features"

for model in "${MODELS[@]}"; do
  for split in "${SPLITS[@]}"; do
    gpu="${GPUS[$((job_index % MAX_PARALLEL))]}"
    log_path="${LOG_DIR}/features_${model}_${split}_gpu${gpu}.log"

    run_feature_job "${gpu}" "${model}" "${split}" > "${log_path}" 2>&1 &
    PIDS+=("$!")
    LABELS+=("${model}/${split}/gpu${gpu}")

    echo "[launch] ${model}/${split} on gpu ${gpu}; log=${log_path}"

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

for model in "${MODELS[@]}"; do
  ref_log="${LOG_DIR}/reference_${model}.log"
  echo "[reference] ${model}; log=${ref_log}"

  PYTHONPATH="${ROOT}/src:${PYTHONPATH:-}" \
    python -m polyalign_data reference-build \
      --records-path "${CURRENT_HDPO_ROOT}/${model}/current_hdpo_train.jsonl" \
      --features-path "${FEATURE_ROOT}/research_models/train/${model}/train_answer_features_dedup.jsonl" \
      --records-path "${CURRENT_HDPO_ROOT}/${model}/current_hdpo_val.jsonl" \
      --features-path "${FEATURE_ROOT}/research_models/val/${model}/val_answer_features_dedup.jsonl" \
      --records-path "${CURRENT_HDPO_ROOT}/${model}/current_hdpo_test.jsonl" \
      --features-path "${FEATURE_ROOT}/research_models/test/${model}/test_answer_features_dedup.jsonl" \
      --output-root "${REF_ROOT}/${model}" \
      --overwrite \
      2>&1 | tee "${ref_log}"
done

echo "[info] validating row counts"

MODELS_STR="${MODELS[*]}" CURRENT_HDPO_ROOT="${CURRENT_HDPO_ROOT}" FEATURE_ROOT="${FEATURE_ROOT}" REF_ROOT="${REF_ROOT}" python - <<'PY'
import json
import os
from pathlib import Path

models = os.environ["MODELS_STR"].split()
splits = ["train", "val", "test"]
current_root = Path(os.environ["CURRENT_HDPO_ROOT"])
feature_root = Path(os.environ["FEATURE_ROOT"])
ref_root = Path(os.environ["REF_ROOT"])

def count_jsonl(path: Path) -> int:
    with path.open("r", encoding="utf-8") as handle:
        return sum(1 for line in handle if line.strip())

report = []

for model in models:
    for split in splits:
        current_path = current_root / model / f"current_hdpo_{split}.jsonl"
        feature_path = feature_root / "research_models" / split / model / f"{split}_answer_features_dedup.jsonl"

        current_rows = count_jsonl(current_path)
        feature_rows = count_jsonl(feature_path)

        if current_rows != feature_rows:
            raise SystemExit(
                f"row mismatch: {model}/{split}: current={current_rows}, features={feature_rows}"
            )

        report.append({
            "model": model,
            "split": split,
            "current_rows": current_rows,
            "feature_rows": feature_rows,
            "feature_path": str(feature_path),
        })

    for required in ["bucket_references.json", "feature_matrix.jsonl", "summary.json"]:
        path = ref_root / model / required
        if not path.exists():
            raise SystemExit(f"missing reference artifact: {path}")

print(json.dumps(report, indent=2, ensure_ascii=False))
print("All HDPO-aligned current files, LM feature files, and reference artifacts are present and row-count aligned.")
PY

echo "[done] all zh HDPO feature extraction and reference builds completed"
echo "[done] logs: ${LOG_DIR}"
BASH