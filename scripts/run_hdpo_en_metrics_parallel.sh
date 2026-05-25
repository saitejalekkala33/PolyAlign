#!/usr/bin/env bash

set -euo pipefail

PROJECT_DIR="/home/umair/TW/PolyAlign"
LOG_DIR="${PROJECT_DIR}/logs/hdpo_en_metrics"

mkdir -p "${LOG_DIR}"
cd "${PROJECT_DIR}"

CUDA_VISIBLE_DEVICES=0 python -m metrics \
    --test-lf-path "${PROJECT_DIR}/data/english/hdpo/hdpo_prepared/qwen25-1-5b/hdpo_test.json" \
    --predictions-path "${PROJECT_DIR}/data/english/merged_sft_dedup/runs/qwen25-1-5b-hdpo-en-ref-conditioned/predictions.jsonl" \
    --output-json "${PROJECT_DIR}/data/metrics/qwen25-1-5b-hdpo-en.json" \
    --current-test-path "${PROJECT_DIR}/data/english/hdpo/current-hdpo-en/qwen25-1-5b/current_hdpo_test.jsonl" \
    --bucket-references-path "${PROJECT_DIR}/data/english/hdpo/reference_artifacts-hdpo/qwen25-1-5b/bucket_references.json" \
    --feature-matrix-path "${PROJECT_DIR}/data/english/hdpo/reference_artifacts-hdpo/qwen25-1-5b/feature_matrix.jsonl" \
    --work-dir "${PROJECT_DIR}/data/metrics/qwen25-1-5b-hdpo-en" \
    --model-alias qwen25_1_5b \
    --device cuda \
    --mauve-device-id 0 \
    --overwrite-artifacts \
    > "${LOG_DIR}/qwen25-1-5b-hdpo-en.log" 2>&1 &

PID_QWEN15=$!

CUDA_VISIBLE_DEVICES=1 python -m metrics \
    --test-lf-path "${PROJECT_DIR}/data/english/hdpo/hdpo_prepared/gemma2-2b/hdpo_test.json" \
    --predictions-path "${PROJECT_DIR}/data/english/merged_sft_dedup/runs/gemma2-2b-hdpo-en-ref-conditioned/predictions.jsonl" \
    --output-json "${PROJECT_DIR}/data/metrics/gemma2-2b-hdpo-en.json" \
    --current-test-path "${PROJECT_DIR}/data/english/hdpo/current-hdpo-en/gemma2-2b/current_hdpo_test.jsonl" \
    --bucket-references-path "${PROJECT_DIR}/data/english/hdpo/reference_artifacts-hdpo/gemma2-2b/bucket_references.json" \
    --feature-matrix-path "${PROJECT_DIR}/data/english/hdpo/reference_artifacts-hdpo/gemma2-2b/feature_matrix.jsonl" \
    --work-dir "${PROJECT_DIR}/data/metrics/gemma2-2b-hdpo-en" \
    --model-alias gemma2_2b \
    --device cuda \
    --mauve-device-id 0 \
    --overwrite-artifacts \
    > "${LOG_DIR}/gemma2-2b-hdpo-en.log" 2>&1 &

PID_GEMMA=$!

CUDA_VISIBLE_DEVICES=2 python -m metrics \
    --test-lf-path "${PROJECT_DIR}/data/english/hdpo/hdpo_prepared/qwen25-3b/hdpo_val.json" \
    --predictions-path "${PROJECT_DIR}/data/english/merged_sft_dedup/runs/qwen25-3b-hdpo-en-ref-conditioned/predictions.jsonl" \
    --output-json "${PROJECT_DIR}/data/metrics/qwen25-3b-hdpo-en.json" \
    --current-test-path "${PROJECT_DIR}/data/english/hdpo/current-hdpo-en/qwen25-3b/current_hdpo_val.jsonl" \
    --bucket-references-path "${PROJECT_DIR}/data/english/hdpo/reference_artifacts-hdpo/qwen25-3b/bucket_references.json" \
    --feature-matrix-path "${PROJECT_DIR}/data/english/hdpo/reference_artifacts-hdpo/qwen25-3b/feature_matrix.jsonl" \
    --work-dir "${PROJECT_DIR}/data/metrics/qwen25-3b-hdpo-en" \
    --model-alias qwen25_3b \
    --device cuda \
    --mauve-device-id 0 \
    --overwrite-artifacts \
    > "${LOG_DIR}/qwen25-3b-hdpo-en.log" 2>&1 &

PID_QWEN3=$!

FAIL=0

wait "${PID_QWEN15}" || FAIL=1
wait "${PID_GEMMA}" || FAIL=1
wait "${PID_QWEN3}" || FAIL=1

if [[ "${FAIL}" -ne 0 ]]; then
    echo "One or more metrics jobs failed. Check logs in ${LOG_DIR}"
    exit 1
fi

echo "All HDPO English metrics jobs completed successfully."
