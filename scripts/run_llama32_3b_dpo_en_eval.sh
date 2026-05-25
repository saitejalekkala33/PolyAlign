#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="/home/umair/TW/PolyAlign"
GPU_ID="6"
PORT="8090"
BASE_URL="http://127.0.0.1:${PORT}"

MODEL_NAME="llama32-3b-dpo-en"
MODEL_ALIAS="llama32_3b"
MODEL_DIR="${PROJECT_DIR}/models/${MODEL_NAME}"
TOKENIZER_DIR="${MODEL_DIR}"

INPUT_PATH="${PROJECT_DIR}/vendor/LlamaFactory/data/test.json"
OUTPUT_DIR="${PROJECT_DIR}/experiments/dpo/runs/${MODEL_NAME}"
PREDICTIONS_PATH="${OUTPUT_DIR}/predictions.jsonl"

METRICS_JSON="${PROJECT_DIR}/data/metrics/llama32-3b-dpo-en-eval.json"
METRICS_WORK_DIR="${PROJECT_DIR}/data/metrics/llama32-3b-dpo-en-eval-artifacts"

CURRENT_TEST_PATH="${PROJECT_DIR}/data/hf/english/merged_sft_dedup/current/test.jsonl"
BUCKET_REFERENCES_PATH="${PROJECT_DIR}/data/reference_artifacts/llama32_3b/bucket_references.json"
FEATURE_MATRIX_PATH="${PROJECT_DIR}/data/reference_artifacts/llama32_3b/feature_matrix.jsonl"

HF_DATASET_REPO="saiteja33/PolyAlign-All"
HF_PATH_IN_REPO="english/merged_sft_dedup/runs/${MODEL_NAME}"

LOG_DIR="${PROJECT_DIR}/logs"
VLLM_LOG="${LOG_DIR}/${MODEL_NAME}_vllm_${PORT}.log"
RUN_LOG="${LOG_DIR}/${MODEL_NAME}_full_run.log"

cd "${PROJECT_DIR}"

mkdir -p "${OUTPUT_DIR}"
mkdir -p "${METRICS_WORK_DIR}"
mkdir -p "${LOG_DIR}"
mkdir -p "$(dirname "${INPUT_PATH}")"
mkdir -p "$(dirname "${CURRENT_TEST_PATH}")"
mkdir -p "$(dirname "${BUCKET_REFERENCES_PATH}")"
mkdir -p "$(dirname "${FEATURE_MATRIX_PATH}")"

exec > >(tee -a "${RUN_LOG}") 2>&1

hf auth whoami >/dev/null

cp "$(hf download "${HF_DATASET_REPO}" english/merged_sft_dedup/llamafactory/test.json --repo-type dataset)" "${INPUT_PATH}"
cp "$(hf download "${HF_DATASET_REPO}" english/merged_sft_dedup/current/test.jsonl --repo-type dataset)" "${CURRENT_TEST_PATH}"
cp "$(hf download "${HF_DATASET_REPO}" english/reference_artifacts/llama32_3b/bucket_references.json --repo-type dataset)" "${BUCKET_REFERENCES_PATH}"
cp "$(hf download "${HF_DATASET_REPO}" english/reference_artifacts/llama32_3b/feature_matrix.jsonl --repo-type dataset)" "${FEATURE_MATRIX_PATH}"

test -d "${MODEL_DIR}"
test -f "${MODEL_DIR}/config.json"
test -f "${MODEL_DIR}/tokenizer.json"
test -f "${MODEL_DIR}/tokenizer_config.json"
test -f "${INPUT_PATH}"
test -f "${CURRENT_TEST_PATH}"
test -f "${BUCKET_REFERENCES_PATH}"
test -f "${FEATURE_MATRIX_PATH}"

python - <<PY
from transformers import AutoTokenizer

tok = AutoTokenizer.from_pretrained(
    "${TOKENIZER_DIR}",
    use_fast=True,
    trust_remote_code=True,
)

print(type(tok))
print(tok.eos_token, tok.eos_token_id)
PY

VLLM_PID=""

kill_vllm() {
    if [[ -n "${VLLM_PID:-}" ]] && kill -0 "${VLLM_PID}" 2>/dev/null; then
        kill "${VLLM_PID}" || true
        wait "${VLLM_PID}" || true
    fi
}

trap kill_vllm EXIT

CUDA_VISIBLE_DEVICES="${GPU_ID}" vllm serve "${MODEL_DIR}" \
    --served-model-name "${MODEL_NAME}" \
    --tokenizer "${TOKENIZER_DIR}" \
    --host 127.0.0.1 \
    --port "${PORT}" \
    --gpu-memory-utilization 0.75 \
    --trust-remote-code \
    > "${VLLM_LOG}" 2>&1 &

VLLM_PID="$!"

for i in $(seq 1 180); do
    if curl -fsS "${BASE_URL}/v1/models" >/dev/null 2>&1; then
        break
    fi

    if ! kill -0 "${VLLM_PID}" 2>/dev/null; then
        tail -n 160 "${VLLM_LOG}" || true
        exit 1
    fi

    sleep 5

    if [[ "${i}" == "180" ]]; then
        tail -n 160 "${VLLM_LOG}" || true
        exit 1
    fi
done

python experiments/dpo/run_vllm_dpo.py \
    --input-path "${INPUT_PATH}" \
    --output-dir "${OUTPUT_DIR}" \
    --model-name "${MODEL_NAME}" \
    --tokenizer-name-or-path "${TOKENIZER_DIR}" \
    --base-url "${BASE_URL}" \
    --sample-size 0 \
    --batch-size 4 \
    --resume \
    --trust-remote-code

kill_vllm
VLLM_PID=""
trap - EXIT

test -f "${PREDICTIONS_PATH}"

# hf upload "${HF_DATASET_REPO}" "${OUTPUT_DIR}" "${HF_PATH_IN_REPO}" \
#     --repo-type dataset \
#     --commit-message "Upload ${MODEL_NAME} English predictions"

CUDA_VISIBLE_DEVICES="${GPU_ID}" python -m metrics \
    --test-lf-path "${INPUT_PATH}" \
    --predictions-path "${PREDICTIONS_PATH}" \
    --output-json "${METRICS_JSON}" \
    --current-test-path "${CURRENT_TEST_PATH}" \
    --bucket-references-path "${BUCKET_REFERENCES_PATH}" \
    --feature-matrix-path "${FEATURE_MATRIX_PATH}" \
    --work-dir "${METRICS_WORK_DIR}" \
    --model-alias "${MODEL_ALIAS}" \
    --device cuda \
    --mauve-device-id 0 \
    # --overwrite-artifacts

git add "${OUTPUT_DIR}" "${METRICS_JSON}" "${METRICS_WORK_DIR}" || true

if ! git diff --cached --quiet; then
    git commit -m "Add llama32-3b DPO English predictions and metrics"
fi

LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu:/lib/x86_64-linux-gnu git push