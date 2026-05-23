#!/usr/bin/env bash
set -euo pipefail

cd /home/umair/TW/PolyAlign

LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu:/lib/x86_64-linux-gnu git pull


CUDA_VISIBLE_DEVICES=1 python -m metrics \
  --test-lf-path /home/umair/TW/PolyAlign/data/english/merged_sft_dedup/llamafactory/test.json \
  --predictions-path /home/umair/TW/PolyAlign/data/english/merged_sft_dedup/pred/qwen25-1-5b-dpo-en.jsonl \
  --output-json /home/umair/TW/PolyAlign/data/metrics/qwen25-1-5b-en-dpo-test.json \
  --current-test-path /home/umair/TW/PolyAlign/data/english/merged_sft_dedup/current/test.jsonl \
  --bucket-references-path /home/umair/TW/PolyAlign/data/english/reference_artifacts/qwen25_1_5b/bucket_references.json \
  --feature-matrix-path /home/umair/TW/PolyAlign/data/english/reference_artifacts/qwen25_1_5b/feature_matrix.jsonl \
  --work-dir /home/umair/TW/PolyAlign/data/metrics/qwen25-1-5b-dpo-artifacts \
  --model-alias qwen25_1_5b \
  --device cuda \
  --mauve-device-id 0

LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu:/lib/x86_64-linux-gnu git pull

LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu:/lib/x86_64-linux-gnu git add \
  /home/umair/TW/PolyAlign/data/metrics/llama32-3b-en-dist-sft-test.json \
  /home/umair/TW/PolyAlign/data/metrics/gemma2-2b-en-dpo-test.json \
  /home/umair/TW/PolyAlign/data/metrics/qwen25-1-5b-en-dpo-test.json \
  /home/umair/TW/PolyAlign/data/metrics/qwen25-3b-en-dpo-test.json

if ! LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu:/lib/x86_64-linux-gnu git diff --cached --quiet; then
  LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu:/lib/x86_64-linux-gnu git commit -m "add english metrics outputs"
  LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu:/lib/x86_64-linux-gnu git push
fi