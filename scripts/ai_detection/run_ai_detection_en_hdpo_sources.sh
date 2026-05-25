#!/usr/bin/env bash
set -euo pipefail

REPO="${REPO:-/home/umair/TW/PolyAlign}"
cd "$REPO"

CONFIG_DIR="${CONFIG_DIR:-$REPO/logs/ai-detection-en-hdpo-sources}"
mkdir -p "$CONFIG_DIR"

SOURCE_CONFIG_FILE="$CONFIG_DIR/en_hdpo_source_config.json"
cat > "$SOURCE_CONFIG_FILE" <<'JSON'
{
  "en": {
    "hf_prefix": "english",
    "human_hf_path": "english/merged_sft_dedup/llamafactory/test.json",
    "current_hf_path": "english/merged_sft_dedup/current/test.jsonl",
    "sources": [
      {
        "source_id": "qwen25_1_5b__hdpo",
        "model_key": "qwen25_1_5b",
        "stage": "hdpo",
        "path": "english/merged_sft_dedup/runs/qwen25-1-5b-hdpo-en-ref-conditioned/predictions.jsonl"
      },
      {
        "source_id": "gemma2_2b__hdpo",
        "model_key": "gemma2_2b",
        "stage": "hdpo",
        "path": "english/merged_sft_dedup/runs/gemma2-2b-hdpo-en-ref-conditioned/predictions.jsonl"
      },
      {
        "source_id": "qwen25_3b__hdpo",
        "model_key": "qwen25_3b",
        "stage": "hdpo",
        "path": "english/merged_sft_dedup/runs/qwen25-3b-hdpo-en-ref-conditioned/predictions.jsonl"
      }
    ]
  }
}
JSON

export POLYALIGN_AI_SOURCE_CONFIG_FILE="$SOURCE_CONFIG_FILE"
export LANGS="en"
export DETECTORS="radar binoculars fast_detect_gpt"
export RESUME="${RESUME:-1}"
export UPLOAD_TO_HF="${UPLOAD_TO_HF:-1}"
export GIT_PUSH_SUMMARIES="${GIT_PUSH_SUMMARIES:-1}"
export CLEANUP_DETECTOR_MODELS="${CLEANUP_DETECTOR_MODELS:-1}"
export LOG_ROOT="${LOG_ROOT:-$REPO/logs/ai-detection-en-hdpo-sources}"

bash "$REPO/scripts/ai_detection/run_ai_detection_en_two_sources.sh"
