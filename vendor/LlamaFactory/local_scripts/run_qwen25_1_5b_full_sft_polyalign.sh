#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

if [[ ! -f "data/train.json" ]]; then
  echo "Missing data/train.json in $ROOT_DIR/data" >&2
  exit 1
fi

if [[ ! -f "data/val.json" ]]; then
  echo "Missing data/val.json in $ROOT_DIR/data" >&2
  exit 1
fi

CONFIG_PATH="examples/train_full/qwen25_1_5b_full_sft_polyalign.yaml"

llamafactory-cli train "$CONFIG_PATH"
