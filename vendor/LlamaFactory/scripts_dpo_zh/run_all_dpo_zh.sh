#!/bin/bash
set -euo pipefail

cd ~/PolyAlign/vendor/LlamaFactory

source ~/miniforge3/etc/profile.d/conda.sh
conda activate polyalign

export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}"

export WANDB_PROJECT=polyalign-dpo-zh
export WANDB_ENTITY=saitejalekkala33
# unset WANDB_ENTITY
export WANDB_MODE=online

mkdir -p logs

echo "Host: $(hostname)"
echo "SLURM_JOB_ID=${SLURM_JOB_ID:-unset}"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
nvidia-smi

run_dpo () {
    local name="$1"
    local yaml="$2"
    local log_file="$3"

    echo ""
    echo "============================================================"
    echo "Starting DPO run: ${name}"
    echo "YAML: ${yaml}"
    echo "Log: ${log_file}"
    echo "============================================================"
    echo ""

    export WANDB_NAME="${name}"

    LD_LIBRARY_PATH="$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}" \
    CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}" \
    FORCE_TORCHRUN=1 \
    llamafactory-cli train "${yaml}" 2>&1 | tee "${log_file}"

    echo ""
    echo "Finished DPO run: ${name}"
    echo ""
}

run_dpo "qwen25-15b-dpo-zh"  "examples/train_dpo_zh/qwen25-15b-dpo-zh.yaml"  "logs/qwen25-15b-dpo-zh.txt"
run_dpo "gemma2-2b-dpo-zh"   "examples/train_dpo_zh/gemma2-2b-dpo-zh.yaml"   "logs/gemma2-2b-dpo-zh.txt"
run_dpo "qwen25-3b-dpo-zh"   "examples/train_dpo_zh/qwen25-3b-dpo-zh.yaml"   "logs/qwen25-3b-dpo-zh.txt"
run_dpo "llama32-3b-dpo-zh"  "examples/train_dpo_zh/llama32-3b-dpo-zh.yaml"  "logs/llama32-3b-dpo-zh.txt"

echo ""
echo "All DPO runs completed."
