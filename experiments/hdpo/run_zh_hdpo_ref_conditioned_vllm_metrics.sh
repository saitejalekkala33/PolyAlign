#!/usr/bin/env bash
set -euo pipefail

REPO=${REPO:-/home/umair/TW/PolyAlign}
RUN_ID=${RUN_ID:-$(date +%Y%m%d_%H%M%S)}

export REPO
export RUN_ID
export RUN_SUFFIX=${RUN_SUFFIX:--ref-conditioned}
export INFERENCE_SCRIPT=${INFERENCE_SCRIPT:-experiments/hdpo/run_vllm_hdpo_ref_conditioned.py}
export LOG_DIR=${LOG_DIR:-$REPO/logs/hdpo_ref_conditioned_vllm_zh/$RUN_ID}
export REFERENCE_CONDITIONING_INSTRUCTION=${REFERENCE_CONDITIONING_INSTRUCTION:-Use the reference answer only as semantic guidance. Write an answer that is similar in meaning and appropriate for the question, but strictly do not output the exact same text as the reference answer.}

cd "$REPO"
exec bash experiments/hdpo/run_zh_hdpo_vllm_metrics.sh
#!/usr/bin/env bash
set -euo pipefail

REPO=${REPO:-/home/umair/TW/PolyAlign}
RUN_ID=${RUN_ID:-$(date +%Y%m%d_%H%M%S)}

export REPO
export RUN_ID
export RUN_SUFFIX=${RUN_SUFFIX:--ref-conditioned}
export INFERENCE_SCRIPT=${INFERENCE_SCRIPT:-experiments/hdpo/run_vllm_hdpo_ref_conditioned.py}
export LOG_DIR=${LOG_DIR:-$REPO/logs/hdpo_ref_conditioned_vllm_zh/$RUN_ID}
export REFERENCE_CONDITIONING_INSTRUCTION=${REFERENCE_CONDITIONING_INSTRUCTION:-Use the reference answer only as semantic guidance. Write an answer that is similar in meaning and appropriate for the question, but strictly do not output the exact same text as the reference answer.}

cd "$REPO"
exec bash experiments/hdpo/run_zh_hdpo_vllm_metrics.sh
