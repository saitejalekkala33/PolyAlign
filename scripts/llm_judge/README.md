# PolyAlign LLM-as-a-Judge Pipeline

This pipeline scores every configured PolyAlign human/model response with large
local judges served through vLLM:

- `qwen3_30b_a3b_instruct_2507`: `Qwen/Qwen3-30B-A3B-Instruct-2507`
- `glm45_air_fp8`: `zai-org/GLM-4.5-Air-FP8`

Execution is sequential by judge model. First Qwen is downloaded and run to
completion; only then GLM is downloaded and run. For each judge model, the
default execution uses two parallel vLLM servers over disjoint source-file
shards:

- server 0: GPUs `0,1,2,3`, port `8100`
- server 1: GPUs `4,5,6,7`, port `8101`

Both servers use the same judge model and write different score files under the
same judge ID. Both judges run with hidden reasoning disabled in the prompt; GLM
also uses `{"enable_thinking": false}` during local chat-template rendering.

## Rubric

The rubric is in `rubric.yaml`. It uses 8 integer dimensions, each scored 1-5:

- `task_success`
- `factual_grounding`
- `instruction_following`
- `reference_alignment`
- `conditional_appropriateness`
- `response_shape_and_length`
- `discourse_naturalness`
- `safety`

The script computes four deterministic 0-100 composites from those raw scores:
`overall`, `utility`, `conditional_naturalness`, and
`distribution_faithfulness`.

## No-Truncation Policy

The runner renders the full prompt, counts tokens with the judge tokenizer, and
fails by default if `prompt_tokens + max_tokens + safety_margin` exceeds
`MAX_MODEL_LEN`. It never truncates input text. If you want to record oversized
rows as errors and continue, set `OVERLENGTH_POLICY=record_error`.

## Run

```bash
REPO=/home/umair/TW/PolyAlign \
HF_DATASET_REPO=saiteja33/PolyAlign-All \
bash scripts/llm_judge/run_llm_judge.sh
```

Useful overrides:

```bash
MAX_MODEL_LEN=65536          # increase context if all prompts fit in memory
BATCH_SIZE=4                 # reduce if either vLLM server is memory-bound
SAMPLE_SIZE=100              # smoke test before full scoring
UPLOAD_TO_HF=0               # local-only run
GIT_PUSH_SUMMARIES=0         # skip git commit/push
DOWNLOAD_JUDGE_MODELS=0      # let vLLM resolve model ids directly
CLEANUP_JUDGE_MODEL_AFTER_RUN=1  # delete each local judge model after scoring
QWEN_GPU_GROUPS='0,1,2,3;4,5,6,7'
GLM_GPU_GROUPS='0,1,2,3;4,5,6,7'
QWEN_PORTS='8100 8101'
GLM_PORTS='8100 8101'
```

Outputs are written under `data/llm_judge/work` and uploaded to the dataset repo
under `english/llm-judge`, `chinese/llm-judge`, and `llm-judge/metrics`.
