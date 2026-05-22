# PolyAlign LLM-as-a-Judge Pipeline

This pipeline scores every configured PolyAlign model response with local judges
served through vLLM. Human responses are loaded as references/context only; they
are not scored as candidates by default.

- `qwen3_8b`: `Qwen/Qwen3-8B`
- `qwen25_7b_instruct`: `Qwen/Qwen2.5-7B-Instruct`
- `mistral_small_3_2_24b_instruct_2506`: `mistralai/Mistral-Small-3.2-24B-Instruct-2506`
- `ministral3_8b_instruct_2512`: `mistralai/Ministral-3-8B-Instruct-2512`

Execution is sequential by judge model. First Qwen is downloaded and run to
completion; only then Mistral is downloaded and run. For each judge model, the
default execution uses two parallel vLLM servers over disjoint source-file
shards for an 8x A100 40GB node:

- server 0: GPUs `0,1,2,3`, port `8100`
- server 1: GPUs `4,5,6,7`, port `8101`

Both servers use the same judge model and write different score files under the
same judge ID. Qwen runs with thinking disabled through both local chat-template
rendering and vLLM server defaults:

```bash
--default-chat-template-kwargs '{"enable_thinking":false}' --generation-config vllm
```

`zai-org/GLM-4.5-Air-FP8` remains configurable through `JUDGES`, but it is not
the default because FP8 GLM is the higher-risk option on A100 40GB hardware.
`ministral3_8b_instruct_2512` is configured for one vLLM server per GPU by
default and uses Mistral-format loading:

```bash
--tokenizer_mode mistral --config_format mistral --load_format mistral
```

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
MAX_NUM_SEQS=16              # default for A100 40GB; lower if vLLM OOMs
SAMPLE_SIZE=100              # smoke test before full scoring
UPLOAD_TO_HF=0               # local-only run
GIT_PUSH_SUMMARIES=0         # skip git commit/push
DOWNLOAD_JUDGE_MODELS=0      # let vLLM resolve model ids directly
CLEANUP_JUDGE_MODEL_AFTER_RUN=0  # keep downloaded judges; default deletes after each judge
INCLUDE_HUMAN_CANDIDATES=1   # optional; normally leave human responses unscored
QWEN_MODEL_ID=/home/umair/TW/Multilingual-Interviewer/models/qwen3-8b
QWEN_GPU_GROUPS='0,1,2,3;4,5,6,7'
QWEN25_7B_GPU_GROUPS='0;1;2;3;4;5;6;7'
MISTRAL_GPU_GROUPS='0,1,2,3;4,5,6,7'
MINISTRAL_GPU_GROUPS='0;1;2;3;4;5;6;7'
QWEN_PORTS='8100 8101'
QWEN25_7B_PORTS='8100 8101 8102 8103 8104 8105 8106 8107'
MISTRAL_PORTS='8100 8101'
MINISTRAL_PORTS='8100 8101 8102 8103 8104 8105 8106 8107'
JUDGES='qwen3_8b'            # run only the Qwen judge
JUDGES='qwen25_7b_instruct'  # run only Qwen2.5-7B-Instruct
JUDGES='ministral3_8b_instruct_2512'  # run only Ministral 3 8B
JUDGES='qwen3_8b glm45_air_fp8'  # optional GLM run if you want to test it
```

Outputs are written under `data/llm_judge/work` and uploaded to the dataset repo
under `english/llm-judge`, `chinese/llm-judge`, and `llm-judge/metrics`.
