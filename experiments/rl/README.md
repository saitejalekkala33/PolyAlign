# Pairwise vLLM Inference Helpers

This directory contains the shared inference implementation used by the DPO and HDPO wrappers:

- `experiments/dpo/run_vllm_dpo.py`
- `experiments/hdpo/run_vllm_hdpo.py`
- `experiments/hdpo/run_vllm_hdpo_ref_conditioned.py`

`vllm_pair_inference.py` reads LLaMA-Factory-style preference JSON/JSONL files with `instruction`, `input`, `chosen`, `rejected`, optional `history`, and PolyAlign metadata. It renders prompts through the tokenizer chat template, calls a vLLM `/v1/completions` endpoint, and writes:

```text
predictions.jsonl
progress.json
summary.json
config.json
```

## Example

Serve a model:

```bash
vllm serve sathiiiii/polyalign-qwen2.5-1.5b-zh-hdpo --served-model-name qwen25_1_5b_zh_hdpo
```

Run HDPO inference:

```bash
python experiments/hdpo/run_vllm_hdpo.py \
  --input-path data/hf/chinese/merged_sft_dedup/hdpo_prepared/qwen25_1_5b/llamafactory/hdpo_test.json \
  --output-dir data/runs/qwen25_1_5b_hdpo_zh \
  --model-name qwen25_1_5b_zh_hdpo \
  --tokenizer-name-or-path sathiiiii/polyalign-qwen2.5-1.5b-zh-hdpo \
  --base-url http://127.0.0.1:8000 \
  --sample-size 32 \
  --overwrite
```

Use `--sample-size <= 0` for the full input file, `--resume` to continue a partially completed run, and `--sample-mode random --seed 42` for deterministic subsampling.

`run_vllm_hdpo_ref_conditioned.py` includes the chosen/reference output in the prompt. It is an oracle-style leakage analysis mode and should not be mixed with normal HDPO evaluation outputs.
