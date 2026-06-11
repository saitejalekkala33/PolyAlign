# PolyAlign: Conditional Human-Distribution Alignment

PolyAlign is the code and artifact workspace for **"PolyAlign: Conditional Human-Distribution Alignment"**. The project studies post-training that preserves human response variation across language, task family, interaction track, and response length, instead of aligning every prompt to one generic assistant style.

The repo contains:

- a Python preprocessing package, `polyalign-data`, for normalizing English and Chinese corpora into a shared schema;
- Bucket-SFT / Dist-SFT and HDPO data exporters;
- a vendored LLaMA-Factory tree with PolyAlign-specific full-parameter SFT, DPO, and HDPO hooks;
- vLLM inference scripts for BaseLM, CoT, SFT, Dist-SFT, DPO, and HDPO checkpoints;
- metric, LLM-as-a-judge, AI-detection, embedding-analysis, and human-evaluation tooling.

## Released Assets

Dataset and generated artifacts:

- Hugging Face dataset: [saiteja33/PolyAlign-All](https://huggingface.co/datasets/saiteja33/PolyAlign-All)

The dataset repo is used as an artifact store with heterogeneous JSON/JSONL files, so the Hugging Face preview may not render the whole repo as one loadable dataset. Prefer `hf download` for exact files or the full tree.

HDPO checkpoints:

| Language | Base family | Model |
| --- | --- | --- |
| English | Qwen2.5-1.5B | [saiteja33/qwen2.5-1.5b-en-hdpo](https://huggingface.co/saiteja33/qwen2.5-1.5b-en-hdpo) |
| English | Gemma-2-2B | [saiteja33/gemma2-2b-hdpo-en](https://huggingface.co/saiteja33/gemma2-2b-hdpo-en) |
| English | Qwen2.5-3B | [saiteja33/qwen2.5-3b-en-hdpo](https://huggingface.co/saiteja33/qwen2.5-3b-en-hdpo) |
| English | Llama-3.2-3B | [saiteja33/llama-3.2-3b-hdpo-en](https://huggingface.co/saiteja33/llama-3.2-3b-hdpo-en) |
| Chinese | Qwen2.5-1.5B | [sathiiiii/polyalign-qwen2.5-1.5b-zh-hdpo](https://huggingface.co/sathiiiii/polyalign-qwen2.5-1.5b-zh-hdpo) |
| Chinese | Gemma-2-2B | [saiteja33/gemma2-2b-hdpo-zh](https://huggingface.co/saiteja33/gemma2-2b-hdpo-zh) |
| Chinese | Qwen2.5-3B | [saiteja33/qwen2.5-3b-hdpo-zh](https://huggingface.co/saiteja33/qwen2.5-3b-hdpo-zh) |
| Chinese | Llama-3.2-3B | [saiteja33/llama-3.2-3b-hdpo-zh](https://huggingface.co/saiteja33/llama-3.2-3b-hdpo-zh) |

Bucket-SFT checkpoints, published with `dist-sft` names:

| Language | Base family | Model |
| --- | --- | --- |
| Chinese | Qwen2.5-1.5B | [sathiiiii/polyalign-qwen2.5-1.5b-zh-dist-sft](https://huggingface.co/sathiiiii/polyalign-qwen2.5-1.5b-zh-dist-sft) |
| Chinese | Gemma-2-2B | [sathiiiii/polyalign-gemma2-2b-zh-dist-sft](https://huggingface.co/sathiiiii/polyalign-gemma2-2b-zh-dist-sft) |
| Chinese | Qwen2.5-3B | [sathiiiii/polyalign-qwen2.5-3b-zh-dist-sft](https://huggingface.co/sathiiiii/polyalign-qwen2.5-3b-zh-dist-sft) |
| Chinese | Llama-3.2-3B | [sathiiiii/polyalign-llama3.2-3b-zh-dist-sft](https://huggingface.co/sathiiiii/polyalign-llama3.2-3b-zh-dist-sft) |
| English | Qwen2.5-1.5B | [sathiiiii/polyalign-qwen2.5-1.5b-en-dist-sft](https://huggingface.co/sathiiiii/polyalign-qwen2.5-1.5b-en-dist-sft) |
| English | Gemma-2-2B | [sathiiiii/polyalign-gemma2-2b-en-dist-sft](https://huggingface.co/sathiiiii/polyalign-gemma2-2b-en-dist-sft) |
| English | Qwen2.5-3B | [sathiiiii/polyalign-qwen2.5-3b-en-dist-sft](https://huggingface.co/sathiiiii/polyalign-qwen2.5-3b-en-dist-sft) |
| English | Llama-3.2-3B | [sathiiiii/polyalign-llama3.2-3b-en-dist-sft](https://huggingface.co/sathiiiii/polyalign-llama3.2-3b-en-dist-sft) |

## Method At A Glance

PolyAlign treats alignment as matching a **conditional human response distribution**. Each example is assigned metadata:

```text
language, track, family, style_bucket, length_bin, bucket_id
```

The preprint defines buckets primarily from `(language, track, family, length_bin)` and uses bucket-specific human feature statistics for training and evaluation.

```text
raw corpora
  -> normalized PolyAlign schema
  -> deduped train / val / test splits
  -> current JSONL + LLaMA-Factory views
  -> bucket references + feature matrices
  -> Bucket-SFT / Dist-SFT
  -> HDPO critic targets and scored preference pairs
  -> HDPO policy training
  -> utility, naturalness, judge, detector, and human-eval analysis
```

Bucket-SFT gives each bucket equal optimization mass. HDPO starts from the supervised checkpoint, trains a bucket-conditioned critic over human-support distance, scores chosen/rejected pairs, and optimizes a DPO-style objective with distributional regularization.

## Corpus Scope

The preprint reports this bilingual corpus inventory after normalization and deduplication:

| Dataset | Lang | Situation | Track | Train / Val / Test | Total |
| --- | --- | --- | --- | --- | --- |
| Dolly | en | assistant_like | single | 13,525 / 713 / 757 | 14,995 |
| ELI5 | en | longform_qa | single | 91,772 / 5,446 / 7,786 | 105,004 |
| DailyDialog | en | open_chat | multi | 37,377 / 3,774 / 3,681 | 44,832 |
| MS MARCO | en | qa_search | single | 80,143 / 9,754 / 9,399 | 99,296 |
| CoQA | en | qa_search | multi | 98,015 / 10,620 / 7,983 | 116,618 |
| SQuAD v2 | en | qa_search | single | 117,444 / 12,832 / 11,870 | 142,146 |
| Natural Questions | en | qa_search | single | 90,133 / 5,081 / 5,017 | 100,231 |
| MultiWOZ | en | task_dialogue | multi | 56,013 / 7,355 / 7,362 | 70,730 |
| COIG-CQIA | zh | assistant_like | single | 9,536 / 509 / 579 | 10,624 |
| HC3-Chinese | zh | longform_qa | single | 19,990 / 1,152 / 1,058 | 22,200 |
| OASST2-zh | zh | open_chat | multi | 3,809 / 527 / 225 | 4,561 |
| CMRC2018 | zh | qa_search | single | 10,142 / 3,219 / 1,002 | 14,363 |
| DRCD | zh | qa_search | single | 26,936 / 3,524 / 3,493 | 33,953 |
| DuReader | zh | qa_search | single | 15,923 / 1,956 / - | 17,879 |

The checked-in scope files are [configs/scope_freeze.json](configs/scope_freeze.json), [docs/scope_freeze.md](docs/scope_freeze.md), and [docs/dedup_policy.md](docs/dedup_policy.md).

## Repository Layout

```text
configs/                  Dataset and pipeline configs.
docs/                     Scope and deduplication policy notes.
src/polyalign_data/       Installable preprocessing, export, feature, and HDPO critic package.
vendor/LlamaFactory/      Vendored LLaMA-Factory with PolyAlign training additions.
experiments/              vLLM inference wrappers for base, SFT, DPO, and HDPO models.
metrics/                  Utility, diversity, BNG, HCR, MAUVE, TDM, NUF, and aggregate metrics.
scripts/llm_judge/        Local vLLM LLM-as-a-judge pipeline and rubric.
scripts/ai_detection/     AI-detector input building, scoring, and summary helpers.
human-eval/               Static blind human-evaluation UI.
sample/                   Small sample train/val/test files.
```

## Setup

Use Python 3.11 or newer. The lightweight data pipeline can run on CPU; full training and vLLM evaluation require a Linux GPU environment.

```bash
python -m venv .venv
# Linux/macOS:
source .venv/bin/activate
# Windows PowerShell:
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e .
```

For training with the vendored LLaMA-Factory code:

```bash
python -m pip install -e vendor/LlamaFactory
```

For evaluation pipelines, install the extra packages required by the scripts you use, such as `vllm`, `mauve-text`, `scikit-learn`, and `pyyaml`. GPU stack details depend on CUDA or ROCm; the paper experiments used single-node 8-GPU training with PyTorch/ROCm and AMD Instinct MI210 GPUs.

Download released artifacts:

```bash
hf download saiteja33/PolyAlign-All --repo-type dataset --local-dir data/hf
```

Download a checkpoint:

```bash
hf download saiteja33/qwen2.5-1.5b-en-hdpo --local-dir models/qwen25_1_5b_en_hdpo
```

## Reproducibility

### 1. Build normalized data locally

Install the package first, then run formatters. The English registry is exposed by `--all`; the checked-in full pipeline config is currently Chinese-specific.

```bash
python -m polyalign_data format --all --output-root data/formatted --cache-dir data/cache
python -m polyalign_data format --config configs/format_job_chinese.json
```

Run the checked-in Chinese end-to-end preprocessing pipeline:

```bash
python -m polyalign_data pipeline --config configs/pipeline_chinese.json
```

The pipeline writes formatted data, deduped data, merged SFT views, feature files, bucket references, and a summary under `data/chinese/` according to [configs/pipeline_chinese.json](configs/pipeline_chinese.json).

### 2. Export LLaMA-Factory views

Standard SFT merged view:

```bash
python -m polyalign_data.export_sft_views \
  --input-root data/chinese/formatted_dedup \
  --output-root data/chinese/merged_sft_dedup
```

Bucket-SFT / Dist-SFT view:

```bash
python -m polyalign_data.export_dist_sft_views \
  --input-root data/chinese/formatted_dedup \
  --output-root data/chinese/merged_sft_dedup/llamafactory
```

DPO views require aligned SFT prediction files generated from the same `current` records. The concrete run directories depend on the model and evaluation run you produced or downloaded.

```bash
python -m polyalign_data export-dpo-views \
  --records-root data/chinese/merged_sft_dedup/current \
  --output-root data/chinese/dpo \
  --model-alias qwen25_1_5b \
  --language-tag zh \
  --train-predictions <path-to-train-sft-predictions.jsonl> \
  --val-predictions <path-to-val-sft-predictions.jsonl> \
  --test-predictions <path-to-test-sft-predictions.jsonl>
```

### 3. Train with LLaMA-Factory

Example YAMLs live under [vendor/LlamaFactory/examples](vendor/LlamaFactory/examples). They are runnable templates, not a complete manifest for every released checkpoint.

```bash
cd vendor/LlamaFactory
llamafactory-cli train examples/train_full/qwen25_1_5b_full_sft_polyalign.yaml
llamafactory-cli train examples/train_full/qwen25_1_5b_full_dist_sft_polyalign.yaml
llamafactory-cli train examples/train_full/qwen25_1_5b_full_hdpo_polyalign.yaml
```

Some YAMLs and shell launchers include machine-specific absolute paths from the original experiment environment. Update `model_name_or_path`, `dataset_dir`, `output_dir`, and any critic paths before rerunning them.

### 4. Build HDPO data and critic scores

The generic CLI has transformer-critic commands. Prediction, feature, and reference paths must all correspond to the same language, split, and model alias.

```bash
python -m polyalign_data hdpo-build-pairs \
  --record-path data/chinese/merged_sft_dedup/current/train.jsonl \
  --prediction-path <path-to-train-sft-predictions.jsonl> \
  --output-root data/chinese/hdpo_work/qwen25_1_5b/pairs_raw \
  --split-name train \
  --merged-output-path data/chinese/hdpo_work/qwen25_1_5b/pairs_raw/hdpo_pairs_raw_train_all.jsonl

python -m polyalign_data hdpo-critic-prepare \
  --record-path data/chinese/merged_sft_dedup/current/train.jsonl \
  --feature-path <path-to-train-answer-features.jsonl> \
  --references-path <path-to-bucket_references.json> \
  --output-path data/chinese/hdpo_work/qwen25_1_5b/critic/critic_train.jsonl

python -m polyalign_data hdpo-critic-train \
  --train-path data/chinese/hdpo_work/qwen25_1_5b/critic/critic_train.jsonl \
  --output-dir data/chinese/hdpo_work/qwen25_1_5b/critic/bundle \
  --encoder-name-or-path <local-dist-sft-or-base-encoder> \
  --device cuda
```

The MLP-only critic workflow used by some checked-in launchers is exposed through:

```bash
python -m polyalign_data.hdpo_critic_mlp --help
```

Reference shell launchers are in [scripts/run_zh_hdpo_mlp.sh](scripts/run_zh_hdpo_mlp.sh), [scripts/run_zh_hdpo_features_refs.sh](scripts/run_zh_hdpo_features_refs.sh), and related scripts. Treat them as environment-specific run books.

### 5. Run generation with vLLM

Serve a released model:

```bash
vllm serve sathiiiii/polyalign-qwen2.5-1.5b-zh-hdpo --served-model-name qwen25_1_5b_zh_hdpo
```

Run HDPO-style pair inference against a vLLM completions endpoint:

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

Other inference entry points:

```bash
python experiments/base_lm/run_vllm_baseline.py --help
python experiments/base_lm/run_vllm_cot.py --help
python experiments/sft/run_vllm_sft.py --help
python experiments/sft/run_vllm_dist_sft.py --help
python experiments/dpo/run_vllm_dpo.py --help
python experiments/hdpo/run_vllm_hdpo_ref_conditioned.py --help
```

`run_vllm_hdpo_ref_conditioned.py` intentionally includes the chosen/reference answer in the prompt and is only for oracle or leakage analysis. Keep those outputs separate from normal evaluation.

### 6. Compute metrics

The metrics package expects a LLaMA-Factory test file and an aligned `predictions.jsonl` with `source_index`.

```bash
python -m metrics \
  --test-lf-path data/hf/chinese/merged_sft_dedup/llamafactory/test.json \
  --predictions-path data/runs/qwen25_1_5b_hdpo_zh/predictions.jsonl \
  --current-test-path data/hf/chinese/merged_sft_dedup/current/test.jsonl \
  --human-feature-path data/hf/chinese/hdpo/features-hdpo/research_models/test/qwen25_1_5b/test_answer_features_dedup.jsonl \
  --bucket-references-path data/hf/chinese/hdpo/reference_artifacts-hdpo/qwen25_1_5b/bucket_references.json \
  --feature-matrix-path data/hf/chinese/hdpo/reference_artifacts-hdpo/qwen25_1_5b/feature_matrix.jsonl \
  --output-json data/metrics/qwen25_1_5b_hdpo_zh.json \
  --model-alias qwen25_1_5b \
  --device cuda
```

See [metrics/README.md](metrics/README.md) for metric details and companion-file auto-discovery.

### 7. Run judge and detector analyses

LLM-as-a-judge:

```bash
REPO=$PWD HF_DATASET_REPO=saiteja33/PolyAlign-All bash scripts/llm_judge/run_llm_judge.sh
```

See [scripts/llm_judge/README.md](scripts/llm_judge/README.md). The rubric scores eight 1-5 dimensions and reports 0-100 composites for overall quality, utility, conditional naturalness, and distribution faithfulness.

AI-detection helper commands:

```bash
python scripts/ai_detection/polyalign_ai_detection.py list-downloads --lang all --include-human
python scripts/ai_detection/polyalign_ai_detection.py build-inputs --raw-root data/hf --output-root data/ai_detection/work --lang zh
python scripts/ai_detection/polyalign_ai_detection.py score-lang --help
```

### 8. Run the human-evaluation UI

```bash
cd human-eval
python -m http.server 8088
```

Open `http://localhost:8088`. See [human-eval/README.md](human-eval/README.md).

## Output Schema

Normalized records use this schema:

```json
{
  "id": "unique_id",
  "dataset": "dolly",
  "split": "train",
  "language": "en",
  "track": "single",
  "family": "assistant",
  "style_bucket": "assistant_like",
  "length_bin": "medium",
  "question": "prompt text",
  "context": "",
  "dialogue_history": [],
  "human_answer": "target response",
  "bucket_id": "en|single|assistant|medium",
  "meta": {
    "source_dataset": "databricks/databricks-dolly-15k",
    "source_split": "train",
    "source_id": "dolly-train-000001",
    "length_tokens": 84
  }
}
```

## Notes And Caveats

- No top-level repository license file is present in this checkout. Check the licenses of the underlying source datasets, base models, released model cards, and the vendored LLaMA-Factory and AI-detector subtrees before redistribution.
- `vendor/LlamaFactory/` is a vendored training dependency with PolyAlign additions; its own README and Apache-2.0 license remain under that directory.
- Some notebooks and generated plots are present in the workspace, but the reproducible command-line entry points are the Python modules and shell scripts documented above.
- Shell launchers were written for Linux GPU nodes and may require path, GPU, and environment-variable edits.

## Citation

No arXiv identifier is present in the local preprint PDF used for this README. Until an official BibTeX entry is released, cite the preprint title and authors:

```bibtex
@misc{polyalign2026,
  title = {PolyAlign: Conditional Human-Distribution Alignment},
  author = {L. D. M. S. Sai Teja and Ufaq Khan and Sathira Silva and Xiao Wu and Muhammad Haris Khan},
  year = {2026},
  note = {Preprint}
}
```
