# PolyAlign Data Pipeline

This workspace contains the PolyAlign dataset-normalization package for both the existing English corpus and a separate Chinese corpus pipeline.

English formatters currently cover:

- `databricks/databricks-dolly-15k`
- `microsoft/ms_marco` (`v1.1`)
- `stanfordnlp/coqa`
- `rexarski/eli5_category`
- `rajpurkar/squad_v2`
- `sentence-transformers/natural-questions`
- `roskoN/dailydialog`
- `pfb30/multi_woz_v22`

Chinese formatters currently cover:

- `OpenAssistant/oasst2` filtered to reviewed Chinese branches
- `luozhouyang/dureader` (`checklist` + `robust`)
- `hfl/cmrc2018`
- `voidful/DRCD`
- `Hello-SimpleAI/HC3-Chinese`
- `m-a-p/COIG-CQIA`

## What It Does

- Normalizes each dataset into one JSONL schema.
- Applies dataset-specific split policy.
- Writes one directory per dataset with one JSONL file per split.
- Emits a dataset manifest for reproducibility.
- Builds an initial bucket/reference summary over formatted outputs.
- Can run the full pipeline from formatted data through deduped merged SFT views, feature files, and full bucket reference artifacts.

## Unified Output Schema

Each normalized row follows this structure:

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

## Scope Decisions

- `SQuAD v2` and `sentence-transformers/natural-questions` are preserved as QA/reference-style corpora even though the answers are extractive/passage-like.
- `CoQA` is unfolded turn-by-turn at the conversation level.
- `DailyDialog` is converted into response-prediction examples by alternating speakers and using odd-numbered utterances as targets.
- `ELI5-Category` keeps `validation2` as an auxiliary split instead of collapsing it into dev/test.
- Chinese outputs are intended to live under `data/chinese/...` so they remain separate from the existing English artifacts under `data/...`.

## Usage

Install the package in editable mode if you want the `polyalign-data` command:

```bash
python -m pip install -e .
```

Format datasets from the editable config file:

```bash
python -m polyalign_data format --config configs/format_job.json
```

Format the Chinese datasets into their own root:

```bash
python -m polyalign_data format --config configs/format_job_chinese.json
```

Format a single dataset directly:

```bash
python -m polyalign_data format --dataset dolly --output-root data/formatted
```

Format all datasets:

```bash
python -m polyalign_data format --all --output-root data/formatted
```

Build the initial reference summary after formatting:

```bash
python -m polyalign_data reference --input-root data/formatted --output-path data/reference_artifacts/prior-reference_summary.json
```

Run the full Chinese pipeline into separate artifact roots:

```bash
python -m polyalign_data pipeline --config configs/pipeline_chinese.json
```

## Config Files

- `configs/scope_freeze.json`: frozen project scope and split policy.
- `configs/format_job.json`: editable run config where you can choose dataset names.
- `configs/format_job_chinese.json`: Chinese-only formatting config that writes to `data/chinese/formatted`.
- `configs/pipeline_chinese.json`: Chinese-only full pipeline config that writes formatted, deduped, merged, feature, and reference artifacts under `data/chinese/`.

## Package Layout

```text
configs/
docs/
src/polyalign_data/
  datasets/
```
