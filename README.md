# PolyAlign Data Pipeline

This workspace now contains the first scope-freeze and dataset-normalization package for PolyAlign.

It covers the eight datasets you selected and excludes `OpenAssistant/oasst2` for now:

- `databricks/databricks-dolly-15k`
- `microsoft/ms_marco` (`v1.1`)
- `stanfordnlp/coqa`
- `rexarski/eli5_category`
- `rajpurkar/squad_v2`
- `sentence-transformers/natural-questions`
- `roskoN/dailydialog`
- `pfb30/multi_woz_v22`

## What It Does

- Normalizes each dataset into one JSONL schema.
- Applies dataset-specific split policy.
- Writes one directory per dataset with one JSONL file per split.
- Emits a dataset manifest for reproducibility.
- Builds an initial bucket/reference summary over formatted outputs.

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

- English-only preprocessing for now.
- `OASST2` intentionally excluded.
- `SQuAD v2` and `sentence-transformers/natural-questions` are preserved as QA/reference-style corpora even though the answers are extractive/passage-like.
- `CoQA` is unfolded turn-by-turn at the conversation level.
- `DailyDialog` is converted into response-prediction examples by alternating speakers and using odd-numbered utterances as targets.
- `ELI5-Category` keeps `validation2` as an auxiliary split instead of collapsing it into dev/test.

## Usage

Install the package in editable mode if you want the `polyalign-data` command:

```bash
python -m pip install -e .
```

Format datasets from the editable config file:

```bash
python -m polyalign_data format --config configs/format_job.json
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
python -m polyalign_data reference --input-root data/formatted --output-path data/reference/reference_summary.json
```

## Config Files

- `configs/scope_freeze.json`: frozen project scope and split policy.
- `configs/format_job.json`: editable run config where you can choose dataset names.

## Package Layout

```text
configs/
docs/
src/polyalign_data/
  datasets/
```
