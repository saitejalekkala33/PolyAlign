# Metrics Package

This package evaluates PolyAlign generations from a LlamaFactory `test.json` and a matching `predictions.jsonl`.

It produces one JSON report covering:

- utility metrics: raw EM, normalized EM, token F1, ROUGE-L
- diversity metrics: Distinct-1/2/3 and Self-BLEU-4
- naturalness metrics: BNG, HCR, global MAUVE, conditional MAUVE
- multi-turn dynamics: non-heuristic TDM based on continuous turn coupling and latent transition structure
- NUF summaries and Pareto frontier outputs

## Inputs

Required:

- `--test-lf-path`
- `--predictions-path`
- `--output-json`

Optional companion files are auto-discovered from the `data/` root when possible:

- `merged_sft_dedup/current/test.jsonl`
- `features/research_models/test/<model_alias>/test_answer_features_dedup.jsonl`
- `reference_artifacts/bucket_references.json`
- `reference_artifacts/feature_matrix.jsonl`

## Example

```bash
python -m metrics \
  --test-lf-path /TW/PolyAlign/data/merged_sft_dedup/llamafactory/test.json \
  --predictions-path /TW/PolyAlign/data/merged_sft_dedup/runs/qwen25_1_5b/predictions.jsonl \
  --output-json /TW/PolyAlign/data/metrics/qwen25_1_5b_eval.json \
  --model-alias qwen25_1_5b \
  --device cuda
```

## Notes

- By default, prediction-side LM features are computed with the same research model alias used for the human feature files.
- MAUVE requires `mauve-text`.
- For BNG, the default reference source is the aligned human test feature file. Pass `--bng-reference-source feature_matrix` to use `feature_matrix.jsonl` instead.
