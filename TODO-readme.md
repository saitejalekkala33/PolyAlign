# TODO

## Finish Feature Extraction

- Add prompt-conditioned interaction features that are not in the current answer-only extractor:
  - prompt-answer lexical overlap
  - prompt-answer accommodation / alignment
  - history-answer accommodation for multi-turn data
  - answer-to-previous-turn length ratio
- Add deeper syntactic features beyond the current shallow POS/readability coverage:
  - dependency depth
  - clause density
  - subordinate clause rate
  - tree-complexity features
- Add semantic / representation features:
  - embedding-based semantic features
  - prompt-answer semantic similarity
  - bucket-level semantic prototype distance
- Export paper-ready reference-builder artifacts from extracted features:
  - per-example feature matrix
  - bucket prototypes
  - q10-q90 support regions
  - q25-q75 support regions

## Add LM-Based Features

- Add a separate `extract_lm_features.py` pipeline for model-based scoring features that are not covered by the current text-only linguistic extractor.
- Include unconditional answer scoring features:
  - perplexity
  - negative log-likelihood
  - mean token surprisal
  - max token surprisal
  - surprisal variance / standard deviation
- Include conditional scoring features:
  - `P(answer | question + context)` for single-turn examples
  - `P(answer | dialogue_history + question + context)` for multi-turn examples
  - conditional perplexity
  - conditional negative log-likelihood
  - prompt-to-answer log-probability gap
- Keep the implementation separate from `extract_linguistic_features.py` so text-native features and LM-native features remain cleanly separated.
- Decide and document the scoring backbone:
  - base LM
  - vanilla SFT model
  - both, if comparative scoring is needed
- For multi-turn data, score only the target answer tokens while conditioning on the full prior interaction context.
- Save outputs in both:
  - JSONL with per-example feature dictionaries
  - CSV with one row per example
