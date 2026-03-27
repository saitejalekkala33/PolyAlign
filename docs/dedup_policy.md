# Dedup Policy

This file defines the deduplication policy for PolyAlign preprocessing.

The goal is not just to remove repeated rows. The real goal is to prevent:

- exact duplicates inside a dataset
- near-duplicates that inflate counts
- train/dev/test leakage
- cross-dataset leakage into evaluation

## Why This Matters

Without deduplication, the pipeline can overestimate:

- effective training size
- bucket coverage
- naturalness statistics
- downstream evaluation quality

For this project, dedup must be done before final scope-freeze statistics are treated as stable.

## Dedup Levels

### 1. Exact Example Dedup

Remove examples that are exactly the same after normalization.

For single-turn examples, compare:

- `question`
- `context`
- `human_answer`

For multi-turn examples, compare:

- `dialogue_history`
- `question`
- `context`
- `human_answer`

This is the minimum dedup stage and should always be enabled.

### 2. Group-Level Dedup

For conversation datasets, dedup should respect the original source unit.

Use the source group as the protected unit:

- `coqa`: conversation / story
- `dailydialog`: dialogue
- `multiwoz`: dialogue

This prevents related turns from being split across train/dev/test after dedup.

### 3. Cross-Split Leak Prevention

After splitting, no normalized example should appear in more than one split.

This applies both:

- within a dataset
- across merged corpora

If a duplicate appears in multiple splits, keep the example only in the highest-priority split:

1. `train`
2. `dev`
3. `test`

For evaluation protection, a stricter option is preferred:

1. keep `test`
2. keep `dev`
3. keep `train`

For this project, use the stricter evaluation-safe order:

1. `test`
2. `dev`
3. `train`

That means if a train example duplicates a test example, the train copy should be dropped.

### 4. Cross-Dataset Exact Dedup

Dedup should also run across datasets after formatting.

This matters especially for:

- `ms_marco`
- `squad_v2`
- `natural_questions`
- `coqa`
- `eli5_category`

because some prompt-answer pairs or passage-answer pairs can overlap semantically or exactly.

At minimum, exact normalized duplicates across datasets should be removed from:

- `train`
- `dev`

if they are already present in:

- `test`
- `validation2`

### 5. Near-Duplicate Dedup

This is optional for the first pass, but should be planned.

Near-duplicate detection means examples are not string-identical, but are effectively the same.

Recommended future methods:

- normalized edit distance
- MinHash / shingling
- sentence embedding cosine similarity

This is not required for the first clean release, but exact dedup is required.

## Canonicalization Rules

Dedup must never compare raw text directly. It should compare normalized text.

Use the same normalization policy for all datasets:

1. Unicode normalize if needed
2. lowercase
3. strip leading/trailing whitespace
4. collapse repeated internal whitespace
5. normalize line endings
6. remove empty history turns
7. normalize list/dictionary ordering when serializing structured fields

## Canonical Dedup Keys

### Single-Turn Key

```text
key = hash(
  language,
  track,
  family,
  normalize(question),
  normalize(context),
  normalize(human_answer)
)
```

### Multi-Turn Key

```text
key = hash(
  language,
  track,
  family,
  normalize(dialogue_history),
  normalize(question),
  normalize(context),
  normalize(human_answer)
)
```

### Group Key

For grouped datasets, also keep a source-level key:

```text
group_key = source conversation / dialogue / story id
```

## Dataset-Specific Rules

### dolly

- exact dedup on normalized `instruction + context + response`
- no group-level handling needed

### ms_marco

- exact dedup on normalized `query + selected/fallback context + chosen answer`
- preserve `query_id` in metadata for traceability

### coqa

- dedup at two levels:
  - conversation-level protection
  - example-level turn dedup
- never split duplicated turns from the same conversation across splits

### eli5_category

- exact dedup on normalized `title + selftext + best_answer`
- preserve `q_id`
- never let `validation2` leak into `train`

### squad_v2

- exact dedup on normalized `question + context + answer`
- unanswerable examples with empty answer should still be deduped

### natural_questions

- exact dedup on normalized `query + answer`
- because this is a train-only release, dedup must happen before local split assignment

### dailydialog

- protect the full dialogue as the group
- dedup target assistant turns at example level

### multiwoz

- protect the full dialogue as the group
- dedup system turns at example level

## Recommended Order In The Pipeline

Use this order:

1. load raw dataset
2. normalize into unified schema
3. build canonical example key
4. build canonical group key if applicable
5. remove exact duplicates
6. assign splits using group-aware logic
7. run cross-split leak check
8. run cross-dataset evaluation leak check
9. write final JSONL files
10. emit a dedup report

## What The Dedup Report Should Contain

Every run should save a report with:

- original count
- exact duplicates removed
- cross-split duplicates removed
- cross-dataset train/test overlaps removed
- final count
- duplicate rate
- counts by dataset
- counts by split

## Minimum First Implementation

The first implementation should do these three things:

1. exact dedup inside each dataset after normalization
2. group-aware split protection for conversation datasets
3. cross-split and cross-dataset exact leak removal with `test > dev > train` priority

That is enough for a first paper-safe preprocessing release.

## Future Extensions

- near-duplicate detection with MinHash or embeddings
- semantic overlap reports
- separate dedup policies for optimization vs reference corpora
- bucket-aware duplicate analysis
