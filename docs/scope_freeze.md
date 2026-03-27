# Scope Freeze

Frozen on `2026-03-27`.

## Included

- `dolly`
- `ms_marco`
- `coqa`
- `eli5_category`
- `squad_v2`
- `natural_questions`
- `dailydialog`
- `multiwoz`

## Excluded

- `OpenAssistant/oasst2`

## Main Rules

- English-only preprocessing in this stage.
- Keep one formatter per dataset.
- Normalize into one schema with `track`, `family`, `style_bucket`, `length_bin`, and `bucket_id`.
- Do not mix official and local split logic implicitly; every formatter declares its policy in the emitted manifest.

## Dataset-Level Notes

- `dolly`: train-only source release, so it gets a deterministic local 90/5/5 split.
- `ms_marco`: use HF `v1.1`; validation is renamed to `dev`.
- `coqa`: split at the conversation level to prevent turn leakage between train and dev.
- `eli5_category`: preserve `validation2` as an auxiliary robustness split.
- `squad_v2`: preserve unanswerable examples with empty `human_answer` and `meta.is_unanswerable=true`.
- `natural_questions`: this workspace targets the `sentence-transformers/natural-questions` HF variant, which is train-only and passage-like.
- `dailydialog`: projected into next-response examples by alternating speakers from utterance 0.
- `multiwoz`: only `SYSTEM` turns are used as target responses.
