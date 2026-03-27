from __future__ import annotations

from typing import Any

from polyalign_data.text import length_bin_from_count, normalize_text, token_count


def build_bucket_id(language: str, track: str, family: str, length_bin: str) -> str:
    return "|".join([language, track, family, length_bin])


def build_record(
    *,
    example_id: str,
    dataset: str,
    split: str,
    language: str,
    track: str,
    family: str,
    style_bucket: str,
    question: str,
    context: str = "",
    dialogue_history: list[dict[str, str]] | None = None,
    human_answer: str,
    meta: dict[str, Any] | None = None,
) -> dict[str, Any]:
    normalized_answer = normalize_text(human_answer)
    answer_tokens = token_count(normalized_answer)
    length_bin = length_bin_from_count(answer_tokens)
    payload_meta = dict(meta or {})
    payload_meta["length_tokens"] = answer_tokens
    return {
        "id": example_id,
        "dataset": dataset,
        "split": split,
        "language": language,
        "track": track,
        "family": family,
        "style_bucket": style_bucket,
        "length_bin": length_bin,
        "question": normalize_text(question),
        "context": normalize_text(context),
        "dialogue_history": [
            {"role": turn["role"], "text": normalize_text(turn["text"])}
            for turn in (dialogue_history or [])
        ],
        "human_answer": normalized_answer,
        "bucket_id": build_bucket_id(language, track, family, length_bin),
        "meta": payload_meta,
    }
