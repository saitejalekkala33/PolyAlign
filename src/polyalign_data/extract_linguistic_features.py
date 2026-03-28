from __future__ import annotations

import argparse
import csv
import json
import math
import re
from collections import Counter
from pathlib import Path
from statistics import pstdev
from typing import Any

from nltk.corpus import stopwords
from nltk.sentiment import SentimentIntensityAnalyzer
from nltk.tag import pos_tag
from nltk.tag.mapping import map_tag
from nltk.tokenize import sent_tokenize, word_tokenize
from tqdm.auto import tqdm

from polyalign_data.io_utils import ensure_dir
from polyalign_data.lm_registry import ResearchModelSpec, model_aliases, resolve_model_aliases
from polyalign_data.lm_scoring import (
    DEFAULT_LM_MAX_SEQ_LENGTH,
    HFLMFeatureExtractor,
    derive_model_output_paths,
)
from polyalign_data.text import normalize_text


WORD_RE = re.compile(r"\b[\w']+\b", re.UNICODE)
ALPHA_RE = re.compile(r"[A-Za-z]+(?:'[A-Za-z]+)?")
URL_RE = re.compile(r"https?://\S+|www\.\S+", re.IGNORECASE)
EMAIL_RE = re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.-]+\b")
YEAR_RE = re.compile(r"\b(?:1[5-9]\d{2}|20\d{2}|2100)\b")
ORDINAL_RE = re.compile(r"\b\d+(?:st|nd|rd|th)\b", re.IGNORECASE)
LIST_BULLET_RE = re.compile(r"^\s*[-*•]\s+", re.MULTILINE)
ENUM_LINE_RE = re.compile(r"^\s*(?:\d+|[A-Za-z])[.)]\s+", re.MULTILINE)
ELLIPSIS_RE = re.compile(r"\.\.\.|…")
REPEATED_CHAR_RE = re.compile(r"(.)\1{2,}")

STOPWORDS = frozenset(stopwords.words("english"))
VADER = SentimentIntensityAnalyzer()


UNIVERSAL_TAG_FEATURES = (
    "ADJ",
    "ADP",
    "ADV",
    "CONJ",
    "DET",
    "NOUN",
    "NUM",
    "PRON",
    "PRT",
    "VERB",
    "X",
)

PTB_TAG_FEATURES = (
    "MD",
    "NNP",
    "NNPS",
    "VB",
    "VBD",
    "VBG",
    "VBN",
    "VBP",
    "VBZ",
    "WDT",
    "WP",
    "WP$",
    "WRB",
)


def _safe_div(numerator: float, denominator: float) -> float:
    if denominator == 0:
        return 0.0
    return numerator / denominator


def _round(value: float | int) -> float | int:
    if isinstance(value, float):
        return round(value, 6)
    return value


def _tokenize_words(text: str) -> list[str]:
    return WORD_RE.findall(text)


def _alphabetic_tokens(text: str) -> list[str]:
    return ALPHA_RE.findall(text)


def _split_sentences(text: str) -> list[str]:
    normalized = normalize_text(text)
    if not normalized:
        return []
    try:
        sentences = [sentence.strip() for sentence in sent_tokenize(normalized) if sentence.strip()]
    except LookupError:
        sentences = []
    if sentences:
        return sentences
    parts = re.split(r"(?<=[.!?])\s+|\n+", normalized)
    return [part.strip() for part in parts if part.strip()] or [normalized]


def _word_tokenize_safe(text: str) -> list[str]:
    try:
        return word_tokenize(text)
    except LookupError:
        return _tokenize_words(text)


def _pos_tag_safe(tokens: list[str]) -> list[tuple[str, str]]:
    if not tokens:
        return []
    try:
        return pos_tag(tokens)
    except LookupError:
        return [(token, "X") for token in tokens]


def _count_syllables(word: str) -> int:
    cleaned = re.sub(r"[^a-z]", "", word.lower())
    if not cleaned:
        return 0
    if len(cleaned) <= 3:
        return 1
    vowels = "aeiouy"
    count = 0
    previous_is_vowel = False
    for char in cleaned:
        is_vowel = char in vowels
        if is_vowel and not previous_is_vowel:
            count += 1
        previous_is_vowel = is_vowel
    if cleaned.endswith("e") and not cleaned.endswith(("le", "ye")) and count > 1:
        count -= 1
    return max(1, count)


def _shannon_entropy(counter: Counter[str], total: int) -> float:
    if total == 0:
        return 0.0
    entropy = 0.0
    for count in counter.values():
        probability = count / total
        entropy -= probability * math.log2(probability)
    return entropy


def _universal_pos_counts(tagged_tokens: list[tuple[str, str]]) -> Counter[str]:
    counts: Counter[str] = Counter()
    for _token, ptb_tag in tagged_tokens:
        try:
            universal = map_tag("en-ptb", "universal", ptb_tag)
        except Exception:
            universal = "X"
        counts[universal] += 1
    return counts


def extract_linguistic_features(text: str) -> dict[str, Any]:
    normalized = normalize_text(text)
    chars = len(normalized)
    chars_no_space = len(normalized.replace(" ", "").replace("\n", ""))
    whitespace_count = sum(1 for char in normalized if char.isspace())
    punctuation_count = sum(1 for char in normalized if re.match(r"[^\w\s]", char))
    digit_char_count = sum(1 for char in normalized if char.isdigit())
    alphabetic_char_count = sum(1 for char in normalized if char.isalpha())
    uppercase_char_count = sum(1 for char in normalized if char.isupper())
    lowercase_char_count = sum(1 for char in normalized if char.islower())
    non_ascii_char_count = sum(1 for char in normalized if ord(char) > 127)
    newline_count = normalized.count("\n")
    paragraph_count = len([chunk for chunk in normalized.split("\n\n") if chunk.strip()]) if normalized else 0

    sentences = _split_sentences(normalized)
    sentence_char_lengths = [len(sentence) for sentence in sentences]
    word_tokens = _word_tokenize_safe(normalized)
    alpha_tokens = _alphabetic_tokens(normalized)
    alpha_lower = [token.lower() for token in alpha_tokens]
    sentence_tokens = [_word_tokenize_safe(sentence) for sentence in sentences]
    sentence_lengths = [len(tokens) for tokens in sentence_tokens if tokens]
    tagged_tokens = _pos_tag_safe(word_tokens)
    pos_counts = _universal_pos_counts(tagged_tokens)
    ptb_counts = Counter(tag for _token, tag in tagged_tokens)

    word_count = len(alpha_lower)
    token_count = len(word_tokens)
    sentence_count = len(sentences)
    unique_count = len(set(alpha_lower))
    token_counter = Counter(alpha_lower)
    hapax_count = sum(1 for value in token_counter.values() if value == 1)
    dislegomena_count = sum(1 for value in token_counter.values() if value == 2)
    bigrams = list(zip(alpha_lower, alpha_lower[1:]))
    trigrams = list(zip(alpha_lower, alpha_lower[1:], alpha_lower[2:]))
    bigram_counter = Counter(bigrams)
    syllables = [_count_syllables(token) for token in alpha_lower]
    total_syllables = sum(syllables)
    multisyllabic_word_count = sum(1 for count in syllables if count >= 3)
    avg_word_length = _safe_div(sum(len(token) for token in alpha_lower), word_count)
    avg_syllables_per_word = _safe_div(total_syllables, word_count)

    words_per_sentence = _safe_div(word_count, sentence_count)
    flesch_reading_ease = 206.835 - (1.015 * words_per_sentence) - (84.6 * avg_syllables_per_word) if word_count else 0.0
    flesch_kincaid_grade = (0.39 * words_per_sentence) + (11.8 * avg_syllables_per_word) - 15.59 if word_count else 0.0
    gunning_fog = 0.4 * (words_per_sentence + 100 * _safe_div(multisyllabic_word_count, word_count)) if word_count else 0.0
    smog_index = (
        1.043 * math.sqrt(multisyllabic_word_count * (30 / sentence_count)) + 3.1291
        if sentence_count and multisyllabic_word_count
        else 0.0
    )
    coleman_liau = (
        0.0588 * (_safe_div(alphabetic_char_count, word_count) * 100)
        - 0.296 * (_safe_div(sentence_count, word_count) * 100)
        - 15.8
        if word_count
        else 0.0
    )
    automated_readability = (
        4.71 * _safe_div(chars_no_space, word_count) + 0.5 * words_per_sentence - 21.43
        if word_count
        else 0.0
    )

    stopword_count = sum(1 for token in alpha_lower if token in STOPWORDS)
    content_word_count = sum(pos_counts[tag] for tag in ("ADJ", "ADV", "NOUN", "VERB"))
    sentiment = VADER.polarity_scores(normalized) if normalized else {"neg": 0.0, "neu": 0.0, "pos": 0.0, "compound": 0.0}

    question_mark_count = normalized.count("?")
    exclamation_mark_count = normalized.count("!")
    comma_count = normalized.count(",")
    semicolon_count = normalized.count(";")
    colon_count = normalized.count(":")
    period_count = normalized.count(".")
    ellipsis_count = len(ELLIPSIS_RE.findall(normalized))
    quote_count = normalized.count('"') + normalized.count("“") + normalized.count("”") + normalized.count("‘") + normalized.count("’")
    apostrophe_count = normalized.count("'")
    hyphen_count = normalized.count("-") + normalized.count("–") + normalized.count("—")
    slash_count = normalized.count("/")
    parenthesis_count = normalized.count("(") + normalized.count(")")
    bracket_count = normalized.count("[") + normalized.count("]") + normalized.count("{") + normalized.count("}")
    sentences_with_question = sum(1 for sentence in sentences if sentence.endswith("?"))
    sentences_with_exclamation = sum(1 for sentence in sentences if sentence.endswith("!"))
    sentences_with_terminal = sum(1 for sentence in sentences if sentence and sentence[-1] in ".!?")
    sentence_initial_capitals = sum(1 for sentence in sentences if sentence[:1].isupper())

    all_caps_token_count = sum(1 for token in word_tokens if token.isupper() and len(token) > 1)
    title_case_token_count = sum(1 for token in word_tokens if token[:1].isupper() and token[1:].islower())
    lowercase_token_count = sum(1 for token in word_tokens if token.islower())
    contraction_count = sum(1 for token in word_tokens if "'" in token)
    number_token_count = sum(1 for token in word_tokens if any(char.isdigit() for char in token))
    year_token_count = len(YEAR_RE.findall(normalized))
    ordinal_token_count = len(ORDINAL_RE.findall(normalized))
    url_count = len(URL_RE.findall(normalized))
    email_count = len(EMAIL_RE.findall(normalized))

    short_word_count = sum(1 for token in alpha_lower if len(token) <= 3)
    medium_word_count = sum(1 for token in alpha_lower if 4 <= len(token) <= 6)
    long_word_count = sum(1 for token in alpha_lower if len(token) >= 7)
    very_long_word_count = sum(1 for token in alpha_lower if len(token) >= 11)

    bullet_line_count = len(LIST_BULLET_RE.findall(normalized))
    enumerated_line_count = len(ENUM_LINE_RE.findall(normalized))
    code_fence_count = normalized.count("```")
    inline_code_marker_count = normalized.count("`")
    repeated_char_sequence_count = len(REPEATED_CHAR_RE.findall(normalized))

    features: dict[str, Any] = {
        "char_count": chars,
        "char_no_space_count": chars_no_space,
        "whitespace_count": whitespace_count,
        "newline_count": newline_count,
        "paragraph_count": paragraph_count,
        "punctuation_count": punctuation_count,
        "digit_char_count": digit_char_count,
        "alphabetic_char_count": alphabetic_char_count,
        "uppercase_char_count": uppercase_char_count,
        "lowercase_char_count": lowercase_char_count,
        "non_ascii_char_count": non_ascii_char_count,
        "token_count": token_count,
        "word_token_count": word_count,
        "unique_word_count": unique_count,
        "sentence_count": sentence_count,
        "avg_char_per_token": _safe_div(chars_no_space, token_count),
        "avg_char_per_sentence": _safe_div(chars, sentence_count),
        "avg_word_length": avg_word_length,
        "avg_tokens_per_sentence": _safe_div(sum(sentence_lengths), len(sentence_lengths)),
        "min_tokens_per_sentence": min(sentence_lengths) if sentence_lengths else 0,
        "max_tokens_per_sentence": max(sentence_lengths) if sentence_lengths else 0,
        "std_tokens_per_sentence": pstdev(sentence_lengths) if len(sentence_lengths) > 1 else 0.0,
        "avg_sentence_char_length": _safe_div(sum(sentence_char_lengths), len(sentence_char_lengths)),
        "min_sentence_char_length": min(sentence_char_lengths) if sentence_char_lengths else 0,
        "max_sentence_char_length": max(sentence_char_lengths) if sentence_char_lengths else 0,
        "std_sentence_char_length": pstdev(sentence_char_lengths) if len(sentence_char_lengths) > 1 else 0.0,
        "short_word_count": short_word_count,
        "medium_word_count": medium_word_count,
        "long_word_count": long_word_count,
        "very_long_word_count": very_long_word_count,
        "short_word_ratio": _safe_div(short_word_count, word_count),
        "long_word_ratio": _safe_div(long_word_count, word_count),
        "very_long_word_ratio": _safe_div(very_long_word_count, word_count),
        "type_token_ratio": _safe_div(unique_count, word_count),
        "root_ttr": _safe_div(unique_count, math.sqrt(word_count)) if word_count else 0.0,
        "corrected_ttr": _safe_div(unique_count, math.sqrt(2 * word_count)) if word_count else 0.0,
        "hapax_ratio": _safe_div(hapax_count, word_count),
        "dislegomena_ratio": _safe_div(dislegomena_count, word_count),
        "distinct_1": _safe_div(len(set(alpha_lower)), word_count),
        "distinct_2": _safe_div(len(set(bigrams)), len(bigrams)),
        "distinct_3": _safe_div(len(set(trigrams)), len(trigrams)),
        "repeated_token_ratio": _safe_div(sum(value - 1 for value in token_counter.values() if value > 1), word_count),
        "repeated_bigram_ratio": _safe_div(sum(value - 1 for value in bigram_counter.values() if value > 1), len(bigrams)),
        "adjacent_repeat_count": sum(1 for left, right in zip(alpha_lower, alpha_lower[1:]) if left == right),
        "max_token_frequency": max(token_counter.values(), default=0),
        "max_token_frequency_ratio": _safe_div(max(token_counter.values(), default=0), word_count),
        "shannon_entropy": _shannon_entropy(token_counter, word_count),
        "syllable_count": total_syllables,
        "avg_syllables_per_word": avg_syllables_per_word,
        "multisyllabic_word_count": multisyllabic_word_count,
        "multisyllabic_word_ratio": _safe_div(multisyllabic_word_count, word_count),
        "flesch_reading_ease": flesch_reading_ease,
        "flesch_kincaid_grade": flesch_kincaid_grade,
        "gunning_fog": gunning_fog,
        "smog_index": smog_index,
        "coleman_liau": coleman_liau,
        "automated_readability_index": automated_readability,
        "question_mark_count": question_mark_count,
        "exclamation_mark_count": exclamation_mark_count,
        "comma_count": comma_count,
        "semicolon_count": semicolon_count,
        "colon_count": colon_count,
        "period_count": period_count,
        "ellipsis_count": ellipsis_count,
        "quote_count": quote_count,
        "apostrophe_count": apostrophe_count,
        "hyphen_count": hyphen_count,
        "slash_count": slash_count,
        "parenthesis_count": parenthesis_count,
        "bracket_count": bracket_count,
        "punctuation_density": _safe_div(punctuation_count, token_count),
        "question_sentence_ratio": _safe_div(sentences_with_question, sentence_count),
        "exclam_sentence_ratio": _safe_div(sentences_with_exclamation, sentence_count),
        "terminal_punctuation_ratio": _safe_div(sentences_with_terminal, sentence_count),
        "sentence_initial_capital_ratio": _safe_div(sentence_initial_capitals, sentence_count),
        "all_caps_token_count": all_caps_token_count,
        "title_case_token_count": title_case_token_count,
        "lowercase_token_count": lowercase_token_count,
        "all_caps_token_ratio": _safe_div(all_caps_token_count, token_count),
        "title_case_token_ratio": _safe_div(title_case_token_count, token_count),
        "contraction_count": contraction_count,
        "number_token_count": number_token_count,
        "year_token_count": year_token_count,
        "ordinal_token_count": ordinal_token_count,
        "url_count": url_count,
        "email_count": email_count,
        "stopword_count": stopword_count,
        "stopword_ratio": _safe_div(stopword_count, word_count),
        "content_word_count": content_word_count,
        "content_word_ratio": _safe_div(content_word_count, token_count),
        "lexical_density": _safe_div(content_word_count, max(1, stopword_count + content_word_count)),
        "modal_count": ptb_counts["MD"],
        "modal_ratio": _safe_div(ptb_counts["MD"], token_count),
        "wh_word_count": sum(ptb_counts[tag] for tag in ("WDT", "WP", "WP$", "WRB")),
        "wh_word_ratio": _safe_div(sum(ptb_counts[tag] for tag in ("WDT", "WP", "WP$", "WRB")), token_count),
        "proper_noun_count": ptb_counts["NNP"] + ptb_counts["NNPS"],
        "proper_noun_ratio": _safe_div(ptb_counts["NNP"] + ptb_counts["NNPS"], token_count),
        "past_verb_count": ptb_counts["VBD"],
        "gerund_verb_count": ptb_counts["VBG"],
        "past_participle_count": ptb_counts["VBN"],
        "non_3sg_present_verb_count": ptb_counts["VBP"],
        "third_person_present_verb_count": ptb_counts["VBZ"],
        "base_verb_count": ptb_counts["VB"],
        "vader_neg": sentiment["neg"],
        "vader_neu": sentiment["neu"],
        "vader_pos": sentiment["pos"],
        "vader_compound": sentiment["compound"],
        "bullet_line_count": bullet_line_count,
        "enumerated_line_count": enumerated_line_count,
        "code_fence_count": code_fence_count,
        "inline_code_marker_count": inline_code_marker_count,
        "repeated_char_sequence_count": repeated_char_sequence_count,
    }

    for tag in UNIVERSAL_TAG_FEATURES:
        lower = tag.lower()
        features[f"pos_{lower}_count"] = pos_counts[tag]
        features[f"pos_{lower}_ratio"] = _safe_div(pos_counts[tag], token_count)

    return {name: _round(value) for name, value in features.items()}


def _iter_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                yield json.loads(line)


def _count_jsonl_rows(path: Path) -> int:
    count = 0
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                count += 1
    return count


def _write_feature_row(
    jsonl_handle,
    csv_writer: csv.DictWriter | None,
    row: dict[str, Any],
) -> None:
    jsonl_handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    if csv_writer is None:
        return
    flat_row = {
        "id": row.get("id", ""),
        "dataset": row.get("dataset", ""),
        "split": row.get("split", ""),
        "field_name": row.get("field_name", ""),
    }
    if "model_alias" in row:
        flat_row["model_alias"] = row["model_alias"]
        flat_row["model_name"] = row["model_name"]
    flat_row.update(row["features"])
    csv_writer.writerow(flat_row)


def _open_csv_writer(
    csv_path: Path | None,
    feature_names: list[str],
    *,
    include_model_columns: bool = False,
):
    if csv_path is None:
        return None, None
    ensure_dir(csv_path.parent)
    handle = csv_path.open("w", encoding="utf-8", newline="")
    fieldnames = ["id", "dataset", "split", "field_name"]
    if include_model_columns:
        fieldnames.extend(["model_alias", "model_name"])
    fieldnames.extend(feature_names)
    writer = csv.DictWriter(handle, fieldnames=fieldnames)
    writer.writeheader()
    return handle, writer


def extract_features_file(
    input_path: str | Path,
    output_jsonl: str | Path,
    *,
    text_field: str = "human_answer",
    output_csv: str | Path | None = None,
    include_text: bool = False,
) -> dict[str, Any]:
    input_file = Path(input_path)
    output_file = Path(output_jsonl)
    ensure_dir(output_file.parent)
    total_rows = _count_jsonl_rows(input_file)

    rows: list[dict[str, Any]] = []
    progress = tqdm(
        _iter_jsonl(input_file),
        total=total_rows,
        desc=f"text-features:{input_file.stem}",
        unit="row",
    )
    for record in progress:
        text_value = normalize_text(record.get(text_field, ""))
        feature_row = {
            "id": record.get("id", ""),
            "dataset": record.get("dataset", ""),
            "split": record.get("split", ""),
            "field_name": text_field,
            "features": extract_linguistic_features(text_value),
        }
        if include_text:
            feature_row["text"] = text_value
        rows.append(feature_row)

    with output_file.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    if output_csv:
        csv_path = Path(output_csv)
        ensure_dir(csv_path.parent)
        feature_names = sorted(rows[0]["features"].keys()) if rows else []
        with csv_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=["id", "dataset", "split", "field_name", *feature_names])
            writer.writeheader()
            for row in rows:
                flat_row = {
                    "id": row["id"],
                    "dataset": row["dataset"],
                    "split": row["split"],
                    "field_name": row["field_name"],
                }
                flat_row.update(row["features"])
                writer.writerow(flat_row)

    return {
        "input_path": str(input_file),
        "output_jsonl": str(output_file),
        "output_csv": str(output_csv) if output_csv else None,
        "records": len(rows),
        "feature_count": len(rows[0]["features"]) if rows else 0,
        "text_field": text_field,
    }


def extract_model_features_file(
    input_path: str | Path,
    output_jsonl: str | Path,
    *,
    model_spec: ResearchModelSpec,
    text_field: str = "human_answer",
    output_csv: str | Path | None = None,
    include_text: bool = False,
    device: str = "auto",
    dtype: str = "auto",
    max_seq_length: int = DEFAULT_LM_MAX_SEQ_LENGTH,
    trust_remote_code: bool = False,
) -> dict[str, Any]:
    input_file = Path(input_path)
    output_file = Path(output_jsonl)
    ensure_dir(output_file.parent)
    total_rows = _count_jsonl_rows(input_file)

    extractor = HFLMFeatureExtractor(
        model_spec,
        device=device,
        dtype=dtype,
        max_seq_length=max_seq_length,
        trust_remote_code=trust_remote_code,
    )

    rows = 0
    feature_count = 0
    csv_handle = None
    csv_writer = None

    try:
        with output_file.open("w", encoding="utf-8") as jsonl_handle:
            progress = tqdm(
                _iter_jsonl(input_file),
                total=total_rows,
                desc=f"lm-features:{model_spec.alias}:{input_file.stem}",
                unit="row",
            )
            for record in progress:
                text_value = normalize_text(record.get(text_field, ""))
                features = extract_linguistic_features(text_value)
                features.update(extractor.score_record(record, text_field=text_field))
                feature_count = feature_count or len(features)
                if output_csv and csv_writer is None:
                    csv_path = Path(output_csv)
                    csv_handle, csv_writer = _open_csv_writer(
                        csv_path,
                        sorted(features.keys()),
                        include_model_columns=True,
                    )
                feature_row = {
                    "id": record.get("id", ""),
                    "dataset": record.get("dataset", ""),
                    "split": record.get("split", ""),
                    "field_name": text_field,
                    "model_alias": model_spec.alias,
                    "model_name": model_spec.model_id,
                    "features": features,
                }
                if include_text:
                    feature_row["text"] = text_value
                _write_feature_row(jsonl_handle, csv_writer, feature_row)
                rows += 1
    finally:
        if csv_handle is not None:
            csv_handle.close()

    return {
        "input_path": str(input_file),
        "output_jsonl": str(output_file),
        "output_csv": str(output_csv) if output_csv else None,
        "records": rows,
        "feature_count": feature_count,
        "text_field": text_field,
        "model_alias": model_spec.alias,
        "model_name": model_spec.model_id,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Extract text-based and optional model-scored answer features from a merged PolyAlign JSONL file."
    )
    parser.add_argument("--input-path", help="Input JSONL file in the PolyAlign current-format schema.")
    parser.add_argument("--output-jsonl", help="Output JSONL path for text-only extracted features.")
    parser.add_argument(
        "--text-field",
        default="human_answer",
        help="Field name to featurize. For the current schema this is usually `human_answer`.",
    )
    parser.add_argument("--output-csv", help="Optional wide CSV export with one row per example and one column per feature.")
    parser.add_argument("--include-text", action="store_true", help="Include the featurized text in the output JSONL.")
    parser.add_argument(
        "--lm-model",
        action="append",
        choices=["all", *model_aliases()],
        help="Optional research LM alias to score with. Repeat to emit per-model outputs.",
    )
    parser.add_argument(
        "--lm-output-root",
        help="Output root for per-model feature files. Each model alias gets its own subdirectory.",
    )
    parser.add_argument(
        "--lm-write-csv",
        action="store_true",
        help="Also write one wide CSV per LM-specific output directory.",
    )
    parser.add_argument(
        "--lm-device",
        default="auto",
        help="Execution device for Hugging Face LM scoring. Use `auto`, `cpu`, `cuda`, or a torch device string.",
    )
    parser.add_argument(
        "--lm-dtype",
        default="auto",
        choices=["auto", "float16", "bfloat16", "float32"],
        help="Torch dtype for LM scoring.",
    )
    parser.add_argument(
        "--lm-max-seq-length",
        type=int,
        default=DEFAULT_LM_MAX_SEQ_LENGTH,
        help="Maximum sequence length used for LM scoring. Prefix tokens are truncated from the left first.",
    )
    parser.add_argument(
        "--lm-trust-remote-code",
        action="store_true",
        help="Pass trust_remote_code=True when loading Hugging Face models and tokenizers.",
    )
    parser.add_argument(
        "--list-lm-models",
        action="store_true",
        help="Print the supported LM aliases and exit.",
    )
    args = parser.parse_args()
    if args.list_lm_models:
        print(
            json.dumps(
                {alias: {"model_id": resolve_model_aliases([alias])[0].model_id} for alias in model_aliases()},
                indent=2,
                ensure_ascii=False,
            )
        )
        return

    if not args.input_path:
        parser.error("--input-path is required unless --list-lm-models is used.")
    if not args.output_jsonl and not args.lm_model:
        parser.error("Provide --output-jsonl for text-only features, or use --lm-model with --lm-output-root.")
    if args.lm_model and not args.lm_output_root:
        parser.error("--lm-output-root is required when --lm-model is used.")

    summaries: dict[str, Any] = {"input_path": args.input_path, "text_field": args.text_field}
    input_file = Path(args.input_path)

    if args.output_jsonl:
        summaries["text_features"] = extract_features_file(
            args.input_path,
            args.output_jsonl,
            text_field=args.text_field,
            output_csv=args.output_csv,
            include_text=args.include_text,
        )

    if args.lm_model:
        base_jsonl_name = Path(args.output_jsonl).name if args.output_jsonl else None
        base_csv_name = Path(args.output_csv).name if args.output_csv else None
        model_summaries = []
        for model_spec in resolve_model_aliases(args.lm_model):
            model_output_jsonl, model_output_csv = derive_model_output_paths(
                input_file,
                output_root=Path(args.lm_output_root),
                model_alias=model_spec.alias,
                text_field=args.text_field,
                base_jsonl_name=base_jsonl_name,
                write_csv=args.lm_write_csv or bool(args.output_csv),
                base_csv_name=base_csv_name,
            )
            model_summaries.append(
                extract_model_features_file(
                    args.input_path,
                    model_output_jsonl,
                    model_spec=model_spec,
                    text_field=args.text_field,
                    output_csv=model_output_csv,
                    include_text=args.include_text,
                    device=args.lm_device,
                    dtype=args.lm_dtype,
                    max_seq_length=args.lm_max_seq_length,
                    trust_remote_code=args.lm_trust_remote_code,
                )
            )
        summaries["lm_features"] = model_summaries

    print(json.dumps(summaries, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
