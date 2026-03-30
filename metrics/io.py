from __future__ import annotations

import json
import string
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable


ARTICLES = {"a", "an", "the"}
PUNCT_TABLE = str.maketrans("", "", string.punctuation)


@dataclass(frozen=True)
class EvaluationPaths:
    test_lf_path: Path
    predictions_path: Path
    current_test_path: Path
    human_feature_path: Path | None
    bucket_references_path: Path
    feature_matrix_path: Path | None
    output_json_path: Path
    work_dir: Path
    model_alias: str


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def iter_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                yield json.loads(line)


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return list(iter_jsonl(path))


def load_concatenated_json(path: Path) -> list[dict[str, Any]]:
    text = path.read_text(encoding="utf-8")
    decoder = json.JSONDecoder()
    rows: list[dict[str, Any]] = []
    index = 0
    text_length = len(text)
    while index < text_length:
        while index < text_length and text[index].isspace():
            index += 1
        if index >= text_length:
            break
        obj, end = decoder.raw_decode(text, index)
        if not isinstance(obj, dict):
            raise ValueError(f"Expected JSON object in {path}, found {type(obj).__name__}.")
        rows.append(obj)
        index = end
    return rows


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def normalize_text(text: Any) -> str:
    if text is None:
        return ""
    raw = str(text).replace("\r\n", "\n").replace("\r", "\n").strip()
    normalized_lines = [" ".join(line.strip().split()) for line in raw.split("\n")]
    return "\n".join(normalized_lines).strip()


def normalize_answer(text: Any) -> str:
    value = normalize_text(text).lower().translate(PUNCT_TABLE)
    tokens = [token for token in value.split() if token not in ARTICLES]
    return " ".join(tokens)


def flatten_dialogue_history(dialogue_history: list[dict[str, str]] | None) -> list[list[str]]:
    pairs: list[list[str]] = []
    current_user: str | None = None
    for turn in dialogue_history or []:
        role = turn.get("role")
        text = turn.get("text", "")
        if role == "user":
            current_user = text
        elif role == "assistant" and current_user is not None:
            pairs.append([current_user, text])
            current_user = None
    return pairs


def find_data_root(*paths: Path) -> Path | None:
    for path in paths:
        current = path if path.is_dir() else path.parent
        for candidate in [current, *current.parents]:
            if candidate.name == "data":
                return candidate
    return None


def resolve_model_alias(model_alias: str | None, predictions_path: Path) -> str:
    if model_alias:
        return model_alias
    candidate = predictions_path.parent.name
    return candidate or "qwen25_1_5b"


def resolve_path(explicit: str | Path | None, candidate: Path | None) -> Path | None:
    if explicit:
        return Path(explicit)
    return candidate


def resolve_evaluation_paths(
    *,
    test_lf_path: str | Path,
    predictions_path: str | Path,
    output_json_path: str | Path,
    current_test_path: str | Path | None = None,
    human_feature_path: str | Path | None = None,
    bucket_references_path: str | Path | None = None,
    feature_matrix_path: str | Path | None = None,
    work_dir: str | Path | None = None,
    model_alias: str | None = None,
) -> EvaluationPaths:
    test_lf = Path(test_lf_path)
    predictions = Path(predictions_path)
    output_json = Path(output_json_path)
    resolved_model_alias = resolve_model_alias(model_alias, predictions)
    data_root = find_data_root(test_lf, predictions)

    current_candidate = data_root / "merged_sft_dedup" / "current" / "test.jsonl" if data_root else None
    human_feature_candidate = (
        data_root / "features" / "research_models" / "test" / resolved_model_alias / "test_answer_features_dedup.jsonl"
        if data_root
        else None
    )
    bucket_refs_candidate = data_root / "reference_artifacts" / "bucket_references.json" if data_root else None
    feature_matrix_candidate = data_root / "reference_artifacts" / "feature_matrix.jsonl" if data_root else None
    work_dir_candidate = output_json.parent / f"{output_json.stem}_artifacts"

    resolved_current = resolve_path(current_test_path, current_candidate)
    resolved_bucket_refs = resolve_path(bucket_references_path, bucket_refs_candidate)
    resolved_human_features = resolve_path(human_feature_path, human_feature_candidate)
    if resolved_human_features is not None and not resolved_human_features.exists():
        csv_fallback = resolved_human_features.with_suffix(".csv")
        if csv_fallback.exists():
            resolved_human_features = csv_fallback
    resolved_feature_matrix = resolve_path(feature_matrix_path, feature_matrix_candidate)
    resolved_work_dir = Path(work_dir) if work_dir else work_dir_candidate

    if resolved_current is None:
        raise ValueError("Could not resolve current test JSONL. Pass --current-test-path explicitly.")
    if resolved_bucket_refs is None:
        raise ValueError("Could not resolve bucket_references.json. Pass --bucket-references-path explicitly.")

    return EvaluationPaths(
        test_lf_path=test_lf,
        predictions_path=predictions,
        current_test_path=resolved_current,
        human_feature_path=resolved_human_features,
        bucket_references_path=resolved_bucket_refs,
        feature_matrix_path=resolved_feature_matrix,
        output_json_path=output_json,
        work_dir=resolved_work_dir,
        model_alias=resolved_model_alias,
    )
