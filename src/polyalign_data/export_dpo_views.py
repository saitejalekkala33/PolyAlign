from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from polyalign_data.io_utils import ensure_dir
from polyalign_data.text import normalize_text


DEFAULT_PREDICTION_FILENAME = "predictions.jsonl"
SUPPORTED_SPLITS = ("train", "val", "test")
REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATASET_INFO_PATH = REPO_ROOT / "vendor" / "LlamaFactory" / "data" / "dataset_info.json"


def log_step(message: str) -> None:
    print(f"[export_dpo] {message}", flush=True)


def _decode_json_values(text: str, *, path: Path, line_number: int | None = None) -> list[dict[str, Any]]:
    decoder = json.JSONDecoder()
    values: list[dict[str, Any]] = []
    index = 0
    text_length = len(text)
    while index < text_length:
        while index < text_length and text[index].isspace():
            index += 1
        if index >= text_length:
            break

        try:
            value, next_index = decoder.raw_decode(text, index)
        except json.JSONDecodeError as exc:
            location = f"{path}"
            if line_number is not None:
                location += f" (line {line_number})"
            raise ValueError(f"Failed to decode JSON from {location}: {exc}") from exc

        if not isinstance(value, dict):
            location = f"{path}"
            if line_number is not None:
                location += f" (line {line_number})"
            raise ValueError(f"Expected JSON object in {location}, got {type(value).__name__}.")

        values.append(value)
        index = next_index

    return values


def _read_records(path: Path) -> list[dict[str, Any]]:
    if path.suffix.lower() == ".json":
        payload = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(payload, list):
            raise ValueError(f"Expected a JSON array in {path}.")
        return [dict(item) for item in payload]

    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            line = raw_line.strip()
            if line:
                parsed_values = _decode_json_values(line, path=path, line_number=line_number)
                if len(parsed_values) > 1:
                    log_step(f"detected {len(parsed_values)} concatenated JSON objects in {path.name} line {line_number}")
                records.extend(parsed_values)
    return records


def _resolve_split_record_path(records_root: Path, split_name: str) -> Path:
    candidates = [
        records_root / f"{split_name}.jsonl",
        records_root / f"{split_name}.json",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate

    raise FileNotFoundError(
        f"Could not find a `{split_name}.jsonl` or `{split_name}.json` file under {records_root}."
    )


def _resolve_prediction_path(path_or_dir: str | Path, prediction_filename: str) -> Path:
    candidate = Path(path_or_dir)
    if candidate.is_dir():
        candidate = candidate / prediction_filename
    if not candidate.exists():
        raise FileNotFoundError(f"Could not find predictions at {candidate}.")
    return candidate


def _slugify_name(value: str) -> str:
    normalized = normalize_text(value).lower()
    safe = normalized
    for source, target in (
        ("\\", "_"),
        ("/", "_"),
        (":", "_"),
        (" ", "_"),
        ("-", "_"),
    ):
        safe = safe.replace(source, target)
    while "__" in safe:
        safe = safe.replace("__", "_")
    return safe.strip("_")


def _normalize_history(history: Any) -> list[dict[str, str]]:
    if not isinstance(history, list):
        return []

    normalized: list[dict[str, str]] = []
    for turn in history:
        if isinstance(turn, dict):
            role = normalize_text(turn.get("role", ""))
            text = normalize_text(turn.get("text", ""))
            if role or text:
                normalized.append({"role": role, "text": text})
            continue
        elif isinstance(turn, (list, tuple)) and len(turn) == 2:
            first = normalize_text(turn[0])
            second = normalize_text(turn[1])
            if first or second:
                normalized.append({"role": "user", "text": first})
                normalized.append({"role": "assistant", "text": second})
            continue
        else:
            continue

    return normalized


def _history_to_llamafactory_pairs(record: dict[str, Any]) -> list[list[str]]:
    history_turns = _normalize_history(record.get("dialogue_history", record.get("history", [])))
    question = normalize_text(record.get("question", record.get("instruction", "")))

    if history_turns:
        last_turn = history_turns[-1]
        if last_turn.get("role", "") == "user" and last_turn.get("text", "") == question:
            history_turns = history_turns[:-1]

    pairs: list[list[str]] = []
    pending_user: str | None = None
    for turn in history_turns:
        role = turn.get("role", "")
        text = turn.get("text", "")
        if not text:
            continue
        if role == "user":
            pending_user = text
        elif role == "assistant" and pending_user is not None:
            pairs.append([pending_user, text])
            pending_user = None

    return pairs


def _normalized_value(payload: dict[str, Any], *keys: str) -> str:
    for key in keys:
        if key in payload and payload.get(key) is not None:
            return normalize_text(payload.get(key))
    return ""


def _collect_alignment_errors(record: dict[str, Any], prediction: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    comparisons = [
        ("instruction", _normalized_value(record, "question", "instruction"), _normalized_value(prediction, "instruction", "question")),
        ("input", _normalized_value(record, "context", "input"), _normalized_value(prediction, "input", "context")),
        (
            "reference_output",
            _normalized_value(record, "human_answer", "reference_output", "output"),
            _normalized_value(prediction, "reference_output", "human_answer", "output"),
        ),
    ]
    for field_name, expected, observed in comparisons:
        if observed and expected != observed:
            errors.append(field_name)

    prediction_history = _history_to_llamafactory_pairs(prediction)
    if prediction_history:
        record_history = _history_to_llamafactory_pairs(record)
        if prediction_history != record_history:
            errors.append("history")

    return errors


def _build_dpo_item(
    *,
    record: dict[str, Any],
    prediction: dict[str, Any],
    source_index: int,
    split_name: str,
    language_tag: str,
    system_prompt: str,
    prediction_text_field: str,
) -> dict[str, Any]:
    chosen = _normalized_value(record, "human_answer", "reference_output", "output")
    rejected = _normalized_value(prediction, prediction_text_field)
    item: dict[str, Any] = {
        "instruction": _normalized_value(record, "question", "instruction"),
        "input": _normalized_value(record, "context", "input"),
        "chosen": chosen,
        "rejected": rejected,
        "system": system_prompt,
        "id": record.get("id", ""),
        "dataset": _normalized_value(record, "dataset"),
        "split": split_name,
        "language": _normalized_value(record, "language") or language_tag,
        "track": _normalized_value(record, "track"),
        "family": _normalized_value(record, "family"),
        "style_bucket": _normalized_value(record, "style_bucket"),
        "length_bin": _normalized_value(record, "length_bin"),
        "bucket_id": _normalized_value(record, "bucket_id"),
        "reference_output": chosen,
        "source_index": source_index,
        "pair_type": "human_over_sft_prediction",
    }
    history = _history_to_llamafactory_pairs(record)
    if history:
        item["history"] = history

    rejected_model_name = _normalized_value(prediction, "model_name")
    if rejected_model_name:
        item["rejected_model_name"] = rejected_model_name

    rejected_finish_reason = _normalized_value(prediction, "finish_reason")
    if rejected_finish_reason:
        item["rejected_finish_reason"] = rejected_finish_reason

    return item


def _build_dataset_info_entry(file_name: str) -> dict[str, Any]:
    return {
        "file_name": file_name,
        "ranking": True,
        "formatting": "alpaca",
        "columns": {
            "prompt": "instruction",
            "query": "input",
            "history": "history",
            "system": "system",
            "chosen": "chosen",
            "rejected": "rejected",
        },
    }


def _update_dataset_info(dataset_info_path: Path, dataset_name_to_file_name: dict[str, str]) -> dict[str, Any]:
    log_step(f"updating dataset_info: {dataset_info_path}")
    if dataset_info_path.exists():
        payload = json.loads(dataset_info_path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise ValueError(f"Expected a JSON object in {dataset_info_path}.")
    else:
        payload = {}

    updated_entries: list[str] = []
    for dataset_name, file_name in sorted(dataset_name_to_file_name.items()):
        payload[dataset_name] = _build_dataset_info_entry(file_name)
        updated_entries.append(dataset_name)

    ensure_dir(dataset_info_path.parent)
    dataset_info_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    log_step(f"dataset_info updated: {', '.join(updated_entries)}")
    return {
        "dataset_info_path": str(dataset_info_path),
        "updated_entries": updated_entries,
    }


def _export_split(
    *,
    split_name: str,
    record_path: Path,
    prediction_path: Path,
    output_root: Path,
    model_alias: str,
    language_tag: str,
    system_prompt: str,
    prediction_text_field: str,
    keep_exact_match: bool,
) -> dict[str, Any]:
    log_step(f"[{split_name}] loading source records: {record_path}")
    records = _read_records(record_path)
    log_step(f"[{split_name}] loading predictions: {prediction_path}")
    predictions = _read_records(prediction_path)
    log_step(f"[{split_name}] validating alignment for {len(records)} records and {len(predictions)} predictions")

    seen_indices: set[int] = set()
    duplicate_indices: list[int] = []
    out_of_range_indices: list[int] = []
    missing_source_index_rows = 0
    mismatch_examples: list[dict[str, Any]] = []
    exported_items: list[dict[str, Any]] = []
    skipped_exact_match = 0
    skipped_empty_chosen = 0
    skipped_empty_rejected = 0

    for row_number, prediction in enumerate(predictions):
        source_index = prediction.get("source_index")
        if not isinstance(source_index, int):
            missing_source_index_rows += 1
            continue
        if source_index < 0 or source_index >= len(records):
            out_of_range_indices.append(source_index)
            continue
        if source_index in seen_indices:
            duplicate_indices.append(source_index)
            continue

        seen_indices.add(source_index)
        record = records[source_index]
        mismatch_fields = _collect_alignment_errors(record, prediction)
        if mismatch_fields:
            mismatch_examples.append(
                {
                    "source_index": source_index,
                    "row_number": row_number,
                    "fields": mismatch_fields,
                }
            )
            continue

        chosen = _normalized_value(record, "human_answer", "reference_output", "output")
        rejected = _normalized_value(prediction, prediction_text_field)
        if not chosen:
            skipped_empty_chosen += 1
            continue
        if not rejected:
            skipped_empty_rejected += 1
            continue
        if not keep_exact_match and chosen == rejected:
            skipped_exact_match += 1
            continue

        exported_items.append(
            _build_dpo_item(
                record=record,
                prediction=prediction,
                source_index=source_index,
                split_name=split_name,
                language_tag=language_tag,
                system_prompt=system_prompt,
                prediction_text_field=prediction_text_field,
            )
        )

    missing_indices = sorted(set(range(len(records))) - seen_indices)
    if missing_source_index_rows or out_of_range_indices or duplicate_indices or mismatch_examples or missing_indices:
        error_payload = {
            "split": split_name,
            "record_path": str(record_path),
            "prediction_path": str(prediction_path),
            "source_records": len(records),
            "prediction_rows": len(predictions),
            "missing_source_index_rows": missing_source_index_rows,
            "out_of_range_indices": out_of_range_indices[:20],
            "duplicate_indices": duplicate_indices[:20],
            "missing_indices": missing_indices[:20],
            "mismatch_examples": mismatch_examples[:20],
        }
        raise ValueError(
            f"DPO export alignment failed for split `{split_name}`: {json.dumps(error_payload, ensure_ascii=False)}"
        )

    output_name = f"{_slugify_name(model_alias)}_{split_name}_dpo_{_slugify_name(language_tag)}.json"
    output_path = output_root / output_name
    log_step(f"[{split_name}] alignment passed; writing {len(exported_items)} DPO pairs to {output_path}")
    ensure_dir(output_root)
    output_path.write_text(json.dumps(exported_items, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    log_step(f"[{split_name}] export complete")

    return {
        "split": split_name,
        "record_path": str(record_path),
        "prediction_path": str(prediction_path),
        "output_path": str(output_path),
        "source_records": len(records),
        "prediction_rows": len(predictions),
        "covered_source_indices": len(seen_indices),
        "exported_pairs": len(exported_items),
        "skipped_exact_match": skipped_exact_match,
        "skipped_empty_chosen": skipped_empty_chosen,
        "skipped_empty_rejected": skipped_empty_rejected,
        "dataset_name": output_path.stem,
        "file_name": output_path.name,
    }


def export_dpo_views(
    *,
    records_root: str | Path,
    output_root: str | Path,
    model_alias: str,
    language_tag: str,
    train_predictions: str | Path | None = None,
    val_predictions: str | Path | None = None,
    test_predictions: str | Path | None = None,
    prediction_filename: str = DEFAULT_PREDICTION_FILENAME,
    system_prompt: str = "",
    prediction_text_field: str = "prediction",
    keep_exact_match: bool = False,
    dataset_info_path: str | Path | None = DEFAULT_DATASET_INFO_PATH,
    update_dataset_info: bool = True,
) -> dict[str, Any]:
    records_dir = Path(records_root)
    output_dir = Path(output_root)
    normalized_language = _slugify_name(language_tag) or "xx"
    normalized_system_prompt = normalize_text(system_prompt)
    resolved_dataset_info_path = Path(dataset_info_path) if dataset_info_path is not None else None

    log_step(f"records root: {records_dir}")
    log_step(f"output root: {output_dir}")
    log_step(f"model alias: {_slugify_name(model_alias)}")
    log_step(f"language tag: {normalized_language}")

    split_predictions = {
        "train": train_predictions,
        "val": val_predictions,
        "test": test_predictions,
    }
    summaries: dict[str, Any] = {}
    dataset_name_to_file_name: dict[str, str] = {}
    for split_name, prediction_path_or_dir in split_predictions.items():
        if prediction_path_or_dir is None:
            continue

        log_step(f"starting split: {split_name}")
        record_path = _resolve_split_record_path(records_dir, split_name)
        prediction_path = _resolve_prediction_path(prediction_path_or_dir, prediction_filename)
        split_summary = _export_split(
            split_name=split_name,
            record_path=record_path,
            prediction_path=prediction_path,
            output_root=output_dir,
            model_alias=model_alias,
            language_tag=normalized_language,
            system_prompt=normalized_system_prompt,
            prediction_text_field=prediction_text_field,
            keep_exact_match=keep_exact_match,
        )
        summaries[split_name] = split_summary
        dataset_name_to_file_name[split_summary["dataset_name"]] = split_summary["file_name"]
        log_step(f"finished split: {split_name}")

    if not summaries:
        raise ValueError("No prediction paths were provided. Pass at least one of train/val/test predictions.")

    result = {
        "records_root": str(records_dir),
        "output_root": str(output_dir),
        "model_alias": _slugify_name(model_alias),
        "language_tag": normalized_language,
        "prediction_filename": prediction_filename,
        "prediction_text_field": prediction_text_field,
        "keep_exact_match": keep_exact_match,
        "splits": summaries,
    }
    if update_dataset_info:
        if resolved_dataset_info_path is None:
            raise ValueError("`update_dataset_info=True` requires a dataset_info_path.")
        result["dataset_info_update"] = _update_dataset_info(resolved_dataset_info_path, dataset_name_to_file_name)
    else:
        result["dataset_info_update"] = {
            "dataset_info_path": str(resolved_dataset_info_path) if resolved_dataset_info_path is not None else None,
            "updated_entries": [],
            "skipped": True,
        }

    log_step("all requested DPO exports completed")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Export LlamaFactory-compatible DPO views from current split files and aligned SFT prediction files."
    )
    parser.add_argument("--records-root", required=True, help="Directory containing train/val/test current-format files.")
    parser.add_argument("--output-root", required=True, help="Directory where exported DPO JSON files will be written.")
    parser.add_argument("--model-alias", required=True, help="Model alias used in exported filenames.")
    parser.add_argument("--language-tag", required=True, help="Language tag used in exported filenames.")
    parser.add_argument("--train-predictions", help="Train prediction file or run directory containing predictions.jsonl.")
    parser.add_argument("--val-predictions", help="Val prediction file or run directory containing predictions.jsonl.")
    parser.add_argument("--test-predictions", help="Test prediction file or run directory containing predictions.jsonl.")
    parser.add_argument(
        "--prediction-filename",
        default=DEFAULT_PREDICTION_FILENAME,
        help="Prediction filename to use when a prediction path points to a directory.",
    )
    parser.add_argument(
        "--system-prompt",
        default="",
        help="Optional fixed system prompt written into every exported DPO record.",
    )
    parser.add_argument(
        "--prediction-text-field",
        default="prediction",
        help="Field in the prediction rows used as the rejected response.",
    )
    parser.add_argument(
        "--keep-exact-match",
        action="store_true",
        help="Keep pairs where the rejected prediction exactly matches the human answer after normalization.",
    )
    parser.add_argument(
        "--dataset-info-path",
        default=str(DEFAULT_DATASET_INFO_PATH),
        help="Path to the LlamaFactory dataset_info.json file to update after export.",
    )
    parser.add_argument(
        "--skip-dataset-info-update",
        action="store_true",
        help="Do not update dataset_info.json after exporting the DPO files.",
    )
    args = parser.parse_args()
    summary = export_dpo_views(
        records_root=args.records_root,
        output_root=args.output_root,
        model_alias=args.model_alias,
        language_tag=args.language_tag,
        train_predictions=args.train_predictions,
        val_predictions=args.val_predictions,
        test_predictions=args.test_predictions,
        prediction_filename=args.prediction_filename,
        system_prompt=args.system_prompt,
        prediction_text_field=args.prediction_text_field,
        keep_exact_match=args.keep_exact_match,
        dataset_info_path=args.dataset_info_path,
        update_dataset_info=(not args.skip_dataset_info_update),
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
