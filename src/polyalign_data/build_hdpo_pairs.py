from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from polyalign_data.io_utils import ensure_dir, write_jsonl
from polyalign_data.text import normalize_text


SPLIT_ALIASES = {
    "val": "dev",
}


def _read_records(path: Path) -> list[dict[str, Any]]:
    if path.suffix.lower() == ".json":
        payload = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(payload, list):
            raise ValueError(f"Expected a JSON array in {path}.")
        return [dict(item) for item in payload]

    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if line:
                records.append(json.loads(line))
    return records


def _write_records(path: Path, records: list[dict[str, Any]]) -> None:
    ensure_dir(path.parent)
    if path.suffix.lower() == ".json":
        path.write_text(json.dumps(records, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        return
    write_jsonl(path, records)


def _resolve_split_name(value: str) -> str:
    normalized = normalize_text(value).lower()
    normalized = SPLIT_ALIASES.get(normalized, normalized)
    if normalized not in {"train", "dev", "test", "validation2"}:
        raise ValueError(f"Unsupported HDPO pair split name: {value}")
    return normalized


def _normalized_value(payload: dict[str, Any], key: str) -> str:
    value = payload.get(key)
    return normalize_text(value) if value is not None else ""


def _collect_mismatched_fields(record: dict[str, Any], prediction: dict[str, Any]) -> list[str]:
    mismatched_fields: list[str] = []
    comparisons = [
        ("instruction", _normalized_value(record, "question"), _normalized_value(prediction, "instruction")),
        ("input", _normalized_value(record, "context"), _normalized_value(prediction, "input")),
        ("reference_output", _normalized_value(record, "human_answer"), _normalized_value(prediction, "reference_output")),
    ]
    for field_name, expected, observed in comparisons:
        if observed and expected != observed:
            mismatched_fields.append(field_name)
    return mismatched_fields


def _copy_history(record: dict[str, Any]) -> list[dict[str, Any]]:
    history = record.get("dialogue_history", [])
    if not isinstance(history, list):
        return []
    copied: list[dict[str, Any]] = []
    for turn in history:
        if isinstance(turn, dict):
            copied.append(dict(turn))
    return copied


def build_hdpo_pair_files(
    record_path: str | Path,
    prediction_path: str | Path,
    output_root: str | Path,
    *,
    split_name: str,
    pair_type: str = "global",
    prediction_text_field: str = "prediction",
    keep_exact_match: bool = False,
    keep_mismatched: bool = False,
    merged_output_path: str | Path | None = None,
) -> dict[str, Any]:
    record_file = Path(record_path)
    prediction_file = Path(prediction_path)
    output_dir = Path(output_root)
    split_name = _resolve_split_name(split_name)

    records = _read_records(record_file)
    predictions = _read_records(prediction_file)
    grouped_records: dict[str, list[dict[str, Any]]] = defaultdict(list)
    merged_records: list[dict[str, Any]] = []
    dataset_counts: Counter[str] = Counter()
    counters: Counter[str] = Counter()
    used_source_indices: set[int] = set()

    for prediction in predictions:
        source_index = prediction.get("source_index")
        if not isinstance(source_index, int):
            counters["skipped_missing_source_index"] += 1
            continue
        if source_index < 0 or source_index >= len(records):
            counters["skipped_out_of_range_source_index"] += 1
            continue

        record = records[source_index]
        mismatched_fields = _collect_mismatched_fields(record, prediction)
        if mismatched_fields and not keep_mismatched:
            counters["skipped_mismatch"] += 1
            for field_name in mismatched_fields:
                counters[f"mismatch_{field_name}"] += 1
            continue

        chosen = _normalized_value(record, "human_answer")
        rejected = _normalized_value(prediction, prediction_text_field)
        if not chosen:
            counters["skipped_empty_chosen"] += 1
            continue
        if not rejected:
            counters["skipped_empty_rejected"] += 1
            continue
        if not keep_exact_match and chosen == rejected:
            counters["skipped_exact_match"] += 1
            continue

        dataset_name = _normalized_value(record, "dataset") or "__unknown_dataset__"
        pair_record: dict[str, Any] = {
            "id": record.get("id", ""),
            "dataset": dataset_name,
            "split": _normalized_value(record, "split") or split_name,
            "language": _normalized_value(record, "language"),
            "track": _normalized_value(record, "track"),
            "family": _normalized_value(record, "family"),
            "style_bucket": _normalized_value(record, "style_bucket"),
            "length_bin": _normalized_value(record, "length_bin"),
            "bucket_id": _normalized_value(record, "bucket_id"),
            "question": _normalized_value(record, "question"),
            "context": _normalized_value(record, "context"),
            "dialogue_history": _copy_history(record),
            "human_answer": chosen,
            "chosen": chosen,
            "rejected": rejected,
            "pair_type": normalize_text(pair_type).lower() or "global",
            "source_index": source_index,
        }
        model_name = _normalized_value(prediction, "model_name")
        if model_name:
            pair_record["rejected_model_name"] = model_name
        finish_reason = _normalized_value(prediction, "finish_reason")
        if finish_reason:
            pair_record["rejected_finish_reason"] = finish_reason
        meta = record.get("meta")
        if isinstance(meta, dict):
            pair_record["meta"] = dict(meta)
        if mismatched_fields:
            pair_record["prediction_mismatched_fields"] = mismatched_fields

        grouped_records[dataset_name].append(pair_record)
        merged_records.append(pair_record)
        dataset_counts[dataset_name] += 1
        used_source_indices.add(source_index)
        counters["paired_records"] += 1

    output_paths: dict[str, str] = {}
    for dataset_name, dataset_records in sorted(grouped_records.items()):
        output_path = output_dir / dataset_name / f"{split_name}.jsonl"
        _write_records(output_path, dataset_records)
        output_paths[dataset_name] = str(output_path)

    merged_output = None
    if merged_output_path is not None:
        merged_output = Path(merged_output_path)
        _write_records(merged_output, merged_records)

    return {
        "record_path": str(record_file),
        "prediction_path": str(prediction_file),
        "output_root": str(output_dir),
        "split_name": split_name,
        "pair_type": normalize_text(pair_type).lower() or "global",
        "prediction_text_field": prediction_text_field,
        "source_records": len(records),
        "prediction_rows": len(predictions),
        "covered_source_indices": len(used_source_indices),
        "missing_source_indices": max(0, len(records) - len(used_source_indices)),
        "dataset_counts": dict(sorted(dataset_counts.items())),
        "output_paths": output_paths,
        "merged_output_path": str(merged_output) if merged_output is not None else None,
        "skipped_missing_source_index": counters["skipped_missing_source_index"],
        "skipped_out_of_range_source_index": counters["skipped_out_of_range_source_index"],
        "skipped_mismatch": counters["skipped_mismatch"],
        "mismatch_instruction": counters["mismatch_instruction"],
        "mismatch_input": counters["mismatch_input"],
        "mismatch_reference_output": counters["mismatch_reference_output"],
        "skipped_empty_chosen": counters["skipped_empty_chosen"],
        "skipped_empty_rejected": counters["skipped_empty_rejected"],
        "skipped_exact_match": counters["skipped_exact_match"],
        "paired_records": counters["paired_records"],
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build raw HDPO pair files by aligning current-format records with model prediction JSONL files."
    )
    parser.add_argument("--record-path", required=True, help="Current-format merged JSONL/JSON file for a single split.")
    parser.add_argument(
        "--prediction-path",
        required=True,
        help="Prediction JSONL/JSON file aligned to the same split, containing `source_index` and a prediction field.",
    )
    parser.add_argument(
        "--output-root",
        required=True,
        help="Output root where per-dataset raw pair files are written as `<dataset>/<split>.jsonl`.",
    )
    parser.add_argument(
        "--split-name",
        required=True,
        help="Source split name to write (`train`, `dev`, `test`, or `validation2`). `val` is accepted and mapped to `dev`.",
    )
    parser.add_argument("--pair-type", default="global", help="Pair type label written into every exported record.")
    parser.add_argument(
        "--prediction-text-field",
        default="prediction",
        help="Field in the prediction rows used as the rejected response text.",
    )
    parser.add_argument(
        "--keep-exact-match",
        action="store_true",
        help="Keep pairs where the rejected prediction exactly matches the human answer after normalization.",
    )
    parser.add_argument(
        "--keep-mismatched",
        action="store_true",
        help="Keep rows even when instruction/input/reference_output do not match the aligned source record.",
    )
    parser.add_argument(
        "--merged-output-path",
        help="Optional merged JSONL/JSON file containing all exported pair rows for this split.",
    )
    args = parser.parse_args()
    summary = build_hdpo_pair_files(
        args.record_path,
        args.prediction_path,
        args.output_root,
        split_name=args.split_name,
        pair_type=args.pair_type,
        prediction_text_field=args.prediction_text_field,
        keep_exact_match=args.keep_exact_match,
        keep_mismatched=args.keep_mismatched,
        merged_output_path=args.merged_output_path,
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
