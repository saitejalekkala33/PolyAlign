from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

from polyalign_data.text import normalize_text


def _read_current_jsonl(path: str | Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
    return records


def _read_json_records(path: str | Path) -> list[dict[str, Any]]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError(f"Expected JSON array in {path}, found {type(payload).__name__}.")
    return payload


def _history_to_llamafactory_pairs(record: dict[str, Any]) -> list[list[str]]:
    history_turns = record.get("dialogue_history", record.get("history", [])) or []
    question = normalize_text(record.get("question", record.get("instruction", "")))
    if history_turns:
        last_turn = history_turns[-1]
        if (
            isinstance(last_turn, dict)
            and normalize_text(last_turn.get("role", "")) == "user"
            and normalize_text(last_turn.get("text", "")) == question
        ):
            history_turns = history_turns[:-1]

    pairs: list[list[str]] = []
    pending_user: str | None = None
    for turn in history_turns:
        if isinstance(turn, dict):
            role = normalize_text(turn.get("role", ""))
            text = normalize_text(turn.get("text", ""))
        elif isinstance(turn, list) and len(turn) == 2:
            user_text = normalize_text(turn[0])
            assistant_text = normalize_text(turn[1])
            if user_text or assistant_text:
                pairs.append([user_text, assistant_text])
            continue
        else:
            continue

        if not text:
            continue
        if role == "user":
            pending_user = text
        elif role == "assistant" and pending_user is not None:
            pairs.append([pending_user, text])
            pending_user = None

    return pairs


def _bucket_key(record: dict[str, Any]) -> str:
    bucket_id = normalize_text(record.get("bucket_id", ""))
    if bucket_id:
        return bucket_id
    return "__missing_bucket__"


def _compute_bucket_weights(records: list[dict[str, Any]]) -> dict[str, float]:
    bucket_counts = Counter(_bucket_key(record) for record in records)
    if not bucket_counts:
        return {}

    total_examples = sum(bucket_counts.values())
    num_buckets = len(bucket_counts)
    return {
        bucket: total_examples / (num_buckets * count)
        for bucket, count in bucket_counts.items()
    }


def _build_expected_system_prompt(record: dict[str, Any]) -> str:
    prompt_parts = [
        ("family", normalize_text(record.get("family", ""))),
        ("track", normalize_text(record.get("track", ""))),
        ("style", normalize_text(record.get("style_bucket", ""))),
        ("length", normalize_text(record.get("length_bin", ""))),
    ]
    profile = "; ".join(f"{name}={value}" for name, value in prompt_parts if value)
    if not profile:
        return "You are a helpful assistant."

    return f"You are a helpful assistant. Follow the target response profile when answering. {profile}."


def _compare_record_pair(
    current_record: dict[str, Any],
    sample_record: dict[str, Any],
    *,
    expected_weight: float | None,
) -> list[str]:
    mismatched_fields: list[str] = []

    field_pairs = [
        ("instruction", normalize_text(current_record.get("question", "")), normalize_text(sample_record.get("instruction", current_record.get("instruction", "")))),
        ("input", normalize_text(current_record.get("context", "")), normalize_text(sample_record.get("input", current_record.get("input", "")))),
        ("output", normalize_text(current_record.get("human_answer", "")), normalize_text(sample_record.get("output", sample_record.get("reference_output", "")))),
    ]
    for field_name, expected_value, observed_value in field_pairs:
        if expected_value != observed_value:
            mismatched_fields.append(field_name)

    expected_history = _history_to_llamafactory_pairs(current_record)
    observed_history = _history_to_llamafactory_pairs(sample_record)
    if expected_history != observed_history:
        mismatched_fields.append("history")

    optional_fields = ["bucket_id", "style_bucket", "length_bin", "family", "track"]
    for field_name in optional_fields:
        if field_name in sample_record:
            if normalize_text(current_record.get(field_name, "")) != normalize_text(sample_record.get(field_name, "")):
                mismatched_fields.append(field_name)

    if "system" in sample_record:
        expected_system = _build_expected_system_prompt(current_record)
        observed_system = normalize_text(sample_record.get("system", ""))
        if expected_system != observed_system:
            mismatched_fields.append("system")

    if "dist_sft_weight" in sample_record and expected_weight is not None:
        observed_weight = float(sample_record.get("dist_sft_weight", 0.0))
        rounded_expected = round(float(expected_weight), 8)
        if abs(observed_weight - rounded_expected) > 1e-8:
            mismatched_fields.append("dist_sft_weight")

    return mismatched_fields


def compare_current_and_sample(
    current_path: str | Path,
    sample_path: str | Path,
    *,
    max_mismatches: int = 20,
) -> dict[str, Any]:
    current_records = _read_current_jsonl(current_path)
    sample_records = _read_json_records(sample_path)
    bucket_weights = _compute_bucket_weights(current_records)

    comparison_length = min(len(current_records), len(sample_records))
    mismatch_examples: list[dict[str, Any]] = []
    mismatched_rows = 0

    for row_index in range(comparison_length):
        current_record = current_records[row_index]
        sample_record = sample_records[row_index]
        mismatched_fields = _compare_record_pair(
            current_record,
            sample_record,
            expected_weight=bucket_weights.get(_bucket_key(current_record)),
        )
        if mismatched_fields:
            mismatched_rows += 1
            if len(mismatch_examples) < max_mismatches:
                mismatch_examples.append(
                    {
                        "row_index": row_index,
                        "id": normalize_text(current_record.get("id", "")),
                        "fields": mismatched_fields,
                    }
                )

    count_mismatch = len(current_records) != len(sample_records)
    return {
        "current_path": str(current_path),
        "sample_path": str(sample_path),
        "current_records": len(current_records),
        "sample_records": len(sample_records),
        "compared_rows": comparison_length,
        "count_match": not count_mismatch,
        "mismatched_rows": mismatched_rows,
        "matches_exactly": (not count_mismatch) and mismatched_rows == 0,
        "mismatch_examples": mismatch_examples,
    }


def _resolve_current_file(current_root: Path, split: str) -> Path:
    return current_root / f"{split}.jsonl"


def _resolve_sample_file(sample_root: Path, split: str, kind: str) -> Path:
    if kind == "sample":
        return sample_root / f"{split}-sample.json"
    if kind == "dist_sft":
        return sample_root / f"dist_sft_{split}.json"
    raise ValueError(f"Unsupported kind: {kind}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Check whether sampled current JSONL files match sampled LlamaFactory JSON files in content and order."
    )
    parser.add_argument("--current-root", required=True, help="Directory containing sampled current-format train/val/test JSONL files.")
    parser.add_argument("--sample-root", required=True, help="Directory containing sampled JSON arrays such as train-sample.json or dist_sft_train.json.")
    parser.add_argument(
        "--kind",
        choices=["sample", "dist_sft"],
        required=True,
        help="Which sampled JSON family to compare against the current JSONL splits.",
    )
    parser.add_argument(
        "--split",
        choices=["train", "val", "test", "all"],
        default="all",
        help="Which split to compare. Use all to compare train/val/test together.",
    )
    parser.add_argument("--max-mismatches", type=int, default=20, help="Maximum mismatch examples to include per split.")
    args = parser.parse_args()

    current_root = Path(args.current_root)
    sample_root = Path(args.sample_root)
    splits = ["train", "val", "test"] if args.split == "all" else [args.split]

    summary: dict[str, Any] = {
        "current_root": str(current_root),
        "sample_root": str(sample_root),
        "kind": args.kind,
        "splits": {},
    }
    all_match = True

    for split in splits:
        current_path = _resolve_current_file(current_root, split)
        sample_path = _resolve_sample_file(sample_root, split, args.kind)
        split_summary = compare_current_and_sample(
            current_path,
            sample_path,
            max_mismatches=args.max_mismatches,
        )
        summary["splits"][split] = split_summary
        all_match = all_match and split_summary["matches_exactly"]

    summary["all_match"] = all_match
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
