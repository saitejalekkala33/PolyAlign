from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

from polyalign_data.io_utils import ensure_dir
from polyalign_data.text import normalize_text


SOURCE_SPLIT_TO_TARGET = {
    "train": "train",
    "dev": "val",
    "test": "test",
}


def _iter_dataset_split_files(input_root: Path, source_split: str):
    for dataset_dir in sorted(path for path in input_root.iterdir() if path.is_dir()):
        split_path = dataset_dir / f"{source_split}.jsonl"
        if split_path.exists():
            yield dataset_dir.name, split_path


def _iter_split_records(input_root: Path, source_split: str) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for _dataset_name, split_path in _iter_dataset_split_files(input_root, source_split):
        with split_path.open("r", encoding="utf-8") as source_handle:
            for raw_line in source_handle:
                line = raw_line.strip()
                if line:
                    records.append(json.loads(line))

    return records


def _history_to_llamafactory_pairs(record: dict[str, Any]) -> list[list[str]]:
    history_turns = list(record.get("dialogue_history", []))
    question = normalize_text(record.get("question", ""))

    if history_turns:
        last_turn = history_turns[-1]
        last_role = last_turn.get("role", "")
        last_text = normalize_text(last_turn.get("text", ""))
        if last_role == "user" and last_text == question:
            history_turns = history_turns[:-1]

    pairs: list[list[str]] = []
    pending_user: str | None = None
    for turn in history_turns:
        role = turn.get("role", "")
        text = normalize_text(turn.get("text", ""))
        if not text:
            continue
        if role == "user":
            pending_user = text
        elif role == "assistant":
            if pending_user is None:
                continue
            pairs.append([pending_user, text])
            pending_user = None

    return pairs


def _bucket_key(record: dict[str, Any]) -> str:
    bucket_id = normalize_text(record.get("bucket_id", ""))
    return bucket_id or "__missing_bucket__"


def _compute_bucket_weights(records: list[dict[str, Any]]) -> dict[str, float]:
    bucket_counts = Counter(_bucket_key(record) for record in records)
    if not bucket_counts:
        return {}

    total_examples = sum(bucket_counts.values())
    num_buckets = len(bucket_counts)
    return {
        bucket: total_examples / (num_buckets * count)
        for bucket, count in bucket_counts.items()
        if count > 0
    }


def _build_system_prompt(record: dict[str, Any]) -> str:
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


def _target_filename(target_split: str) -> str:
    return f"dist_sft_{target_split}.json"


def _to_dist_sft_alpaca(record: dict[str, Any], bucket_weight: float) -> dict[str, Any]:
    item = {
        "instruction": normalize_text(record.get("question", "")),
        "input": normalize_text(record.get("context", "")),
        "output": normalize_text(record.get("human_answer", "")),
        "system": _build_system_prompt(record),
        "dist_sft_weight": round(float(bucket_weight), 8),
        "bucket_id": normalize_text(record.get("bucket_id", "")),
        "style_bucket": normalize_text(record.get("style_bucket", "")),
        "length_bin": normalize_text(record.get("length_bin", "")),
        "family": normalize_text(record.get("family", "")),
        "track": normalize_text(record.get("track", "")),
    }
    history = _history_to_llamafactory_pairs(record)
    if history:
        item["history"] = history
    return item


def export_dist_sft_views(
    input_root: str | Path,
    output_root: str | Path,
    *,
    include_validation2: bool = False,
) -> dict[str, dict[str, Any]]:
    input_dir = Path(input_root)
    output_dir = Path(output_root)
    ensure_dir(output_dir)

    source_splits = ["train", "dev", "test"]
    if include_validation2:
        source_splits.append("validation2")

    summary: dict[str, dict[str, Any]] = {"llamafactory_dist_sft": {}}

    for source_split in source_splits:
        target_split = SOURCE_SPLIT_TO_TARGET.get(source_split, source_split)
        output_path = output_dir / _target_filename(target_split)
        records = _iter_split_records(input_dir, source_split)
        bucket_weights = _compute_bucket_weights(records)
        items = [
            _to_dist_sft_alpaca(record, bucket_weights.get(_bucket_key(record), 1.0))
            for record in records
        ]

        output_path.write_text(json.dumps(items, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

        summary["llamafactory_dist_sft"][target_split] = {
            "records": len(records),
            "buckets": len(bucket_weights),
            "output_path": str(output_path),
            "format": "llamafactory_alpaca_json_array",
        }

    return summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Export a separate LlamaFactory-compatible Dist-SFT view with bucket conditioning and weights."
    )
    parser.add_argument("--input-root", required=True, help="Root directory containing per-dataset formatted JSONL files.")
    parser.add_argument(
        "--output-root",
        required=True,
        help="Output directory. The script writes `dist_sft_train.json`, `dist_sft_val.json` and `dist_sft_test.json`.",
    )
    parser.add_argument(
        "--include-validation2",
        action="store_true",
        help="Also export the ELI5 auxiliary validation2 split as `validation2.jsonl`.",
    )
    args = parser.parse_args()
    summary = export_dist_sft_views(
        args.input_root,
        args.output_root,
        include_validation2=args.include_validation2,
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
