from __future__ import annotations

import argparse
import json
import random
from collections import Counter
from pathlib import Path
from typing import Any

from polyalign_data.io_utils import ensure_dir
from polyalign_data.text import normalize_text


DEFAULT_SPLITS = ("train", "val", "test")


def _iter_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                yield json.loads(line)


def convert_llamafactory_jsonl_to_json(
    input_path: str | Path,
    output_path: str | Path,
) -> dict[str, Any]:
    input_file = Path(input_path)
    output_file = Path(output_path)
    ensure_dir(output_file.parent)

    records = list(_iter_jsonl(input_file))
    output_file.write_text(json.dumps(records, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    return {
        "input_path": str(input_file),
        "output_path": str(output_file),
        "records": len(records),
        "format": "llamafactory_alpaca_json_array",
    }


def convert_llamafactory_split_dir(
    input_root: str | Path,
    output_root: str | Path,
    *,
    include_validation2: bool = False,
) -> dict[str, Any]:
    input_dir = Path(input_root)
    output_dir = Path(output_root)
    ensure_dir(output_dir)

    splits = list(DEFAULT_SPLITS)
    if include_validation2:
        splits.append("validation2")

    summary: dict[str, Any] = {"input_root": str(input_dir), "output_root": str(output_dir), "splits": {}}
    for split in splits:
        input_path = input_dir / f"{split}.jsonl"
        if not input_path.exists():
            continue
        output_path = output_dir / f"{split}.json"
        summary["splits"][split] = convert_llamafactory_jsonl_to_json(input_path, output_path)

    return summary


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
        elif role == "assistant" and pending_user is not None:
            pairs.append([pending_user, text])
            pending_user = None

    return pairs


def _current_to_llamafactory_alpaca(record: dict[str, Any]) -> dict[str, Any]:
    item = {
        "instruction": normalize_text(record.get("question", "")),
        "input": normalize_text(record.get("context", "")),
        "output": normalize_text(record.get("human_answer", "")),
    }
    history = _history_to_llamafactory_pairs(record)
    if history:
        item["history"] = history
    return item


def _group_current_rows_by_bucket(
    input_file: Path,
) -> dict[str, list[tuple[int, dict[str, Any]]]]:
    bucket_rows: dict[str, list[tuple[int, dict[str, Any]]]] = {}
    for source_index, record in enumerate(_iter_jsonl(input_file)):
        bucket_id = normalize_text(record.get("bucket_id", "")) or "__missing_bucket__"
        bucket_rows.setdefault(bucket_id, []).append((source_index, record))
    return bucket_rows


def _allocate_balanced_bucket_counts(capacities: dict[str, int], target_total: int) -> dict[str, int]:
    bucket_ids = sorted(bucket_id for bucket_id, count in capacities.items() if count > 0)
    allocations = {bucket_id: 0 for bucket_id in bucket_ids}
    remaining = min(max(target_total, 0), sum(capacities.values()))
    active = list(bucket_ids)

    while remaining > 0 and active:
        active = [bucket_id for bucket_id in active if allocations[bucket_id] < capacities[bucket_id]]
        if not active:
            break

        base, remainder = divmod(remaining, len(active))
        if base == 0:
            for bucket_id in active[:remaining]:
                allocations[bucket_id] += 1
            remaining = 0
            break

        consumed = 0
        next_active: list[str] = []
        for ordinal, bucket_id in enumerate(active):
            desired = base + (1 if ordinal < remainder else 0)
            room = capacities[bucket_id] - allocations[bucket_id]
            take = min(desired, room)
            if take:
                allocations[bucket_id] += take
                consumed += take
            if allocations[bucket_id] < capacities[bucket_id]:
                next_active.append(bucket_id)

        if consumed == 0:
            break
        remaining -= consumed
        active = next_active

    return allocations


def sample_current_jsonl_to_llamafactory_json(
    input_path: str | Path,
    output_path: str | Path,
    *,
    sample_size: int,
    seed: int = 42,
) -> dict[str, Any]:
    input_file = Path(input_path)
    output_file = Path(output_path)
    ensure_dir(output_file.parent)

    bucket_rows = _group_current_rows_by_bucket(input_file)
    selected_rows, allocations = _sample_grouped_rows(
        bucket_rows,
        sample_size=sample_size,
        seed=seed,
    )
    sampled_records = [_current_to_llamafactory_alpaca(record) for _source_index, record in selected_rows]
    output_file.write_text(json.dumps(sampled_records, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    nonzero_bucket_counts = {bucket_id: count for bucket_id, count in allocations.items() if count > 0}
    bucket_counter = Counter(nonzero_bucket_counts.values())
    written = len(sampled_records)

    return {
        "input_path": str(input_file),
        "output_path": str(output_file),
        "requested_records": sample_size,
        "written_records": written,
        "available_records": sum(capacities.values()),
        "unique_bucket_ids": len(capacities),
        "sampled_bucket_ids": len(nonzero_bucket_counts),
        "min_bucket_count": min(nonzero_bucket_counts.values()) if nonzero_bucket_counts else 0,
        "max_bucket_count": max(nonzero_bucket_counts.values()) if nonzero_bucket_counts else 0,
        "bucket_counts": dict(sorted(nonzero_bucket_counts.items())),
        "bucket_histogram": dict(sorted(bucket_counter.items())),
        "format": "llamafactory_alpaca_json_array",
        "sampling_strategy": "balanced_by_bucket_id_with_capacity_caps",
        "seed": seed,
    }


def _sample_grouped_rows(
    bucket_rows: dict[str, list[tuple[int, dict[str, Any]]]],
    *,
    sample_size: int,
    seed: int,
) -> tuple[list[tuple[int, dict[str, Any]]], dict[str, int]]:
    capacities = {bucket_id: len(rows) for bucket_id, rows in bucket_rows.items()}
    allocations = _allocate_balanced_bucket_counts(capacities, sample_size)
    rng = random.Random(seed)

    selected_rows: list[tuple[int, dict[str, Any]]] = []
    for bucket_id in sorted(bucket_rows):
        rows = list(bucket_rows[bucket_id])
        rng.shuffle(rows)
        selected_rows.extend(rows[: allocations.get(bucket_id, 0)])

    selected_rows.sort(key=lambda item: item[0])
    return selected_rows, allocations


def sample_current_jsonl_to_current_jsonl(
    input_path: str | Path,
    output_path: str | Path,
    *,
    sample_size: int,
    seed: int = 42,
) -> dict[str, Any]:
    input_file = Path(input_path)
    output_file = Path(output_path)
    ensure_dir(output_file.parent)

    bucket_rows = _group_current_rows_by_bucket(input_file)
    selected_rows, allocations = _sample_grouped_rows(
        bucket_rows,
        sample_size=sample_size,
        seed=seed,
    )
    with output_file.open("w", encoding="utf-8") as handle:
        for _source_index, record in selected_rows:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    nonzero_bucket_counts = {bucket_id: count for bucket_id, count in allocations.items() if count > 0}
    bucket_counter = Counter(nonzero_bucket_counts.values())

    return {
        "input_path": str(input_file),
        "output_path": str(output_file),
        "requested_records": sample_size,
        "written_records": len(selected_rows),
        "available_records": sum(len(rows) for rows in bucket_rows.values()),
        "unique_bucket_ids": len(bucket_rows),
        "sampled_bucket_ids": len(nonzero_bucket_counts),
        "min_bucket_count": min(nonzero_bucket_counts.values()) if nonzero_bucket_counts else 0,
        "max_bucket_count": max(nonzero_bucket_counts.values()) if nonzero_bucket_counts else 0,
        "bucket_counts": dict(sorted(nonzero_bucket_counts.items())),
        "bucket_histogram": dict(sorted(bucket_counter.items())),
        "format": "current_jsonl",
        "sampling_strategy": "balanced_by_bucket_id_with_capacity_caps",
        "seed": seed,
    }


def sample_current_split_dir_to_llamafactory_json(
    input_root: str | Path,
    output_root: str | Path,
    *,
    train_size: int,
    val_size: int,
    test_size: int,
    seed: int = 42,
) -> dict[str, Any]:
    input_dir = Path(input_root)
    output_dir = Path(output_root)
    ensure_dir(output_dir)

    requested_sizes = {
        "train": train_size,
        "val": val_size,
        "test": test_size,
    }
    summary: dict[str, Any] = {
        "input_root": str(input_dir),
        "output_root": str(output_dir),
        "seed": seed,
        "splits": {},
    }

    for split, sample_size in requested_sizes.items():
        input_path = input_dir / f"{split}.jsonl"
        if not input_path.exists():
            continue
        output_path = output_dir / f"{split}-sample.json"
        summary["splits"][split] = sample_current_jsonl_to_llamafactory_json(
            input_path,
            output_path,
            sample_size=sample_size,
            seed=seed,
        )

    return summary


def sample_current_split_dir_to_current_jsonl(
    input_root: str | Path,
    output_root: str | Path,
    *,
    train_size: int,
    val_size: int,
    test_size: int,
    seed: int = 42,
) -> dict[str, Any]:
    input_dir = Path(input_root)
    output_dir = Path(output_root)
    ensure_dir(output_dir)

    requested_sizes = {
        "train": train_size,
        "val": val_size,
        "test": test_size,
    }
    summary: dict[str, Any] = {
        "input_root": str(input_dir),
        "output_root": str(output_dir),
        "seed": seed,
        "splits": {},
    }

    for split, sample_size in requested_sizes.items():
        input_path = input_dir / f"{split}.jsonl"
        if not input_path.exists():
            continue
        output_path = output_dir / f"{split}.jsonl"
        summary["splits"][split] = sample_current_jsonl_to_current_jsonl(
            input_path,
            output_path,
            sample_size=sample_size,
            seed=seed,
        )

    return summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert LlamaFactory-format JSONL files into JSON array files accepted by LlamaFactory."
    )
    parser.add_argument("--input-path", help="Single LlamaFactory JSONL file to convert.")
    parser.add_argument("--output-path", help="Output JSON path for single-file conversion.")
    parser.add_argument("--input-root", help="Directory containing train/val/test LlamaFactory JSONL files.")
    parser.add_argument("--output-root", help="Output directory for converted JSON split files.")
    parser.add_argument("--sample-current-root", help="Directory containing current-format train/val/test JSONL files.")
    parser.add_argument(
        "--sample-output-root",
        help="Output directory for sampled LlamaFactory JSON arrays from current-format JSONL files.",
    )
    parser.add_argument(
        "--sample-current-jsonl-root",
        help="Directory containing current-format train/val/test JSONL files to sample into new current-format JSONL files.",
    )
    parser.add_argument(
        "--sample-current-jsonl-output-root",
        help="Output directory for sampled current-format train/val/test JSONL files.",
    )
    parser.add_argument("--train-size", type=int, default=2000, help="Requested train sample size for current-format sampling.")
    parser.add_argument("--val-size", type=int, default=1000, help="Requested val sample size for current-format sampling.")
    parser.add_argument("--test-size", type=int, default=1000, help="Requested test sample size for current-format sampling.")
    parser.add_argument("--seed", type=int, default=42, help="Sampling seed for current-format balanced sampling.")
    parser.add_argument(
        "--include-validation2",
        action="store_true",
        help="Also convert validation2.jsonl if it exists.",
    )
    args = parser.parse_args()

    if args.input_path:
        if not args.output_path:
            parser.error("--output-path is required when --input-path is used.")
        summary = convert_llamafactory_jsonl_to_json(args.input_path, args.output_path)
    elif args.input_root:
        if not args.output_root:
            parser.error("--output-root is required when --input-root is used.")
        summary = convert_llamafactory_split_dir(
            args.input_root,
            args.output_root,
            include_validation2=args.include_validation2,
        )
    elif args.sample_current_root:
        if not args.sample_output_root:
            parser.error("--sample-output-root is required when --sample-current-root is used.")
        summary = sample_current_split_dir_to_llamafactory_json(
            args.sample_current_root,
            args.sample_output_root,
            train_size=args.train_size,
            val_size=args.val_size,
            test_size=args.test_size,
            seed=args.seed,
        )
    elif args.sample_current_jsonl_root:
        if not args.sample_current_jsonl_output_root:
            parser.error(
                "--sample-current-jsonl-output-root is required when --sample-current-jsonl-root is used."
            )
        summary = sample_current_split_dir_to_current_jsonl(
            args.sample_current_jsonl_root,
            args.sample_current_jsonl_output_root,
            train_size=args.train_size,
            val_size=args.val_size,
            test_size=args.test_size,
            seed=args.seed,
        )
    else:
        parser.error(
            "Use either --input-path/--output-path, --input-root/--output-root, "
            "--sample-current-root/--sample-output-root, "
            "or --sample-current-jsonl-root/--sample-current-jsonl-output-root."
        )

    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
