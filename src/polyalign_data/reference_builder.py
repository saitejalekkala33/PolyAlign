from __future__ import annotations

import argparse
import json
import math
import shutil
from array import array
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from polyalign_data.io_utils import ensure_dir


def _read_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                yield json.loads(line)


def _quantile(values: list[int], probability: float) -> int:
    if not values:
        return 0
    sorted_values = sorted(values)
    index = round((len(sorted_values) - 1) * probability)
    return sorted_values[index]


def _quantile_float(sorted_values: list[float], probability: float) -> float:
    if not sorted_values:
        return 0.0
    index = round((len(sorted_values) - 1) * probability)
    return float(sorted_values[index])


def build_reference_summary(input_root: str | Path, output_path: str | Path) -> dict:
    input_dir = Path(input_root)
    bucket_counts = Counter()
    dataset_split_counts = Counter()
    style_counts = Counter()
    bucket_lengths: dict[str, list[int]] = defaultdict(list)

    for dataset_dir in sorted(path for path in input_dir.iterdir() if path.is_dir()):
        for jsonl_path in sorted(dataset_dir.glob("*.jsonl")):
            split_name = jsonl_path.stem
            for record in _read_jsonl(jsonl_path):
                bucket_id = record["bucket_id"]
                dataset_split_counts[(record["dataset"], split_name)] += 1
                bucket_counts[bucket_id] += 1
                style_counts[record["style_bucket"]] += 1
                bucket_lengths[bucket_id].append(record["meta"]["length_tokens"])

    summary = {
        "dataset_split_counts": {
            f"{dataset}:{split}": count
            for (dataset, split), count in sorted(dataset_split_counts.items())
        },
        "bucket_counts": dict(sorted(bucket_counts.items())),
        "style_counts": dict(sorted(style_counts.items())),
        "bucket_length_stats": {
            bucket_id: {
                "count": len(lengths),
                "q10": _quantile(lengths, 0.10),
                "q50": _quantile(lengths, 0.50),
                "q90": _quantile(lengths, 0.90),
                "min": min(lengths) if lengths else 0,
                "max": max(lengths) if lengths else 0,
            }
            for bucket_id, lengths in sorted(bucket_lengths.items())
        },
    }
    output_file = Path(output_path)
    ensure_dir(output_file.parent)
    output_file.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return summary


def _iter_paired_records(record_path: Path, feature_path: Path):
    record_iter = _read_jsonl(record_path)
    feature_iter = _read_jsonl(feature_path)
    for index, (record, feature_row) in enumerate(zip(record_iter, feature_iter, strict=True)):
        if record.get("id") != feature_row.get("id"):
            raise ValueError(
                f"Mismatched ids at row {index}: record={record.get('id')} feature={feature_row.get('id')}"
            )
        yield record, feature_row


def _stats_from_values(values: array) -> dict[str, float | int]:
    if not values:
        return {
            "count": 0,
            "min": 0.0,
            "max": 0.0,
            "mean": 0.0,
            "std": 0.0,
            "q10": 0.0,
            "q25": 0.0,
            "q50": 0.0,
            "q75": 0.0,
            "q90": 0.0,
        }
    count = len(values)
    minimum = float(min(values))
    maximum = float(max(values))
    mean = float(sum(values) / count)
    variance = sum((float(value) - mean) ** 2 for value in values) / count
    sorted_values = sorted(float(value) for value in values)
    return {
        "count": count,
        "min": minimum,
        "max": maximum,
        "mean": mean,
        "std": math.sqrt(variance),
        "q10": _quantile_float(sorted_values, 0.10),
        "q25": _quantile_float(sorted_values, 0.25),
        "q50": _quantile_float(sorted_values, 0.50),
        "q75": _quantile_float(sorted_values, 0.75),
        "q90": _quantile_float(sorted_values, 0.90),
    }


def build_bucket_references(
    record_paths: list[str | Path],
    feature_paths: list[str | Path],
    output_root: str | Path,
    *,
    min_bucket_size: int = 20,
    overwrite: bool = False,
) -> dict[str, Any]:
    if len(record_paths) != len(feature_paths):
        raise ValueError("`record_paths` and `feature_paths` must have the same length.")

    output_dir = Path(output_root)
    if output_dir.exists() and any(output_dir.iterdir()) and not overwrite:
        raise FileExistsError(f"{output_dir} already exists and is not empty. Pass overwrite=True to replace it.")
    if output_dir.exists() and overwrite:
        shutil.rmtree(output_dir)
    ensure_dir(output_dir)

    temp_dir = output_dir / "_bucket_tmp"
    ensure_dir(temp_dir)
    feature_matrix_path = output_dir / "feature_matrix.jsonl"

    bucket_counts = Counter()
    split_counts = Counter()
    style_counts = Counter()
    dataset_counts = Counter()
    feature_names: set[str] = set()
    bucket_temp_handles: dict[str, Any] = {}
    total_records = 0

    try:
        with feature_matrix_path.open("w", encoding="utf-8") as feature_matrix_handle:
            for record_path, feature_path in zip(record_paths, feature_paths, strict=True):
                record_file = Path(record_path)
                feature_file = Path(feature_path)
                for record, feature_row in _iter_paired_records(record_file, feature_file):
                    total_records += 1
                    bucket_id = record["bucket_id"]
                    bucket_counts[bucket_id] += 1
                    split_counts[record["split"]] += 1
                    style_counts[record["style_bucket"]] += 1
                    dataset_counts[record["dataset"]] += 1
                    feature_names.update(feature_row["features"].keys())

                    joined = {
                        "id": record["id"],
                        "dataset": record["dataset"],
                        "split": record["split"],
                        "language": record["language"],
                        "track": record["track"],
                        "family": record["family"],
                        "style_bucket": record["style_bucket"],
                        "length_bin": record["length_bin"],
                        "bucket_id": bucket_id,
                        "features": feature_row["features"],
                    }
                    feature_matrix_handle.write(json.dumps(joined, ensure_ascii=False) + "\n")

                    if bucket_id not in bucket_temp_handles:
                        bucket_temp_handles[bucket_id] = (temp_dir / f"{bucket_id.replace('|', '__')}.jsonl").open(
                            "w", encoding="utf-8"
                        )
                    bucket_temp_handles[bucket_id].write(json.dumps(joined, ensure_ascii=False) + "\n")
    finally:
        for handle in bucket_temp_handles.values():
            handle.close()

    bucket_references: dict[str, Any] = {}
    skipped_buckets: dict[str, int] = {}

    for temp_path in sorted(temp_dir.glob("*.jsonl")):
        bucket_rows = _read_jsonl(temp_path)
        bucket_feature_values: dict[str, array] = {}
        bucket_style_counts = Counter()
        bucket_dataset_counts = Counter()
        bucket_split_counts = Counter()
        bucket_total = 0
        bucket_id = ""

        for row in bucket_rows:
            bucket_id = row["bucket_id"]
            bucket_total += 1
            bucket_style_counts[row["style_bucket"]] += 1
            bucket_dataset_counts[row["dataset"]] += 1
            bucket_split_counts[row["split"]] += 1
            for feature_name, value in row["features"].items():
                if feature_name not in bucket_feature_values:
                    bucket_feature_values[feature_name] = array("d")
                bucket_feature_values[feature_name].append(float(value))

        if bucket_total < min_bucket_size:
            skipped_buckets[bucket_id] = bucket_total
            continue

        feature_stats = {feature_name: _stats_from_values(values) for feature_name, values in sorted(bucket_feature_values.items())}
        prototype_mean = {feature_name: stats["mean"] for feature_name, stats in feature_stats.items()}
        prototype_median = {feature_name: stats["q50"] for feature_name, stats in feature_stats.items()}
        support_q10_q90 = {
            feature_name: {"low": stats["q10"], "high": stats["q90"]}
            for feature_name, stats in feature_stats.items()
        }
        support_q25_q75 = {
            feature_name: {"low": stats["q25"], "high": stats["q75"]}
            for feature_name, stats in feature_stats.items()
        }

        bucket_references[bucket_id] = {
            "count": bucket_total,
            "feature_count": len(feature_stats),
            "prototype_mean": prototype_mean,
            "prototype_median": prototype_median,
            "support_q10_q90": support_q10_q90,
            "support_q25_q75": support_q25_q75,
            "feature_stats": feature_stats,
            "style_counts": dict(sorted(bucket_style_counts.items())),
            "dataset_counts": dict(sorted(bucket_dataset_counts.items())),
            "split_counts": dict(sorted(bucket_split_counts.items())),
        }
        temp_path.unlink(missing_ok=True)

    references_path = output_dir / "bucket_references.json"
    references_path.write_text(json.dumps(bucket_references, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    summary = {
        "record_paths": [str(Path(path)) for path in record_paths],
        "feature_paths": [str(Path(path)) for path in feature_paths],
        "feature_matrix_path": str(feature_matrix_path),
        "bucket_references_path": str(references_path),
        "total_records": total_records,
        "feature_count": len(feature_names),
        "min_bucket_size": min_bucket_size,
        "bucket_counts": dict(sorted(bucket_counts.items())),
        "split_counts": dict(sorted(split_counts.items())),
        "style_counts": dict(sorted(style_counts.items())),
        "dataset_counts": dict(sorted(dataset_counts.items())),
        "retained_bucket_count": len(bucket_references),
        "skipped_buckets": dict(sorted(skipped_buckets.items())),
    }
    summary_path = output_dir / "reference_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    if temp_dir.exists():
        shutil.rmtree(temp_dir, ignore_errors=True)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Reference-builder utilities for PolyAlign.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    summary_parser = subparsers.add_parser("summary", help="Build a bucket/reference summary from formatted dataset folders.")
    summary_parser.add_argument("--input-root", required=True, help="Root directory containing formatted dataset folders.")
    summary_parser.add_argument("--output-path", required=True, help="Path to the output summary JSON file.")

    build_parser = subparsers.add_parser("build", help="Build bucket reference artifacts from records and matching feature files.")
    build_parser.add_argument("--records-path", action="append", required=True, help="Path to a merged current-format JSONL file. Repeatable.")
    build_parser.add_argument("--features-path", action="append", required=True, help="Path to a matching feature JSONL file. Repeatable.")
    build_parser.add_argument("--output-root", required=True, help="Output directory for reference artifacts.")
    build_parser.add_argument("--min-bucket-size", type=int, default=20, help="Minimum examples required to keep a bucket.")
    build_parser.add_argument("--overwrite", action="store_true", help="Overwrite the output directory if it exists.")

    args = parser.parse_args()
    if args.command == "summary":
        build_reference_summary(args.input_root, args.output_path)
    else:
        build_bucket_references(
            args.records_path,
            args.features_path,
            args.output_root,
            min_bucket_size=args.min_bucket_size,
            overwrite=args.overwrite,
        )


if __name__ == "__main__":
    main()
