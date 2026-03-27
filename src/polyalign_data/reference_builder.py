from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

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


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a bucket/reference summary from formatted JSONL data.")
    parser.add_argument("--input-root", required=True, help="Root directory containing formatted dataset folders.")
    parser.add_argument("--output-path", required=True, help="Path to the output summary JSON file.")
    args = parser.parse_args()
    build_reference_summary(args.input_root, args.output_path)


if __name__ == "__main__":
    main()
