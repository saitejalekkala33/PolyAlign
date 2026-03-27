from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from collections import Counter
from pathlib import Path
from typing import Any

from polyalign_data.io_utils import ensure_dir, write_json
from polyalign_data.text import normalize_text


SPLIT_PRIORITY = {
    "test": 0,
    "validation2": 1,
    "dev": 2,
    "train": 3,
}

GROUP_DATASETS = {"coqa", "dailydialog", "multiwoz"}


def _iter_dataset_split_files(input_root: Path):
    for dataset_dir in sorted(path for path in input_root.iterdir() if path.is_dir()):
        for jsonl_path in sorted(dataset_dir.glob("*.jsonl")):
            yield dataset_dir.name, jsonl_path.stem, jsonl_path


def _iter_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                yield json.loads(line)


def _priority(split_name: str) -> int:
    return SPLIT_PRIORITY.get(split_name, 99)


def _canonical_history(history: list[dict[str, Any]] | None) -> list[dict[str, str]]:
    canonical_turns: list[dict[str, str]] = []
    for turn in history or []:
        role = normalize_text(turn.get("role", "")).lower()
        text = normalize_text(turn.get("text", "")).lower()
        if role and text:
            canonical_turns.append({"role": role, "text": text})
    return canonical_turns


def _example_key(record: dict[str, Any]) -> str:
    payload = {
        "language": normalize_text(record.get("language", "")).lower(),
        "track": normalize_text(record.get("track", "")).lower(),
        "family": normalize_text(record.get("family", "")).lower(),
        "question": normalize_text(record.get("question", "")).lower(),
        "context": normalize_text(record.get("context", "")).lower(),
        "human_answer": normalize_text(record.get("human_answer", "")).lower(),
        "dialogue_history": _canonical_history(record.get("dialogue_history")),
    }
    encoded = json.dumps(payload, sort_keys=True, ensure_ascii=False).encode("utf-8")
    return hashlib.sha1(encoded).hexdigest()


def _group_key(dataset_name: str, record: dict[str, Any]) -> str | None:
    if dataset_name not in GROUP_DATASETS:
        return None
    meta = record.get("meta", {})
    if dataset_name == "coqa":
        return meta.get("conversation_id")
    return meta.get("dialogue_id")


def _initial_group_best_split(input_root: Path) -> tuple[dict[tuple[str, str], str], Counter, Counter]:
    group_best_split: dict[tuple[str, str], str] = {}
    original_counts = Counter()
    dataset_split_counts = Counter()

    for dataset_name, split_name, split_path in _iter_dataset_split_files(input_root):
        for record in _iter_jsonl(split_path):
            original_counts["total"] += 1
            original_counts[f"split:{split_name}"] += 1
            dataset_split_counts[(dataset_name, split_name)] += 1
            group_key = _group_key(dataset_name, record)
            if group_key is None:
                continue
            key = (dataset_name, group_key)
            best_split = group_best_split.get(key)
            if best_split is None or _priority(split_name) < _priority(best_split):
                group_best_split[key] = split_name

    return group_best_split, original_counts, dataset_split_counts


def dedup_formatted_corpus(
    input_root: str | Path,
    output_root: str | Path,
    report_path: str | Path,
    *,
    overwrite: bool = False,
) -> dict[str, Any]:
    input_dir = Path(input_root)
    output_dir = Path(output_root)
    report_file = Path(report_path)

    if input_dir.resolve() == output_dir.resolve():
        raise ValueError("Input and output roots must be different for dedup.")

    if output_dir.exists() and any(output_dir.iterdir()) and not overwrite:
        raise FileExistsError(f"{output_dir} already exists and is not empty. Pass overwrite=True to replace it.")
    if output_dir.exists() and overwrite:
        shutil.rmtree(output_dir)
    ensure_dir(output_dir)

    group_best_split, original_counts, original_dataset_split_counts = _initial_group_best_split(input_dir)

    seen_example_keys: dict[str, tuple[str, str]] = {}
    output_handles: dict[tuple[str, str], Any] = {}
    final_dataset_split_counts = Counter()
    removed_counts = Counter()

    split_order = sorted(
        list(_iter_dataset_split_files(input_dir)),
        key=lambda item: (_priority(item[1]), item[0], item[2].name),
    )

    try:
        for dataset_name, split_name, split_path in split_order:
            output_dataset_dir = output_dir / dataset_name
            ensure_dir(output_dataset_dir)
            output_path = output_dataset_dir / split_path.name
            handle_key = (dataset_name, split_name)
            if handle_key not in output_handles:
                output_handles[handle_key] = output_path.open("w", encoding="utf-8")

            output_handle = output_handles[handle_key]
            for record in _iter_jsonl(split_path):
                group_key = _group_key(dataset_name, record)
                if group_key is not None:
                    best_split = group_best_split.get((dataset_name, group_key))
                    if best_split is not None and split_name != best_split:
                        removed_counts["group_split_conflicts_removed"] += 1
                        continue

                example_key = _example_key(record)
                if example_key in seen_example_keys:
                    previous_dataset, previous_split = seen_example_keys[example_key]
                    removed_counts["exact_duplicates_removed"] += 1
                    if previous_split == split_name:
                        removed_counts["same_split_duplicates_removed"] += 1
                    else:
                        removed_counts["cross_split_duplicates_removed"] += 1
                    if previous_dataset != dataset_name:
                        removed_counts["cross_dataset_duplicates_removed"] += 1
                    continue

                seen_example_keys[example_key] = (dataset_name, split_name)
                output_handle.write(json.dumps(record, ensure_ascii=False) + "\n")
                final_dataset_split_counts[(dataset_name, split_name)] += 1
    finally:
        for handle in output_handles.values():
            handle.close()

    per_dataset_manifests: dict[str, dict[str, Any]] = {}
    for dataset_name in sorted({name for name, _split in original_dataset_split_counts}):
        input_manifest_path = input_dir / dataset_name / "manifest.json"
        base_manifest = {}
        if input_manifest_path.exists():
            base_manifest = json.loads(input_manifest_path.read_text(encoding="utf-8"))
        final_counts = {
            split: final_dataset_split_counts[(dataset_name, split)]
            for split in sorted(split for name, split in final_dataset_split_counts if name == dataset_name)
        }
        original_counts_for_dataset = {
            split: original_dataset_split_counts[(dataset_name, split)]
            for split in sorted(split for name, split in original_dataset_split_counts if name == dataset_name)
        }
        manifest = {
            **base_manifest,
            "source_root": str(input_dir),
            "dedup_applied": True,
            "dedup_policy": "exact example dedup + group-aware split protection + evaluation-safe split priority",
            "original_split_counts": original_counts_for_dataset,
            "split_counts": final_counts,
            "removed_count": sum(original_counts_for_dataset.values()) - sum(final_counts.values()),
        }
        per_dataset_manifests[dataset_name] = manifest
        write_json(output_dir / dataset_name / "manifest.json", manifest)

    final_total = sum(final_dataset_split_counts.values())
    report = {
        "input_root": str(input_dir),
        "output_root": str(output_dir),
        "policy": {
            "exact_example_dedup": True,
            "group_aware_split_protection": True,
            "evaluation_safe_priority": ["test", "validation2", "dev", "train"],
            "group_datasets": sorted(GROUP_DATASETS),
        },
        "counts": {
            "original_total": original_counts["total"],
            "final_total": final_total,
            "removed_total": original_counts["total"] - final_total,
            "duplicate_rate": round(
                (original_counts["total"] - final_total) / original_counts["total"], 6
            )
            if original_counts["total"]
            else 0.0,
            **{key: value for key, value in removed_counts.items()},
        },
        "original_dataset_split_counts": {
            f"{dataset}:{split}": count
            for (dataset, split), count in sorted(original_dataset_split_counts.items())
        },
        "final_dataset_split_counts": {
            f"{dataset}:{split}": count
            for (dataset, split), count in sorted(final_dataset_split_counts.items())
        },
    }

    write_json(report_file, report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Deduplicate formatted PolyAlign JSONL datasets with evaluation-safe split priority."
    )
    parser.add_argument("--input-root", required=True, help="Root directory containing formatted per-dataset JSONL files.")
    parser.add_argument("--output-root", required=True, help="Output directory for the deduplicated corpus.")
    parser.add_argument("--report-path", required=True, help="Output JSON path for the dedup report.")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite the output root if it already exists.")
    args = parser.parse_args()
    report = dedup_formatted_corpus(
        args.input_root,
        args.output_root,
        args.report_path,
        overwrite=args.overwrite,
    )
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
