from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

from polyalign_data.dedup import dedup_formatted_corpus
from polyalign_data.export_sft_views import export_merged_sft_views
from polyalign_data.extract_linguistic_features import extract_features_file
from polyalign_data.io_utils import ensure_dir, write_json
from polyalign_data.reference_builder import build_bucket_references, build_reference_summary
from polyalign_data.registry import create_formatter


def _remove_tree(path: Path) -> None:
    if path.exists():
        shutil.rmtree(path)


def _write_temp_summary(config: dict[str, Any], formatted_root: Path) -> tuple[dict[str, Any], Path]:
    dedup_report_path = Path(config["dedup_report_path"])
    temp_path = dedup_report_path.parent / "_prior_reference_summary.tmp.json"
    summary = build_reference_summary(formatted_root, temp_path)
    return summary, temp_path


def run_pipeline(config: dict[str, Any]) -> dict[str, Any]:
    datasets = config["datasets"]
    seed = int(config.get("seed", 42))
    cache_dir = str(config.get("cache_dir", "data/cache"))
    overwrite = bool(config.get("overwrite", False))

    formatted_root = Path(config["format_output_root"])
    dedup_root = Path(config["dedup_output_root"])
    dedup_report_path = Path(config["dedup_report_path"])
    merged_root = Path(config["merged_output_root"])
    features_root = Path(config["features_output_root"])
    reference_root = Path(config["reference_output_root"])
    prior_reference_filename = str(config.get("prior_reference_filename", "prior-reference_summary.json"))
    feature_text_field = str(config.get("feature_text_field", "human_answer"))
    feature_include_text = bool(config.get("feature_include_text", False))
    feature_write_csv = bool(config.get("feature_write_csv", False))
    reference_min_bucket_size = int(config.get("reference_min_bucket_size", 20))
    pipeline_summary_path = Path(config["pipeline_summary_path"]) if config.get("pipeline_summary_path") else None

    if overwrite:
        for output_root in (formatted_root, dedup_root, merged_root, features_root, reference_root):
            _remove_tree(output_root)

    manifests = []
    for dataset_name in datasets:
        formatter = create_formatter(dataset_name, seed=seed, cache_dir=cache_dir)
        manifests.append(formatter.write(formatted_root, overwrite=overwrite))

    prior_reference_summary, temp_prior_path = _write_temp_summary(config, formatted_root)
    dedup_report = dedup_formatted_corpus(
        formatted_root,
        dedup_root,
        dedup_report_path,
        overwrite=overwrite,
    )
    merge_summary = export_merged_sft_views(dedup_root, merged_root)

    current_root = merged_root / "current"
    feature_current_root = features_root / "current"
    ensure_dir(feature_current_root)

    feature_summaries: dict[str, dict[str, Any]] = {}
    record_paths: list[str] = []
    feature_paths: list[str] = []
    for split in ("train", "val", "test"):
        input_path = current_root / f"{split}.jsonl"
        output_jsonl = feature_current_root / f"{split}.jsonl"
        output_csv = feature_current_root / f"{split}.csv" if feature_write_csv else None
        feature_summaries[split] = extract_features_file(
            input_path,
            output_jsonl,
            text_field=feature_text_field,
            output_csv=output_csv,
            include_text=feature_include_text,
        )
        record_paths.append(str(input_path))
        feature_paths.append(str(output_jsonl))

    reference_summary = build_bucket_references(
        record_paths,
        feature_paths,
        reference_root,
        min_bucket_size=reference_min_bucket_size,
        overwrite=overwrite,
    )
    write_json(reference_root / prior_reference_filename, prior_reference_summary)

    summary = {
        "datasets": manifests,
        "formatted_output_root": str(formatted_root),
        "formatted_reference_summary": prior_reference_summary,
        "dedup_report": dedup_report,
        "merged_sft": merge_summary,
        "features": feature_summaries,
        "reference_build": reference_summary,
    }

    if pipeline_summary_path is not None:
        write_json(pipeline_summary_path, summary)

    temp_prior_path.unlink(missing_ok=True)
    return summary
