from __future__ import annotations

import argparse
import json
from pathlib import Path

from polyalign_data.dedup import dedup_formatted_corpus
from polyalign_data.reference_builder import build_bucket_references, build_reference_summary
from polyalign_data.registry import canonical_dataset_names, create_formatter


def _load_json(path: str | None) -> dict:
    if not path:
        return {}
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _resolve_datasets(args, config: dict) -> list[str]:
    if args.all:
        return canonical_dataset_names()
    if args.dataset:
        return args.dataset
    if config.get("datasets"):
        return config["datasets"]
    raise ValueError("No datasets selected. Use --dataset, --all, or --config.")


def _cmd_format(args) -> None:
    config = _load_json(args.config)
    datasets = _resolve_datasets(args, config)
    output_root = args.output_root or config.get("output_root", "data/formatted")
    cache_dir = args.cache_dir or config.get("cache_dir", "data/cache")
    seed = args.seed if args.seed is not None else config.get("seed", 42)
    overwrite = args.overwrite or config.get("overwrite", False)

    manifests = []
    total = len(datasets)
    for index, dataset_name in enumerate(datasets, start=1):
        print(f"[{index}/{total}] {dataset_name}: starting", flush=True)
        formatter = create_formatter(dataset_name, seed=seed, cache_dir=cache_dir)
        manifest = formatter.write(output_root, overwrite=overwrite)
        manifests.append(manifest)
        split_counts = manifest.get("split_counts", {})
        counts_summary = ", ".join(f"{split}={count}" for split, count in split_counts.items())
        status = manifest.get("status", "written")
        if status == "skipped":
            print(f"[{index}/{total}] {dataset_name}: skipped", flush=True)
        else:
            suffix = f" ({counts_summary})" if counts_summary else ""
            print(f"[{index}/{total}] {dataset_name}: done{suffix}", flush=True)

    print(json.dumps({"output_root": output_root, "datasets": manifests}, indent=2, ensure_ascii=False))


def _cmd_reference(args) -> None:
    summary = build_reference_summary(args.input_root, args.output_path)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


def _cmd_reference_build(args) -> None:
    summary = build_bucket_references(
        args.records_path,
        args.features_path,
        args.output_root,
        min_bucket_size=args.min_bucket_size,
        overwrite=args.overwrite,
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))


def _cmd_dedup(args) -> None:
    report = dedup_formatted_corpus(
        args.input_root,
        args.output_root,
        args.report_path,
        overwrite=args.overwrite,
    )
    print(json.dumps(report, indent=2, ensure_ascii=False))


def _cmd_pipeline(args) -> None:
    from polyalign_data.pipeline import run_pipeline

    config = _load_json(args.config)
    if args.overwrite:
        config["overwrite"] = True
    summary = run_pipeline(config)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="PolyAlign dataset preprocessing CLI.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    format_parser = subparsers.add_parser("format", help="Format one or more datasets into the unified schema.")
    format_parser.add_argument("--config", help="Path to a JSON config file.")
    format_parser.add_argument("--dataset", action="append", help="Dataset name to format. Repeatable.")
    format_parser.add_argument("--all", action="store_true", help="Format all registered datasets.")
    format_parser.add_argument("--output-root", help="Directory where formatted JSONL files will be written.")
    format_parser.add_argument("--cache-dir", help="Directory used for raw downloads and cache.")
    format_parser.add_argument("--seed", type=int, help="Deterministic split seed.")
    format_parser.add_argument("--overwrite", action="store_true", help="Overwrite existing dataset output directories.")
    format_parser.set_defaults(func=_cmd_format)

    reference_parser = subparsers.add_parser("reference", help="Build the initial bucket/reference summary.")
    reference_parser.add_argument("--input-root", required=True, help="Formatted dataset root directory.")
    reference_parser.add_argument("--output-path", required=True, help="Output JSON path for the summary.")
    reference_parser.set_defaults(func=_cmd_reference)

    reference_build_parser = subparsers.add_parser(
        "reference-build",
        help="Build full bucket reference artifacts from merged current JSONL files and matching feature files.",
    )
    reference_build_parser.add_argument("--records-path", action="append", required=True, help="Path to a merged current-format JSONL file. Repeatable.")
    reference_build_parser.add_argument("--features-path", action="append", required=True, help="Path to a matching feature JSONL file. Repeatable.")
    reference_build_parser.add_argument("--output-root", required=True, help="Output directory for reference artifacts.")
    reference_build_parser.add_argument("--min-bucket-size", type=int, default=20, help="Minimum examples required to keep a bucket.")
    reference_build_parser.add_argument("--overwrite", action="store_true", help="Overwrite the output directory if it already exists.")
    reference_build_parser.set_defaults(func=_cmd_reference_build)

    dedup_parser = subparsers.add_parser("dedup", help="Deduplicate formatted datasets with evaluation-safe priority.")
    dedup_parser.add_argument("--input-root", required=True, help="Formatted dataset root directory.")
    dedup_parser.add_argument("--output-root", required=True, help="Output root for deduplicated dataset files.")
    dedup_parser.add_argument("--report-path", required=True, help="Output JSON path for the dedup report.")
    dedup_parser.add_argument("--overwrite", action="store_true", help="Overwrite the output root if it already exists.")
    dedup_parser.set_defaults(func=_cmd_dedup)

    pipeline_parser = subparsers.add_parser(
        "pipeline",
        help="Run the full preprocessing pipeline: format, formatted summary, dedup, merged SFT views, features, and reference build.",
    )
    pipeline_parser.add_argument("--config", required=True, help="Path to a JSON pipeline config file.")
    pipeline_parser.add_argument("--overwrite", action="store_true", help="Override the config and rebuild all pipeline outputs.")
    pipeline_parser.set_defaults(func=_cmd_pipeline)

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)
