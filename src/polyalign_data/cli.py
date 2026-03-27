from __future__ import annotations

import argparse
import json
from pathlib import Path

from polyalign_data.reference_builder import build_reference_summary
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
    for dataset_name in datasets:
        formatter = create_formatter(dataset_name, seed=seed, cache_dir=cache_dir)
        manifest = formatter.write(output_root, overwrite=overwrite)
        manifests.append(manifest)

    print(json.dumps({"output_root": output_root, "datasets": manifests}, indent=2, ensure_ascii=False))


def _cmd_reference(args) -> None:
    summary = build_reference_summary(args.input_root, args.output_path)
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

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)
