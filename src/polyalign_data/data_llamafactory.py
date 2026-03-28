from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from polyalign_data.io_utils import ensure_dir


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


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert LlamaFactory-format JSONL files into JSON array files accepted by LlamaFactory."
    )
    parser.add_argument("--input-path", help="Single LlamaFactory JSONL file to convert.")
    parser.add_argument("--output-path", help="Output JSON path for single-file conversion.")
    parser.add_argument("--input-root", help="Directory containing train/val/test LlamaFactory JSONL files.")
    parser.add_argument("--output-root", help="Output directory for converted JSON split files.")
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
    else:
        parser.error("Use either --input-path/--output-path or --input-root/--output-root.")

    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
