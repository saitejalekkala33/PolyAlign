from __future__ import annotations

import argparse
import json
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


def _iter_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                yield json.loads(line)


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


def _to_llamafactory_alpaca(record: dict[str, Any]) -> dict[str, Any]:
    item = {
        "instruction": normalize_text(record.get("question", "")),
        "input": normalize_text(record.get("context", "")),
        "output": normalize_text(record.get("human_answer", "")),
    }
    history = _history_to_llamafactory_pairs(record)
    if history:
        item["history"] = history
    return item


def export_merged_sft_views(
    input_root: str | Path,
    output_root: str | Path,
    *,
    include_validation2: bool = False,
) -> dict[str, dict[str, int]]:
    input_dir = Path(input_root)
    output_dir = Path(output_root)
    current_dir = output_dir / "current"
    llamafactory_dir = output_dir / "llamafactory"
    ensure_dir(current_dir)
    ensure_dir(llamafactory_dir)

    source_splits = ["train", "dev", "test"]
    if include_validation2:
        source_splits.append("validation2")

    summary: dict[str, dict[str, int]] = {"current": {}, "llamafactory": {}}

    for source_split in source_splits:
        target_split = SOURCE_SPLIT_TO_TARGET.get(source_split, source_split)
        current_output_path = current_dir / f"{target_split}.jsonl"
        llamafactory_output_path = llamafactory_dir / f"{target_split}.jsonl"
        total_count = 0

        with current_output_path.open("w", encoding="utf-8") as current_handle, llamafactory_output_path.open(
            "w", encoding="utf-8"
        ) as llama_handle:
            for _dataset_name, split_path in _iter_dataset_split_files(input_dir, source_split):
                with split_path.open("r", encoding="utf-8") as source_handle:
                    for raw_line in source_handle:
                        line = raw_line.strip()
                        if not line:
                            continue
                        current_handle.write(line + "\n")
                        record = json.loads(line)
                        llama_handle.write(json.dumps(_to_llamafactory_alpaca(record), ensure_ascii=False) + "\n")
                        total_count += 1

        summary["current"][target_split] = total_count
        summary["llamafactory"][target_split] = total_count

    return summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Merge dataset-specific formatted JSONL files into combined train/val/test corpora "
        "in both the PolyAlign schema and the Alpaca-style LlamaFactory schema."
    )
    parser.add_argument("--input-root", required=True, help="Root directory containing per-dataset formatted JSONL files.")
    parser.add_argument(
        "--output-root",
        required=True,
        help="Output directory. The script writes `current/*.jsonl` and `llamafactory/*.jsonl` inside it.",
    )
    parser.add_argument(
        "--include-validation2",
        action="store_true",
        help="Also export the ELI5 auxiliary validation2 split as `validation2.jsonl`.",
    )
    args = parser.parse_args()
    summary = export_merged_sft_views(
        args.input_root,
        args.output_root,
        include_validation2=args.include_validation2,
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
