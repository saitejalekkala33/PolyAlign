from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import math
import os
import random
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable


HF_DATASET_REPO = "saiteja33/PolyAlign-All"


SOURCE_CONFIGS: dict[str, dict[str, Any]] = {
    "en": {
        "hf_prefix": "english",
        "human_hf_path": "english/merged_sft_dedup/llamafactory/test.json",
        "current_hf_path": "english/merged_sft_dedup/current/test.jsonl",
        "sources": [
            # ("qwen25_1_5b__baselm", "qwen25_1_5b", "baselm", "english/merged_sft_dedup/runs/qwen25_1_5b/predictions.jsonl"),
            # ("qwen25_1_5b__sft", "qwen25_1_5b", "sft", "english/merged_sft_dedup/runs/qwen25_1_5_sft/predictions.jsonl"),
            # ("qwen25_1_5b__dpo", "qwen25_1_5b", "dpo", "english/merged_sft_dedup/runs/qwen25-1-5b-dpo-en/predictions.jsonl"),
            # ("qwen25_1_5b__dist_sft", "qwen25_1_5b", "dist_sft", "english/merged_sft_dedup/runs/qwen25_1_5b_dist_sft_test-en/predictions.jsonl"),
            # ("gemma2_2b__baselm", "gemma2_2b", "baselm", "english/merged_sft_dedup/runs/gemma_2_2b/predictions.jsonl"),
            # ("gemma2_2b__sft", "gemma2_2b", "sft", "english/merged_sft_dedup/runs/gemma2_2b_sft-test-en/predictions.jsonl"),
            # ("gemma2_2b__dpo", "gemma2_2b", "dpo", "english/merged_sft_dedup/runs/gemma2-2b-dpo-en/predictions.jsonl"),
            # ("gemma2_2b__dist_sft", "gemma2_2b", "dist_sft", "english/merged_sft_dedup/runs/gemma2_2b_dist-sft-test-en/predictions.jsonl"),
            # ("qwen25_3b__baselm", "qwen25_3b", "baselm", "english/merged_sft_dedup/runs/qwen25_3b/predictions.jsonl"),
            # ("qwen25_3b__sft", "qwen25_3b", "sft", "english/merged_sft_dedup/runs/qwen25_3b_sft-test-en/predictions.jsonl"),
            ("qwen25_3b__dpo", "qwen25_3b", "dpo", "english/merged_sft_dedup/runs/qwen25-3b-dpo-en/predictions.jsonl"),
            # ("qwen25_3b__dist_sft", "qwen25_3b", "dist_sft", "english/merged_sft_dedup/runs/qwen_3b_dist-sft-test-en/predictions.jsonl"),
            # ("llama32_3b__baselm", "llama32_3b", "baselm", "english/merged_sft_dedup/runs/llama32_3b/predictions.jsonl"),
            # ("llama32_3b__sft", "llama32_3b", "sft", "english/merged_sft_dedup/runs/llama32_3b_sft-test-en/predictions.jsonl"),
            ("llama32_3b__dpo", "llama32_3b", "dpo", "english/merged_sft_dedup/runs/llama32-3b-dpo-en/predictions.jsonl"),
            ("llama32_3b__dist_sft", "llama32_3b", "dist_sft", "english/merged_sft_dedup/runs/llama32_3b_dist-sft-test-en/predictions.jsonl"),
        ],
    },
    "zh": {
        "hf_prefix": "chinese",
        "human_hf_path": "chinese/merged_sft_dedup/llamafactory/test.json",
        "current_hf_path": "chinese/merged_sft_dedup/current/test.jsonl",
        "sources": [
            (
                "qwen25_1_5b__hdpo",
                "qwen25_1_5b",
                "hdpo",
                "chinese/merged_sft_dedup/runs/qwen25-1-5b-hdpo-zh-ref-conditioned/predictions.jsonl",
            ),
            (
                "gemma2_2b__hdpo",
                "gemma2_2b",
                "hdpo",
                "chinese/merged_sft_dedup/runs/gemma2-2b-hdpo-zh-ref-conditioned/predictions.jsonl",
            ),
            (
                "qwen25_3b__hdpo",
                "qwen25_3b",
                "hdpo",
                "chinese/merged_sft_dedup/runs/qwen25-3b-hdpo-zh-ref-conditioned/predictions.jsonl",
            ),
            (
                "llama32_3b__hdpo",
                "llama32_3b",
                "hdpo",
                "chinese/merged_sft_dedup/runs/llama32-3b-hdpo-zh-ref-conditioned/predictions.jsonl",
            ),
        ],
    },
}


def _normalize_source_entry(entry: Any) -> tuple[str, str, str, str]:
    if isinstance(entry, dict):
        values = (
            entry.get("source_id"),
            entry.get("model_key"),
            entry.get("stage"),
            entry.get("path") or entry.get("rel_path") or entry.get("hf_path"),
        )
    else:
        values = tuple(entry) if isinstance(entry, (list, tuple)) else ()
    if len(values) != 4 or not all(isinstance(value, str) and value for value in values):
        raise ValueError(f"Invalid source config entry: {entry!r}")
    return values  # type: ignore[return-value]


def _apply_source_config_overrides() -> None:
    config_text = os.environ.get("POLYALIGN_AI_SOURCE_CONFIG_JSON")
    config_file = os.environ.get("POLYALIGN_AI_SOURCE_CONFIG_FILE")
    if config_file:
        config_text = Path(config_file).read_text(encoding="utf-8")
    if not config_text:
        return

    raw_config = json.loads(config_text)
    if not isinstance(raw_config, dict):
        raise ValueError("POLYALIGN_AI_SOURCE_CONFIG must decode to a JSON object.")

    for lang, override in raw_config.items():
        if lang not in SOURCE_CONFIGS:
            raise ValueError(f"Unknown language in source config override: {lang}")
        if not isinstance(override, dict):
            raise ValueError(f"Source config override for {lang} must be an object.")

        merged = dict(SOURCE_CONFIGS[lang])
        for key in ("hf_prefix", "human_hf_path", "current_hf_path"):
            if key in override:
                value = override[key]
                if not isinstance(value, str) or not value:
                    raise ValueError(f"{lang}.{key} must be a non-empty string.")
                merged[key] = value

        if "sources" in override:
            sources = override["sources"]
            if not isinstance(sources, list):
                raise ValueError(f"{lang}.sources must be a list.")
            merged["sources"] = [_normalize_source_entry(entry) for entry in sources]

        SOURCE_CONFIGS[lang] = merged


_apply_source_config_overrides()


DETECTOR_ORDER = ("binoculars", "fast_detect_gpt", "ghostbuster", "radar", "detect_gpt")
STAGE_ORDER = ("baselm", "sft", "dpo", "dist_sft", "hdpo")


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def write_json(path: Path, payload: Any) -> None:
    ensure_dir(path.parent)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def iter_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    decoder = json.JSONDecoder()
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue

            index = 0
            while index < len(line):
                while index < len(line) and line[index].isspace():
                    index += 1
                if index >= len(line):
                    break
                item, end = decoder.raw_decode(line, index)
                if not isinstance(item, dict):
                    raise ValueError(f"Expected JSON object in {path}, found {type(item).__name__}.")
                yield item
                index = end


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]], *, append: bool = False) -> None:
    ensure_dir(path.parent)
    mode = "a" if append else "w"
    with path.open(mode, encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def load_json_array(path: Path) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError(f"Expected JSON array at {path}")
    return payload


def load_concatenated_json(path: Path) -> list[dict[str, Any]]:
    if path.suffix.lower() == ".jsonl":
        return list(iter_jsonl(path))
    text = path.read_text(encoding="utf-8")
    decoder = json.JSONDecoder()
    rows: list[dict[str, Any]] = []
    index = 0
    while index < len(text):
        while index < len(text) and text[index].isspace():
            index += 1
        if index >= len(text):
            break
        item, end = decoder.raw_decode(text, index)
        if not isinstance(item, dict):
            raise ValueError(f"Expected JSON object in {path}")
        rows.append(item)
        index = end
    return rows


def count_json_records(path: Path, *, json_array: bool = False) -> int:
    if json_array:
        return len(load_json_array(path))
    if path.suffix.lower() == ".jsonl":
        return sum(1 for _ in iter_jsonl(path))
    return len(load_concatenated_json(path))


def sha1_text(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()


def normalize_text(value: Any) -> str:
    if value is None:
        return ""
    return str(value).replace("\r\n", "\n").replace("\r", "\n").strip()


def normalize_for_compare(value: Any) -> str:
    return " ".join(normalize_text(value).split())


def canonical_history(value: Any) -> list[list[str]]:
    if not isinstance(value, list):
        return []
    pairs: list[list[str]] = []
    for item in value:
        if isinstance(item, (list, tuple)) and len(item) == 2:
            user_text = normalize_for_compare(item[0])
            assistant_text = normalize_for_compare(item[1])
            if user_text or assistant_text:
                pairs.append([user_text, assistant_text])
    return pairs


def list_downloads(args: argparse.Namespace) -> None:
    paths: list[str] = []
    languages = ["en", "zh"] if args.lang == "all" else [args.lang]
    for lang in languages:
        config = SOURCE_CONFIGS[lang]
        paths.append(config["current_hf_path"])
        if args.include_human:
            paths.append(config["human_hf_path"])
        for _source_id, _model_key, _stage, rel_path in config["sources"]:
            paths.append(rel_path)
    for path in dict.fromkeys(paths):
        print(path)


def _history_from_current(row: dict[str, Any]) -> list[list[str]]:
    history = row.get("dialogue_history")
    if not isinstance(history, list):
        return []
    pairs: list[list[str]] = []
    pending: str | None = None
    question = normalize_text(row.get("question"))
    turns = list(history)
    if turns and turns[-1].get("role") == "user" and normalize_text(turns[-1].get("text")) == question:
        turns = turns[:-1]
    for turn in turns:
        role = turn.get("role")
        text = normalize_text(turn.get("text", turn.get("content", "")))
        if role == "user":
            pending = text
        elif role == "assistant" and pending is not None:
            pairs.append([pending, text])
            pending = None
    return pairs


def _prediction_index_map(path: Path, expected_len: int) -> tuple[dict[int, dict[str, Any]], dict[str, int]]:
    rows = load_concatenated_json(path)
    indexed: dict[int, dict[str, Any]] = {}
    bad_index_rows = 0
    duplicate_rows = 0
    empty_prediction_rows = 0
    for row in rows:
        source_index = row.get("source_index")
        if not isinstance(source_index, int) or not (0 <= source_index < expected_len):
            bad_index_rows += 1
            continue
        if source_index in indexed:
            duplicate_rows += 1
            continue
        prediction = normalize_text(row.get("prediction", ""))
        if not prediction:
            empty_prediction_rows += 1
            continue
        indexed[source_index] = row
    report = {
        "rows": len(rows),
        "usable_rows": len(indexed),
        "bad_index_rows": bad_index_rows,
        "duplicate_rows_ignored": duplicate_rows,
        "empty_prediction_rows": empty_prediction_rows,
    }
    return indexed, report


def _valid_prediction_indices(
    *,
    prediction_map: dict[int, dict[str, Any]],
    human_rows: list[dict[str, Any]],
    current_rows: list[dict[str, Any]],
) -> tuple[set[int], dict[str, int]]:
    valid_indices: set[int] = set()
    report = {
        "instruction_mismatches": 0,
        "input_mismatches": 0,
    }
    for source_index, row in prediction_map.items():
        human = human_rows[source_index]
        current = current_rows[source_index]
        row_valid = True
        expected_instruction = current.get("question", human.get("instruction", ""))
        expected_input = current.get("context", human.get("input", ""))
        if "instruction" in row and normalize_for_compare(row.get("instruction")) != normalize_for_compare(expected_instruction):
            report["instruction_mismatches"] += 1
            row_valid = False
        if "input" in row and normalize_for_compare(row.get("input")) != normalize_for_compare(expected_input):
            report["input_mismatches"] += 1
            row_valid = False
        if row_valid:
            valid_indices.add(source_index)
    return valid_indices, report


def _human_path_for_lang(args: argparse.Namespace, lang: str, raw_root: Path) -> Path:
    explicit = getattr(args, f"human_{lang}", None)
    if explicit:
        return Path(explicit)
    return raw_root / SOURCE_CONFIGS[lang]["human_hf_path"]


def check_row_counts(args: argparse.Namespace) -> None:
    raw_root = Path(args.raw_root)
    languages = ["en", "zh"] if args.lang == "all" else [args.lang]
    combined: dict[str, Any] = {}
    failed = False

    for lang in languages:
        config = SOURCE_CONFIGS[lang]
        human_path = _human_path_for_lang(args, lang, raw_root)
        records: list[dict[str, Any]] = []

        if human_path.exists():
            human_count = count_json_records(human_path, json_array=True)
            records.append(
                {
                    "source_id": "human",
                    "model_key": "human",
                    "stage": "human",
                    "label": "human",
                    "path": str(human_path),
                    "row_count": human_count,
                    "status": "present",
                }
            )
        else:
            records.append(
                {
                    "source_id": "human",
                    "model_key": "human",
                    "stage": "human",
                    "label": "human",
                    "path": str(human_path),
                    "row_count": None,
                    "status": "missing",
                }
            )

        for source_id, model_key, stage, rel_path in config["sources"]:
            path = raw_root / rel_path
            if path.exists():
                row_count = count_json_records(path)
                status = "present"
            else:
                row_count = None
                status = "missing"
            records.append(
                {
                    "source_id": source_id,
                    "model_key": model_key,
                    "stage": stage,
                    "label": "ai",
                    "path": str(path),
                    "row_count": row_count,
                    "status": status,
                }
            )

        present_counts = [record["row_count"] for record in records if record["row_count"] is not None]
        missing_count = sum(1 for record in records if record["row_count"] is None)
        expected_count = present_counts[0] if present_counts else None
        all_present_same_count = len(set(present_counts)) <= 1 if present_counts else False
        ok = all_present_same_count and (args.allow_missing or missing_count == 0)
        failed = failed or not ok

        summary = {
            "language": lang,
            "status": "ok" if ok else "mismatch_or_missing",
            "expected_count": expected_count,
            "all_present_same_count": all_present_same_count,
            "missing_count": missing_count,
            "sources": records,
        }
        combined[lang] = summary

        print()
        print(f"[row-counts] {lang}: {summary['status']}")
        print(f"  present_same_count={all_present_same_count} expected_count={expected_count} missing={missing_count}")
        print("  source_id                         label  rows      status")
        for record in records:
            count_text = "-" if record["row_count"] is None else str(record["row_count"])
            print(f"  {record['source_id'][:32]:32s} {record['label'][:5]:5s} {count_text:9s} {record['status']}")

    if args.output_json:
        write_json(Path(args.output_json), combined)

    if failed:
        raise SystemExit(1)


def _row_at(rows: list[dict[str, Any]], index: int) -> dict[str, Any]:
    if 0 <= index < len(rows):
        return rows[index]
    return {}


def _source_index_for_row(row: dict[str, Any], row_number: int) -> tuple[int, Any, bool]:
    original = row.get("source_index")
    if isinstance(original, int) and original >= 0:
        return original, original, True
    return row_number, original, False


def build_inputs(args: argparse.Namespace) -> None:
    raw_root = Path(args.raw_root)
    output_root = Path(args.output_root)
    languages = ["en", "zh"] if args.lang == "all" else [args.lang]
    combined_summary: dict[str, Any] = {}

    for lang in languages:
        config = SOURCE_CONFIGS[lang]
        human_path = _human_path_for_lang(args, lang, raw_root)
        current_path = raw_root / config["current_hf_path"]
        if not human_path.exists():
            raise FileNotFoundError(f"Missing human test file for {lang}: {human_path}")
        if not current_path.exists():
            raise FileNotFoundError(f"Missing current test file for {lang}: {current_path}")

        human_rows = load_json_array(human_path)
        current_rows = list(iter_jsonl(current_path))

        source_meta: dict[str, dict[str, Any]] = {}
        missing_source_files: list[dict[str, Any]] = []
        per_source_reports: dict[str, Any] = {}

        for source_id, model_key, stage, rel_path in config["sources"]:
            path = raw_root / rel_path
            if not path.exists():
                missing_source_files.append(
                    {"source_id": source_id, "model_key": model_key, "stage": stage, "path": str(path)}
                )
                continue
            source_meta[source_id] = {
                "source_id": source_id,
                "model_key": model_key,
                "stage": stage,
                "path": str(path),
                "label": 1,
            }
            prediction_rows = load_concatenated_json(path)
            seen_indices: set[int] = set()
            missing_source_index_rows = 0
            duplicate_source_index_rows = 0
            empty_text_rows = 0
            for row_number, prediction_row in enumerate(prediction_rows):
                source_index, _original_source_index, index_is_valid = _source_index_for_row(prediction_row, row_number)
                if not index_is_valid:
                    missing_source_index_rows += 1
                if source_index in seen_indices:
                    duplicate_source_index_rows += 1
                seen_indices.add(source_index)
                if not normalize_text(prediction_row.get("prediction", "")):
                    empty_text_rows += 1
            per_source_reports[source_id] = {
                "rows": len(prediction_rows),
                "missing_or_bad_source_index_rows": missing_source_index_rows,
                "duplicate_source_index_rows": duplicate_source_index_rows,
                "empty_text_rows": empty_text_rows,
            }

        if not source_meta:
            raise ValueError(f"{lang}: no prediction files were found; cannot build detector inputs")

        inputs_dir = output_root / lang / "inputs"
        ensure_dir(inputs_dir)
        manifest_sources = [
            {
                "source_id": "human",
                "model_key": "human",
                "stage": "human",
                "path": str(human_path),
                "label": 0,
                "input_path": str(inputs_dir / "human.jsonl"),
                "row_count": len(human_rows),
            }
        ]

        def base_payload(source_index: int, *, source_row_index: int, source_id: str, source_row: dict[str, Any] | None = None) -> dict[str, Any]:
            current = _row_at(current_rows, source_index)
            human = _row_at(human_rows, source_index)
            source_row = source_row or {}
            return {
                "source_index": source_index,
                "source_row_index": source_row_index,
                "source_uid": f"{source_id}:{source_row_index}",
                "id": current.get("id", source_row.get("id", human.get("id", ""))),
                "dataset": current.get("dataset", source_row.get("dataset", human.get("dataset", ""))),
                "split": current.get("split", source_row.get("split", human.get("split", ""))),
                "language": lang,
                "track": current.get("track", source_row.get("track", human.get("track", ""))),
                "family": current.get("family", source_row.get("family", human.get("family", ""))),
                "style_bucket": current.get("style_bucket", source_row.get("style_bucket", human.get("style_bucket", ""))),
                "length_bin": current.get("length_bin", source_row.get("length_bin", human.get("length_bin", ""))),
                "bucket_id": current.get("bucket_id", source_row.get("bucket_id", human.get("bucket_id", ""))),
                "instruction": source_row.get("instruction", human.get("instruction", current.get("question", ""))),
                "input": source_row.get("input", human.get("input", current.get("context", ""))),
                "history": source_row.get("history", human.get("history", _history_from_current(current))),
            }

        human_rows_out: list[dict[str, Any]] = []
        for source_index, human_row in enumerate(human_rows):
            current = _row_at(current_rows, source_index)
            payload = base_payload(source_index, source_row_index=source_index, source_id="human", source_row=human_row)
            text = normalize_text(human_row.get("output", current.get("human_answer", "")))
            payload.update(
                {
                    "source_id": "human",
                    "model_key": "human",
                    "stage": "human",
                    "label": 0,
                    "text": text,
                    "text_sha1": sha1_text(text),
                }
            )
            human_rows_out.append(payload)
        write_jsonl(inputs_dir / "human.jsonl", human_rows_out)

        for source_id, meta in source_meta.items():
            prediction_rows = load_concatenated_json(Path(meta["path"]))
            meta = source_meta[source_id]
            rows_out = []
            for row_number, pred_row in enumerate(prediction_rows):
                source_index, original_source_index, index_is_valid = _source_index_for_row(pred_row, row_number)
                payload = base_payload(source_index, source_row_index=row_number, source_id=source_id, source_row=pred_row)
                text = normalize_text(pred_row.get("prediction", ""))
                payload.update(
                    {
                        "source_id": source_id,
                        "model_key": meta["model_key"],
                        "stage": meta["stage"],
                        "label": 1,
                        "text": text,
                        "text_sha1": sha1_text(text),
                        "finish_reason": pred_row.get("finish_reason", ""),
                        "prediction_model_name": pred_row.get("model_name", ""),
                        "original_source_index": original_source_index,
                        "source_index_is_valid": index_is_valid,
                    }
                )
                rows_out.append(payload)
            input_path = inputs_dir / f"{source_id}.jsonl"
            write_jsonl(input_path, rows_out)
            manifest_sources.append({**meta, "input_path": str(input_path), "row_count": len(rows_out)})

        summary = {
            "language": lang,
            "mode": "all_rows",
            "human_path": str(human_path),
            "current_path": str(current_path),
            "human_rows": len(human_rows),
            "current_rows": len(current_rows),
            "present_prediction_sources": len(source_meta),
            "missing_prediction_sources": missing_source_files,
            "per_source_reports": per_source_reports,
            "sources": manifest_sources,
        }
        write_json(output_root / lang / "input_summary.json", summary)
        write_json(output_root / lang / "manifest.json", {"language": lang, "sources": manifest_sources})
        combined_summary[lang] = summary
        print(
            f"[inputs] {lang}: wrote all rows for human ({len(human_rows)}) "
            f"and {len(source_meta)} prediction sources"
        )

    write_json(output_root / "input_summary.json", combined_summary)


@dataclass
class ScoreResult:
    raw_score: float | None
    score_ai: float | None
    predicted_ai: bool | None
    error: str | None = None
    extra: dict[str, Any] | None = None


class BaseScorer:
    detector_name = "base"

    def score_batch(self, texts: list[str]) -> list[ScoreResult]:
        return [self.score_one(text) for text in texts]

    def score_one(self, text: str) -> ScoreResult:
        raise NotImplementedError


class RadarScorer(BaseScorer):
    detector_name = "radar"

    def __init__(self, model_path: Path, device: str, batch_size: int, max_length: int) -> None:
        import torch
        import torch.nn.functional as F
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        self.torch = torch
        self.F = F
        self.device = device
        self.batch_size = batch_size
        self.max_length = max_length
        self.tokenizer = AutoTokenizer.from_pretrained(str(model_path), local_files_only=True)
        self.model = AutoModelForSequenceClassification.from_pretrained(str(model_path), local_files_only=True).to(device)
        self.model.eval()

    def score_batch(self, texts: list[str]) -> list[ScoreResult]:
        results: list[ScoreResult] = []
        with self.torch.no_grad():
            for start in range(0, len(texts), self.batch_size):
                batch = texts[start : start + self.batch_size]
                inputs = self.tokenizer(
                    batch,
                    padding=True,
                    truncation=True,
                    max_length=self.max_length,
                    return_tensors="pt",
                )
                inputs = {key: value.to(self.device) for key, value in inputs.items()}
                probs = self.F.log_softmax(self.model(**inputs).logits, -1)[:, 0].exp().detach().cpu().tolist()
                for prob in probs:
                    results.append(ScoreResult(raw_score=float(prob), score_ai=float(prob), predicted_ai=prob >= 0.5))
        return results


class BinocularsScorer(BaseScorer):
    detector_name = "binoculars"

    def __init__(
        self,
        detector_root: Path,
        observer_path: Path,
        performer_path: Path,
        batch_size: int,
        max_length: int,
        mode: str,
    ) -> None:
        sys.path.insert(0, str(detector_root))
        from binoculars import Binoculars

        self.detector = Binoculars(
            observer_name_or_path=str(observer_path),
            performer_name_or_path=str(performer_path),
            max_token_observed=max_length,
            mode=mode,
        )
        self.batch_size = batch_size
        self.threshold = float(self.detector.threshold)

    def score_batch(self, texts: list[str]) -> list[ScoreResult]:
        results: list[ScoreResult] = []
        for start in range(0, len(texts), self.batch_size):
            batch = texts[start : start + self.batch_size]
            raw_scores = self.detector.compute_score(batch)
            if isinstance(raw_scores, float):
                raw_scores = [raw_scores]
            for raw in raw_scores:
                raw_float = float(raw)
                predicted_ai = raw_float < self.threshold
                results.append(
                    ScoreResult(
                        raw_score=raw_float,
                        score_ai=-raw_float,
                        predicted_ai=predicted_ai,
                        extra={"threshold_raw": self.threshold, "higher_raw_score": "more_human"},
                    )
                )
        return results


def _normal_pdf(x: float, mu: float, sigma: float) -> float:
    return math.exp(-0.5 * ((x - mu) / sigma) ** 2) / (sigma * math.sqrt(2.0 * math.pi))


class FastDetectGPTScorer(BaseScorer):
    detector_name = "fast_detect_gpt"

    CLASSIFIER = {"mu0": 0.1603, "sigma0": 1.0791, "mu1": 2.4686, "sigma1": 2.1582}

    def __init__(self, sampling_path: Path, scoring_path: Path, device: str, batch_size: int, max_length: int) -> None:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.torch = torch
        self.device = device
        self.batch_size = batch_size
        self.max_length = max_length
        self.scoring_tokenizer = AutoTokenizer.from_pretrained(str(scoring_path), local_files_only=True)
        self.sampling_tokenizer = AutoTokenizer.from_pretrained(str(sampling_path), local_files_only=True)
        if self.scoring_tokenizer.pad_token is None:
            self.scoring_tokenizer.pad_token = self.scoring_tokenizer.eos_token
        if self.sampling_tokenizer.pad_token is None:
            self.sampling_tokenizer.pad_token = self.sampling_tokenizer.eos_token
        self.scoring_model = AutoModelForCausalLM.from_pretrained(
            str(scoring_path),
            local_files_only=True,
            torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
            device_map={"": device},
        )
        self.sampling_model = AutoModelForCausalLM.from_pretrained(
            str(sampling_path),
            local_files_only=True,
            torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
            device_map={"": device},
        )
        self.scoring_model.eval()
        self.sampling_model.eval()

    def _prob(self, criterion: float) -> float:
        p = self.CLASSIFIER
        pdf0 = _normal_pdf(criterion, p["mu0"], p["sigma0"])
        pdf1 = _normal_pdf(criterion, p["mu1"], p["sigma1"])
        denom = pdf0 + pdf1
        return float(pdf1 / denom) if denom > 0 else 0.0

    def score_batch(self, texts: list[str]) -> list[ScoreResult]:
        results: list[ScoreResult] = []
        for text in texts:
            try:
                results.append(self.score_one(text))
            except Exception as exc:
                results.append(ScoreResult(raw_score=None, score_ai=None, predicted_ai=None, error=f"{type(exc).__name__}: {exc}"))
        return results

    def score_one(self, text: str) -> ScoreResult:
        torch = self.torch
        score_tokens = self.scoring_tokenizer(
            text,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
            padding=True,
            return_token_type_ids=False,
        ).to(self.device)
        labels = score_tokens.input_ids[:, 1:]
        with torch.no_grad():
            logits_score = self.scoring_model(**score_tokens).logits[:, :-1]
            sample_tokens = self.sampling_tokenizer(
                text,
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt",
                padding=True,
                return_token_type_ids=False,
            ).to(self.device)
            logits_ref = self.sampling_model(**sample_tokens).logits[:, :-1]

            vocab_size = min(logits_ref.size(-1), logits_score.size(-1))
            logits_ref = logits_ref[:, :, :vocab_size]
            logits_score = logits_score[:, :, :vocab_size]
            labels = labels.clamp(max=vocab_size - 1)

            lprobs_score = torch.log_softmax(logits_score, dim=-1)
            probs_ref = torch.softmax(logits_ref, dim=-1)
            log_likelihood = lprobs_score.gather(dim=-1, index=labels.unsqueeze(-1)).squeeze(-1)
            mean_ref = (probs_ref * lprobs_score).sum(dim=-1)
            var_ref = (probs_ref * torch.square(lprobs_score)).sum(dim=-1) - torch.square(mean_ref)
            discrepancy = (log_likelihood.sum(dim=-1) - mean_ref.sum(dim=-1)) / var_ref.sum(dim=-1).sqrt().clamp_min(1e-6)
            criterion = float(discrepancy.item())
        prob = self._prob(criterion)
        return ScoreResult(raw_score=criterion, score_ai=prob, predicted_ai=prob >= 0.5, extra={"prob_ai": prob})


class DetectGPTScorer(BaseScorer):
    detector_name = "detect_gpt"

    def __init__(
        self,
        base_path: Path,
        mask_path: Path,
        device: str,
        batch_size: int,
        max_length: int,
        n_perturbations: int,
        pct_words_masked: float,
        span_length: int,
    ) -> None:
        import torch
        from transformers import AutoModelForCausalLM, AutoModelForSeq2SeqLM, AutoTokenizer

        self.torch = torch
        self.device = device
        self.batch_size = batch_size
        self.max_length = max_length
        self.n_perturbations = n_perturbations
        self.pct_words_masked = pct_words_masked
        self.span_length = span_length
        self.base_tokenizer = AutoTokenizer.from_pretrained(str(base_path), local_files_only=True)
        if self.base_tokenizer.pad_token is None:
            self.base_tokenizer.pad_token = self.base_tokenizer.eos_token
        self.base_model = AutoModelForCausalLM.from_pretrained(
            str(base_path),
            local_files_only=True,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            device_map={"": device},
        )
        self.mask_tokenizer = AutoTokenizer.from_pretrained(str(mask_path), local_files_only=True)
        self.mask_model = AutoModelForSeq2SeqLM.from_pretrained(
            str(mask_path),
            local_files_only=True,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            device_map={"": device},
        )
        self.base_model.eval()
        self.mask_model.eval()

    def _logprob(self, text: str) -> float:
        torch = self.torch
        tokens = self.base_tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=self.max_length,
            padding=True,
            return_token_type_ids=False,
        ).to(self.device)
        labels = tokens.input_ids[:, 1:]
        with torch.no_grad():
            logits = self.base_model(**tokens).logits[:, :-1]
            lprobs = torch.log_softmax(logits, dim=-1)
            gathered = lprobs.gather(dim=-1, index=labels.unsqueeze(-1)).squeeze(-1)
            return float(gathered.mean().item())

    def _mask_text(self, text: str) -> tuple[str, int]:
        words = text.split()
        if len(words) < 8:
            return text, 0
        n_spans = max(1, int(self.pct_words_masked * len(words) / max(self.span_length, 1)))
        n_spans = min(n_spans, max(1, len(words) // max(self.span_length + 1, 1)))
        used = [False] * len(words)
        starts: list[int] = []
        for _ in range(n_spans * 5):
            if len(starts) >= n_spans:
                break
            start = random.randint(0, max(0, len(words) - self.span_length))
            end = min(len(words), start + self.span_length)
            if any(used[start:end]):
                continue
            for i in range(start, end):
                used[i] = True
            starts.append(start)
        starts.sort()
        if not starts:
            return text, 0
        output: list[str] = []
        cursor = 0
        for sentinel_id, start in enumerate(starts):
            end = min(len(words), start + self.span_length)
            output.extend(words[cursor:start])
            output.append(f"<extra_id_{sentinel_id}>")
            cursor = end
        output.extend(words[cursor:])
        return " ".join(output), len(starts)

    def _extract_t5_fills(self, decoded: str, n_masks: int) -> list[str] | None:
        fills: list[str] = []
        for i in range(n_masks):
            start_token = f"<extra_id_{i}>"
            end_token = f"<extra_id_{i + 1}>"
            start = decoded.find(start_token)
            if start == -1:
                return None
            start += len(start_token)
            end = decoded.find(end_token, start)
            if end == -1:
                end = len(decoded)
            fills.append(decoded[start:end].strip())
        return fills

    def _perturb_once(self, text: str) -> str:
        masked, n_masks = self._mask_text(text)
        if n_masks == 0:
            return text
        torch = self.torch
        inputs = self.mask_tokenizer(masked, return_tensors="pt", truncation=True, max_length=512).to(self.device)
        with torch.no_grad():
            outputs = self.mask_model.generate(
                **inputs,
                do_sample=True,
                top_p=0.96,
                max_new_tokens=128,
                num_return_sequences=1,
            )
        decoded = self.mask_tokenizer.decode(outputs[0], skip_special_tokens=False)
        fills = self._extract_t5_fills(decoded, n_masks)
        if not fills:
            return text
        result = masked
        for i, fill in enumerate(fills):
            result = result.replace(f"<extra_id_{i}>", fill, 1)
        result = re.sub(r"<extra_id_\d+>", "", result)
        return " ".join(result.split()) or text

    def score_batch(self, texts: list[str]) -> list[ScoreResult]:
        results: list[ScoreResult] = []
        for text in texts:
            try:
                results.append(self.score_one(text))
            except Exception as exc:
                results.append(ScoreResult(raw_score=None, score_ai=None, predicted_ai=None, error=f"{type(exc).__name__}: {exc}"))
        return results

    def score_one(self, text: str) -> ScoreResult:
        original_ll = self._logprob(text)
        perturbed_lls = [self._logprob(self._perturb_once(text)) for _ in range(self.n_perturbations)]
        mean_pert = sum(perturbed_lls) / len(perturbed_lls)
        if len(perturbed_lls) > 1:
            variance = sum((value - mean_pert) ** 2 for value in perturbed_lls) / (len(perturbed_lls) - 1)
            std_pert = math.sqrt(max(variance, 1e-12))
        else:
            std_pert = 1.0
        z_score = (original_ll - mean_pert) / std_pert
        return ScoreResult(
            raw_score=z_score,
            score_ai=z_score,
            predicted_ai=z_score > 0.0,
            extra={"original_logprob": original_ll, "perturbed_logprob_mean": mean_pert, "perturbed_logprob_std": std_pert},
        )


def build_scorer(args: argparse.Namespace) -> BaseScorer:
    models_root = Path(args.models_root)
    detector_root = Path(args.detector_root)
    device = args.device
    if args.detector == "radar":
        return RadarScorer(models_root / "radar" / "RADAR-Vicuna-7B", device, args.batch_size, args.max_length)
    if args.detector == "binoculars":
        return BinocularsScorer(
            detector_root / "Binoculars",
            models_root / "binoculars" / "falcon-7b",
            models_root / "binoculars" / "falcon-7b-instruct",
            args.batch_size,
            args.max_length,
            args.binoculars_mode,
        )
    if args.detector == "fast_detect_gpt":
        return FastDetectGPTScorer(
            models_root / "fast-detect-gpt" / "Meta-Llama-3-8B",
            models_root / "fast-detect-gpt" / "Meta-Llama-3-8B-Instruct",
            device,
            args.batch_size,
            args.max_length,
        )
    if args.detector == "detect_gpt":
        return DetectGPTScorer(
            models_root / "detect-gpt" / "gpt-neo-2.7B",
            models_root / "detect-gpt" / "t5-mask",
            device,
            args.batch_size,
            args.max_length,
            args.detectgpt_perturbations,
            args.detectgpt_pct_words_masked,
            args.detectgpt_span_length,
        )
    raise ValueError(f"Unsupported detector for scoring: {args.detector}")


def _completion_keys(row: dict[str, Any]) -> list[tuple[str, Any]]:
    keys: list[tuple[str, Any]] = []
    source_uid = row.get("source_uid")
    if source_uid:
        keys.append(("uid", source_uid))
    source_index = row.get("source_index")
    if isinstance(source_index, int):
        keys.append(("idx", source_index))
    return keys


def completed_keys(path: Path) -> set[tuple[str, Any]]:
    if not path.exists():
        return set()
    done: set[tuple[str, Any]] = set()
    for row in iter_jsonl(path):
        done.update(_completion_keys(row))
    return done


def _clear_cuda_cache_after_error() -> None:
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        return


def detector_error_result(exc: Exception, text: str) -> ScoreResult:
    message = f"[DETECTOR_ERROR] {type(exc).__name__}: {exc}"
    return ScoreResult(
        raw_score=None,
        score_ai=None,
        predicted_ai=None,
        error=message[:1000],
        extra={
            "detector_error_type": type(exc).__name__,
            "text_length_chars": len(text),
            "text_word_count": len(text.split()),
        },
    )


def score_texts_safely(scorer: BaseScorer, texts: list[str]) -> list[ScoreResult]:
    try:
        results = scorer.score_batch(texts)
        if len(results) != len(texts):
            raise ValueError(f"{scorer.detector_name} returned {len(results)} scores for {len(texts)} texts")
        return results
    except Exception as exc:
        _clear_cuda_cache_after_error()
        if len(texts) <= 1:
            return [detector_error_result(exc, texts[0] if texts else "")]
        midpoint = max(1, len(texts) // 2)
        return score_texts_safely(scorer, texts[:midpoint]) + score_texts_safely(scorer, texts[midpoint:])


def score_file(scorer: BaseScorer, input_path: Path, output_path: Path, batch_size: int, resume: bool) -> dict[str, Any]:
    done = completed_keys(output_path) if resume else set()
    rows = [row for row in iter_jsonl(input_path) if not any(key in done for key in _completion_keys(row))]
    mode_append = resume and output_path.exists()
    total = len(rows)
    written = 0
    ensure_dir(output_path.parent)
    with output_path.open("a" if mode_append else "w", encoding="utf-8", buffering=1) as handle:
        for start in range(0, len(rows), batch_size):
            batch_rows = rows[start : start + batch_size]
            texts = [normalize_text(row.get("text", "")) for row in batch_rows]
            results = score_texts_safely(scorer, texts)
            for row, result in zip(batch_rows, results, strict=True):
                payload = {
                    "source_index": row.get("source_index"),
                    "id": row.get("id", ""),
                    "language": row.get("language", ""),
                    "source_uid": row.get("source_uid", ""),
                    "source_row_index": row.get("source_row_index"),
                    "source_id": row.get("source_id", ""),
                    "model_key": row.get("model_key", ""),
                    "stage": row.get("stage", ""),
                    "label": row.get("label"),
                    "detector": scorer.detector_name,
                    "raw_score": result.raw_score,
                    "score_ai": result.score_ai,
                    "predicted_ai": result.predicted_ai,
                    "error": result.error,
                    "text_sha1": row.get("text_sha1", ""),
                }
                if result.extra:
                    payload.update(result.extra)
                handle.write(json.dumps(payload, ensure_ascii=False) + "\n")
                written += 1
            print(f"[score] {scorer.detector_name} {input_path.name}: {written}/{total}", flush=True)
    return {"input_path": str(input_path), "output_path": str(output_path), "resumed_keys": len(done), "written_rows": written}


def score_lang(args: argparse.Namespace) -> None:
    work_dir = Path(args.work_dir)
    manifest_path = work_dir / args.lang / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    sources = manifest["sources"]
    selected = [source for idx, source in enumerate(sources) if idx % args.num_shards == args.shard_index]
    if not selected:
        print(f"[score] no sources for shard {args.shard_index}/{args.num_shards}")
        return

    scorer = build_scorer(args)
    summaries = []
    for source in selected:
        input_path = Path(source["input_path"])
        output_path = work_dir / args.lang / "scores" / args.detector / f"{source['source_id']}.jsonl"
        summaries.append(score_file(scorer, input_path, output_path, args.batch_size, args.resume))
    write_json(work_dir / args.lang / "scores" / args.detector / f"shard_{args.shard_index}_summary.json", summaries)


def _load_scores(path: Path) -> list[dict[str, Any]]:
    return [row for row in iter_jsonl(path) if row.get("score_ai") is not None and row.get("predicted_ai") is not None]


def _roc_metrics(labels: list[int], scores: list[float]) -> dict[str, Any]:
    try:
        from sklearn.metrics import roc_auc_score, roc_curve

        auroc = float(roc_auc_score(labels, scores))
        fpr, tpr, thresholds = roc_curve(labels, scores)
        fpr_l = [float(x) for x in fpr]
        tpr_l = [float(x) for x in tpr]
        thresholds_l = [float(x) for x in thresholds]
    except Exception:
        paired = sorted(zip(scores, labels), key=lambda item: item[0])
        n_pos = sum(labels)
        n_neg = len(labels) - n_pos
        rank_sum = sum(rank for rank, (_score, label) in enumerate(paired, start=1) if label == 1)
        auroc = (rank_sum - n_pos * (n_pos + 1) / 2) / max(n_pos * n_neg, 1)
        fpr_l, tpr_l, thresholds_l = [0.0, 1.0], [0.0, 1.0], [float("inf"), float("-inf")]

    def tpr_at(max_fpr: float) -> float:
        values = [t for f, t in zip(fpr_l, tpr_l, strict=True) if f <= max_fpr]
        return max(values) if values else 0.0

    return {
        "auroc": auroc,
        "tpr_at_fpr_0_01": tpr_at(0.01),
        "tpr_at_fpr_0_05": tpr_at(0.05),
        "tpr_at_fpr_0_10": tpr_at(0.10),
        "roc": {"fpr": fpr_l, "tpr": tpr_l, "thresholds": thresholds_l},
    }


def summarize(args: argparse.Namespace) -> None:
    work_dir = Path(args.work_dir)
    manifest = json.loads((work_dir / args.lang / "manifest.json").read_text(encoding="utf-8"))
    score_dir = work_dir / args.lang / "scores" / args.detector
    output_path = work_dir / args.lang / "metrics" / args.detector / "summary.json"
    existing_summary: dict[str, Any] = {}
    if args.merge_existing and output_path.exists():
        existing_summary = json.loads(output_path.read_text(encoding="utf-8"))
    human_scores = _load_scores(score_dir / "human.jsonl")
    source_summaries: dict[str, Any] = dict(existing_summary.get("source_summaries", {}))
    pair_metrics: dict[str, Any] = dict(existing_summary.get("pair_metrics_vs_human", {}))

    for source in manifest["sources"]:
        source_id = source["source_id"]
        path = score_dir / f"{source_id}.jsonl"
        if not path.exists():
            source_summaries[source_id] = {"status": "missing_scores", "path": str(path)}
            continue
        rows = _load_scores(path)
        pred_ai = sum(1 for row in rows if row.get("predicted_ai") is True)
        pred_human = sum(1 for row in rows if row.get("predicted_ai") is False)
        source_summaries[source_id] = {
            "status": "ok",
            "source_id": source_id,
            "model_key": source["model_key"],
            "stage": source["stage"],
            "label": source["label"],
            "n_scored": len(rows),
            "n_errors_or_unscored": sum(1 for _ in iter_jsonl(path)) - len(rows),
            "predicted_ai_count": pred_ai,
            "predicted_human_count": pred_human,
            "predicted_ai_rate": pred_ai / len(rows) if rows else None,
            "score_ai_mean": sum(float(row["score_ai"]) for row in rows) / len(rows) if rows else None,
        }
        if source_id == "human" or not human_scores:
            continue

        n = min(len(human_scores), len(rows))
        human_by_index = {row["source_index"]: row for row in human_scores}
        ai_by_index = {row["source_index"]: row for row in rows}
        common_indices = sorted(set(human_by_index) & set(ai_by_index))
        labels = [0] * len(common_indices) + [1] * len(common_indices)
        scores = [float(human_by_index[i]["score_ai"]) for i in common_indices] + [
            float(ai_by_index[i]["score_ai"]) for i in common_indices
        ]
        metrics = _roc_metrics(labels, scores)
        human_pred_ai = sum(1 for i in common_indices if human_by_index[i].get("predicted_ai") is True)
        human_pred_human = sum(1 for i in common_indices if human_by_index[i].get("predicted_ai") is False)
        ai_pred_ai = sum(1 for i in common_indices if ai_by_index[i].get("predicted_ai") is True)
        ai_pred_human = sum(1 for i in common_indices if ai_by_index[i].get("predicted_ai") is False)
        metrics.update(
            {
                "n_common": len(common_indices),
                "confusion_matrix_at_detector_threshold": {
                    "human_pred_human": human_pred_human,
                    "human_pred_ai": human_pred_ai,
                    "ai_pred_human": ai_pred_human,
                    "ai_pred_ai": ai_pred_ai,
                },
                "ai_detected_rate_at_detector_threshold": ai_pred_ai / len(common_indices) if common_indices else None,
                "human_false_positive_rate_at_detector_threshold": human_pred_ai / len(common_indices) if common_indices else None,
            }
        )
        pair_metrics[source_id] = metrics

    trends: dict[str, Any] = {}
    model_keys = sorted(
        {
            summary.get("model_key")
            for summary in source_summaries.values()
            if isinstance(summary, dict) and summary.get("model_key") not in {None, "", "human"}
        }
    )
    for model_key in model_keys:
        points = []
        for stage in STAGE_ORDER:
            source_id = f"{model_key}__{stage}"
            summary = source_summaries.get(source_id)
            if summary and summary.get("status") == "ok":
                points.append({"stage": stage, "predicted_ai_count": summary["predicted_ai_count"], "predicted_ai_rate": summary["predicted_ai_rate"]})
        counts = [point["predicted_ai_count"] for point in points]
        trends[model_key] = {
            "available_stages": [point["stage"] for point in points],
            "predicted_ai_counts": counts,
            "predicted_ai_rates": [point["predicted_ai_rate"] for point in points],
            "non_increasing_ai_count": all(a >= b for a, b in zip(counts, counts[1:])),
        }

    summary = {
        "language": args.lang,
        "detector": args.detector,
        "score_dir": str(score_dir),
        "merge_existing": bool(args.merge_existing),
        "source_summaries": source_summaries,
        "pair_metrics_vs_human": pair_metrics,
        "trend_by_model": trends,
    }
    write_json(output_path, summary)
    print(f"[summary] wrote {output_path}")


def mark_skipped(args: argparse.Namespace) -> None:
    work_dir = Path(args.work_dir)
    payload = {
        "language": args.lang,
        "detector": args.detector,
        "status": "skipped",
        "reason": args.reason,
    }
    write_json(work_dir / args.lang / "metrics" / args.detector / "summary.json", payload)
    print(f"[skip] {args.lang}/{args.detector}: {args.reason}")


def combine_summaries(args: argparse.Namespace) -> None:
    work_dir = Path(args.work_dir)
    output_path = work_dir / "metrics" / "combined_summary.json"
    combined: dict[str, Any] = {}
    if args.merge_existing and output_path.exists():
        combined = json.loads(output_path.read_text(encoding="utf-8"))
    languages = ["en", "zh"] if args.lang == "all" else [args.lang]
    for lang in languages:
        combined[lang] = {}
        for detector in DETECTOR_ORDER:
            path = work_dir / lang / "metrics" / detector / "summary.json"
            if path.exists():
                combined[lang][detector] = json.loads(path.read_text(encoding="utf-8"))
            else:
                combined[lang][detector] = {"status": "missing"}
    write_json(output_path, combined)
    print(f"[summary] wrote {output_path}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="PolyAlign AI detector analysis pipeline helpers.")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("list-downloads")
    p.add_argument("--lang", choices=["all", "en", "zh"], default="all")
    p.add_argument("--include-human", action="store_true")
    p.set_defaults(func=list_downloads)

    p = sub.add_parser("check-row-counts")
    p.add_argument("--raw-root", required=True)
    p.add_argument("--lang", choices=["all", "en", "zh"], default="all")
    p.add_argument("--human-en")
    p.add_argument("--human-zh")
    p.add_argument("--allow-missing", action="store_true")
    p.add_argument("--output-json")
    p.set_defaults(func=check_row_counts)

    p = sub.add_parser("build-inputs")
    p.add_argument("--raw-root", required=True)
    p.add_argument("--output-root", required=True)
    p.add_argument("--lang", choices=["all", "en", "zh"], default="all")
    p.add_argument("--human-en")
    p.add_argument("--human-zh")
    p.set_defaults(func=build_inputs)

    p = sub.add_parser("score-lang")
    p.add_argument("--work-dir", required=True)
    p.add_argument("--lang", choices=["en", "zh"], required=True)
    p.add_argument("--detector", choices=["binoculars", "fast_detect_gpt", "radar", "detect_gpt"], required=True)
    p.add_argument("--models-root", required=True)
    p.add_argument("--detector-root", required=True)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--max-length", type=int, default=512)
    p.add_argument("--shard-index", type=int, default=0)
    p.add_argument("--num-shards", type=int, default=1)
    p.add_argument("--resume", action="store_true")
    p.add_argument("--binoculars-mode", choices=["low-fpr", "accuracy"], default="low-fpr")
    p.add_argument("--detectgpt-perturbations", type=int, default=10)
    p.add_argument("--detectgpt-pct-words-masked", type=float, default=0.3)
    p.add_argument("--detectgpt-span-length", type=int, default=2)
    p.set_defaults(func=score_lang)

    p = sub.add_parser("summarize")
    p.add_argument("--work-dir", required=True)
    p.add_argument("--lang", choices=["en", "zh"], required=True)
    p.add_argument("--detector", choices=list(DETECTOR_ORDER), required=True)
    p.add_argument("--merge-existing", action="store_true")
    p.set_defaults(func=summarize)

    p = sub.add_parser("mark-skipped")
    p.add_argument("--work-dir", required=True)
    p.add_argument("--lang", choices=["en", "zh"], required=True)
    p.add_argument("--detector", required=True)
    p.add_argument("--reason", required=True)
    p.set_defaults(func=mark_skipped)

    p = sub.add_parser("combine-summaries")
    p.add_argument("--work-dir", required=True)
    p.add_argument("--lang", choices=["all", "en", "zh"], default="all")
    p.add_argument("--merge-existing", action="store_true")
    p.set_defaults(func=combine_summaries)

    return parser


def main() -> None:
    args = build_parser().parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
