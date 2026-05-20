from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import os
import random
import re
import time
from pathlib import Path
from typing import Any, Iterable
from urllib import error, request


HF_DATASET_REPO = "saiteja33/PolyAlign-All"


SOURCE_CONFIGS: dict[str, dict[str, Any]] = {
    "en": {
        "hf_prefix": "english",
        "human_hf_path": "english/merged_sft_dedup/llamafactory/test.json",
        "current_hf_path": "english/merged_sft_dedup/current/test.jsonl",
        "sources": [
            ("qwen25_1_5b__baselm", "qwen25_1_5b", "baselm", "english/merged_sft_dedup/runs/qwen25_1_5b/predictions.jsonl"),
            ("qwen25_1_5b__sft", "qwen25_1_5b", "sft", "english/merged_sft_dedup/runs/qwen25_1_5_sft/predictions.jsonl"),
            ("qwen25_1_5b__dpo", "qwen25_1_5b", "dpo", "english/merged_sft_dedup/runs/qwen25-1-5b-dpo-en/predictions.jsonl"),
            ("qwen25_1_5b__dist_sft", "qwen25_1_5b", "dist_sft", "english/merged_sft_dedup/runs/qwen25_1_5b_dist_sft_test-en/predictions.jsonl"),
            ("gemma2_2b__baselm", "gemma2_2b", "baselm", "english/merged_sft_dedup/runs/gemma_2_2b/predictions.jsonl"),
            ("gemma2_2b__sft", "gemma2_2b", "sft", "english/merged_sft_dedup/runs/gemma2_2b_sft-test-en/predictions.jsonl"),
            ("gemma2_2b__dpo", "gemma2_2b", "dpo", "english/merged_sft_dedup/runs/gemma2-2b-dpo-en/predictions.jsonl"),
            ("gemma2_2b__dist_sft", "gemma2_2b", "dist_sft", "english/merged_sft_dedup/runs/gemma2_2b_dist-sft-test-en/predictions.jsonl"),
            ("qwen25_3b__baselm", "qwen25_3b", "baselm", "english/merged_sft_dedup/runs/qwen25_3b/predictions.jsonl"),
            ("qwen25_3b__sft", "qwen25_3b", "sft", "english/merged_sft_dedup/runs/qwen25_3b_sft-test-en/predictions.jsonl"),
            ("qwen25_3b__dist_sft", "qwen25_3b", "dist_sft", "english/merged_sft_dedup/runs/qwen_3b_dist-sft-test-en/predictions.jsonl"),
            ("llama32_3b__baselm", "llama32_3b", "baselm", "english/merged_sft_dedup/runs/llama32_3b/predictions.jsonl"),
            ("llama32_3b__sft", "llama32_3b", "sft", "english/merged_sft_dedup/runs/llama32_3b_sft-test-en/predictions.jsonl"),
        ],
    },
    "zh": {
        "hf_prefix": "chinese",
        "human_hf_path": "chinese/merged_sft_dedup/llamafactory/test.json",
        "current_hf_path": "chinese/merged_sft_dedup/current/test.jsonl",
        "sources": [
            ("qwen25_1_5b__baselm", "qwen25_1_5b", "baselm", "chinese/merged_sft_dedup/runs/qwen25_1_5b_zh/predictions.jsonl"),
            ("qwen25_1_5b__sft", "qwen25_1_5b", "sft", "chinese/merged_sft_dedup/runs/qwen25_1_5_sft-zh/predictions.jsonl"),
            ("qwen25_1_5b__dpo", "qwen25_1_5b", "dpo", "chinese/merged_sft_dedup/runs/qwen25-15b-dpo-zh/predictions.jsonl"),
            ("qwen25_1_5b__dist_sft", "qwen25_1_5b", "dist_sft", "chinese/merged_sft_dedup/runs/qwen25_1_5b_dist_sft_zh_test/predictions.jsonl"),
            ("gemma2_2b__baselm", "gemma2_2b", "baselm", "chinese/merged_sft_dedup/runs/gemma_2_2b_zh/predictions.jsonl"),
            ("gemma2_2b__sft", "gemma2_2b", "sft", "chinese/merged_sft_dedup/runs/gemma2-2b-sft-zh/predictions.jsonl"),
            ("gemma2_2b__dpo", "gemma2_2b", "dpo", "chinese/merged_sft_dedup/runs/gemma2-2b-dpo-zh/predictions.jsonl"),
            ("gemma2_2b__dist_sft", "gemma2_2b", "dist_sft", "chinese/merged_sft_dedup/runs/gemma2_2b_dist-sft-zh-test/predictions.jsonl"),
            ("qwen25_3b__baselm", "qwen25_3b", "baselm", "chinese/merged_sft_dedup/runs/qwen25_3b_zh/predictions.jsonl"),
            ("qwen25_3b__sft", "qwen25_3b", "sft", "chinese/merged_sft_dedup/runs/qwen25_3b_sft-zh-test/predictions.jsonl"),
            ("qwen25_3b__dpo", "qwen25_3b", "dpo", "chinese/merged_sft_dedup/runs/qwen25-3b-dpo-zh/predictions.jsonl"),
            ("qwen25_3b__dist_sft", "qwen25_3b", "dist_sft", "chinese/merged_sft_dedup/runs/qwen25_3b_dist-sft-zh-test/predictions.jsonl"),
            ("llama32_3b__baselm", "llama32_3b", "baselm", "chinese/merged_sft_dedup/runs/llama32_3b_zh/predictions.jsonl"),
            ("llama32_3b__sft", "llama32_3b", "sft", "chinese/merged_sft_dedup/runs/llama3_2-3b_sft-zh-test/predictions.jsonl"),
            ("llama32_3b__dpo", "llama32_3b", "dpo", "chinese/merged_sft_dedup/runs/llama32-3b-dpo-zh/predictions.jsonl"),
            ("llama32_3b__dist_sft", "llama32_3b", "dist_sft", "chinese/merged_sft_dedup/runs/llama32_3b_dist_sft_zh_test/predictions.jsonl"),
        ],
    },
}


STAGE_ORDER = ("human", "baselm", "sft", "dpo", "dist_sft", "hdpo")
DEFAULT_PROMPTS_PATH = Path(__file__).with_name("llm-judge-prompts.py")
DEFAULT_RUBRIC_PATH = Path(__file__).with_name("rubric.yaml")


try:
    from tqdm.auto import tqdm
except Exception:  # pragma: no cover - optional progress dependency
    def tqdm(iterable: Iterable[Any], **_: Any) -> Iterable[Any]:
        return iterable


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def write_json(path: Path, payload: Any) -> None:
    ensure_dir(path.parent)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]], *, append: bool = False) -> None:
    ensure_dir(path.parent)
    mode = "a" if append else "w"
    with path.open(mode, encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


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


def normalize_text(value: Any) -> str:
    if value is None:
        return ""
    return str(value).replace("\r\n", "\n").replace("\r", "\n").strip()


def normalize_for_compare(value: Any) -> str:
    return " ".join(normalize_text(value).split())


def sha1_text(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()


def parse_json_object(value: str | None, *, default: dict[str, Any] | None = None) -> dict[str, Any]:
    if not value:
        return dict(default or {})
    payload = json.loads(value)
    if not isinstance(payload, dict):
        raise ValueError("Expected a JSON object.")
    return payload


def load_rubric(path: Path) -> dict[str, Any]:
    try:
        import yaml
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("PyYAML is required for rubric loading. Install with `pip install pyyaml`.") from exc
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected YAML mapping at {path}")
    return payload


def load_prompts_module(path: Path) -> Any:
    spec = importlib.util.spec_from_file_location("polyalign_llm_judge_prompts", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load prompt module from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


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
    if turns and isinstance(turns[-1], dict) and turns[-1].get("role") == "user":
        if normalize_text(turns[-1].get("text", turns[-1].get("content", ""))) == question:
            turns = turns[:-1]
    for turn in turns:
        if not isinstance(turn, dict):
            continue
        role = turn.get("role")
        text = normalize_text(turn.get("text", turn.get("content", "")))
        if role == "user":
            pending = text
        elif role == "assistant" and pending is not None:
            pairs.append([pending, text])
            pending = None
    return pairs


def _row_at(rows: list[dict[str, Any]], index: int) -> dict[str, Any]:
    if 0 <= index < len(rows):
        return rows[index]
    return {}


def _source_index_for_row(row: dict[str, Any], row_number: int, expected_len: int) -> tuple[int, Any, bool, bool]:
    original = row.get("source_index")
    if isinstance(original, int) and original >= 0:
        return original, original, True, original < expected_len
    return row_number, original, False, row_number < expected_len


def _human_path_for_lang(args: argparse.Namespace, lang: str, raw_root: Path) -> Path:
    explicit = getattr(args, f"human_{lang}", None)
    if explicit:
        return Path(explicit)
    return raw_root / SOURCE_CONFIGS[lang]["human_hf_path"]


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
        expected_len = min(len(human_rows), len(current_rows))

        inputs_dir = output_root / lang / "inputs"
        ensure_dir(inputs_dir)

        manifest_sources: list[dict[str, Any]] = [
            {
                "source_id": "human",
                "model_key": "human",
                "stage": "human",
                "source_type": "human",
                "label": 0,
                "path": str(human_path),
                "input_path": str(inputs_dir / "human.jsonl"),
                "row_count": len(human_rows),
            }
        ]

        missing_source_files: list[dict[str, Any]] = []
        per_source_reports: dict[str, Any] = {}

        def base_payload(
            source_index: int,
            *,
            source_row_index: int,
            source_id: str,
            source_row: dict[str, Any] | None = None,
        ) -> dict[str, Any]:
            current = _row_at(current_rows, source_index)
            human = _row_at(human_rows, source_index)
            source_row = source_row or {}
            reference_output = normalize_text(human.get("output", current.get("human_answer", "")))
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
                "reference_output": reference_output,
                "reference_sha1": sha1_text(reference_output),
            }

        human_output_rows: list[dict[str, Any]] = []
        for source_index, human_row in enumerate(human_rows):
            payload = base_payload(source_index, source_row_index=source_index, source_id="human", source_row=human_row)
            candidate_text = normalize_text(human_row.get("output", payload["reference_output"]))
            payload.update(
                {
                    "source_id": "human",
                    "model_key": "human",
                    "stage": "human",
                    "source_type": "human",
                    "label": 0,
                    "candidate_text": candidate_text,
                    "candidate_sha1": sha1_text(candidate_text),
                    "source_index_is_valid": True,
                    "source_index_in_range": source_index < expected_len,
                }
            )
            human_output_rows.append(payload)
        write_jsonl(inputs_dir / "human.jsonl", human_output_rows)

        for source_id, model_key, stage, rel_path in config["sources"]:
            path = raw_root / rel_path
            if not path.exists():
                missing_source_files.append(
                    {"source_id": source_id, "model_key": model_key, "stage": stage, "path": str(path)}
                )
                continue

            prediction_rows = load_concatenated_json(path)
            rows_out: list[dict[str, Any]] = []
            seen_indices: set[int] = set()
            missing_source_index_rows = 0
            out_of_range_rows = 0
            duplicate_source_index_rows = 0
            empty_candidate_rows = 0
            instruction_mismatches = 0
            input_mismatches = 0

            for row_number, pred_row in enumerate(prediction_rows):
                source_index, original_source_index, index_is_valid, index_in_range = _source_index_for_row(
                    pred_row,
                    row_number,
                    expected_len,
                )
                if not index_is_valid:
                    missing_source_index_rows += 1
                if not index_in_range:
                    out_of_range_rows += 1
                if source_index in seen_indices:
                    duplicate_source_index_rows += 1
                seen_indices.add(source_index)

                current = _row_at(current_rows, source_index)
                human = _row_at(human_rows, source_index)
                if "instruction" in pred_row:
                    expected_instruction = current.get("question", human.get("instruction", ""))
                    if normalize_for_compare(pred_row.get("instruction")) != normalize_for_compare(expected_instruction):
                        instruction_mismatches += 1
                if "input" in pred_row:
                    expected_input = current.get("context", human.get("input", ""))
                    if normalize_for_compare(pred_row.get("input")) != normalize_for_compare(expected_input):
                        input_mismatches += 1

                payload = base_payload(source_index, source_row_index=row_number, source_id=source_id, source_row=pred_row)
                candidate_text = normalize_text(pred_row.get("prediction", ""))
                if not candidate_text:
                    empty_candidate_rows += 1
                payload.update(
                    {
                        "source_id": source_id,
                        "model_key": model_key,
                        "stage": stage,
                        "source_type": "model",
                        "label": 1,
                        "candidate_text": candidate_text,
                        "candidate_sha1": sha1_text(candidate_text),
                        "finish_reason": pred_row.get("finish_reason", ""),
                        "prediction_model_name": pred_row.get("model_name", ""),
                        "original_source_index": original_source_index,
                        "source_index_is_valid": index_is_valid,
                        "source_index_in_range": index_in_range,
                    }
                )
                rows_out.append(payload)

            input_path = inputs_dir / f"{source_id}.jsonl"
            write_jsonl(input_path, rows_out)
            manifest_sources.append(
                {
                    "source_id": source_id,
                    "model_key": model_key,
                    "stage": stage,
                    "source_type": "model",
                    "label": 1,
                    "path": str(path),
                    "input_path": str(input_path),
                    "row_count": len(rows_out),
                }
            )
            per_source_reports[source_id] = {
                "rows": len(prediction_rows),
                "missing_or_bad_source_index_rows": missing_source_index_rows,
                "out_of_range_source_index_rows": out_of_range_rows,
                "duplicate_source_index_rows": duplicate_source_index_rows,
                "empty_candidate_rows": empty_candidate_rows,
                "instruction_mismatches": instruction_mismatches,
                "input_mismatches": input_mismatches,
            }

        summary = {
            "language": lang,
            "mode": "full_context_llm_judge_inputs",
            "human_path": str(human_path),
            "current_path": str(current_path),
            "human_rows": len(human_rows),
            "current_rows": len(current_rows),
            "expected_aligned_rows": expected_len,
            "present_prediction_sources": len(manifest_sources) - 1,
            "missing_prediction_sources": missing_source_files,
            "per_source_reports": per_source_reports,
            "sources": manifest_sources,
        }
        write_json(output_root / lang / "input_summary.json", summary)
        write_json(output_root / lang / "manifest.json", {"language": lang, "sources": manifest_sources})
        combined_summary[lang] = summary
        print(f"[inputs] {lang}: wrote human plus {len(manifest_sources) - 1} prediction source files", flush=True)

    write_json(output_root / "input_summary.json", combined_summary)


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


def sample_rows(rows: list[dict[str, Any]], *, sample_size: int, sample_mode: str, seed: int) -> list[dict[str, Any]]:
    if sample_size <= 0 or sample_size >= len(rows):
        return rows
    if sample_mode == "random":
        rng = random.Random(seed)
        indexed = list(enumerate(rows))
        selected = rng.sample(indexed, sample_size)
        selected.sort(key=lambda item: item[0])
        return [row for _index, row in selected]
    return rows[:sample_size]


def extract_choice_map(response: dict[str, Any], batch_size: int) -> dict[int, dict[str, Any]]:
    mapped: dict[int, dict[str, Any]] = {}
    for ordinal, choice in enumerate(response.get("choices") or []):
        raw_index = choice.get("index", ordinal)
        index = raw_index if isinstance(raw_index, int) and 0 <= raw_index < batch_size else ordinal
        mapped.setdefault(index, choice)
    return mapped


def call_vllm_completion_batch(
    *,
    base_url: str,
    api_key: str | None,
    model_name: str,
    prompts: list[str],
    max_tokens: int,
    temperature: float,
    top_p: float,
    timeout: float,
    repetition_penalty: float,
    frequency_penalty: float,
    presence_penalty: float,
    stop: list[str],
    extra_body: dict[str, Any],
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "model": model_name,
        "prompt": prompts,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "top_p": top_p,
        "n": 1,
        "repetition_penalty": repetition_penalty,
        "frequency_penalty": frequency_penalty,
        "presence_penalty": presence_penalty,
        "stop": stop,
    }
    payload.update(extra_body)
    req = request.Request(
        url=base_url.rstrip("/") + "/v1/completions",
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Content-Type": "application/json",
            **({"Authorization": f"Bearer {api_key}"} if api_key else {}),
        },
        method="POST",
    )
    try:
        with request.urlopen(req, timeout=timeout) as response:
            return json.loads(response.read().decode("utf-8"))
    except error.HTTPError as exc:
        error_body = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"vLLM HTTP {exc.code}: {error_body}") from exc
    except error.URLError as exc:
        raise RuntimeError(f"Could not reach vLLM server at {base_url}: {exc}") from exc


def extract_json_payload(text: str) -> dict[str, Any]:
    text = normalize_text(text)
    fenced = re.search(r"```(?:json)?\s*(.*?)```", text, flags=re.DOTALL | re.IGNORECASE)
    if fenced:
        text = fenced.group(1).strip()
    if not text.startswith("{"):
        start = text.find("{")
        end = text.rfind("}")
        if start != -1 and end != -1 and end > start:
            text = text[start : end + 1]
    payload = json.loads(text)
    if not isinstance(payload, dict):
        raise ValueError("Judge output was not a JSON object.")
    return payload


def dimension_ids(rubric: dict[str, Any]) -> list[str]:
    return [dimension["id"] for dimension in rubric.get("dimensions", [])]


def validate_scores(payload: dict[str, Any], rubric: dict[str, Any]) -> dict[str, int]:
    raw_scores = payload.get("scores")
    if not isinstance(raw_scores, dict):
        raise ValueError("Judge output missing object field `scores`.")
    score_min = int(rubric.get("score_min", 1))
    score_max = int(rubric.get("score_max", 5))
    scores: dict[str, int] = {}
    missing: list[str] = []
    bad: list[str] = []
    for dim_id in dimension_ids(rubric):
        value = raw_scores.get(dim_id)
        if value is None:
            missing.append(dim_id)
            continue
        if isinstance(value, str) and value.strip().isdigit():
            value = int(value.strip())
        if not isinstance(value, int) or not (score_min <= value <= score_max):
            bad.append(dim_id)
            continue
        scores[dim_id] = value
    if missing or bad:
        raise ValueError(f"Invalid score object. missing={missing} bad={bad}")
    return scores


def score_to_100(value: int, rubric: dict[str, Any]) -> float:
    score_min = float(rubric.get("score_min", 1))
    score_max = float(rubric.get("score_max", 5))
    return 100.0 * (float(value) - score_min) / max(score_max - score_min, 1.0)


def compute_composites(scores: dict[str, int], rubric: dict[str, Any]) -> dict[str, float]:
    composites: dict[str, float] = {}
    for name, weights in (rubric.get("composites") or {}).items():
        if not isinstance(weights, dict):
            continue
        total_weight = 0.0
        weighted_sum = 0.0
        for dim_id, weight in weights.items():
            if dim_id not in scores:
                continue
            numeric_weight = float(weight)
            total_weight += numeric_weight
            weighted_sum += numeric_weight * score_to_100(scores[dim_id], rubric)
        composites[name] = round(weighted_sum / total_weight, 4) if total_weight else 0.0
    return composites


def build_error_row(
    row: dict[str, Any],
    *,
    judge_id: str,
    judge_model_name: str,
    error_message: str,
    prompt_tokens: int | None = None,
    prompt_sha1: str | None = None,
    raw_judge_response: str = "",
    finish_reason: Any = None,
    usage: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "source_index": row.get("source_index"),
        "source_row_index": row.get("source_row_index"),
        "source_uid": row.get("source_uid", ""),
        "id": row.get("id", ""),
        "language": row.get("language", ""),
        "dataset": row.get("dataset", ""),
        "track": row.get("track", ""),
        "family": row.get("family", ""),
        "style_bucket": row.get("style_bucket", ""),
        "length_bin": row.get("length_bin", ""),
        "bucket_id": row.get("bucket_id", ""),
        "source_id": row.get("source_id", ""),
        "model_key": row.get("model_key", ""),
        "stage": row.get("stage", ""),
        "source_type": row.get("source_type", ""),
        "label": row.get("label"),
        "judge_id": judge_id,
        "judge_model_name": judge_model_name,
        "candidate_sha1": row.get("candidate_sha1", ""),
        "reference_sha1": row.get("reference_sha1", ""),
        "prompt_tokens": prompt_tokens,
        "prompt_sha1": prompt_sha1,
        "raw_judge_response": raw_judge_response,
        "finish_reason": finish_reason,
        "usage": usage or {},
        "scores": None,
        "computed_scores": None,
        "major_errors": [],
        "rationale": "",
        "error": error_message,
    }


def build_success_row(
    row: dict[str, Any],
    *,
    judge_id: str,
    judge_model_name: str,
    prompt_tokens: int,
    prompt_sha1: str,
    raw_judge_response: str,
    finish_reason: Any,
    usage: dict[str, Any],
    parsed_payload: dict[str, Any],
    scores: dict[str, int],
    computed_scores: dict[str, float],
) -> dict[str, Any]:
    major_errors = parsed_payload.get("major_errors", [])
    if not isinstance(major_errors, list):
        major_errors = [str(major_errors)]
    rationale = normalize_text(parsed_payload.get("rationale", ""))
    return {
        "source_index": row.get("source_index"),
        "source_row_index": row.get("source_row_index"),
        "source_uid": row.get("source_uid", ""),
        "id": row.get("id", ""),
        "language": row.get("language", ""),
        "dataset": row.get("dataset", ""),
        "track": row.get("track", ""),
        "family": row.get("family", ""),
        "style_bucket": row.get("style_bucket", ""),
        "length_bin": row.get("length_bin", ""),
        "bucket_id": row.get("bucket_id", ""),
        "source_id": row.get("source_id", ""),
        "model_key": row.get("model_key", ""),
        "stage": row.get("stage", ""),
        "source_type": row.get("source_type", ""),
        "label": row.get("label"),
        "judge_id": judge_id,
        "judge_model_name": judge_model_name,
        "candidate_sha1": row.get("candidate_sha1", ""),
        "reference_sha1": row.get("reference_sha1", ""),
        "prompt_tokens": prompt_tokens,
        "prompt_sha1": prompt_sha1,
        "raw_judge_response": raw_judge_response,
        "finish_reason": finish_reason,
        "usage": usage,
        "scores": scores,
        "computed_scores": computed_scores,
        "major_errors": [normalize_text(item) for item in major_errors],
        "rationale": rationale,
        "error": None,
    }


def retry_single_prompt(
    *,
    prompt: str,
    row: dict[str, Any],
    rubric: dict[str, Any],
    judge_id: str,
    judge_model_name: str,
    prompt_tokens: int,
    prompt_sha1: str,
    args: argparse.Namespace,
    stop_sequences: list[str],
    extra_body: dict[str, Any],
    error_prefix: str,
) -> dict[str, Any]:
    last_error = error_prefix
    for _attempt in range(args.max_retries):
        try:
            response = call_vllm_completion_batch(
                base_url=args.base_url,
                api_key=args.api_key,
                model_name=args.judge_model_name,
                prompts=[prompt],
                max_tokens=args.max_tokens,
                temperature=args.temperature,
                top_p=args.top_p,
                timeout=args.timeout,
                repetition_penalty=args.repetition_penalty,
                frequency_penalty=args.frequency_penalty,
                presence_penalty=args.presence_penalty,
                stop=stop_sequences,
                extra_body=extra_body,
            )
            choice = extract_choice_map(response, 1).get(0, {})
            raw_text = choice.get("text") or ""
            parsed = extract_json_payload(raw_text)
            scores = validate_scores(parsed, rubric)
            computed = compute_composites(scores, rubric)
            return build_success_row(
                row,
                judge_id=judge_id,
                judge_model_name=judge_model_name,
                prompt_tokens=prompt_tokens,
                prompt_sha1=prompt_sha1,
                raw_judge_response=raw_text,
                finish_reason=choice.get("finish_reason"),
                usage=response.get("usage", {}),
                parsed_payload=parsed,
                scores=scores,
                computed_scores=computed,
            )
        except Exception as exc:
            last_error = f"{type(exc).__name__}: {exc}"
    return build_error_row(
        row,
        judge_id=judge_id,
        judge_model_name=judge_model_name,
        error_message=last_error,
        prompt_tokens=prompt_tokens,
        prompt_sha1=prompt_sha1,
    )


def score_input_file(
    *,
    input_path: Path,
    output_path: Path,
    rubric: dict[str, Any],
    prompts_module: Any,
    tokenizer: Any,
    chat_template_kwargs: dict[str, Any],
    args: argparse.Namespace,
) -> dict[str, Any]:
    all_rows = list(iter_jsonl(input_path))
    selected_rows = sample_rows(all_rows, sample_size=args.sample_size, sample_mode=args.sample_mode, seed=args.seed)
    done = completed_keys(output_path) if args.resume else set()
    rows = [row for row in selected_rows if not any(key in done for key in _completion_keys(row))]

    ensure_dir(output_path.parent)
    mode = "a" if args.resume and output_path.exists() else "w"
    stop_sequences = ["<|im_end|>", "<|endoftext|>", "<|eot_id|>", "\n<|im_start|>"]

    extra_body = parse_json_object(args.extra_body_json)
    if args.guided_json:
        extra_body.setdefault("guided_json", prompts_module.judge_json_schema(rubric))

    started = time.time()
    written = 0
    errors = 0
    token_max = 0
    token_sum = 0

    with output_path.open(mode, encoding="utf-8", buffering=1) as handle:
        total_batches = (len(rows) + args.batch_size - 1) // args.batch_size if rows else 0
        for batch_start in tqdm(range(0, len(rows), args.batch_size), total=total_batches, desc=f"judge {input_path.stem}", unit="batch"):
            batch_rows = rows[batch_start : batch_start + args.batch_size]
            valid_items: list[tuple[dict[str, Any], str, int, str]] = []
            prompts: list[str] = []

            for row in batch_rows:
                try:
                    messages = prompts_module.build_judge_messages(row, rubric)
                    prompt = tokenizer.apply_chat_template(
                        messages,
                        tokenize=False,
                        add_generation_prompt=True,
                        **chat_template_kwargs,
                    )
                    prompt_tokens = len(tokenizer.encode(prompt, add_special_tokens=False))
                    prompt_sha1 = sha1_text(prompt)
                    token_max = max(token_max, prompt_tokens)
                    token_sum += prompt_tokens
                    if args.max_model_len > 0 and prompt_tokens + args.max_tokens + args.prompt_safety_margin > args.max_model_len:
                        message = (
                            f"Prompt has {prompt_tokens} tokens; max_model_len={args.max_model_len}, "
                            f"max_tokens={args.max_tokens}, safety_margin={args.prompt_safety_margin}. "
                            "No truncation was applied."
                        )
                        if args.overlength_policy == "fail":
                            raise ValueError(message)
                        out_row = build_error_row(
                            row,
                            judge_id=args.judge_id,
                            judge_model_name=args.judge_model_name,
                            error_message=message,
                            prompt_tokens=prompt_tokens,
                            prompt_sha1=prompt_sha1,
                        )
                        handle.write(json.dumps(out_row, ensure_ascii=False) + "\n")
                        written += 1
                        errors += 1
                        continue
                    valid_items.append((row, prompt, prompt_tokens, prompt_sha1))
                    prompts.append(prompt)
                except Exception as exc:
                    out_row = build_error_row(
                        row,
                        judge_id=args.judge_id,
                        judge_model_name=args.judge_model_name,
                        error_message=f"{type(exc).__name__}: {exc}",
                    )
                    if args.overlength_policy == "fail":
                        raise
                    handle.write(json.dumps(out_row, ensure_ascii=False) + "\n")
                    written += 1
                    errors += 1

            if not valid_items:
                handle.flush()
                os.fsync(handle.fileno())
                continue

            response = call_vllm_completion_batch(
                base_url=args.base_url,
                api_key=args.api_key,
                model_name=args.judge_model_name,
                prompts=prompts,
                max_tokens=args.max_tokens,
                temperature=args.temperature,
                top_p=args.top_p,
                timeout=args.timeout,
                repetition_penalty=args.repetition_penalty,
                frequency_penalty=args.frequency_penalty,
                presence_penalty=args.presence_penalty,
                stop=stop_sequences,
                extra_body=extra_body,
            )
            choice_map = extract_choice_map(response, len(valid_items))
            batch_usage = response.get("usage", {})

            for valid_index, (row, prompt, prompt_tokens, prompt_sha1) in enumerate(valid_items):
                choice = choice_map.get(valid_index, {})
                raw_text = choice.get("text") or ""
                try:
                    parsed = extract_json_payload(raw_text)
                    scores = validate_scores(parsed, rubric)
                    computed = compute_composites(scores, rubric)
                    out_row = build_success_row(
                        row,
                        judge_id=args.judge_id,
                        judge_model_name=args.judge_model_name,
                        prompt_tokens=prompt_tokens,
                        prompt_sha1=prompt_sha1,
                        raw_judge_response=raw_text,
                        finish_reason=choice.get("finish_reason"),
                        usage=batch_usage,
                        parsed_payload=parsed,
                        scores=scores,
                        computed_scores=computed,
                    )
                except Exception as exc:
                    out_row = retry_single_prompt(
                        prompt=prompt,
                        row=row,
                        rubric=rubric,
                        judge_id=args.judge_id,
                        judge_model_name=args.judge_model_name,
                        prompt_tokens=prompt_tokens,
                        prompt_sha1=prompt_sha1,
                        args=args,
                        stop_sequences=stop_sequences,
                        extra_body=extra_body,
                        error_prefix=f"{type(exc).__name__}: {exc}",
                    )
                if out_row.get("error"):
                    errors += 1
                handle.write(json.dumps(out_row, ensure_ascii=False) + "\n")
                written += 1

            handle.flush()
            os.fsync(handle.fileno())
            print(f"[judge] {args.judge_id} {input_path.name}: {written}/{len(rows)} new rows", flush=True)

    return {
        "input_path": str(input_path),
        "output_path": str(output_path),
        "rows_total": len(all_rows),
        "rows_selected": len(selected_rows),
        "rows_previously_completed": len(selected_rows) - len(rows),
        "rows_written": written,
        "error_rows_written": errors,
        "prompt_tokens_max": token_max,
        "prompt_tokens_mean_new_rows": round(token_sum / max(written, 1), 4) if written else 0.0,
        "elapsed_seconds": round(time.time() - started, 3),
    }


def judge(args: argparse.Namespace) -> None:
    from transformers import AutoTokenizer

    if args.source_num_shards <= 0:
        raise ValueError("--source-num-shards must be > 0")
    if not (0 <= args.source_shard_index < args.source_num_shards):
        raise ValueError("--source-shard-index must satisfy 0 <= index < --source-num-shards")

    work_dir = Path(args.work_dir)
    rubric = load_rubric(Path(args.rubric_yaml))
    prompts_module = load_prompts_module(Path(args.prompts_py))
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name_or_path, trust_remote_code=args.trust_remote_code)
    if not getattr(tokenizer, "chat_template", None):
        raise ValueError("Judge tokenizer must provide a chat_template for prompt rendering.")
    chat_template_kwargs = parse_json_object(args.chat_template_kwargs_json)

    languages = ["en", "zh"] if args.lang == "all" else [args.lang]
    run_summaries: dict[str, Any] = {}
    for lang in languages:
        manifest_path = work_dir / lang / "manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        summaries: list[dict[str, Any]] = []
        selected_sources = [
            source
            for source_index, source in enumerate(manifest["sources"])
            if source_index % args.source_num_shards == args.source_shard_index
        ]
        for source in selected_sources:
            input_path = Path(source["input_path"])
            output_path = work_dir / lang / "scores" / args.judge_id / f"{source['source_id']}.jsonl"
            summaries.append(
                {
                    "source_id": source["source_id"],
                    "model_key": source["model_key"],
                    "stage": source["stage"],
                    **score_input_file(
                        input_path=input_path,
                        output_path=output_path,
                        rubric=rubric,
                        prompts_module=prompts_module,
                        tokenizer=tokenizer,
                        chat_template_kwargs=chat_template_kwargs,
                        args=args,
                    ),
                }
            )
        if args.source_num_shards == 1:
            summary_filename = "run_summary.json"
        else:
            summary_filename = f"run_summary.shard_{args.source_shard_index}_of_{args.source_num_shards}.json"
        summary_path = work_dir / lang / "scores" / args.judge_id / summary_filename
        write_json(
            summary_path,
            {
                "language": lang,
                "judge_id": args.judge_id,
                "source_shard_index": args.source_shard_index,
                "source_num_shards": args.source_num_shards,
                "sources_total": len(manifest["sources"]),
                "sources_selected": len(selected_sources),
                "sources": summaries,
            },
        )
        run_summaries[lang] = summaries
    print(json.dumps({"judge_id": args.judge_id, "languages": run_summaries}, indent=2, ensure_ascii=False))


def _mean(values: list[float]) -> float | None:
    if not values:
        return None
    return sum(values) / len(values)


def _valid_score_rows(path: Path) -> tuple[list[dict[str, Any]], int]:
    valid: list[dict[str, Any]] = []
    total = 0
    if not path.exists():
        return valid, total
    for row in iter_jsonl(path):
        total += 1
        if row.get("error") is None and isinstance(row.get("scores"), dict):
            valid.append(row)
    return valid, total


def summarize_judge_lang(work_dir: Path, lang: str, judge_id: str, rubric: dict[str, Any]) -> dict[str, Any]:
    manifest = json.loads((work_dir / lang / "manifest.json").read_text(encoding="utf-8"))
    score_dir = work_dir / lang / "scores" / judge_id
    dim_ids = dimension_ids(rubric)
    composite_names = list((rubric.get("composites") or {}).keys())

    source_summaries: dict[str, Any] = {}
    csv_rows: list[dict[str, Any]] = []
    for source in manifest["sources"]:
        source_id = source["source_id"]
        path = score_dir / f"{source_id}.jsonl"
        rows, total_rows = _valid_score_rows(path)
        dim_means = {
            dim_id: _mean([float(row["scores"][dim_id]) for row in rows if dim_id in row["scores"]])
            for dim_id in dim_ids
        }
        composite_means = {
            name: _mean([float(row["computed_scores"][name]) for row in rows if isinstance(row.get("computed_scores"), dict) and name in row["computed_scores"]])
            for name in composite_names
        }
        summary = {
            "status": "ok" if path.exists() else "missing_scores",
            "source_id": source_id,
            "model_key": source["model_key"],
            "stage": source["stage"],
            "source_type": source.get("source_type", ""),
            "label": source.get("label"),
            "score_path": str(path),
            "n_rows": total_rows,
            "n_valid": len(rows),
            "n_errors": total_rows - len(rows),
            "dimension_means": dim_means,
            "composite_means_0_100": composite_means,
            "prompt_tokens_mean": _mean([float(row["prompt_tokens"]) for row in rows if row.get("prompt_tokens") is not None]),
            "prompt_tokens_max": max([int(row["prompt_tokens"]) for row in rows if row.get("prompt_tokens") is not None], default=None),
        }
        source_summaries[source_id] = summary
        csv_row = {
            "language": lang,
            "judge_id": judge_id,
            "source_id": source_id,
            "model_key": source["model_key"],
            "stage": source["stage"],
            "source_type": source.get("source_type", ""),
            "n_rows": total_rows,
            "n_valid": len(rows),
            "n_errors": total_rows - len(rows),
        }
        for dim_id, value in dim_means.items():
            csv_row[f"mean_{dim_id}"] = "" if value is None else round(value, 6)
        for name, value in composite_means.items():
            csv_row[f"{name}_0_100"] = "" if value is None else round(value, 6)
        csv_rows.append(csv_row)

    trends: dict[str, Any] = {}
    for model_key in sorted({source["model_key"] for source in manifest["sources"]}):
        points = []
        for stage in STAGE_ORDER:
            source_id = "human" if model_key == "human" and stage == "human" else f"{model_key}__{stage}"
            summary = source_summaries.get(source_id)
            if summary and summary["status"] == "ok":
                points.append(
                    {
                        "stage": stage,
                        "n_valid": summary["n_valid"],
                        "overall_0_100": summary["composite_means_0_100"].get("overall"),
                        "utility_0_100": summary["composite_means_0_100"].get("utility"),
                        "conditional_naturalness_0_100": summary["composite_means_0_100"].get("conditional_naturalness"),
                        "distribution_faithfulness_0_100": summary["composite_means_0_100"].get("distribution_faithfulness"),
                    }
                )
        if points:
            trends[model_key] = {"available_stages": [point["stage"] for point in points], "points": points}

    output_dir = work_dir / lang / "metrics" / "llm_judge" / judge_id
    ensure_dir(output_dir)
    output = {
        "language": lang,
        "judge_id": judge_id,
        "rubric": {
            "name": rubric.get("name"),
            "version": rubric.get("version"),
            "score_min": rubric.get("score_min"),
            "score_max": rubric.get("score_max"),
            "dimensions": dim_ids,
            "composites": list((rubric.get("composites") or {}).keys()),
        },
        "source_summaries": source_summaries,
        "trend_by_model": trends,
    }
    write_json(output_dir / "summary.json", output)
    if csv_rows:
        with (output_dir / "summary.csv").open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(csv_rows[0].keys()))
            writer.writeheader()
            writer.writerows(csv_rows)
    print(f"[summary] wrote {output_dir / 'summary.json'}", flush=True)
    return output


def summarize(args: argparse.Namespace) -> None:
    work_dir = Path(args.work_dir)
    rubric = load_rubric(Path(args.rubric_yaml))
    languages = ["en", "zh"] if args.lang == "all" else [args.lang]
    if args.judge_id == "all":
        judge_ids = sorted({path.name for lang in languages for path in (work_dir / lang / "scores").glob("*") if path.is_dir()})
    else:
        judge_ids = [args.judge_id]
    combined: dict[str, Any] = {}
    for lang in languages:
        combined[lang] = {}
        for judge_id in judge_ids:
            combined[lang][judge_id] = summarize_judge_lang(work_dir, lang, judge_id, rubric)
    write_json(work_dir / "metrics" / "llm_judge_summaries.json", combined)


def _pearson(xs: list[float], ys: list[float]) -> float | None:
    if len(xs) < 2 or len(xs) != len(ys):
        return None
    mean_x = sum(xs) / len(xs)
    mean_y = sum(ys) / len(ys)
    num = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys, strict=True))
    den_x = sum((x - mean_x) ** 2 for x in xs) ** 0.5
    den_y = sum((y - mean_y) ** 2 for y in ys) ** 0.5
    if den_x == 0.0 or den_y == 0.0:
        return None
    return num / (den_x * den_y)


def combine_summaries(args: argparse.Namespace) -> None:
    work_dir = Path(args.work_dir)
    languages = ["en", "zh"] if args.lang == "all" else [args.lang]
    combined: dict[str, Any] = {}
    csv_rows: list[dict[str, Any]] = []

    for lang in languages:
        metric_root = work_dir / lang / "metrics" / "llm_judge"
        judge_summaries: dict[str, Any] = {}
        for summary_path in sorted(metric_root.glob("*/summary.json")):
            judge_summaries[summary_path.parent.name] = json.loads(summary_path.read_text(encoding="utf-8"))
        source_ids = sorted({source_id for summary in judge_summaries.values() for source_id in summary["source_summaries"]})
        lang_payload: dict[str, Any] = {"judge_ids": sorted(judge_summaries), "source_summaries": {}}

        for source_id in source_ids:
            per_judge = {
                judge_id: summary["source_summaries"][source_id]
                for judge_id, summary in judge_summaries.items()
                if source_id in summary["source_summaries"]
            }
            if not per_judge:
                continue
            first = next(iter(per_judge.values()))
            composite_names = sorted(
                {
                    name
                    for source_summary in per_judge.values()
                    for name in source_summary.get("composite_means_0_100", {})
                }
            )
            composite_means = {
                name: _mean(
                    [
                        float(source_summary["composite_means_0_100"][name])
                        for source_summary in per_judge.values()
                        if source_summary.get("composite_means_0_100", {}).get(name) is not None
                    ]
                )
                for name in composite_names
            }
            agreement: dict[str, Any] = {}
            if len(per_judge) == 2:
                judge_items = list(per_judge.items())
                left_scores: list[float] = []
                right_scores: list[float] = []
                for name in composite_names:
                    left = judge_items[0][1].get("composite_means_0_100", {}).get(name)
                    right = judge_items[1][1].get("composite_means_0_100", {}).get(name)
                    if left is not None and right is not None:
                        left_scores.append(float(left))
                        right_scores.append(float(right))
                agreement = {
                    "judge_pair": [judge_items[0][0], judge_items[1][0]],
                    "pearson_across_source_composites": _pearson(left_scores, right_scores),
                    "mean_absolute_difference_across_source_composites": (
                        _mean([abs(a - b) for a, b in zip(left_scores, right_scores, strict=True)])
                        if left_scores
                        else None
                    ),
                }
            source_payload = {
                "source_id": source_id,
                "model_key": first.get("model_key"),
                "stage": first.get("stage"),
                "source_type": first.get("source_type"),
                "n_valid_by_judge": {judge_id: summary["n_valid"] for judge_id, summary in per_judge.items()},
                "mean_composites_0_100": composite_means,
                "agreement": agreement,
                "per_judge": per_judge,
            }
            lang_payload["source_summaries"][source_id] = source_payload
            csv_row = {
                "language": lang,
                "source_id": source_id,
                "model_key": first.get("model_key"),
                "stage": first.get("stage"),
                "source_type": first.get("source_type"),
                "judges": ",".join(sorted(per_judge)),
            }
            for name, value in composite_means.items():
                csv_row[f"mean_{name}_0_100"] = "" if value is None else round(value, 6)
            csv_rows.append(csv_row)

        combined[lang] = lang_payload

    output_root = work_dir / "metrics"
    write_json(output_root / "llm_judge_combined_summary.json", combined)
    if csv_rows:
        fieldnames = sorted({key for row in csv_rows for key in row})
        ordered = ["language", "source_id", "model_key", "stage", "source_type", "judges"]
        fieldnames = ordered + [key for key in fieldnames if key not in ordered]
        with (output_root / "llm_judge_combined_summary.csv").open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(csv_rows)
    print(f"[summary] wrote {output_root / 'llm_judge_combined_summary.json'}", flush=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="PolyAlign LLM-as-a-judge evaluation helpers.")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("list-downloads")
    p.add_argument("--lang", choices=["all", "en", "zh"], default="all")
    p.add_argument("--include-human", action="store_true")
    p.set_defaults(func=list_downloads)

    p = sub.add_parser("build-inputs")
    p.add_argument("--raw-root", required=True)
    p.add_argument("--output-root", required=True)
    p.add_argument("--lang", choices=["all", "en", "zh"], default="all")
    p.add_argument("--human-en")
    p.add_argument("--human-zh")
    p.set_defaults(func=build_inputs)

    p = sub.add_parser("judge")
    p.add_argument("--work-dir", required=True)
    p.add_argument("--lang", choices=["all", "en", "zh"], default="all")
    p.add_argument("--judge-id", required=True)
    p.add_argument("--judge-model-name", required=True)
    p.add_argument("--tokenizer-name-or-path", required=True)
    p.add_argument("--base-url", required=True)
    p.add_argument("--api-key")
    p.add_argument("--rubric-yaml", default=str(DEFAULT_RUBRIC_PATH))
    p.add_argument("--prompts-py", default=str(DEFAULT_PROMPTS_PATH))
    p.add_argument("--sample-size", type=int, default=0)
    p.add_argument("--sample-mode", choices=["first", "random"], default="first")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--source-shard-index",
        type=int,
        default=0,
        help="Source-file shard index for parallel judging. Sharding is over manifest sources, not rows.",
    )
    p.add_argument(
        "--source-num-shards",
        type=int,
        default=1,
        help="Number of source-file shards for parallel judging.",
    )
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--max-tokens", type=int, default=384)
    p.add_argument("--max-model-len", type=int, default=65536)
    p.add_argument("--prompt-safety-margin", type=int, default=64)
    p.add_argument("--overlength-policy", choices=["fail", "record_error"], default="fail")
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--top-p", type=float, default=1.0)
    p.add_argument("--repetition-penalty", type=float, default=1.0)
    p.add_argument("--frequency-penalty", type=float, default=0.0)
    p.add_argument("--presence-penalty", type=float, default=0.0)
    p.add_argument("--timeout", type=float, default=600.0)
    p.add_argument("--max-retries", type=int, default=2)
    p.add_argument("--extra-body-json")
    p.add_argument("--chat-template-kwargs-json")
    p.add_argument("--disable-guided-json", dest="guided_json", action="store_false")
    p.set_defaults(guided_json=True)
    p.add_argument("--trust-remote-code", action="store_true")
    p.add_argument("--resume", action="store_true")
    p.set_defaults(func=judge)

    p = sub.add_parser("summarize")
    p.add_argument("--work-dir", required=True)
    p.add_argument("--lang", choices=["all", "en", "zh"], default="all")
    p.add_argument("--judge-id", default="all")
    p.add_argument("--rubric-yaml", default=str(DEFAULT_RUBRIC_PATH))
    p.set_defaults(func=summarize)

    p = sub.add_parser("combine-summaries")
    p.add_argument("--work-dir", required=True)
    p.add_argument("--lang", choices=["all", "en", "zh"], default="all")
    p.set_defaults(func=combine_summaries)

    return parser


def main() -> None:
    args = build_parser().parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
