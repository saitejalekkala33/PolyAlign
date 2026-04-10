from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from statistics import mean
from typing import Any

from polyalign_data.io_utils import ensure_dir
from polyalign_data.text import normalize_text


SOURCE_SPLIT_TO_TARGET = {
    "train": "train",
    "dev": "val",
    "test": "test",
}

PAIR_TYPE_DEFAULT_SCORES = {
    "global": 0.0,
    "distribution": 1.0,
    "local": 0.5,
}


def _iter_dataset_split_files(input_root: Path, source_split: str):
    for dataset_dir in sorted(path for path in input_root.iterdir() if path.is_dir()):
        split_path = dataset_dir / f"{source_split}.jsonl"
        if split_path.exists():
            yield dataset_dir.name, split_path


def _iter_split_records(input_root: Path, source_split: str) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for _dataset_name, split_path in _iter_dataset_split_files(input_root, source_split):
        with split_path.open("r", encoding="utf-8") as source_handle:
            for raw_line in source_handle:
                line = raw_line.strip()
                if line:
                    records.append(json.loads(line))

    return records


def _first_value(record: dict[str, Any], *keys: str) -> Any:
    for key in keys:
        if key in record and record[key] is not None:
            return record[key]
    return None


def _first_text(record: dict[str, Any], *keys: str) -> str:
    value = _first_value(record, *keys)
    return normalize_text(value) if value is not None else ""


def _float_value(record: dict[str, Any], *keys: str, default: float = 0.0) -> float:
    value = _first_value(record, *keys)
    if value is None:
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _history_to_llamafactory_pairs(record: dict[str, Any]) -> list[list[str]]:
    existing_history = record.get("history")
    if isinstance(existing_history, list) and all(isinstance(item, list) and len(item) == 2 for item in existing_history):
        return [[normalize_text(pair[0]), normalize_text(pair[1])] for pair in existing_history]

    history_turns = list(record.get("dialogue_history", []))
    question = _first_text(record, "question", "instruction")

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


def _language_key(record: dict[str, Any]) -> str:
    language = _first_text(record, "language")
    return language or "__missing_language__"


def _bucket_key(record: dict[str, Any]) -> str:
    bucket_id = _first_text(record, "bucket_id")
    if bucket_id:
        return bucket_id

    parts = [
        _first_text(record, "language"),
        _first_text(record, "track"),
        _first_text(record, "family"),
        _first_text(record, "style_bucket"),
        _first_text(record, "length_bin"),
    ]
    if any(parts):
        return "|".join(part or "_" for part in parts)
    return "__missing_bucket__"


def _normalized_inverse_frequency(records: list[dict[str, Any]], key_fn) -> dict[str, float]:
    counts = Counter(key_fn(record) for record in records)
    if not counts:
        return {}

    total_examples = sum(counts.values())
    num_keys = len(counts)
    raw_scores = {
        key: max(0.0, (total_examples / (num_keys * count)) - 1.0)
        for key, count in counts.items()
        if count > 0
    }
    max_score = max(raw_scores.values(), default=0.0)
    if max_score <= 0:
        return {key: 0.0 for key in raw_scores}

    return {key: value / max_score for key, value in raw_scores.items()}


def _build_system_prompt(record: dict[str, Any]) -> str:
    prompt_parts = [
        ("family", _first_text(record, "family")),
        ("track", _first_text(record, "track")),
        ("style", _first_text(record, "style_bucket")),
        ("length", _first_text(record, "length_bin")),
    ]
    profile = "; ".join(f"{name}={value}" for name, value in prompt_parts if value)
    if not profile:
        return "You are a helpful assistant."

    return f"You are a helpful assistant. Follow the target response profile when answering. {profile}."


def _target_filename(target_split: str) -> str:
    return f"hdpo_{target_split}.json"


def _normalize_pair_type(value: str) -> str:
    normalized = normalize_text(value).lower()
    if normalized in {"dist", "distributional"}:
        return "distribution"
    if normalized in {"local_alignment", "locality"}:
        return "local"
    if normalized in {"global_utility", "quality"}:
        return "global"
    if normalized in PAIR_TYPE_DEFAULT_SCORES:
        return normalized
    return "global"


def _resolve_chosen(record: dict[str, Any]) -> str:
    return _first_text(record, "chosen", "chosen_answer", "chosen_output", "human_answer", "output")


def _resolve_rejected(record: dict[str, Any]) -> str:
    return _first_text(record, "rejected", "rejected_answer", "rejected_output", "model_rejected")


def _clip(value: float, min_value: float, max_value: float) -> float:
    return max(min_value, min(max_value, value))


def _build_weight_tables(train_records: list[dict[str, Any]]) -> tuple[dict[str, float], dict[str, float]]:
    bucket_gap_table = _normalized_inverse_frequency(train_records, _bucket_key)
    language_weight_table = _normalized_inverse_frequency(train_records, _language_key)
    return bucket_gap_table, language_weight_table


def _to_hdpo_alpaca(
    record: dict[str, Any],
    *,
    bucket_gap_table: dict[str, float],
    language_weight_table: dict[str, float],
    alpha: float,
    beta: float,
    gamma: float,
    eta: float,
    w_min: float,
    w_max: float,
    pair_type_scores: dict[str, float],
) -> dict[str, Any]:
    chosen = _resolve_chosen(record)
    rejected = _resolve_rejected(record)
    if not chosen or not rejected:
        raise ValueError("HDPO export requires both chosen and rejected responses.")

    pair_type = _normalize_pair_type(_first_text(record, "pair_type"))
    bucket_id = _bucket_key(record)
    language = _language_key(record)
    chosen_dist_score = _float_value(
        record, "chosen_dist_score", "chosen_distribution_score", "chosen_critic_score", default=0.0
    )
    rejected_dist_score = _float_value(
        record, "rejected_dist_score", "rejected_distribution_score", "rejected_critic_score", default=0.0
    )
    baseline_bucket_gap = _float_value(record, "baseline_bucket_gap", default=bucket_gap_table.get(bucket_id, 0.0))
    lang_weight = _float_value(record, "lang_weight", default=language_weight_table.get(language, 0.0))
    dist_advantage = max(0.0, rejected_dist_score - chosen_dist_score)
    pair_bias = pair_type_scores.get(pair_type, 0.0)
    hdpo_weight = _clip(
        1.0 + alpha * baseline_bucket_gap + beta * dist_advantage + gamma * lang_weight + eta * pair_bias,
        w_min,
        w_max,
    )

    item = {
        "instruction": _first_text(record, "question", "instruction"),
        "input": _first_text(record, "context", "input"),
        "chosen": chosen,
        "rejected": rejected,
        "system": _build_system_prompt(record),
        "hdpo_weight": round(float(hdpo_weight), 8),
        "bucket_id": bucket_id,
        "language": _first_text(record, "language"),
        "family": _first_text(record, "family"),
        "track": _first_text(record, "track"),
        "style_bucket": _first_text(record, "style_bucket"),
        "length_bin": _first_text(record, "length_bin"),
        "pair_type": pair_type,
        "baseline_bucket_gap": round(float(baseline_bucket_gap), 8),
        "chosen_dist_score": round(float(chosen_dist_score), 8),
        "rejected_dist_score": round(float(rejected_dist_score), 8),
        "lang_weight": round(float(lang_weight), 8),
    }
    critic_bucket_id = _first_value(record, "critic_bucket_id")
    if critic_bucket_id is not None:
        try:
            item["critic_bucket_id"] = int(critic_bucket_id)
        except (TypeError, ValueError):
            pass
    history = _history_to_llamafactory_pairs(record)
    if history:
        item["history"] = history
    return item


def export_hdpo_views(
    input_root: str | Path,
    output_root: str | Path,
    *,
    include_validation2: bool = False,
    alpha: float = 0.5,
    beta: float = 1.0,
    gamma: float = 0.25,
    eta: float = 0.25,
    w_min: float = 0.5,
    w_max: float = 3.0,
    global_pair_score: float = 0.0,
    distribution_pair_score: float = 1.0,
    local_pair_score: float = 0.5,
) -> dict[str, dict[str, Any]]:
    input_dir = Path(input_root)
    output_dir = Path(output_root)
    ensure_dir(output_dir)

    source_splits = ["train", "dev", "test"]
    if include_validation2:
        source_splits.append("validation2")

    split_records = {source_split: _iter_split_records(input_dir, source_split) for source_split in source_splits}
    train_records = split_records.get("train", [])
    bucket_gap_table, language_weight_table = _build_weight_tables(train_records)
    pair_type_scores = {
        "global": global_pair_score,
        "distribution": distribution_pair_score,
        "local": local_pair_score,
    }

    summary: dict[str, dict[str, Any]] = {"llamafactory_hdpo": {}}
    for source_split in source_splits:
        target_split = SOURCE_SPLIT_TO_TARGET.get(source_split, source_split)
        output_path = output_dir / _target_filename(target_split)
        records = split_records[source_split]
        exported_items: list[dict[str, Any]] = []
        skipped = 0
        for record in records:
            try:
                exported_items.append(
                    _to_hdpo_alpaca(
                        record,
                        bucket_gap_table=bucket_gap_table,
                        language_weight_table=language_weight_table,
                        alpha=alpha,
                        beta=beta,
                        gamma=gamma,
                        eta=eta,
                        w_min=w_min,
                        w_max=w_max,
                        pair_type_scores=pair_type_scores,
                    )
                )
            except ValueError:
                skipped += 1

        output_path.write_text(json.dumps(exported_items, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

        weights = [item["hdpo_weight"] for item in exported_items]
        summary["llamafactory_hdpo"][target_split] = {
            "records": len(exported_items),
            "skipped": skipped,
            "output_path": str(output_path),
            "format": "llamafactory_pairwise_json_array",
            "weight_mean": round(mean(weights), 8) if weights else 0.0,
            "weight_min": round(min(weights), 8) if weights else 0.0,
            "weight_max": round(max(weights), 8) if weights else 0.0,
        }

    return summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Export a separate LlamaFactory-compatible HDPO view with bucket-aware pair weighting."
    )
    parser.add_argument("--input-root", required=True, help="Root directory containing per-dataset pairwise JSONL files.")
    parser.add_argument(
        "--output-root",
        required=True,
        help="Output directory. The script writes `hdpo_train.json`, `hdpo_val.json` and `hdpo_test.json`.",
    )
    parser.add_argument("--alpha", type=float, default=0.5, help="Coefficient for baseline bucket gap.")
    parser.add_argument("--beta", type=float, default=1.0, help="Coefficient for distribution critic gap.")
    parser.add_argument("--gamma", type=float, default=0.25, help="Coefficient for language weight.")
    parser.add_argument("--eta", type=float, default=0.25, help="Coefficient for pair-type bias.")
    parser.add_argument("--w-min", type=float, default=0.5, help="Minimum HDPO weight after clipping.")
    parser.add_argument("--w-max", type=float, default=3.0, help="Maximum HDPO weight after clipping.")
    parser.add_argument("--global-pair-score", type=float, default=0.0, help="Bias term for global pairs.")
    parser.add_argument(
        "--distribution-pair-score",
        type=float,
        default=1.0,
        help="Bias term for distribution pairs.",
    )
    parser.add_argument("--local-pair-score", type=float, default=0.5, help="Bias term for local pairs.")
    parser.add_argument(
        "--include-validation2",
        action="store_true",
        help="Also export the ELI5 auxiliary validation2 split as `validation2.jsonl`.",
    )
    args = parser.parse_args()
    summary = export_hdpo_views(
        args.input_root,
        args.output_root,
        include_validation2=args.include_validation2,
        alpha=args.alpha,
        beta=args.beta,
        gamma=args.gamma,
        eta=args.eta,
        w_min=args.w_min,
        w_max=args.w_max,
        global_pair_score=args.global_pair_score,
        distribution_pair_score=args.distribution_pair_score,
        local_pair_score=args.local_pair_score,
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
