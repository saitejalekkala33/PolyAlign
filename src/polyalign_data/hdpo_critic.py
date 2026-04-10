from __future__ import annotations

import argparse
import importlib.util
import json
import math
import random
import sys
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from statistics import mean
from typing import Any, Optional

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from polyalign_data.io_utils import ensure_dir, read_json
from polyalign_data.text import normalize_text


REPO_ROOT = Path(__file__).resolve().parents[2]
HDPO_CRITIC_MODULE_PATH = REPO_ROOT / "vendor" / "LlamaFactory" / "src" / "llamafactory" / "train" / "hdpo" / "critic.py"
_hdpo_critic_spec = importlib.util.spec_from_file_location("polyalign_hdpo_vendor_critic", HDPO_CRITIC_MODULE_PATH)
if _hdpo_critic_spec is None or _hdpo_critic_spec.loader is None:
    raise ImportError(f"Unable to load HDPO critic module from {HDPO_CRITIC_MODULE_PATH}.")
_hdpo_critic_module = importlib.util.module_from_spec(_hdpo_critic_spec)
sys.modules[_hdpo_critic_spec.name] = _hdpo_critic_module
_hdpo_critic_spec.loader.exec_module(_hdpo_critic_module)

HDPOCriticBundleConfig = _hdpo_critic_module.HDPOCriticBundleConfig
UNKNOWN_BUCKET = _hdpo_critic_module.UNKNOWN_BUCKET
BucketConditionedDistributionCritic = _hdpo_critic_module.BucketConditionedDistributionCritic
bucket_ids_from_names = _hdpo_critic_module.bucket_ids_from_names
compute_hdpo_critic_loss = _hdpo_critic_module.compute_hdpo_critic_loss
encode_response_texts = _hdpo_critic_module.encode_response_texts
load_hdpo_critic_bundle = _hdpo_critic_module.load_hdpo_critic_bundle
predict_hdpo_critic_scores = _hdpo_critic_module.predict_hdpo_critic_scores
save_hdpo_critic_bundle = _hdpo_critic_module.save_hdpo_critic_bundle


SOURCE_SPLIT_TO_TARGET = {
    "train": "train",
    "dev": "val",
    "test": "test",
}


def _read_records(path: Path) -> list[dict[str, Any]]:
    if path.suffix.lower() == ".json":
        payload = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(payload, list):
            raise ValueError(f"Expected a JSON array in {path}.")
        return [dict(item) for item in payload]

    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def _write_records(path: Path, records: list[dict[str, Any]]) -> None:
    ensure_dir(path.parent)
    if path.suffix.lower() == ".json":
        path.write_text(json.dumps(records, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        return

    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def _iter_dataset_split_files(input_root: Path, source_split: str) -> Iterator[tuple[str, Path]]:
    for dataset_dir in sorted(path for path in input_root.iterdir() if path.is_dir()):
        split_path = dataset_dir / f"{source_split}.jsonl"
        if split_path.exists():
            yield dataset_dir.name, split_path


def _resolve_split_output_name(source_split: str) -> str:
    return SOURCE_SPLIT_TO_TARGET.get(source_split, source_split)


def _resolve_text(record: dict[str, Any], *keys: str) -> str:
    for key in keys:
        value = record.get(key)
        if value is not None:
            text = normalize_text(value)
            if text:
                return text
    return ""


def _resolve_bucket_id(record: dict[str, Any]) -> str:
    bucket_id = normalize_text(record.get("bucket_id", ""))
    if bucket_id:
        return bucket_id

    parts = [
        normalize_text(record.get("language", "")),
        normalize_text(record.get("track", "")),
        normalize_text(record.get("family", "")),
        normalize_text(record.get("style_bucket", "")),
        normalize_text(record.get("length_bin", "")),
    ]
    if any(parts):
        return "|".join(part or "_" for part in parts)
    return ""


def _feature_support_distance(
    features: dict[str, Any],
    bucket_reference: dict[str, Any],
    *,
    support_band: str = "q10_q90",
    min_scale: float = 1e-6,
) -> float:
    if support_band == "q25_q75":
        support = bucket_reference.get("support_q25_q75", {})
    else:
        support = bucket_reference.get("support_q10_q90", {})
    feature_stats = bucket_reference.get("feature_stats", {})

    distances: list[float] = []
    for feature_name, raw_value in features.items():
        if feature_name not in support or feature_name not in feature_stats:
            continue
        try:
            value = float(raw_value)
        except (TypeError, ValueError):
            continue

        band = support[feature_name]
        stats = feature_stats[feature_name]
        low = float(band["low"])
        high = float(band["high"])
        std = max(float(stats.get("std", 0.0)), min_scale)
        scale = max(high - low, std, min_scale)

        if value < low:
            distances.append((low - value) / scale)
        elif value > high:
            distances.append((value - high) / scale)
        else:
            distances.append(0.0)

    if not distances:
        return 0.0
    return float(mean(distances))


def prepare_hdpo_critic_targets(
    record_path: str | Path,
    feature_path: str | Path,
    references_path: str | Path,
    output_path: str | Path,
    *,
    text_field: str = "human_answer",
    support_band: str = "q10_q90",
    include_text: bool = True,
) -> dict[str, Any]:
    record_file = Path(record_path)
    feature_file = Path(feature_path)
    output_file = Path(output_path)
    references = read_json(Path(references_path))
    records = _read_records(record_file)
    feature_rows = _read_records(feature_file)
    if len(records) != len(feature_rows):
        raise ValueError("record_path and feature_path must contain the same number of rows.")

    prepared_records: list[dict[str, Any]] = []
    skipped_missing_bucket = 0
    skipped_missing_text = 0
    for index, (record, feature_row) in enumerate(zip(records, feature_rows, strict=True)):
        record_id = record.get("id")
        if record_id != feature_row.get("id"):
            raise ValueError(
                f"Mismatched ids at row {index}: record={record.get('id')} feature={feature_row.get('id')}"
            )

        bucket_id = _resolve_bucket_id(record)
        if not bucket_id or bucket_id not in references:
            skipped_missing_bucket += 1
            continue

        response_text = _resolve_text(record, text_field)
        if not response_text:
            skipped_missing_text += 1
            continue

        target_score = _feature_support_distance(
            feature_row.get("features", {}),
            references[bucket_id],
            support_band=support_band,
        )
        prepared_record = {
            "id": record_id,
            "dataset": record.get("dataset", ""),
            "split": record.get("split", ""),
            "bucket_id": bucket_id,
            "target_score": round(target_score, 8),
            "source_text_field": text_field,
        }
        if include_text:
            prepared_record["response_text"] = response_text
        prepared_records.append(prepared_record)

    _write_records(output_file, prepared_records)
    return {
        "record_path": str(record_file),
        "feature_path": str(feature_file),
        "references_path": str(Path(references_path)),
        "output_path": str(output_file),
        "records": len(prepared_records),
        "skipped_missing_bucket": skipped_missing_bucket,
        "skipped_missing_text": skipped_missing_text,
        "support_band": support_band,
    }


class ResponseScoreDataset(Dataset):
    def __init__(self, records: list[dict[str, Any]], bucket_to_id: dict[str, int]) -> None:
        self.records = records
        self.bucket_to_id = bucket_to_id

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, Any]:
        record = self.records[index]
        return {
            "text": normalize_text(record.get("response_text", "")),
            "bucket_id": int(self.bucket_to_id.get(record.get("bucket_id", ""), self.bucket_to_id[UNKNOWN_BUCKET])),
            "target_score": float(record.get("target_score", 0.0)),
        }


class PairRankingDataset(Dataset):
    def __init__(self, records: list[dict[str, Any]], bucket_to_id: dict[str, int]) -> None:
        self.records = records
        self.bucket_to_id = bucket_to_id

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, Any]:
        record = self.records[index]
        bucket_id = int(self.bucket_to_id.get(_resolve_bucket_id(record), self.bucket_to_id[UNKNOWN_BUCKET]))
        return {
            "chosen_text": _resolve_text(record, "chosen", "chosen_answer", "chosen_output", "human_answer"),
            "rejected_text": _resolve_text(record, "rejected", "rejected_answer", "rejected_output", "model_rejected"),
            "bucket_id": bucket_id,
        }


def _collate_score_batch(batch: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "texts": [item["text"] for item in batch],
        "bucket_ids": torch.tensor([item["bucket_id"] for item in batch], dtype=torch.long),
        "target_scores": torch.tensor([item["target_score"] for item in batch], dtype=torch.float32),
    }


def _collate_pair_batch(batch: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "chosen_texts": [item["chosen_text"] for item in batch],
        "rejected_texts": [item["rejected_text"] for item in batch],
        "bucket_ids": torch.tensor([item["bucket_id"] for item in batch], dtype=torch.long),
    }


def _load_records_from_paths(paths: list[str | Path]) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for path in paths:
        records.extend(_read_records(Path(path)))
    return records


@dataclass
class CriticTrainingMetrics:
    total_loss: float
    reg_loss: float
    rank_loss: float
    steps: int


def _mean_metrics(metrics: list[CriticTrainingMetrics]) -> dict[str, float]:
    if not metrics:
        return {"loss": 0.0, "reg_loss": 0.0, "rank_loss": 0.0, "steps": 0}
    return {
        "loss": round(mean(metric.total_loss for metric in metrics), 8),
        "reg_loss": round(mean(metric.reg_loss for metric in metrics), 8),
        "rank_loss": round(mean(metric.rank_loss for metric in metrics), 8),
        "steps": sum(metric.steps for metric in metrics),
    }


def train_hdpo_critic(
    *,
    train_paths: list[str | Path],
    output_dir: str | Path,
    encoder_name_or_path: str,
    eval_paths: Optional[list[str | Path]] = None,
    pair_train_paths: Optional[list[str | Path]] = None,
    pair_eval_paths: Optional[list[str | Path]] = None,
    batch_size: int = 16,
    pair_batch_size: int = 8,
    learning_rate: float = 1.0e-3,
    weight_decay: float = 0.0,
    num_epochs: int = 3,
    max_length: int = 512,
    hidden_dim: int = 256,
    bucket_dim: int = 64,
    dropout: float = 0.1,
    margin: float = 0.1,
    reg_lambda: float = 1.0,
    rank_lambda: float = 1.0,
    encoder_learning_rate: Optional[float] = None,
    finetune_encoder: bool = False,
    trust_remote_code: bool = False,
    device: str = "auto",
    seed: int = 42,
) -> dict[str, Any]:
    from transformers import AutoModel, AutoTokenizer

    rng = random.Random(seed)
    torch.manual_seed(seed)

    train_records = _load_records_from_paths(train_paths)
    eval_records = _load_records_from_paths(eval_paths or [])
    pair_train_records = _load_records_from_paths(pair_train_paths or [])
    pair_eval_records = _load_records_from_paths(pair_eval_paths or [])
    if not train_records:
        raise ValueError("HDPO critic training requires at least one regression training record.")
    bucket_names = {UNKNOWN_BUCKET}
    for record in train_records + eval_records + pair_train_records + pair_eval_records:
        bucket_id = _resolve_bucket_id(record)
        if bucket_id:
            bucket_names.add(bucket_id)
    bucket_to_id = {bucket_name: index for index, bucket_name in enumerate(sorted(bucket_names))}

    resolved_device = torch.device(device) if device != "auto" else (
        torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    )

    tokenizer = AutoTokenizer.from_pretrained(
        encoder_name_or_path,
        trust_remote_code=trust_remote_code,
        use_fast=True,
    )
    if tokenizer.pad_token_id is None:
        if tokenizer.eos_token is not None:
            tokenizer.pad_token = tokenizer.eos_token
        elif tokenizer.bos_token is not None:
            tokenizer.pad_token = tokenizer.bos_token

    encoder_model = AutoModel.from_pretrained(
        encoder_name_or_path,
        trust_remote_code=trust_remote_code,
    )
    encoder_model.to(resolved_device)
    encoder_model.train(mode=finetune_encoder)
    if not finetune_encoder:
        for parameter in encoder_model.parameters():
            parameter.requires_grad = False

    hidden_size = int(getattr(encoder_model.config, "hidden_size"))
    critic = BucketConditionedDistributionCritic(
        response_dim=hidden_size,
        num_buckets=len(bucket_to_id),
        bucket_dim=bucket_dim,
        hidden_dim=hidden_dim,
        dropout=dropout,
    )
    critic.to(resolved_device)
    critic.train()

    train_score_loader = DataLoader(
        ResponseScoreDataset(train_records, bucket_to_id),
        batch_size=batch_size,
        shuffle=True,
        collate_fn=_collate_score_batch,
    )
    eval_score_loader = (
        DataLoader(
            ResponseScoreDataset(eval_records, bucket_to_id),
            batch_size=batch_size,
            shuffle=False,
            collate_fn=_collate_score_batch,
        )
        if eval_records
        else None
    )
    train_pair_loader = (
        DataLoader(
            PairRankingDataset(pair_train_records, bucket_to_id),
            batch_size=pair_batch_size,
            shuffle=True,
            collate_fn=_collate_pair_batch,
        )
        if pair_train_records
        else None
    )
    eval_pair_loader = (
        DataLoader(
            PairRankingDataset(pair_eval_records, bucket_to_id),
            batch_size=pair_batch_size,
            shuffle=False,
            collate_fn=_collate_pair_batch,
        )
        if pair_eval_records
        else None
    )

    parameter_groups: list[dict[str, Any]] = [{"params": list(critic.parameters()), "lr": learning_rate}]
    if finetune_encoder:
        parameter_groups.append(
            {
                "params": [parameter for parameter in encoder_model.parameters() if parameter.requires_grad],
                "lr": encoder_learning_rate or learning_rate,
            }
        )
    optimizer = torch.optim.AdamW(parameter_groups, weight_decay=weight_decay)

    def _encode_with_grad(texts: list[str]) -> torch.Tensor:
        tokenized = tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
        tokenized = {key: value.to(resolved_device) for key, value in tokenized.items()}
        outputs = encoder_model(**tokenized, output_hidden_states=False, return_dict=True)
        attention_mask = tokenized["attention_mask"].unsqueeze(-1).to(dtype=outputs.last_hidden_state.dtype)
        pooled = (outputs.last_hidden_state * attention_mask).sum(dim=1) / attention_mask.sum(dim=1).clamp_min(1.0)
        return pooled

    def _run_epoch(
        score_loader: DataLoader,
        pair_loader: Optional[DataLoader],
        *,
        train: bool,
    ) -> dict[str, float]:
        if train:
            critic.train()
            encoder_model.train(mode=finetune_encoder)
        else:
            critic.eval()
            encoder_model.eval()

        pair_batches = list(pair_loader) if pair_loader is not None else []
        if train and pair_batches:
            rng.shuffle(pair_batches)

        metrics: list[CriticTrainingMetrics] = []
        for step_index, score_batch in enumerate(score_loader):
            maybe_pair_batch = pair_batches[step_index % len(pair_batches)] if pair_batches else None
            bucket_ids = score_batch["bucket_ids"].to(resolved_device)
            target_scores = score_batch["target_scores"].to(resolved_device)

            with torch.set_grad_enabled(train):
                response_embeddings = _encode_with_grad(score_batch["texts"])
                pred_scores = critic(response_embeddings, bucket_ids)
                chosen_pred_scores = None
                rejected_pred_scores = None
                if maybe_pair_batch is not None:
                    pair_bucket_ids = maybe_pair_batch["bucket_ids"].to(resolved_device)
                    chosen_embeddings = _encode_with_grad(maybe_pair_batch["chosen_texts"])
                    rejected_embeddings = _encode_with_grad(maybe_pair_batch["rejected_texts"])
                    chosen_pred_scores = critic(chosen_embeddings, pair_bucket_ids)
                    rejected_pred_scores = critic(rejected_embeddings, pair_bucket_ids)

                critic_loss = compute_hdpo_critic_loss(
                    pred_scores,
                    target_scores,
                    chosen_pred_scores=chosen_pred_scores,
                    rejected_pred_scores=rejected_pred_scores,
                    margin=margin,
                    reg_lambda=reg_lambda,
                    rank_lambda=rank_lambda,
                )

            if train:
                optimizer.zero_grad(set_to_none=True)
                critic_loss.loss.backward()
                optimizer.step()

            metrics.append(
                CriticTrainingMetrics(
                    total_loss=float(critic_loss.loss.detach().cpu().item()),
                    reg_loss=float(critic_loss.reg_loss.detach().cpu().item()),
                    rank_loss=float(critic_loss.rank_loss.detach().cpu().item()),
                    steps=1,
                )
            )

        return _mean_metrics(metrics)

    history: list[dict[str, Any]] = []
    best_eval_loss: float | None = None
    best_state: dict[str, Any] | None = None
    for epoch in range(1, num_epochs + 1):
        train_metrics = _run_epoch(train_score_loader, train_pair_loader, train=True)
        eval_metrics = _run_epoch(eval_score_loader, eval_pair_loader, train=False) if eval_score_loader else None
        epoch_summary = {"epoch": epoch, "train": train_metrics}
        if eval_metrics is not None:
            epoch_summary["eval"] = eval_metrics
            eval_loss = float(eval_metrics["loss"])
            if best_eval_loss is None or eval_loss < best_eval_loss:
                best_eval_loss = eval_loss
                best_state = {
                    "critic": critic.state_dict(),
                    "encoder": encoder_model.state_dict() if finetune_encoder else None,
                }
        history.append(epoch_summary)

    if best_state is not None:
        critic.load_state_dict(best_state["critic"])
        if finetune_encoder and best_state["encoder"] is not None:
            encoder_model.load_state_dict(best_state["encoder"])

    output_path = Path(output_dir)
    ensure_dir(output_path)
    saved_encoder_name_or_path = encoder_name_or_path
    if finetune_encoder:
        saved_encoder_dir = output_path / "encoder"
        encoder_model.save_pretrained(saved_encoder_dir)
        tokenizer.save_pretrained(saved_encoder_dir)
        saved_encoder_name_or_path = str(saved_encoder_dir)

    bundle_config = HDPOCriticBundleConfig(
        encoder_name_or_path=saved_encoder_name_or_path,
        response_dim=hidden_size,
        num_buckets=len(bucket_to_id),
        max_length=max_length,
        bucket_dim=bucket_dim,
        hidden_dim=hidden_dim,
        dropout=dropout,
        trust_remote_code=trust_remote_code,
    )
    save_hdpo_critic_bundle(
        output_path,
        critic=critic.cpu(),
        bucket_to_id=bucket_to_id,
        config=bundle_config,
    )

    summary = {
        "output_dir": str(output_path),
        "encoder_name_or_path": saved_encoder_name_or_path,
        "train_records": len(train_records),
        "eval_records": len(eval_records),
        "pair_train_records": len(pair_train_records),
        "pair_eval_records": len(pair_eval_records),
        "bucket_count": len(bucket_to_id),
        "finetune_encoder": finetune_encoder,
        "history": history,
    }
    (output_path / "training_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return summary


def score_hdpo_pair_file(
    input_path: str | Path,
    output_path: str | Path,
    *,
    critic_path: str | Path,
    batch_size: int = 16,
    device: str = "auto",
) -> dict[str, Any]:
    input_file = Path(input_path)
    output_file = Path(output_path)
    records = _read_records(input_file)
    bundle = load_hdpo_critic_bundle(critic_path, device=device)

    chosen_texts = [_resolve_text(record, "chosen", "chosen_answer", "chosen_output", "human_answer") for record in records]
    rejected_texts = [_resolve_text(record, "rejected", "rejected_answer", "rejected_output", "model_rejected") for record in records]
    bucket_names = [_resolve_bucket_id(record) for record in records]
    bucket_ids = bucket_ids_from_names(bundle.bucket_to_id, bucket_names)

    all_texts = chosen_texts + rejected_texts
    all_bucket_ids = bucket_ids + bucket_ids
    score_tensor = predict_hdpo_critic_scores(bundle, all_texts, all_bucket_ids, batch_size=batch_size)
    midpoint = len(records)
    chosen_scores = score_tensor[:midpoint].tolist()
    rejected_scores = score_tensor[midpoint:].tolist()

    scored_records: list[dict[str, Any]] = []
    for record, critic_bucket_id, chosen_score, rejected_score in zip(
        records, bucket_ids, chosen_scores, rejected_scores, strict=True
    ):
        updated = dict(record)
        updated["critic_bucket_id"] = int(critic_bucket_id)
        updated["chosen_dist_score"] = round(float(chosen_score), 8)
        updated["rejected_dist_score"] = round(float(rejected_score), 8)
        scored_records.append(updated)

    _write_records(output_file, scored_records)
    return {
        "input_path": str(input_file),
        "output_path": str(output_file),
        "records": len(scored_records),
        "critic_path": str(Path(critic_path)),
    }


def score_hdpo_pair_root(
    input_root: str | Path,
    output_root: str | Path,
    *,
    critic_path: str | Path,
    batch_size: int = 16,
    device: str = "auto",
) -> dict[str, Any]:
    input_dir = Path(input_root)
    output_dir = Path(output_root)
    ensure_dir(output_dir)
    summary: dict[str, Any] = {"input_root": str(input_dir), "output_root": str(output_dir), "splits": {}}
    for source_split in ["train", "dev", "test", "validation2"]:
        split_entries = []
        for dataset_name, split_path in _iter_dataset_split_files(input_dir, source_split):
            target_path = output_dir / dataset_name / f"{source_split}.jsonl"
            result = score_hdpo_pair_file(
                split_path,
                target_path,
                critic_path=critic_path,
                batch_size=batch_size,
                device=device,
            )
            split_entries.append({"dataset": dataset_name, **result})
        if split_entries:
            summary["splits"][_resolve_split_output_name(source_split)] = split_entries

    return summary


def _cmd_prepare(args) -> None:
    summary = prepare_hdpo_critic_targets(
        args.record_path,
        args.feature_path,
        args.references_path,
        args.output_path,
        text_field=args.text_field,
        support_band=args.support_band,
        include_text=(not args.drop_text),
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))


def _cmd_train(args) -> None:
    summary = train_hdpo_critic(
        train_paths=args.train_path,
        output_dir=args.output_dir,
        encoder_name_or_path=args.encoder_name_or_path,
        eval_paths=args.eval_path,
        pair_train_paths=args.pair_train_path,
        pair_eval_paths=args.pair_eval_path,
        batch_size=args.batch_size,
        pair_batch_size=args.pair_batch_size,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        num_epochs=args.num_epochs,
        max_length=args.max_length,
        hidden_dim=args.hidden_dim,
        bucket_dim=args.bucket_dim,
        dropout=args.dropout,
        margin=args.margin,
        reg_lambda=args.reg_lambda,
        rank_lambda=args.rank_lambda,
        encoder_learning_rate=args.encoder_learning_rate,
        finetune_encoder=args.finetune_encoder,
        trust_remote_code=args.trust_remote_code,
        device=args.device,
        seed=args.seed,
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))


def _cmd_score_pairs(args) -> None:
    if args.input_root:
        summary = score_hdpo_pair_root(
            args.input_root,
            args.output_root,
            critic_path=args.critic_path,
            batch_size=args.batch_size,
            device=args.device,
        )
    else:
        summary = score_hdpo_pair_file(
            args.input_path,
            args.output_path,
            critic_path=args.critic_path,
            batch_size=args.batch_size,
            device=args.device,
        )
    print(json.dumps(summary, indent=2, ensure_ascii=False))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="HDPO critic preparation, training, and pair scoring utilities.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    prepare_parser = subparsers.add_parser(
        "prepare",
        help="Prepare critic regression targets from aligned response records, feature rows, and bucket references.",
    )
    prepare_parser.add_argument("--record-path", required=True, help="Current-format JSONL/JSON file containing responses.")
    prepare_parser.add_argument("--feature-path", required=True, help="Aligned feature JSONL/JSON file for the same records.")
    prepare_parser.add_argument("--references-path", required=True, help="Path to bucket_references.json.")
    prepare_parser.add_argument("--output-path", required=True, help="Output JSONL/JSON file for critic regression data.")
    prepare_parser.add_argument("--text-field", default="human_answer", help="Response text field to extract from the input records.")
    prepare_parser.add_argument(
        "--support-band",
        choices=["q10_q90", "q25_q75"],
        default="q10_q90",
        help="Support interval used to compute distance-to-support targets.",
    )
    prepare_parser.add_argument("--drop-text", action="store_true", help="Do not include response_text in the prepared dataset.")
    prepare_parser.set_defaults(func=_cmd_prepare)

    train_parser = subparsers.add_parser("train", help="Train the bucket-conditioned HDPO critic.")
    train_parser.add_argument("--train-path", action="append", required=True, help="Prepared critic regression JSONL/JSON file. Repeatable.")
    train_parser.add_argument("--eval-path", action="append", help="Optional eval regression JSONL/JSON file. Repeatable.")
    train_parser.add_argument("--pair-train-path", action="append", help="Optional pairwise JSONL/JSON file for ranking loss. Repeatable.")
    train_parser.add_argument("--pair-eval-path", action="append", help="Optional pairwise eval JSONL/JSON file for ranking loss. Repeatable.")
    train_parser.add_argument("--output-dir", required=True, help="Directory where the critic bundle will be written.")
    train_parser.add_argument("--encoder-name-or-path", required=True, help="Transformer encoder backbone used for response embeddings.")
    train_parser.add_argument("--batch-size", type=int, default=16, help="Regression batch size.")
    train_parser.add_argument("--pair-batch-size", type=int, default=8, help="Pair ranking batch size.")
    train_parser.add_argument("--learning-rate", type=float, default=1.0e-3, help="Critic learning rate.")
    train_parser.add_argument("--encoder-learning-rate", type=float, help="Optional separate learning rate when fine-tuning the encoder.")
    train_parser.add_argument("--weight-decay", type=float, default=0.0, help="Optimizer weight decay.")
    train_parser.add_argument("--num-epochs", type=int, default=3, help="Number of critic training epochs.")
    train_parser.add_argument("--max-length", type=int, default=512, help="Maximum encoder sequence length.")
    train_parser.add_argument("--hidden-dim", type=int, default=256, help="Hidden size of the critic MLP.")
    train_parser.add_argument("--bucket-dim", type=int, default=64, help="Bucket embedding size.")
    train_parser.add_argument("--dropout", type=float, default=0.1, help="Critic dropout.")
    train_parser.add_argument("--margin", type=float, default=0.1, help="Ranking margin for chosen vs rejected.")
    train_parser.add_argument("--reg-lambda", type=float, default=1.0, help="Regression coefficient.")
    train_parser.add_argument("--rank-lambda", type=float, default=1.0, help="Ranking coefficient.")
    train_parser.add_argument("--finetune-encoder", action="store_true", help="Fine-tune the encoder backbone together with the critic.")
    train_parser.add_argument("--trust-remote-code", action="store_true", help="Pass trust_remote_code=True to AutoModel/AutoTokenizer.")
    train_parser.add_argument("--device", default="auto", help="Torch device for critic training.")
    train_parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    train_parser.set_defaults(func=_cmd_train)

    score_parser = subparsers.add_parser("score-pairs", help="Score chosen/rejected pair files with a trained HDPO critic.")
    score_input = score_parser.add_mutually_exclusive_group(required=True)
    score_input.add_argument("--input-path", help="Single pairwise JSONL/JSON file to score.")
    score_input.add_argument("--input-root", help="Root directory containing per-dataset pairwise split JSONL files.")
    score_parser.add_argument("--output-path", help="Output JSONL/JSON path for single-file scoring.")
    score_parser.add_argument("--output-root", help="Output root for mirrored per-dataset scoring.")
    score_parser.add_argument("--critic-path", required=True, help="Path to a trained critic bundle directory.")
    score_parser.add_argument("--batch-size", type=int, default=16, help="Critic scoring batch size.")
    score_parser.add_argument("--device", default="auto", help="Torch device for critic scoring.")
    score_parser.set_defaults(func=_cmd_score_pairs)

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    if args.command == "score-pairs":
        if args.input_path and not args.output_path:
            parser.error("--output-path is required with --input-path.")
        if args.input_root and not args.output_root:
            parser.error("--output-root is required with --input-root.")
    args.func(args)


if __name__ == "__main__":
    main()
