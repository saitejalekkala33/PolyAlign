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


def _cmd_hdpo_critic_prepare(args) -> None:
    from polyalign_data.hdpo_critic import prepare_hdpo_critic_targets

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


def _cmd_hdpo_build_pairs(args) -> None:
    from polyalign_data.build_hdpo_pairs import build_hdpo_pair_files

    summary = build_hdpo_pair_files(
        args.record_path,
        args.prediction_path,
        args.output_root,
        split_name=args.split_name,
        pair_type=args.pair_type,
        prediction_text_field=args.prediction_text_field,
        keep_exact_match=args.keep_exact_match,
        keep_mismatched=args.keep_mismatched,
        merged_output_path=args.merged_output_path,
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))


def _cmd_export_dpo_views(args) -> None:
    from polyalign_data.export_dpo_views import export_dpo_views

    summary = export_dpo_views(
        records_root=args.records_root,
        output_root=args.output_root,
        model_alias=args.model_alias,
        language_tag=args.language_tag,
        train_predictions=args.train_predictions,
        val_predictions=args.val_predictions,
        test_predictions=args.test_predictions,
        prediction_filename=args.prediction_filename,
        system_prompt=args.system_prompt,
        prediction_text_field=args.prediction_text_field,
        keep_exact_match=args.keep_exact_match,
        dataset_info_path=args.dataset_info_path,
        update_dataset_info=(not args.skip_dataset_info_update),
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))


def _cmd_hdpo_critic_train(args) -> None:
    from polyalign_data.hdpo_critic import train_hdpo_critic

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


def _cmd_hdpo_critic_score_pairs(args) -> None:
    from polyalign_data.hdpo_critic import score_hdpo_pair_file, score_hdpo_pair_root

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

    dpo_export_parser = subparsers.add_parser("export-dpo-views", help="Export LlamaFactory-compatible DPO files from current split files and aligned SFT predictions.",)
    dpo_export_parser.add_argument("--records-root", required=True, help="Directory containing train/val/test current-format files.")
    dpo_export_parser.add_argument("--output-root", required=True, help="Directory where exported DPO JSON files will be written.")
    dpo_export_parser.add_argument("--model-alias", required=True, help="Model alias used in exported filenames.")
    dpo_export_parser.add_argument("--language-tag", required=True, help="Language tag used in exported filenames.")
    dpo_export_parser.add_argument("--train-predictions", help="Train prediction file or run directory containing predictions.jsonl.")
    dpo_export_parser.add_argument("--val-predictions", help="Val prediction file or run directory containing predictions.jsonl.")
    dpo_export_parser.add_argument("--test-predictions", help="Test prediction file or run directory containing predictions.jsonl.")
    dpo_export_parser.add_argument("--prediction-filename", default="predictions.jsonl", help="Prediction filename to use when a prediction path points to a directory.",)
    dpo_export_parser.add_argument("--system-prompt", default="", help="Optional fixed system prompt written into every exported DPO record.",)
    dpo_export_parser.add_argument("--prediction-text-field", default="prediction", help="Field in the prediction rows used as the rejected response text.")
    dpo_export_parser.add_argument("--keep-exact-match", action="store_true", help="Keep rows where the rejected prediction exactly matches the chosen human answer after normalization.")
    dpo_export_parser.add_argument("--dataset-info-path", default="vendor/LlamaFactory/data/dataset_info.json", help="Path to the LlamaFactory dataset_info.json file to update after export.")
    dpo_export_parser.add_argument("--skip-dataset-info-update", action="store_true", help="Do not update dataset_info.json after exporting the DPO files.")
    dpo_export_parser.set_defaults(func=_cmd_export_dpo_views)

    hdpo_pair_parser = subparsers.add_parser("hdpo-build-pairs", help="Build raw HDPO pair files by aligning current-format records with model prediction JSONL files.")
    hdpo_pair_parser.add_argument("--record-path", required=True, help="Current-format merged JSONL/JSON file for a single split.")
    hdpo_pair_parser.add_argument("--prediction-path", required=True, help="Prediction JSONL/JSON file aligned to the same split and containing `source_index`.")
    hdpo_pair_parser.add_argument("--output-root", required=True, help="Output root where per-dataset pair files are written as `<dataset>/<split>.jsonl`.")
    hdpo_pair_parser.add_argument("--split-name", required=True, help="Source split name to write. Accepts `train`, `dev`, `test`, `validation2`, or `val`.")
    hdpo_pair_parser.add_argument("--pair-type", default="global", help="Pair type label written into exported records.")
    hdpo_pair_parser.add_argument("--prediction-text-field", default="prediction", help="Field in the prediction rows used as the rejected response text.")
    hdpo_pair_parser.add_argument("--keep-exact-match", action="store_true", help="Keep rows whose rejected prediction exactly matches the human answer after normalization.")
    hdpo_pair_parser.add_argument("--keep-mismatched", action="store_true", help="Keep rows even when instruction/input/reference_output do not match the aligned source record.")
    hdpo_pair_parser.add_argument("--merged-output-path", help="Optional merged JSONL/JSON file containing all exported pair rows for this split.")
    hdpo_pair_parser.set_defaults(func=_cmd_hdpo_build_pairs)

    hdpo_prepare_parser = subparsers.add_parser("hdpo-critic-prepare", help="Prepare HDPO critic regression targets from aligned records, feature rows, and bucket references.")
    hdpo_prepare_parser.add_argument("--record-path", required=True, help="Current-format JSONL/JSON file containing responses.")
    hdpo_prepare_parser.add_argument("--feature-path", required=True, help="Aligned feature JSONL/JSON file for the same records.")
    hdpo_prepare_parser.add_argument("--references-path", required=True, help="Path to bucket_references.json.")
    hdpo_prepare_parser.add_argument("--output-path", required=True, help="Output JSONL/JSON file for critic regression data.")
    hdpo_prepare_parser.add_argument("--text-field", default="human_answer", help="Response field to extract from the input records.")
    hdpo_prepare_parser.add_argument(
        "--support-band",
        choices=["q10_q90", "q25_q75"],
        default="q10_q90",
        help="Support interval used to compute distance-to-support targets.",
    )
    hdpo_prepare_parser.add_argument("--drop-text", action="store_true", help="Do not include response_text in the prepared dataset.")
    hdpo_prepare_parser.set_defaults(func=_cmd_hdpo_critic_prepare)

    hdpo_train_parser = subparsers.add_parser("hdpo-critic-train", help="Train the HDPO distribution critic.")
    hdpo_train_parser.add_argument("--train-path", action="append", required=True, help="Prepared critic regression JSONL/JSON file. Repeatable.")
    hdpo_train_parser.add_argument("--eval-path", action="append", help="Optional eval regression JSONL/JSON file. Repeatable.")
    hdpo_train_parser.add_argument("--pair-train-path", action="append", help="Optional pairwise JSONL/JSON file for ranking loss. Repeatable.")
    hdpo_train_parser.add_argument("--pair-eval-path", action="append", help="Optional pairwise eval JSONL/JSON file for ranking loss. Repeatable.")
    hdpo_train_parser.add_argument("--output-dir", required=True, help="Directory where the critic bundle will be written.")
    hdpo_train_parser.add_argument("--encoder-name-or-path", required=True, help="Transformer encoder backbone used for response embeddings.")
    hdpo_train_parser.add_argument("--batch-size", type=int, default=16, help="Regression batch size.")
    hdpo_train_parser.add_argument("--pair-batch-size", type=int, default=8, help="Pair ranking batch size.")
    hdpo_train_parser.add_argument("--learning-rate", type=float, default=1.0e-3, help="Critic learning rate.")
    hdpo_train_parser.add_argument("--encoder-learning-rate", type=float, help="Optional separate learning rate when fine-tuning the encoder.")
    hdpo_train_parser.add_argument("--weight-decay", type=float, default=0.0, help="Optimizer weight decay.")
    hdpo_train_parser.add_argument("--num-epochs", type=int, default=3, help="Number of critic training epochs.")
    hdpo_train_parser.add_argument("--max-length", type=int, default=512, help="Maximum encoder sequence length.")
    hdpo_train_parser.add_argument("--hidden-dim", type=int, default=256, help="Hidden size of the critic MLP.")
    hdpo_train_parser.add_argument("--bucket-dim", type=int, default=64, help="Bucket embedding size.")
    hdpo_train_parser.add_argument("--dropout", type=float, default=0.1, help="Critic dropout.")
    hdpo_train_parser.add_argument("--margin", type=float, default=0.1, help="Ranking margin for chosen vs rejected.")
    hdpo_train_parser.add_argument("--reg-lambda", type=float, default=1.0, help="Regression coefficient.")
    hdpo_train_parser.add_argument("--rank-lambda", type=float, default=1.0, help="Ranking coefficient.")
    hdpo_train_parser.add_argument("--finetune-encoder", action="store_true", help="Fine-tune the encoder backbone together with the critic.")
    hdpo_train_parser.add_argument("--trust-remote-code", action="store_true", help="Pass trust_remote_code=True to AutoModel/AutoTokenizer.")
    hdpo_train_parser.add_argument("--device", default="auto", help="Torch device for critic training.")
    hdpo_train_parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    hdpo_train_parser.set_defaults(func=_cmd_hdpo_critic_train)

    hdpo_score_parser = subparsers.add_parser(
        "hdpo-critic-score-pairs",
        help="Score chosen/rejected pair files with a trained HDPO critic.",
    )
    hdpo_score_group = hdpo_score_parser.add_mutually_exclusive_group(required=True)
    hdpo_score_group.add_argument("--input-path", help="Single pairwise JSONL/JSON file to score.")
    hdpo_score_group.add_argument("--input-root", help="Root directory containing per-dataset pairwise split JSONL files.")
    hdpo_score_parser.add_argument("--output-path", help="Output JSONL/JSON path for single-file scoring.")
    hdpo_score_parser.add_argument("--output-root", help="Output root for mirrored per-dataset scoring.")
    hdpo_score_parser.add_argument("--critic-path", required=True, help="Path to a trained critic bundle directory.")
    hdpo_score_parser.add_argument("--batch-size", type=int, default=16, help="Critic scoring batch size.")
    hdpo_score_parser.add_argument("--device", default="auto", help="Torch device for critic scoring.")
    hdpo_score_parser.set_defaults(func=_cmd_hdpo_critic_score_pairs)

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    if args.command == "hdpo-critic-score-pairs":
        if getattr(args, "input_path", None) and not getattr(args, "output_path", None):
            parser.error("--output-path is required when --input-path is used.")
        if getattr(args, "input_root", None) and not getattr(args, "output_root", None):
            parser.error("--output-root is required when --input-root is used.")
    args.func(args)
