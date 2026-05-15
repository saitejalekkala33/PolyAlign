from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Optional

from polyalign_data.hdpo_critic import (
    DEFAULT_MAX_GRAD_NORM,
    prepare_hdpo_critic_targets,
    score_hdpo_pair_file,
    score_hdpo_pair_root,
    train_hdpo_critic,
)


def train_hdpo_mlp_critic(
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
    max_grad_norm: float = DEFAULT_MAX_GRAD_NORM,
    torch_dtype: Optional[str] = None,
    fix_mistral_regex: bool = False,
    trust_remote_code: bool = False,
    device: str = "auto",
    seed: int = 42,
    show_progress: bool = True,
) -> dict[str, Any]:
    summary = train_hdpo_critic(
        train_paths=train_paths,
        output_dir=output_dir,
        encoder_name_or_path=encoder_name_or_path,
        eval_paths=eval_paths,
        pair_train_paths=pair_train_paths,
        pair_eval_paths=pair_eval_paths,
        batch_size=batch_size,
        pair_batch_size=pair_batch_size,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        num_epochs=num_epochs,
        max_length=max_length,
        hidden_dim=hidden_dim,
        bucket_dim=bucket_dim,
        dropout=dropout,
        margin=margin,
        reg_lambda=reg_lambda,
        rank_lambda=rank_lambda,
        encoder_learning_rate=None,
        finetune_encoder=False,
        max_grad_norm=max_grad_norm,
        torch_dtype=torch_dtype,
        gradient_checkpointing=False,
        fix_mistral_regex=fix_mistral_regex,
        trust_remote_code=trust_remote_code,
        device=device,
        seed=seed,
        show_progress=show_progress,
    )
    summary["critic_train_mode"] = "mlp_only"
    return summary


def _cmd_prepare(args: argparse.Namespace) -> None:
    summary = prepare_hdpo_critic_targets(
        args.record_path,
        args.feature_path,
        args.references_path,
        args.output_path,
        text_field=args.text_field,
        support_band=args.support_band,
        include_text=True,
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))


def _cmd_train(args: argparse.Namespace) -> None:
    summary = train_hdpo_mlp_critic(
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
        max_grad_norm=args.max_grad_norm,
        torch_dtype=args.torch_dtype,
        fix_mistral_regex=args.fix_mistral_regex,
        trust_remote_code=args.trust_remote_code,
        device=args.device,
        seed=args.seed,
        show_progress=(not args.no_progress),
    )
    if summary.get("is_main_process", True):
        print(json.dumps(summary, indent=2, ensure_ascii=False))


def _cmd_score_pairs(args: argparse.Namespace) -> None:
    if args.input_root:
        summary = score_hdpo_pair_root(
            args.input_root,
            args.output_root,
            critic_path=args.critic_path,
            batch_size=args.batch_size,
            device=args.device,
            allow_constant_scores=args.allow_constant_scores,
        )
    else:
        summary = score_hdpo_pair_file(
            args.input_path,
            args.output_path,
            critic_path=args.critic_path,
            batch_size=args.batch_size,
            device=args.device,
            allow_constant_scores=args.allow_constant_scores,
        )
    print(json.dumps(summary, indent=2, ensure_ascii=False))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Frozen-encoder MLP-only HDPO critic utilities.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    prepare_parser = subparsers.add_parser(
        "prepare",
        help="Prepare MLP critic regression targets from records, feature rows, and bucket references.",
    )
    prepare_parser.add_argument("--record-path", required=True, help="Current-format JSONL/JSON file containing responses.")
    prepare_parser.add_argument("--feature-path", required=True, help="Aligned feature JSONL/JSON file for the same records.")
    prepare_parser.add_argument("--references-path", required=True, help="Path to bucket_references.json.")
    prepare_parser.add_argument("--output-path", required=True, help="Output JSONL/JSON file for critic regression data.")
    prepare_parser.add_argument("--text-field", default="human_answer", help="Response text field to extract from input records.")
    prepare_parser.add_argument(
        "--support-band",
        choices=["q10_q90", "q25_q75"],
        default="q10_q90",
        help="Support interval used to compute distance-to-support targets.",
    )
    prepare_parser.set_defaults(func=_cmd_prepare)

    train_parser = subparsers.add_parser("train", help="Train only the bucket-conditioned MLP critic.")
    train_parser.add_argument("--train-path", action="append", required=True, help="Prepared critic regression JSONL/JSON file. Repeatable.")
    train_parser.add_argument("--eval-path", action="append", help="Optional eval regression JSONL/JSON file. Repeatable.")
    train_parser.add_argument("--pair-train-path", action="append", help="Optional pairwise JSONL/JSON file for ranking loss. Repeatable.")
    train_parser.add_argument("--pair-eval-path", action="append", help="Optional pairwise eval JSONL/JSON file for ranking loss. Repeatable.")
    train_parser.add_argument("--output-dir", required=True, help="Directory where the MLP critic bundle will be written.")
    train_parser.add_argument("--encoder-name-or-path", required=True, help="Frozen transformer encoder used for response embeddings.")
    train_parser.add_argument("--batch-size", type=int, default=16, help="Regression batch size.")
    train_parser.add_argument("--pair-batch-size", type=int, default=8, help="Pair ranking batch size.")
    train_parser.add_argument("--learning-rate", type=float, default=1.0e-3, help="MLP critic learning rate.")
    train_parser.add_argument("--weight-decay", type=float, default=0.0, help="Optimizer weight decay.")
    train_parser.add_argument("--num-epochs", type=int, default=3, help="Number of critic training epochs.")
    train_parser.add_argument("--max-length", type=int, default=512, help="Maximum encoder sequence length.")
    train_parser.add_argument("--hidden-dim", type=int, default=256, help="Hidden size of the critic MLP.")
    train_parser.add_argument("--bucket-dim", type=int, default=64, help="Bucket embedding size.")
    train_parser.add_argument("--dropout", type=float, default=0.1, help="Critic dropout.")
    train_parser.add_argument("--margin", type=float, default=0.1, help="Ranking margin for chosen vs rejected.")
    train_parser.add_argument("--reg-lambda", type=float, default=1.0, help="Regression coefficient.")
    train_parser.add_argument("--rank-lambda", type=float, default=1.0, help="Ranking coefficient.")
    train_parser.add_argument(
        "--max-grad-norm",
        type=float,
        default=DEFAULT_MAX_GRAD_NORM,
        help="Gradient clipping norm for MLP critic params. Set to 0 to disable clipping.",
    )
    train_parser.add_argument(
        "--torch-dtype",
        choices=["auto", "fp16", "float16", "bf16", "bfloat16", "fp32", "float32"],
        default="auto",
        help="Optional dtype used to load the frozen encoder backbone.",
    )
    train_parser.add_argument(
        "--fix-mistral-regex",
        action="store_true",
        help="Pass fix_mistral_regex=True when loading the tokenizer.",
    )
    train_parser.add_argument("--trust-remote-code", action="store_true", help="Pass trust_remote_code=True to AutoModel/AutoTokenizer.")
    train_parser.add_argument("--device", default="auto", help="Torch device for critic training.")
    train_parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    train_parser.add_argument("--no-progress", action="store_true", help="Disable tqdm batch progress bars.")
    train_parser.set_defaults(func=_cmd_train)

    score_parser = subparsers.add_parser("score-pairs", help="Score chosen/rejected pair files with a trained MLP critic.")
    score_input = score_parser.add_mutually_exclusive_group(required=True)
    score_input.add_argument("--input-path", help="Single pairwise JSONL/JSON file to score.")
    score_input.add_argument("--input-root", help="Root directory containing per-dataset pairwise split JSONL files.")
    score_parser.add_argument("--output-path", help="Output JSONL/JSON path for single-file scoring.")
    score_parser.add_argument("--output-root", help="Output root for mirrored per-dataset scoring.")
    score_parser.add_argument("--critic-path", required=True, help="Path to a trained MLP critic bundle directory.")
    score_parser.add_argument("--batch-size", type=int, default=16, help="Critic scoring batch size.")
    score_parser.add_argument("--device", default="auto", help="Torch device for critic scoring.")
    score_parser.add_argument(
        "--allow-constant-scores",
        action="store_true",
        help="Write output even when score diagnostics detect a collapsed constant critic.",
    )
    score_parser.set_defaults(func=_cmd_score_pairs)

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    if args.command == "score-pairs":
        if getattr(args, "input_path", None) and not getattr(args, "output_path", None):
            parser.error("--output-path is required with --input-path.")
        if getattr(args, "input_root", None) and not getattr(args, "output_root", None):
            parser.error("--output-root is required with --input-root.")
    args.func(args)


if __name__ == "__main__":
    main()
