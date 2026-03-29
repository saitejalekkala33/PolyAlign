from __future__ import annotations

import argparse
import json
from typing import Any

import pandas as pd

from metrics.features import (
    build_reference_feature_frame_from_feature_matrix,
    build_reference_feature_frame_from_human_test,
    ensure_generated_records_file,
    ensure_human_feature_file,
    ensure_prediction_feature_file,
    frame_to_distribution_map,
    load_feature_frame,
    load_primary_rows,
)
from metrics.io import EvaluationPaths, resolve_evaluation_paths, write_json
from metrics.metrics_lib import (
    build_utility_frame,
    compute_bng,
    compute_hcr,
    compute_mauve,
    compute_tdm,
    shared_feature_names,
    summarize_diversity,
    summarize_nuf,
    summarize_utility,
)


def log_step(message: str) -> None:
    print(f"[metrics] {message}", flush=True)


def build_aligned_frame(
    *,
    test_lf_rows: list[dict[str, Any]],
    prediction_rows: list[dict[str, Any]],
    current_test_rows: list[dict[str, Any]],
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for prediction_row in prediction_rows:
        source_index = prediction_row["source_index"]
        test_row = test_lf_rows[source_index]
        current_row = current_test_rows[source_index]
        rows.append(
            {
                "source_index": source_index,
                "id": current_row["id"],
                "dataset": current_row["dataset"],
                "split": current_row["split"],
                "track": current_row["track"],
                "family": current_row["family"],
                "style_bucket": current_row["style_bucket"],
                "length_bin": current_row["length_bin"],
                "bucket_id": current_row["bucket_id"],
                "question": current_row["question"],
                "context": current_row["context"],
                "reference_output": test_row["output"],
                "prediction": prediction_row.get("prediction", "") or "",
                "model_name": prediction_row.get("model_name", ""),
                "finish_reason": prediction_row.get("finish_reason", ""),
            }
        )
    return pd.DataFrame(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate PolyAlign generations against naturalness and utility metrics.")
    parser.add_argument("--test-lf-path", required=True, help="Path to the LlamaFactory test.json file.")
    parser.add_argument("--predictions-path", required=True, help="Path to predictions.jsonl with source_index.")
    parser.add_argument("--output-json", required=True, help="Path to the output metrics JSON file.")
    parser.add_argument("--current-test-path", help="Optional path to current/test.jsonl.")
    parser.add_argument("--human-feature-path", help="Optional path to test_answer_features_dedup.jsonl.")
    parser.add_argument("--bucket-references-path", help="Optional path to bucket_references.json.")
    parser.add_argument("--feature-matrix-path", help="Optional path to feature_matrix.jsonl.")
    parser.add_argument("--work-dir", help="Optional work directory for generated artifacts.")
    parser.add_argument("--model-alias", help="Research model alias used for LM features. Defaults to the run directory name.")
    parser.add_argument("--device", default="auto", help="Device for prediction-side LM feature extraction.")
    parser.add_argument("--dtype", default="auto", help="Torch dtype for prediction-side LM feature extraction.")
    parser.add_argument("--max-seq-length", type=int, default=4096, help="Maximum sequence length for LM feature extraction.")
    parser.add_argument(
        "--bng-reference-source",
        choices=["human_test_features", "feature_matrix"],
        default="human_test_features",
        help="Reference distribution source for BNG.",
    )
    parser.add_argument(
        "--hcr-support-key",
        choices=["support_q10_q90", "support_q25_q75"],
        default="support_q10_q90",
        help="Support region to use for HCR.",
    )
    parser.add_argument("--min-bucket-size", type=int, default=20, help="Minimum bucket size for bucket-level metrics.")
    parser.add_argument("--skip-mauve", action="store_true", help="Skip MAUVE and conditional MAUVE.")
    parser.add_argument("--mauve-featurizer-model", default="gpt2", help="Featurizer model name passed to mauve-text.")
    parser.add_argument("--mauve-device-id", type=int, default=-1, help="Device id for mauve-text. Use -1 for CPU.")
    parser.add_argument("--max-mauve-texts", type=int, default=1000, help="Maximum texts per MAUVE computation.")
    parser.add_argument("--self-bleu-sample-size", type=int, default=500, help="Sample size for Self-BLEU.")
    parser.add_argument(
        "--self-bleu-refs-per-candidate",
        type=int,
        default=100,
        help="Maximum number of reference generations per Self-BLEU candidate.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed for sampling-based metrics.")
    parser.add_argument("--overwrite-artifacts", action="store_true", help="Recompute cached generated records and features.")
    return parser.parse_args()


def evaluate(paths: EvaluationPaths, args: argparse.Namespace) -> dict[str, Any]:
    log_step("Loading primary input files")
    test_lf_rows, prediction_rows, current_test_rows, alignment_report = load_primary_rows(paths)
    log_step("Validating alignment across test, prediction, and current-format files")
    required_zero = [
        "bad_index_rows",
        "duplicate_prediction_rows",
        "missing_index_count",
        "extra_index_count",
        "instruction_mismatches",
        "input_mismatches",
        "history_mismatches",
        "reference_output_mismatches",
        "current_instruction_mismatches",
        "current_input_mismatches",
        "current_output_mismatches",
        "current_history_mismatches",
    ]
    if any(alignment_report[key] != 0 for key in required_zero):
        raise ValueError(f"Input alignment failed: {alignment_report}")
    log_step("Alignment validation passed")

    log_step("Building aligned evaluation table")
    aligned_frame = build_aligned_frame(
        test_lf_rows=test_lf_rows,
        prediction_rows=prediction_rows,
        current_test_rows=current_test_rows,
    )
    log_step(f"Aligned {len(aligned_frame)} examples")

    log_step("Computing utility metrics")
    utility_frame = build_utility_frame(aligned_frame)
    utility_summary = summarize_utility(utility_frame)
    log_step("Utility metrics complete")

    log_step("Computing diversity metrics")
    diversity_summary = summarize_diversity(
        aligned_frame=aligned_frame,
        self_bleu_sample_size=args.self_bleu_sample_size,
        self_bleu_refs_per_candidate=args.self_bleu_refs_per_candidate,
        seed=args.seed,
    )
    log_step("Diversity metrics complete")

    log_step("Preparing generated current-format records")
    generated_records_path = ensure_generated_records_file(
        paths=paths,
        current_test_rows=current_test_rows,
        prediction_rows=prediction_rows,
        overwrite=args.overwrite_artifacts,
    )
    log_step("Preparing prediction-side LM feature file")
    prediction_feature_path = ensure_prediction_feature_file(
        paths=paths,
        generated_records_path=generated_records_path,
        overwrite=args.overwrite_artifacts,
        device=args.device,
        dtype=args.dtype,
        max_seq_length=args.max_seq_length,
    )
    log_step("Preparing human reference feature file")
    human_feature_path = ensure_human_feature_file(
        paths=paths,
        overwrite=args.overwrite_artifacts,
        device=args.device,
        dtype=args.dtype,
        max_seq_length=args.max_seq_length,
    )

    log_step("Loading generated and human feature tables")
    generated_feature_frame = load_feature_frame(prediction_feature_path)
    generated_feature_frame = aligned_frame[["id", "bucket_id", "track", "family", "style_bucket", "length_bin"]].merge(
        generated_feature_frame,
        on="id",
        how="inner",
        validate="one_to_one",
    )
    if len(generated_feature_frame) != len(aligned_frame):
        raise ValueError(
            f"Prediction feature rows ({len(generated_feature_frame)}) do not align one-to-one with predictions ({len(aligned_frame)})."
        )

    human_feature_frame = load_feature_frame(human_feature_path)
    log_step("Building human reference distribution source")
    if args.bng_reference_source == "feature_matrix":
        if paths.feature_matrix_path is None or not paths.feature_matrix_path.exists():
            raise ValueError(
                "BNG reference source was set to feature_matrix, but no readable feature_matrix.jsonl was resolved."
            )
        log_step(f"Using feature matrix for BNG reference source: {paths.feature_matrix_path}")
        human_reference_frame = build_reference_feature_frame_from_feature_matrix(paths.feature_matrix_path)
    else:
        log_step("Using aligned human test feature file for BNG reference source")
        human_reference_frame = build_reference_feature_frame_from_human_test(
            current_test_rows=current_test_rows,
            human_feature_frame=human_feature_frame,
        )

    log_step("Loading bucket references")
    bucket_references = json.loads(paths.bucket_references_path.read_text(encoding="utf-8"))
    feature_names = shared_feature_names(
        generated_feature_frame=generated_feature_frame,
        human_reference_frame=human_reference_frame,
        bucket_references=bucket_references,
    )
    log_step(f"Shared feature count: {len(feature_names)}")
    human_distribution_map = frame_to_distribution_map(
        human_reference_frame,
        feature_columns=feature_names,
    )

    log_step("Computing BNG")
    bng_summary = compute_bng(
        generated_feature_frame=generated_feature_frame,
        human_reference_distribution_map=human_distribution_map,
        feature_names=feature_names,
        min_bucket_size=args.min_bucket_size,
    )
    log_step("Computing HCR")
    hcr_summary = compute_hcr(
        generated_feature_frame=generated_feature_frame,
        bucket_references=bucket_references,
        feature_names=feature_names,
        support_key=args.hcr_support_key,
        min_bucket_size=args.min_bucket_size,
    )
    log_step("Computing MAUVE and conditional MAUVE")
    mauve_summary = compute_mauve(
        aligned_frame=aligned_frame,
        skip_mauve=args.skip_mauve,
        featurize_model_name=args.mauve_featurizer_model,
        device_id=args.mauve_device_id,
        max_texts=args.max_mauve_texts,
        min_bucket_size=args.min_bucket_size,
        seed=args.seed,
    )
    log_step("Computing TDM")
    tdm_summary = compute_tdm(
        aligned_frame=aligned_frame,
        min_bucket_size=args.min_bucket_size,
    )
    log_step("Computing NUF")
    nuf_summary = summarize_nuf(
        utility_summary=utility_summary,
        bng_summary=bng_summary,
        hcr_summary=hcr_summary,
        mauve_summary=mauve_summary,
        tdm_summary=tdm_summary,
    )
    log_step("Metric computation complete")

    return {
        "config": {
            "test_lf_path": str(paths.test_lf_path),
            "predictions_path": str(paths.predictions_path),
            "current_test_path": str(paths.current_test_path),
            "human_feature_path": str(human_feature_path),
            "bucket_references_path": str(paths.bucket_references_path),
            "feature_matrix_path": str(paths.feature_matrix_path) if paths.feature_matrix_path else None,
            "output_json_path": str(paths.output_json_path),
            "work_dir": str(paths.work_dir),
            "model_alias": paths.model_alias,
            "device": args.device,
            "dtype": args.dtype,
            "max_seq_length": args.max_seq_length,
            "bng_reference_source": args.bng_reference_source,
            "hcr_support_key": args.hcr_support_key,
            "min_bucket_size": args.min_bucket_size,
            "skip_mauve": args.skip_mauve,
            "mauve_featurizer_model": args.mauve_featurizer_model,
            "mauve_device_id": args.mauve_device_id,
            "max_mauve_texts": args.max_mauve_texts,
            "self_bleu_sample_size": args.self_bleu_sample_size,
            "self_bleu_refs_per_candidate": args.self_bleu_refs_per_candidate,
            "seed": args.seed,
        },
        "alignment": alignment_report,
        "artifacts": {
            "generated_records_path": str(generated_records_path),
            "prediction_feature_path": str(prediction_feature_path),
            "human_feature_path": str(human_feature_path),
            "shared_feature_count": len(feature_names),
            "shared_feature_names": feature_names,
        },
        "metrics": {
            "utility": utility_summary,
            "diversity": diversity_summary,
            "bng": bng_summary,
            "hcr": hcr_summary,
            "mauve": mauve_summary,
            "tdm": tdm_summary,
            "nuf": nuf_summary,
        },
    }


def main() -> None:
    args = parse_args()
    log_step("Resolving input and companion paths")
    paths = resolve_evaluation_paths(
        test_lf_path=args.test_lf_path,
        predictions_path=args.predictions_path,
        output_json_path=args.output_json,
        current_test_path=args.current_test_path,
        human_feature_path=args.human_feature_path,
        bucket_references_path=args.bucket_references_path,
        feature_matrix_path=args.feature_matrix_path,
        work_dir=args.work_dir,
        model_alias=args.model_alias,
    )
    log_step(f"Working directory: {paths.work_dir}")
    result = evaluate(paths, args)
    log_step(f"Writing final metrics JSON: {paths.output_json_path}")
    write_json(paths.output_json_path, result)
    log_step("Done")


if __name__ == "__main__":
    main()
