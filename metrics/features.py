from __future__ import annotations

from dataclasses import dataclass
from array import array
from pathlib import Path
from typing import Any

import pandas as pd

from metrics.io import (
    EvaluationPaths,
    flatten_dialogue_history,
    iter_jsonl,
    load_concatenated_json,
    load_json,
    load_jsonl,
    write_jsonl,
)
from polyalign_data.extract_linguistic_features import extract_model_features_file
from polyalign_data.lm_registry import resolve_model_aliases


def log_step(message: str) -> None:
    print(f"[metrics] {message}", flush=True)


@dataclass(frozen=True)
class ArtifactInfo:
    path: Path | None
    status: str


def build_alignment_report(
    test_lf_rows: list[dict[str, Any]],
    prediction_rows: list[dict[str, Any]],
    current_test_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    expected_indices = set(range(len(test_lf_rows)))
    observed_indices: list[int] = []
    bad_index_rows = 0
    mismatches = {
        "instruction_mismatches": 0,
        "input_mismatches": 0,
        "history_mismatches": 0,
        "reference_output_mismatches": 0,
        "current_instruction_mismatches": 0,
        "current_input_mismatches": 0,
        "current_output_mismatches": 0,
        "current_history_mismatches": 0,
    }

    if len(test_lf_rows) != len(current_test_rows):
        raise ValueError(
            f"LlamaFactory test rows ({len(test_lf_rows)}) and current test rows ({len(current_test_rows)}) do not match."
        )

    for lf_row, current_row in zip(test_lf_rows, current_test_rows, strict=True):
        if lf_row.get("instruction") != current_row.get("question"):
            mismatches["current_instruction_mismatches"] += 1
        if lf_row.get("input") != current_row.get("context"):
            mismatches["current_input_mismatches"] += 1
        if lf_row.get("output") != current_row.get("human_answer"):
            mismatches["current_output_mismatches"] += 1
        if (lf_row.get("history") or []) != flatten_dialogue_history(current_row.get("dialogue_history") or []):
            mismatches["current_history_mismatches"] += 1

    for row in prediction_rows:
        source_index = row.get("source_index")
        if not isinstance(source_index, int) or not (0 <= source_index < len(test_lf_rows)):
            bad_index_rows += 1
            continue
        observed_indices.append(source_index)
        lf_row = test_lf_rows[source_index]
        if row.get("instruction") != lf_row.get("instruction"):
            mismatches["instruction_mismatches"] += 1
        if row.get("input") != lf_row.get("input"):
            mismatches["input_mismatches"] += 1
        if (row.get("history") or []) != (lf_row.get("history") or []):
            mismatches["history_mismatches"] += 1
        if row.get("reference_output") != lf_row.get("output"):
            mismatches["reference_output_mismatches"] += 1

    observed_set = set(observed_indices)
    return {
        "n_test_rows": len(test_lf_rows),
        "n_prediction_rows": len(prediction_rows),
        "n_current_test_rows": len(current_test_rows),
        "bad_index_rows": bad_index_rows,
        "unique_prediction_indices": len(observed_set),
        "duplicate_prediction_rows": len(observed_indices) - len(observed_set),
        "missing_index_count": len(expected_indices - observed_set),
        "extra_index_count": len(observed_set - expected_indices),
        "min_index": min(observed_set) if observed_set else None,
        "max_index": max(observed_set) if observed_set else None,
        **mismatches,
    }


def load_primary_rows(paths: EvaluationPaths) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    test_lf_rows = load_json(paths.test_lf_path)
    prediction_rows = sorted(load_concatenated_json(paths.predictions_path), key=lambda row: row["source_index"])
    current_test_rows = load_jsonl(paths.current_test_path)
    alignment_report = build_alignment_report(test_lf_rows, prediction_rows, current_test_rows)
    return test_lf_rows, prediction_rows, current_test_rows, alignment_report


def build_generated_current_rows(
    current_test_rows: list[dict[str, Any]],
    prediction_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    generated_rows: list[dict[str, Any]] = []
    for prediction_row in prediction_rows:
        source_index = prediction_row["source_index"]
        base_row = dict(current_test_rows[source_index])
        base_row["model_answer"] = prediction_row.get("prediction", "") or ""
        base_row["reference_output"] = prediction_row.get("reference_output", "") or ""
        base_row["prediction_model_name"] = prediction_row.get("model_name", "")
        generated_rows.append(base_row)
    return generated_rows


def ensure_generated_records_file(
    *,
    paths: EvaluationPaths,
    current_test_rows: list[dict[str, Any]],
    prediction_rows: list[dict[str, Any]],
    overwrite: bool,
) -> ArtifactInfo:
    output_path = paths.work_dir / "generated_current_records.jsonl"
    if output_path.exists() and not overwrite:
        log_step(f"Reusing generated records: {output_path}")
        return ArtifactInfo(path=output_path, status="reused")
    log_step(f"Writing generated current-format records: {output_path}")
    generated_rows = build_generated_current_rows(current_test_rows, prediction_rows)
    write_jsonl(output_path, generated_rows)
    return ArtifactInfo(path=output_path, status="created")


def find_existing_feature_artifact(output_jsonl_path: Path) -> Path | None:
    if output_jsonl_path.exists():
        return output_jsonl_path
    csv_path = output_jsonl_path.with_suffix(".csv")
    if csv_path.exists():
        return csv_path
    return None


def ensure_model_feature_file(
    *,
    input_jsonl_path: Path,
    output_jsonl_path: Path,
    model_alias: str,
    text_field: str,
    overwrite: bool,
    device: str,
    dtype: str,
    max_seq_length: int,
) -> ArtifactInfo:
    existing_artifact = None if overwrite else find_existing_feature_artifact(output_jsonl_path)
    if existing_artifact is not None:
        log_step(f"Reusing feature file: {existing_artifact}")
        return ArtifactInfo(path=existing_artifact, status="reused")
    log_step(
        f"Computing model features with {model_alias} for field `{text_field}` -> {output_jsonl_path}"
    )
    model_specs = resolve_model_aliases([model_alias])
    if len(model_specs) != 1:
        raise ValueError(f"Could not resolve model alias {model_alias!r}.")
    extract_model_features_file(
        input_jsonl_path,
        output_jsonl_path,
        model_spec=model_specs[0],
        text_field=text_field,
        output_csv=output_jsonl_path.with_suffix(".csv"),
        include_text=False,
        device=device,
        dtype=dtype,
        max_seq_length=max_seq_length,
    )
    return ArtifactInfo(path=output_jsonl_path, status="created")


def ensure_prediction_feature_file(
    *,
    paths: EvaluationPaths,
    current_test_rows: list[dict[str, Any]],
    prediction_rows: list[dict[str, Any]],
    overwrite: bool,
    device: str,
    dtype: str,
    max_seq_length: int,
) -> tuple[ArtifactInfo, ArtifactInfo]:
    output_path = paths.work_dir / f"prediction_features_{paths.model_alias}.jsonl"
    existing_artifact = None if overwrite else find_existing_feature_artifact(output_path)
    if existing_artifact is not None:
        log_step(f"Reusing prediction-side feature file: {existing_artifact}")
        return ArtifactInfo(path=existing_artifact, status="reused"), ArtifactInfo(path=None, status="not_required")

    generated_records_info = ensure_generated_records_file(
        paths=paths,
        current_test_rows=current_test_rows,
        prediction_rows=prediction_rows,
        overwrite=overwrite,
    )
    if generated_records_info.path is None:
        raise ValueError("Generated records path is unavailable while computing prediction features.")
    feature_info = ensure_model_feature_file(
        input_jsonl_path=generated_records_info.path,
        output_jsonl_path=output_path,
        model_alias=paths.model_alias,
        text_field="model_answer",
        overwrite=overwrite,
        device=device,
        dtype=dtype,
        max_seq_length=max_seq_length,
    )
    return feature_info, generated_records_info


def ensure_human_feature_file(
    *,
    paths: EvaluationPaths,
    overwrite: bool,
    device: str,
    dtype: str,
    max_seq_length: int,
) -> ArtifactInfo:
    if paths.human_feature_path is not None and paths.human_feature_path.exists():
        log_step(f"Reusing human feature file: {paths.human_feature_path}")
        return ArtifactInfo(path=paths.human_feature_path, status="reused")
    output_path = paths.work_dir / f"human_test_features_{paths.model_alias}.jsonl"
    return ensure_model_feature_file(
        input_jsonl_path=paths.current_test_path,
        output_jsonl_path=output_path,
        model_alias=paths.model_alias,
        text_field="human_answer",
        overwrite=overwrite,
        device=device,
        dtype=dtype,
        max_seq_length=max_seq_length,
    )


def feature_rows_to_frame(feature_rows: list[dict[str, Any]]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for row in feature_rows:
        payload = {
            "id": row.get("id", ""),
            "dataset": row.get("dataset", ""),
            "split": row.get("split", ""),
            "field_name": row.get("field_name", ""),
        }
        if "model_alias" in row:
            payload["model_alias"] = row.get("model_alias", "")
            payload["model_name"] = row.get("model_name", "")
        payload.update({str(name): float(value) for name, value in row.get("features", {}).items()})
        rows.append(payload)
    return pd.DataFrame(rows)


def load_feature_frame(path: Path) -> pd.DataFrame:
    if path.suffix.lower() == ".csv":
        frame = pd.read_csv(path)
        unnamed_columns = [column for column in frame.columns if str(column).startswith("Unnamed:")]
        if unnamed_columns:
            frame = frame.drop(columns=unnamed_columns)
        return frame
    return feature_rows_to_frame(load_jsonl(path))


def build_reference_feature_frame_from_human_test(
    *,
    current_test_rows: list[dict[str, Any]],
    human_feature_frame: pd.DataFrame,
) -> pd.DataFrame:
    metadata_frame = pd.DataFrame(
        [
            {
                "id": row["id"],
                "bucket_id": row["bucket_id"],
                "dataset": row["dataset"],
                "split": row["split"],
                "track": row["track"],
                "family": row["family"],
                "style_bucket": row["style_bucket"],
                "length_bin": row["length_bin"],
            }
            for row in current_test_rows
        ]
    )
    human_payload = human_feature_frame.drop(columns=["dataset", "split", "field_name"], errors="ignore")
    merged = metadata_frame.merge(human_payload, on="id", how="inner", validate="one_to_one")
    if len(merged) != len(current_test_rows):
        log_step(
            "Human feature file is partial; BNG human-test reference distribution will use "
            f"{len(merged)}/{len(current_test_rows)} rows."
        )
    return merged


def build_reference_feature_frame_from_feature_matrix(feature_matrix_path: Path) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for row in iter_jsonl(feature_matrix_path):
        payload = {
            "id": row.get("id", ""),
            "dataset": row.get("dataset", ""),
            "split": row.get("split", ""),
            "track": row.get("track", ""),
            "family": row.get("family", ""),
            "style_bucket": row.get("style_bucket", ""),
            "length_bin": row.get("length_bin", ""),
            "bucket_id": row.get("bucket_id", ""),
        }
        payload.update({str(name): float(value) for name, value in row.get("features", {}).items()})
        rows.append(payload)
    return pd.DataFrame(rows)


def frame_to_distribution_map(frame: pd.DataFrame, *, feature_columns: list[str]) -> dict[str, dict[str, array]]:
    distribution_map: dict[str, dict[str, array]] = {}
    for bucket_id, bucket_frame in frame.groupby("bucket_id"):
        bucket_map: dict[str, array] = {}
        for feature_name in feature_columns:
            bucket_map[feature_name] = array("d", bucket_frame[feature_name].astype(float).tolist())
        distribution_map[str(bucket_id)] = bucket_map
    return distribution_map
