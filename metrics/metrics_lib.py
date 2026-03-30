from __future__ import annotations

import math
import random
from array import array
from collections import Counter
from typing import Any

import numpy as np
import pandas as pd
from nltk.translate.bleu_score import SmoothingFunction, sentence_bleu
from scipy.stats import wasserstein_distance
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler

from metrics.io import normalize_answer


def _trapezoid_integral(y: np.ndarray, x: np.ndarray) -> float:
    if hasattr(np, "trapezoid"):
        return float(np.trapezoid(y, x))
    return float(np.trapz(y, x))


def token_f1(prediction: str, reference: str) -> float:
    prediction_tokens = normalize_answer(prediction).split()
    reference_tokens = normalize_answer(reference).split()
    if not prediction_tokens and not reference_tokens:
        return 1.0
    if not prediction_tokens or not reference_tokens:
        return 0.0
    common = Counter(prediction_tokens) & Counter(reference_tokens)
    overlap = sum(common.values())
    if overlap == 0:
        return 0.0
    precision = overlap / len(prediction_tokens)
    recall = overlap / len(reference_tokens)
    return 2 * precision * recall / (precision + recall)


def rouge_l_f1(prediction: str, reference: str) -> float:
    prediction_tokens = normalize_answer(prediction).split()
    reference_tokens = normalize_answer(reference).split()
    if not prediction_tokens and not reference_tokens:
        return 1.0
    if not prediction_tokens or not reference_tokens:
        return 0.0
    rows = len(prediction_tokens) + 1
    cols = len(reference_tokens) + 1
    dp = [[0] * cols for _ in range(rows)]
    for i in range(1, rows):
        for j in range(1, cols):
            if prediction_tokens[i - 1] == reference_tokens[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])
    lcs = dp[-1][-1]
    precision = lcs / len(prediction_tokens)
    recall = lcs / len(reference_tokens)
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def word_tokens(text: str) -> list[str]:
    return normalize_answer(text).split()


def distinct_n(texts: list[str], n: int) -> float:
    ngrams: list[tuple[str, ...]] = []
    for text in texts:
        tokens = word_tokens(text)
        ngrams.extend(tuple(tokens[index : index + n]) for index in range(len(tokens) - n + 1))
    if not ngrams:
        return 0.0
    return len(set(ngrams)) / len(ngrams)


def self_bleu(
    texts: list[str],
    *,
    n: int,
    sample_size: int,
    refs_per_candidate: int,
    seed: int,
) -> float:
    rng = random.Random(seed)
    sample = list(texts)
    if len(sample) > sample_size:
        sample = rng.sample(sample, sample_size)
    tokenized = [word_tokens(text) for text in sample]
    tokenized = [tokens for tokens in tokenized if tokens]
    if len(tokenized) < 2:
        return 0.0
    weights = tuple([1.0 / n] * n)
    smoothing = SmoothingFunction().method1
    scores: list[float] = []
    for index, candidate in enumerate(tokenized):
        references = tokenized[:index] + tokenized[index + 1 :]
        if len(references) > refs_per_candidate:
            references = rng.sample(references, refs_per_candidate)
        scores.append(sentence_bleu(references, candidate, weights=weights, smoothing_function=smoothing))
    return float(np.mean(scores))


def build_utility_frame(aligned_frame: pd.DataFrame) -> pd.DataFrame:
    utility_frame = aligned_frame.copy()
    utility_frame["raw_em"] = (
        utility_frame["prediction"].fillna("").astype(str) == utility_frame["reference_output"].fillna("").astype(str)
    ).astype(float)
    utility_frame["qa_em"] = utility_frame.apply(
        lambda row: float(normalize_answer(row["prediction"]) == normalize_answer(row["reference_output"])),
        axis=1,
    )
    utility_frame["qa_f1"] = utility_frame.apply(
        lambda row: token_f1(str(row["prediction"]), str(row["reference_output"])),
        axis=1,
    )
    utility_frame["rouge_l_f1"] = utility_frame.apply(
        lambda row: rouge_l_f1(str(row["prediction"]), str(row["reference_output"])),
        axis=1,
    )
    return utility_frame


def summarize_utility(utility_frame: pd.DataFrame) -> dict[str, Any]:
    by_bucket = (
        utility_frame.groupby("bucket_id")[["raw_em", "qa_em", "qa_f1", "rouge_l_f1"]]
        .mean()
        .reset_index()
        .to_dict(orient="records")
    )
    by_dataset = (
        utility_frame.groupby("dataset")[["raw_em", "qa_em", "qa_f1", "rouge_l_f1"]]
        .mean()
        .reset_index()
        .to_dict(orient="records")
    )
    overall = {
        "raw_em": float(utility_frame["raw_em"].mean()),
        "qa_em": float(utility_frame["qa_em"].mean()),
        "qa_f1": float(utility_frame["qa_f1"].mean()),
        "rouge_l_f1": float(utility_frame["rouge_l_f1"].mean()),
    }
    return {"overall": overall, "by_bucket": by_bucket, "by_dataset": by_dataset}


def summarize_diversity(
    *,
    aligned_frame: pd.DataFrame,
    self_bleu_sample_size: int,
    self_bleu_refs_per_candidate: int,
    seed: int,
) -> dict[str, Any]:
    def _summary(texts: list[str]) -> dict[str, float]:
        return {
            "distinct_1": distinct_n(texts, 1),
            "distinct_2": distinct_n(texts, 2),
            "distinct_3": distinct_n(texts, 3),
            "self_bleu_4": self_bleu(
                texts,
                n=4,
                sample_size=self_bleu_sample_size,
                refs_per_candidate=self_bleu_refs_per_candidate,
                seed=seed,
            ),
        }

    human_texts = aligned_frame["reference_output"].astype(str).tolist()
    generated_texts = aligned_frame["prediction"].astype(str).tolist()
    by_bucket = []
    for bucket_id, bucket_frame in aligned_frame.groupby("bucket_id"):
        by_bucket.append(
            {
                "bucket_id": bucket_id,
                "human": _summary(bucket_frame["reference_output"].astype(str).tolist()),
                "generated": _summary(bucket_frame["prediction"].astype(str).tolist()),
            }
        )
    return {
        "overall": {"human": _summary(human_texts), "generated": _summary(generated_texts)},
        "by_bucket": by_bucket,
    }


def shared_feature_names(
    *,
    generated_feature_frame: pd.DataFrame,
    human_reference_frame: pd.DataFrame,
    bucket_references: dict[str, Any],
) -> list[str]:
    excluded = {
        "id",
        "dataset",
        "split",
        "field_name",
        "model_alias",
        "model_name",
        "bucket_id",
        "track",
        "family",
        "style_bucket",
        "length_bin",
    }
    generated_features = {column for column in generated_feature_frame.columns if column not in excluded}
    human_features = {column for column in human_reference_frame.columns if column not in excluded}
    reference_features = set()
    for bucket_payload in bucket_references.values():
        reference_features.update(bucket_payload.get("feature_stats", {}).keys())
    return sorted(generated_features & human_features & reference_features)


def compute_bng(
    *,
    generated_feature_frame: pd.DataFrame,
    human_reference_distribution_map: dict[str, dict[str, array]],
    feature_names: list[str],
    min_bucket_size: int,
) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for bucket_id, bucket_frame in generated_feature_frame.groupby("bucket_id"):
        if bucket_id not in human_reference_distribution_map or len(bucket_frame) < min_bucket_size:
            continue
        bucket_gaps: list[float] = []
        for feature_name in feature_names:
            if feature_name not in human_reference_distribution_map[bucket_id]:
                continue
            human_values = np.array(human_reference_distribution_map[bucket_id][feature_name], dtype=float)
            generated_values = bucket_frame[feature_name].astype(float).to_numpy()
            if human_values.size == 0 or generated_values.size == 0:
                continue
            human_std = float(np.std(human_values))
            if human_std <= 1e-12:
                human_std = 1.0
            gap = wasserstein_distance(generated_values, human_values) / human_std
            bucket_gaps.append(float(gap))
        if bucket_gaps:
            rows.append(
                {
                    "bucket_id": bucket_id,
                    "n_examples": int(len(bucket_frame)),
                    "n_features": int(len(bucket_gaps)),
                    "bng": float(np.mean(bucket_gaps)),
                }
            )
    bucket_frame = pd.DataFrame(rows).sort_values("bucket_id").reset_index(drop=True) if rows else pd.DataFrame()
    overall_macro = float(bucket_frame["bng"].mean()) if not bucket_frame.empty else math.nan
    overall_weighted = (
        float(np.average(bucket_frame["bng"], weights=bucket_frame["n_examples"])) if not bucket_frame.empty else math.nan
    )
    return {
        "overall_macro": overall_macro,
        "overall_weighted": overall_weighted,
        "by_bucket": bucket_frame.to_dict(orient="records"),
    }


def compute_hcr(
    *,
    generated_feature_frame: pd.DataFrame,
    bucket_references: dict[str, Any],
    feature_names: list[str],
    support_key: str,
    min_bucket_size: int,
) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    overall_membership: list[bool] = []
    for bucket_id, bucket_frame in generated_feature_frame.groupby("bucket_id"):
        if bucket_id not in bucket_references or len(bucket_frame) < min_bucket_size:
            continue
        support = bucket_references[bucket_id][support_key]
        usable = [feature_name for feature_name in feature_names if feature_name in support]
        if not usable:
            continue
        feature_values = bucket_frame[usable].astype(float)
        inside_mask = np.ones(len(feature_values), dtype=bool)
        feature_pass_rates: dict[str, float] = {}
        for feature_name in usable:
            low = float(support[feature_name]["low"])
            high = float(support[feature_name]["high"])
            feature_inside = (feature_values[feature_name].to_numpy() >= low) & (feature_values[feature_name].to_numpy() <= high)
            inside_mask &= feature_inside
            feature_pass_rates[feature_name] = float(feature_inside.mean())
        overall_membership.extend(inside_mask.tolist())
        rows.append(
            {
                "bucket_id": bucket_id,
                "n_examples": int(len(bucket_frame)),
                "n_features": int(len(usable)),
                "hcr": float(inside_mask.mean()),
                "mean_feature_inside_rate": float(np.mean(list(feature_pass_rates.values()))),
            }
        )
    bucket_frame = pd.DataFrame(rows).sort_values("bucket_id").reset_index(drop=True) if rows else pd.DataFrame()
    overall_macro = float(bucket_frame["hcr"].mean()) if not bucket_frame.empty else math.nan
    overall_weighted = float(np.mean(overall_membership)) if overall_membership else math.nan
    return {
        "support_region": support_key,
        "overall_macro": overall_macro,
        "overall_weighted": overall_weighted,
        "by_bucket": bucket_frame.to_dict(orient="records"),
    }


def _try_import_mauve() -> Any:
    try:
        import mauve  # type: ignore
    except Exception as exc:
        raise RuntimeError(
            "MAUVE is required for this evaluation. Install it with `pip install mauve-text` or pass --skip-mauve."
        ) from exc
    return mauve


def compute_mauve(
    *,
    aligned_frame: pd.DataFrame,
    skip_mauve: bool,
    featurize_model_name: str,
    device_id: int,
    max_texts: int,
    min_bucket_size: int,
    seed: int,
) -> dict[str, Any]:
    if skip_mauve:
        return {"global": math.nan, "overall_macro": math.nan, "overall_weighted": math.nan, "by_bucket": [], "skipped": True}

    mauve = _try_import_mauve()

    def _score(human_texts: list[str], generated_texts: list[str]) -> float:
        rng = random.Random(seed)
        sample_size = min(len(human_texts), len(generated_texts), max_texts)
        if sample_size < min_bucket_size:
            return math.nan
        if len(human_texts) > sample_size:
            human_texts = rng.sample(human_texts, sample_size)
        if len(generated_texts) > sample_size:
            generated_texts = rng.sample(generated_texts, sample_size)
        result = mauve.compute_mauve(
            p_text=human_texts,
            q_text=generated_texts,
            device_id=device_id,
            max_text_length=256,
            verbose=False,
            featurize_model_name=featurize_model_name,
        )
        return float(result.mauve)

    global_score = _score(
        aligned_frame["reference_output"].astype(str).tolist(),
        aligned_frame["prediction"].astype(str).tolist(),
    )
    rows: list[dict[str, Any]] = []
    for bucket_id, bucket_frame in aligned_frame.groupby("bucket_id"):
        human_texts = bucket_frame["reference_output"].astype(str).tolist()
        generated_texts = bucket_frame["prediction"].astype(str).tolist()
        if min(len(human_texts), len(generated_texts)) < min_bucket_size:
            continue
        rows.append({"bucket_id": bucket_id, "n_examples": int(len(bucket_frame)), "c_mauve": _score(human_texts, generated_texts)})
    bucket_scores = pd.DataFrame(rows)
    overall_macro = float(bucket_scores["c_mauve"].mean()) if not bucket_scores.empty else math.nan
    overall_weighted = (
        float(np.average(bucket_scores["c_mauve"], weights=bucket_scores["n_examples"])) if not bucket_scores.empty else math.nan
    )
    return {
        "global": global_score,
        "overall_macro": overall_macro,
        "overall_weighted": overall_weighted,
        "by_bucket": rows,
        "skipped": False,
    }


def _cosine_similarity_rows(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    left_norm = np.linalg.norm(left, axis=1)
    right_norm = np.linalg.norm(right, axis=1)
    denom = left_norm * right_norm
    denom = np.where(denom <= 1e-12, 1.0, denom)
    return np.sum(left * right, axis=1) / denom


def _sparse_row_cosine_similarity(left: Any, right: Any) -> np.ndarray:
    similarity = left.multiply(right).sum(axis=1)
    return np.asarray(similarity).reshape(-1)


def _stabilized_similarity_from_gap(gap: float) -> float:
    if not math.isfinite(gap):
        return math.nan
    return 1.0 / (1.0 + gap)


def compute_tdm(
    *,
    aligned_frame: pd.DataFrame,
    min_bucket_size: int,
) -> dict[str, Any]:
    multi_turn_frame = aligned_frame[aligned_frame["track"] == "multi"].copy()
    if multi_turn_frame.empty:
        return {"overall_macro": math.nan, "by_bucket": [], "method": "continuous_latent_turn_dynamics"}

    question_texts = multi_turn_frame["question"].astype(str).tolist()
    human_texts = multi_turn_frame["reference_output"].astype(str).tolist()
    generated_texts = multi_turn_frame["prediction"].astype(str).tolist()

    vectorizer = TfidfVectorizer(norm="l2", lowercase=True, token_pattern=r"(?u)\b\w+\b")
    vectorizer.fit(question_texts + human_texts + generated_texts)
    question_matrix = vectorizer.transform(question_texts)
    human_matrix = vectorizer.transform(human_texts)
    generated_matrix = vectorizer.transform(generated_texts)

    max_rank = min(question_matrix.shape[0] - 1, question_matrix.shape[1] - 1)
    if max_rank >= 2:
        svd_rank = min(64, max_rank)
        reducer = TruncatedSVD(n_components=svd_rank, random_state=0)
        reducer.fit(question_matrix)
        question_latent = reducer.transform(question_matrix)
        human_latent = reducer.transform(human_matrix)
        generated_latent = reducer.transform(generated_matrix)
    else:
        question_latent = question_matrix.toarray()
        human_latent = human_matrix.toarray()
        generated_latent = generated_matrix.toarray()

    question_token_counts = np.array([len(normalize_answer(text).split()) for text in question_texts], dtype=float)
    human_token_counts = np.array([len(normalize_answer(text).split()) for text in human_texts], dtype=float)
    generated_token_counts = np.array([len(normalize_answer(text).split()) for text in generated_texts], dtype=float)

    multi_turn_frame["human_length_ratio"] = human_token_counts / np.maximum(question_token_counts, 1.0)
    multi_turn_frame["generated_length_ratio"] = generated_token_counts / np.maximum(question_token_counts, 1.0)

    multi_turn_frame["human_lexical_accommodation"] = _sparse_row_cosine_similarity(question_matrix, human_matrix)
    multi_turn_frame["generated_lexical_accommodation"] = _sparse_row_cosine_similarity(question_matrix, generated_matrix)
    multi_turn_frame["human_semantic_coupling"] = _cosine_similarity_rows(question_latent, human_latent)
    multi_turn_frame["generated_semantic_coupling"] = _cosine_similarity_rows(question_latent, generated_latent)

    question_cols = [f"question_latent_{index}" for index in range(question_latent.shape[1])]
    human_cols = [f"human_response_latent_{index}" for index in range(human_latent.shape[1])]
    generated_cols = [f"generated_response_latent_{index}" for index in range(generated_latent.shape[1])]
    latent_frame = pd.concat(
        [
            pd.DataFrame(question_latent, columns=question_cols, index=multi_turn_frame.index),
            pd.DataFrame(human_latent, columns=human_cols, index=multi_turn_frame.index),
            pd.DataFrame(generated_latent, columns=generated_cols, index=multi_turn_frame.index),
        ],
        axis=1,
    )
    multi_turn_frame = pd.concat([multi_turn_frame, latent_frame], axis=1)

    rows: list[dict[str, Any]] = []
    for bucket_id, bucket_frame in multi_turn_frame.groupby("bucket_id"):
        if len(bucket_frame) < min_bucket_size:
            continue
        human_length = bucket_frame["human_length_ratio"].to_numpy(dtype=float)
        generated_length = bucket_frame["generated_length_ratio"].to_numpy(dtype=float)
        human_accommodation = bucket_frame["human_lexical_accommodation"].to_numpy(dtype=float)
        generated_accommodation = bucket_frame["generated_lexical_accommodation"].to_numpy(dtype=float)
        human_coupling = bucket_frame["human_semantic_coupling"].to_numpy(dtype=float)
        generated_coupling = bucket_frame["generated_semantic_coupling"].to_numpy(dtype=float)

        length_std = float(np.std(human_length))
        accommodation_std = float(np.std(human_accommodation))
        coupling_std = float(np.std(human_coupling))
        if not math.isfinite(length_std) or length_std <= 1e-12:
            length_std = 1.0
        if not math.isfinite(accommodation_std) or accommodation_std <= 1e-12:
            accommodation_std = 1.0
        if not math.isfinite(coupling_std) or coupling_std <= 1e-12:
            coupling_std = 1.0

        length_gap = wasserstein_distance(generated_length, human_length) / length_std
        accommodation_gap = wasserstein_distance(generated_accommodation, human_accommodation) / accommodation_std
        coupling_gap = wasserstein_distance(generated_coupling, human_coupling) / coupling_std

        question_values = bucket_frame[question_cols].to_numpy(dtype=float)
        human_values = bucket_frame[human_cols].to_numpy(dtype=float)
        generated_values = bucket_frame[generated_cols].to_numpy(dtype=float)
        question_centered = question_values - question_values.mean(axis=0, keepdims=True)
        human_centered = human_values - human_values.mean(axis=0, keepdims=True)
        generated_centered = generated_values - generated_values.mean(axis=0, keepdims=True)
        human_transition = question_centered.T @ human_centered / max(len(bucket_frame), 1)
        generated_transition = question_centered.T @ generated_centered / max(len(bucket_frame), 1)
        denominator = float(np.linalg.norm(human_transition, ord="fro"))
        if not math.isfinite(denominator) or denominator <= 1e-12:
            denominator = 1.0
        transition_gap = float(np.linalg.norm(generated_transition - human_transition, ord="fro") / denominator)

        length_similarity = _stabilized_similarity_from_gap(float(length_gap))
        accommodation_similarity = _stabilized_similarity_from_gap(float(accommodation_gap))
        coupling_similarity = _stabilized_similarity_from_gap(float(coupling_gap))
        transition_similarity = _stabilized_similarity_from_gap(float(transition_gap))
        tdm_value = float(np.mean([length_similarity, accommodation_similarity, coupling_similarity, transition_similarity]))
        rows.append(
            {
                "bucket_id": bucket_id,
                "n_examples": int(len(bucket_frame)),
                "length_ratio_similarity": float(length_similarity),
                "lexical_accommodation_similarity": float(accommodation_similarity),
                "semantic_coupling_similarity": float(coupling_similarity),
                "latent_transition_similarity": float(transition_similarity),
                "tdm": tdm_value,
            }
        )

    bucket_frame = pd.DataFrame(rows).sort_values("bucket_id").reset_index(drop=True) if rows else pd.DataFrame()
    overall_macro = float(bucket_frame["tdm"].mean()) if not bucket_frame.empty else math.nan
    return {
        "overall_macro": overall_macro,
        "by_bucket": bucket_frame.to_dict(orient="records"),
        "method": "continuous_latent_turn_dynamics",
    }


def compute_pareto_frontier(
    bucket_frame: pd.DataFrame,
    *,
    utility_column: str,
    naturalness_column: str,
) -> dict[str, Any]:
    clean = bucket_frame.dropna(subset=[utility_column, naturalness_column]).copy()
    if clean.empty:
        return {"frontier_auc": math.nan, "frontier_points": []}
    ordered = clean.sort_values([utility_column, naturalness_column], ascending=[False, False])
    frontier_points: list[dict[str, Any]] = []
    best_naturalness = -math.inf
    for _, row in ordered.iterrows():
        utility_value = float(row[utility_column])
        naturalness_value = float(row[naturalness_column])
        if naturalness_value > best_naturalness:
            frontier_points.append(
                {
                    "bucket_id": row["bucket_id"],
                    utility_column: utility_value,
                    naturalness_column: naturalness_value,
                }
            )
            best_naturalness = naturalness_value
    frontier_df = pd.DataFrame(frontier_points).sort_values(utility_column) if frontier_points else pd.DataFrame()
    frontier_auc = (
        _trapezoid_integral(frontier_df[naturalness_column].to_numpy(), frontier_df[utility_column].to_numpy())
        if len(frontier_df) >= 2
        else math.nan
    )
    return {"frontier_auc": frontier_auc, "frontier_points": frontier_points}


def compute_hypervolume(
    bucket_frame: pd.DataFrame,
    *,
    utility_column: str,
    naturalness_column: str,
    reference_point: tuple[float, float] = (0.0, 0.0),
) -> float:
    clean = bucket_frame.dropna(subset=[utility_column, naturalness_column]).copy()
    if clean.empty:
        return math.nan
    frontier = compute_pareto_frontier(
        clean,
        utility_column=utility_column,
        naturalness_column=naturalness_column,
    )
    if not frontier["frontier_points"]:
        return math.nan
    reference_utility, reference_naturalness = reference_point
    frontier_df = pd.DataFrame(frontier["frontier_points"]).sort_values(utility_column)
    frontier_df = frontier_df[
        (frontier_df[utility_column] > reference_utility)
        & (frontier_df[naturalness_column] > reference_naturalness)
    ]
    if frontier_df.empty:
        return 0.0
    hypervolume = 0.0
    previous_utility = float(reference_utility)
    for _, row in frontier_df.iterrows():
        utility_value = float(row[utility_column])
        naturalness_value = float(row[naturalness_column])
        if utility_value <= previous_utility:
            continue
        hypervolume += (utility_value - previous_utility) * (naturalness_value - reference_naturalness)
        previous_utility = utility_value
    return float(hypervolume)


def _clip01_series(series: pd.Series) -> pd.Series:
    return series.astype(float).clip(lower=0.0, upper=1.0)


def _normalize_global_metric_series(series: pd.Series, *, metric_name: str) -> pd.Series:
    numeric = series.astype(float)
    if metric_name == "bng":
        return 1.0 / (1.0 + numeric.clip(lower=0.0))
    return _clip01_series(numeric)


def _bounded_geometric_mean(values: np.ndarray) -> float:
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return math.nan
    bounded = np.clip(finite, 0.0, 1.0)
    if np.any(bounded == 0.0):
        return 0.0
    return float(np.exp(np.mean(np.log(bounded))))


def _rowwise_bounded_geometric_mean(frame: pd.DataFrame, columns: list[str]) -> pd.Series:
    if not columns:
        return pd.Series([math.nan] * len(frame), index=frame.index, dtype=float)
    matrix = frame[columns].to_numpy(dtype=float)
    result = np.full(len(frame), np.nan, dtype=float)
    valid_rows = np.isfinite(matrix).all(axis=1)
    if not np.any(valid_rows):
        return pd.Series(result, index=frame.index, dtype=float)
    valid_matrix = np.clip(matrix[valid_rows], 0.0, 1.0)
    zero_rows = np.any(valid_matrix == 0.0, axis=1)
    result_valid = np.empty(len(valid_matrix), dtype=float)
    result_valid[zero_rows] = 0.0
    nonzero_rows = ~zero_rows
    if np.any(nonzero_rows):
        result_valid[nonzero_rows] = np.exp(np.mean(np.log(valid_matrix[nonzero_rows]), axis=1))
    result[valid_rows] = result_valid
    return pd.Series(result, index=frame.index, dtype=float)


def build_global_normalized_nuf_summary(
    bucket_frame: pd.DataFrame,
    *,
    utility_column: str,
    metric_columns: list[str],
    reference_point: tuple[float, float] = (0.0, 0.0),
) -> dict[str, Any]:
    if bucket_frame.empty:
        return {
            "normalization": "global_fixed_scale",
            "utility_axis": utility_column,
            "naturalness_components": metric_columns,
            "reference_point": {"utility": reference_point[0], "naturalness": reference_point[1]},
            "frontier_auc": math.nan,
            "hypervolume": math.nan,
            "frontier_points": [],
            "by_bucket": [],
        }

    working = bucket_frame.copy()
    working["utility_score"] = _clip01_series(working[utility_column])
    normalized_columns: list[str] = []
    for metric_column in metric_columns:
        if metric_column not in working:
            continue
        normalized_column = f"normalized__{metric_column}"
        working[normalized_column] = _normalize_global_metric_series(working[metric_column], metric_name=metric_column)
        normalized_columns.append(normalized_column)

    working["naturalness_score"] = _rowwise_bounded_geometric_mean(working, normalized_columns)
    frontier = compute_pareto_frontier(
        working,
        utility_column="utility_score",
        naturalness_column="naturalness_score",
    )
    hypervolume = compute_hypervolume(
        working,
        utility_column="utility_score",
        naturalness_column="naturalness_score",
        reference_point=reference_point,
    )
    return {
        "normalization": "global_fixed_scale",
        "utility_axis": utility_column,
        "naturalness_components": metric_columns,
        "normalized_component_columns": normalized_columns,
        "reference_point": {"utility": reference_point[0], "naturalness": reference_point[1]},
        "frontier_auc": frontier["frontier_auc"],
        "hypervolume": hypervolume,
        "frontier_points": frontier["frontier_points"],
        "by_bucket": working.to_dict(orient="records"),
    }


def build_naturalness_index(
    bucket_frame: pd.DataFrame,
    *,
    utility_column: str,
    metric_columns: list[str],
) -> tuple[pd.DataFrame, dict[str, Any]]:
    if bucket_frame.empty:
        return bucket_frame.copy(), {"frontier_auc": math.nan, "loadings": {}, "frontier_points": []}

    working = bucket_frame.copy()
    signed_columns: list[str] = []
    for metric_column in metric_columns:
        if metric_column not in working:
            continue
        if working[metric_column].dropna().empty:
            continue
        signed_name = f"signed__{metric_column}"
        working[signed_name] = -working[metric_column] if metric_column == "bng" else working[metric_column]
        signed_columns.append(signed_name)
    if not signed_columns:
        working["naturalness_index"] = math.nan
        return working, {"frontier_auc": math.nan, "loadings": {}, "frontier_points": []}

    clean = working.dropna(subset=[utility_column, *signed_columns]).copy()
    if clean.empty:
        working["naturalness_index"] = math.nan
        return working, {"frontier_auc": math.nan, "loadings": {}, "frontier_points": []}

    scaler = StandardScaler()
    matrix = scaler.fit_transform(clean[signed_columns].to_numpy(dtype=float))
    if matrix.shape[1] == 1:
        component = matrix[:, 0]
        loadings = np.array([1.0])
    else:
        _, _, vh = np.linalg.svd(matrix, full_matrices=False)
        loadings = vh[0]
        component = matrix @ loadings
        anchor = matrix.mean(axis=1)
        correlation = float(np.corrcoef(component, anchor)[0, 1])
        if math.isfinite(correlation) and correlation < 0:
            component = -component
            loadings = -loadings

    component_min = float(component.min())
    component_max = float(component.max())
    normalized = np.ones_like(component) if component_max - component_min <= 1e-12 else (component - component_min) / (component_max - component_min)
    clean["naturalness_index"] = normalized
    merged = working.merge(clean[["bucket_id", "naturalness_index"]], on="bucket_id", how="left")
    frontier = compute_pareto_frontier(merged, utility_column=utility_column, naturalness_column="naturalness_index")
    return merged, {
        "frontier_auc": frontier["frontier_auc"],
        "loadings": {column.replace("signed__", ""): float(weight) for column, weight in zip(signed_columns, loadings, strict=True)},
        "frontier_points": frontier["frontier_points"],
    }


def summarize_nuf(
    *,
    utility_summary: dict[str, Any],
    bng_summary: dict[str, Any],
    hcr_summary: dict[str, Any],
    mauve_summary: dict[str, Any],
    tdm_summary: dict[str, Any],
) -> dict[str, Any]:
    utility_bucket_frame = pd.DataFrame(utility_summary["by_bucket"])
    naturalness_frame = utility_bucket_frame[["bucket_id", "qa_f1"]].rename(columns={"qa_f1": "utility_qa_f1"})
    bng_frame = pd.DataFrame(bng_summary["by_bucket"])
    hcr_frame = pd.DataFrame(hcr_summary["by_bucket"])
    mauve_frame = pd.DataFrame(mauve_summary["by_bucket"])
    tdm_frame = pd.DataFrame(tdm_summary["by_bucket"])
    if not bng_frame.empty:
        naturalness_frame = naturalness_frame.merge(bng_frame[["bucket_id", "bng"]], on="bucket_id", how="left")
    if not hcr_frame.empty:
        naturalness_frame = naturalness_frame.merge(hcr_frame[["bucket_id", "hcr"]], on="bucket_id", how="left")
    if not mauve_frame.empty:
        naturalness_frame = naturalness_frame.merge(mauve_frame[["bucket_id", "c_mauve"]], on="bucket_id", how="left")
    if not tdm_frame.empty:
        naturalness_frame = naturalness_frame.merge(tdm_frame[["bucket_id", "tdm"]], on="bucket_id", how="left")

    core_frame, core_frontier = build_naturalness_index(
        naturalness_frame,
        utility_column="utility_qa_f1",
        metric_columns=["bng", "hcr", "c_mauve"],
    )
    legacy_multi_turn_frame, legacy_multi_turn_frontier = build_naturalness_index(
        naturalness_frame.dropna(subset=["tdm"]),
        utility_column="utility_qa_f1",
        metric_columns=["bng", "hcr", "c_mauve", "tdm"],
    )
    overall_summary = build_global_normalized_nuf_summary(
        naturalness_frame,
        utility_column="utility_qa_f1",
        metric_columns=["bng", "c_mauve"],
    )
    multi_turn_summary = build_global_normalized_nuf_summary(
        naturalness_frame.dropna(subset=["tdm"]),
        utility_column="utility_qa_f1",
        metric_columns=["bng", "c_mauve", "tdm"],
    )
    return {
        "core": {
            "frontier_auc": core_frontier["frontier_auc"],
            "loadings": core_frontier["loadings"],
            "frontier_points": core_frontier["frontier_points"],
            "by_bucket": core_frame.to_dict(orient="records"),
        },
        "overall": overall_summary,
        "multi_turn": {
            **multi_turn_summary,
            "legacy_frontier_auc": legacy_multi_turn_frontier["frontier_auc"],
            "legacy_loadings": legacy_multi_turn_frontier["loadings"],
            "legacy_frontier_points": legacy_multi_turn_frontier["frontier_points"],
            "legacy_by_bucket": legacy_multi_turn_frame.to_dict(orient="records"),
        },
    }


def summarize_aggregate(
    *,
    utility_summary: dict[str, Any],
    bng_summary: dict[str, Any],
    mauve_summary: dict[str, Any],
    nuf_summary: dict[str, Any],
) -> dict[str, Any]:
    qa_f1 = float(utility_summary["overall"]["qa_f1"])
    raw_bng_macro = bng_summary["overall_macro"]
    bng_score = _bounded_geometric_mean(
        np.array([_normalize_global_metric_series(pd.Series([raw_bng_macro]), metric_name="bng").iloc[0]], dtype=float)
    )
    c_mauve_macro = mauve_summary.get("overall_macro", math.nan)
    if not math.isfinite(c_mauve_macro):
        c_mauve_macro = mauve_summary.get("global", math.nan)
    nuf_hypervolume = nuf_summary.get("overall", {}).get("hypervolume", math.nan)

    normalized_components = {
        "qa_f1": _bounded_geometric_mean(np.array([qa_f1], dtype=float)),
        "bng_score": bng_score,
        "c_mauve": _bounded_geometric_mean(np.array([c_mauve_macro], dtype=float)),
        "nuf_hypervolume": _bounded_geometric_mean(np.array([nuf_hypervolume], dtype=float)),
    }
    valid_values = [value for value in normalized_components.values() if math.isfinite(value)]
    overall = _bounded_geometric_mean(np.array(valid_values, dtype=float)) if len(valid_values) >= 2 else math.nan

    return {
        "overall": overall,
        "method": "geometric_mean_global_fixed_scale",
        "raw_metrics": {
            "qa_f1": qa_f1,
            "bng_macro": raw_bng_macro,
            "c_mauve": c_mauve_macro,
            "nuf_hypervolume": nuf_hypervolume,
        },
        "components": normalized_components,
        "component_count": len(valid_values),
    }
