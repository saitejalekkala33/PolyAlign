#!/usr/bin/env python3
"""Create Chinese PolyAlign embedding and linguistic-feature plots.

Outputs are written under:
    embed-plots/chinese/<model-family>/

For each model family, the script writes:
    - embedding_pca_comparisons.png/pdf
    - embedding_tsne_comparisons.png/pdf
    - embedding_umap_comparisons.png/pdf, if --reducers includes umap
    - individual/<reducer>_test_vs_*.png
    - linguistic_feature_distributions.png/pdf
    - linguistic_feature_shift_heatmap.png/pdf
    - cache/metadata.json

The Hugging Face embedding model is loaded through a temporary local model
cache and that cache is deleted at the end of the run unless --keep-model-cache
is passed. Input JSON/JSONL files and computed embeddings remain cached.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import multiprocessing as mp
import os
import re
import shutil
import string
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from tqdm.auto import tqdm


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "embed-plots" / "chinese"

HF_BASE = "https://huggingface.co/datasets/saiteja33/PolyAlign-All/resolve/main"
TEST_URL = f"{HF_BASE}/chinese/merged_sft_dedup/llamafactory/test.json"

SOURCE_ORDER = ["test", "baselm", "sft", "dpo", "bucket_sft", "hdpo"]
COMPARISONS = [
    ("baselm", "test-baselm"),
    ("sft", "test-sft"),
    ("dpo", "test-dpo"),
    ("bucket_sft", "test-bucket-sft"),
    ("hdpo", "test-hdpo"),
]
SOURCE_LABELS = {
    "test": "Golden answers",
    "baselm": "BaseLM",
    "sft": "SFT",
    "dpo": "DPO",
    "bucket_sft": "Bucket SFT",
    "hdpo": "HDPO",
}

MODEL_SPECS: dict[str, dict[str, str]] = {
    "qwen25-1-5b": {
        "title": "Qwen2.5-1.5B",
        "baselm": f"{HF_BASE}/chinese/merged_sft_dedup/runs/qwen25_1_5b_zh/predictions.jsonl",
        "sft": f"{HF_BASE}/chinese/merged_sft_dedup/runs/qwen25_1_5_sft-zh/predictions.jsonl",
        "dpo": f"{HF_BASE}/chinese/merged_sft_dedup/runs/qwen25-15b-dpo-zh/predictions.jsonl",
        "bucket_sft": f"{HF_BASE}/chinese/merged_sft_dedup/runs/qwen25_1_5b_dist_sft_zh_test/predictions.jsonl",
        "hdpo": f"{HF_BASE}/chinese/merged_sft_dedup/runs/qwen25-3b-hdpo-zh-ref-conditioned/predictions.jsonl",
    },
    "qwen25-3b": {
        "title": "Qwen2.5-3B",
        "baselm": f"{HF_BASE}/chinese/merged_sft_dedup/runs/qwen25_3b_zh/predictions.jsonl",
        "sft": f"{HF_BASE}/chinese/merged_sft_dedup/runs/qwen25_3b_sft-zh-test/predictions.jsonl",
        "dpo": f"{HF_BASE}/chinese/merged_sft_dedup/runs/qwen25-3b-dpo-zh/predictions.jsonl",
        "bucket_sft": f"{HF_BASE}/chinese/merged_sft_dedup/runs/qwen25_3b_dist-sft-zh-test/predictions.jsonl",
        "hdpo": f"{HF_BASE}/chinese/merged_sft_dedup/runs/qwen25-3b-hdpo-zh-ref-conditioned/predictions.jsonl",
    },
    "llama32-3b": {
        "title": "Llama3.2-3B",
        "baselm": f"{HF_BASE}/chinese/merged_sft_dedup/runs/llama32_3b_zh/predictions.jsonl",
        "sft": f"{HF_BASE}/chinese/merged_sft_dedup/runs/llama3_2-3b_sft-zh-test/predictions.jsonl",
        "dpo": f"{HF_BASE}/chinese/merged_sft_dedup/runs/llama32-3b-dpo-zh/predictions.jsonl",
        "bucket_sft": f"{HF_BASE}/chinese/merged_sft_dedup/runs/llama3_2-3b_sft-zh-test/predictions.jsonl",
        "hdpo": f"{HF_BASE}/chinese/merged_sft_dedup/runs/llama32-3b-hdpo-zh-ref-conditioned/predictions.jsonl",
    },
    "gemma2-2b": {
        "title": "Gemma2-2B",
        "baselm": f"{HF_BASE}/chinese/merged_sft_dedup/runs/gemma_2_2b_zh/predictions.jsonl",
        "sft": f"{HF_BASE}/chinese/merged_sft_dedup/runs/gemma2-2b-sft-zh/predictions.jsonl",
        "dpo": f"{HF_BASE}/chinese/merged_sft_dedup/runs/gemma2-2b-dpo-zh/predictions.jsonl",
        "bucket_sft": f"{HF_BASE}/chinese/merged_sft_dedup/runs/gemma2_2b_dist-sft-zh-test/predictions.jsonl",
        "hdpo": f"{HF_BASE}/chinese/merged_sft_dedup/runs/gemma2-2b-hdpo-zh-ref-conditioned/predictions.jsonl",
    },
}

SOURCE_COLORS = {
    "test": "#23395B",
    "baselm": "#D1495B",
    "sft": "#F59E0B",
    "dpo": "#7C3AED",
    "bucket_sft": "#0891B2",
    "hdpo": "#BE123C",
}
HUMAN_COLOR = SOURCE_COLORS["test"]
MODEL_COLOR = "#C026D3"
SHIFT_LINE_COLOR = "#334155"

CJK_RE = re.compile(r"[\u4e00-\u9fff]")
CONTENT_RE = re.compile(r"[\u4e00-\u9fffA-Za-z0-9]")
CHINESE_PUNCT = set("，。！？；：、（）《》“”‘’【】—…·「」『』")
ASCII_PUNCT = set(string.punctuation)
PUNCT = CHINESE_PUNCT | ASCII_PUNCT

FEATURE_SPECS = [
    ("length_chars", "Answer length"),
    ("context_overlap", "Context overlap"),
    ("char_diversity", "Character diversity"),
    ("char_entropy", "Character entropy"),
    ("chinese_ratio", "Chinese ratio"),
    ("punct_density", "Punctuation density"),
]


@dataclass(frozen=True)
class EmbeddingJob:
    key: str
    display: str
    texts: list[str]
    model_name: str
    batch_size: int
    max_length: int
    gpu_id: int
    position: int
    cache_path: str
    dtype: str
    allow_cpu: bool
    model_cache_dir: str


@dataclass
class LoadedModelData:
    texts: dict[str, list[str]]
    source_indices: dict[str, np.ndarray]
    contexts: list[str]
    instructions: list[str]
    availability: dict[str, dict[str, Any]]
    input_paths: dict[str, Path]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create Chinese PolyAlign embedding and linguistic-feature plots for all model families."
    )
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument(
        "--models",
        default="all",
        help=f"Comma-separated model keys or all. Available: {', '.join(MODEL_SPECS)}",
    )
    parser.add_argument("--embedding-model", default="BAAI/bge-m3")
    parser.add_argument(
        "--gpus",
        default="2,3,4,5,6,7",
        help="Six comma-separated GPU ids: test, BaseLM, SFT, DPO, Bucket SFT, HDPO.",
    )
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument(
        "--dtype",
        choices=("auto", "float16", "bfloat16", "float32"),
        default="auto",
    )
    parser.add_argument(
        "--reducers",
        default="pca,tsne",
        help="Comma-separated reducers from pca,tsne,umap. Use all for pca,tsne,umap.",
    )
    parser.add_argument("--tsne-perplexity", type=float, default=35.0)
    parser.add_argument(
        "--tsne-max-points",
        type=int,
        default=5000,
        help="Maximum aligned rows per pair for t-SNE. 0 uses every available row.",
    )
    parser.add_argument("--umap-neighbors", type=int, default=30)
    parser.add_argument("--umap-min-dist", type=float, default=0.08)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--arrow-sample-size", type=int, default=850)
    parser.add_argument(
        "--point-sample-size",
        type=int,
        default=0,
        help="Optional global aligned-row sample for embedding plots. 0 uses all available rows.",
    )
    parser.add_argument(
        "--embed-sample-only",
        action="store_true",
        help="Embed only --point-sample-size rows. Otherwise all rows are embedded and plotting may be sampled.",
    )
    parser.add_argument("--refresh-data", action="store_true")
    parser.add_argument("--refresh-embeddings", action="store_true")
    parser.add_argument("--allow-cpu", action="store_true")
    parser.add_argument(
        "--keep-model-cache",
        action="store_true",
        help="Do not delete the temporary Hugging Face model cache after the run.",
    )
    return parser.parse_args()


def parse_models(raw: str) -> list[str]:
    if raw.strip().lower() == "all":
        return list(MODEL_SPECS)
    models = [part.strip() for part in raw.split(",") if part.strip()]
    unknown = [model for model in models if model not in MODEL_SPECS]
    if unknown:
        raise ValueError(f"Unknown model keys: {unknown}. Available: {list(MODEL_SPECS)}")
    return models


def parse_reducers(raw: str) -> list[str]:
    if raw.strip().lower() == "all":
        reducers = ["pca", "tsne", "umap"]
    else:
        reducers = [part.strip().lower() for part in raw.split(",") if part.strip()]
    unknown = [name for name in reducers if name not in {"pca", "tsne", "umap"}]
    if unknown:
        raise ValueError(f"Unknown reducers: {unknown}. Use pca, tsne, umap, or all.")
    return reducers


def parse_gpus(raw: str) -> list[int]:
    try:
        gpus = [int(part.strip()) for part in raw.split(",") if part.strip()]
    except ValueError as exc:
        raise ValueError(f"Could not parse --gpus={raw!r}") from exc
    if len(gpus) != len(SOURCE_ORDER):
        raise ValueError(f"--gpus needs {len(SOURCE_ORDER)} ids, got {len(gpus)}: {gpus}")
    return gpus


def slugify(value: str) -> str:
    value = value.lower()
    value = re.sub(r"[^a-z0-9._-]+", "-", value)
    return value.strip("-")


def sha1_texts(texts: list[str], extra: str) -> str:
    h = hashlib.sha1()
    h.update(extra.encode("utf-8"))
    h.update(str(len(texts)).encode("ascii"))
    for text in texts:
        h.update(b"\0")
        h.update(text.encode("utf-8", errors="replace"))
    return h.hexdigest()[:16]


def download_file(url: str, dest: Path, refresh: bool = False) -> Path:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and dest.stat().st_size > 0 and not refresh:
        return dest

    tmp = dest.with_suffix(dest.suffix + ".tmp")
    request = urllib.request.Request(url, headers={"User-Agent": "polyalign-embedding-plot/2.0"})
    with urllib.request.urlopen(request) as response:
        total = int(response.headers.get("Content-Length") or 0)
        with tmp.open("wb") as f, tqdm(
            total=total,
            unit="B",
            unit_scale=True,
            desc=f"download {dest.name}",
            leave=True,
        ) as pbar:
            while True:
                chunk = response.read(1024 * 1024)
                if not chunk:
                    break
                f.write(chunk)
                pbar.update(len(chunk))
    tmp.replace(dest)
    return dest


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSONL in {path} at line {line_no}") from exc
            if not isinstance(row, dict):
                raise ValueError(f"Expected object in {path} at line {line_no}")
            rows.append(row)
    return rows


def as_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value.strip()
    return str(value).strip()


def extract_test_records(rows: Any) -> tuple[list[str], list[str], list[str]]:
    if not isinstance(rows, list):
        raise ValueError("test.json must be a JSON list.")
    outputs: list[str] = []
    contexts: list[str] = []
    instructions: list[str] = []
    for idx, row in enumerate(rows):
        if not isinstance(row, dict):
            raise ValueError(f"test.json row {idx} is not an object.")
        outputs.append(as_text(row.get("output", row.get("reference_output", ""))))
        contexts.append(as_text(row.get("input", "")))
        instructions.append(as_text(row.get("instruction", "")))
    return outputs, contexts, instructions


def extract_predictions(
    rows: list[dict[str, Any]], n_expected: int, source_name: str
) -> tuple[list[str], list[int], dict[str, Any]]:
    by_index: dict[int, str] = {}
    extra_indices: list[int] = []
    has_source_index = any("source_index" in row for row in rows)

    for line_idx, row in enumerate(rows):
        idx_value = row.get("source_index", line_idx) if has_source_index else line_idx
        try:
            source_idx = int(idx_value)
        except (TypeError, ValueError):
            raise ValueError(f"{source_name}: bad source_index={idx_value!r} at JSONL row {line_idx}")

        prediction = row.get(
            "prediction",
            row.get("generated_text", row.get("response", row.get("output", ""))),
        )
        if 0 <= source_idx < n_expected:
            by_index[source_idx] = as_text(prediction)
        else:
            extra_indices.append(source_idx)

    missing = sorted(set(range(n_expected)) - set(by_index))
    extra = sorted(extra_indices)
    if missing:
        preview = ", ".join(map(str, missing[:10]))
        print(f"{source_name}: missing {len(missing)} prediction rows; first missing: {preview}")
    if extra:
        preview = ", ".join(map(str, extra[:10]))
        print(f"{source_name}: ignored {len(extra)} out-of-range source_index values: {preview}")

    indices = sorted(by_index)
    stats = {
        "available": len(indices),
        "missing": len(missing),
        "extra": len(extra),
        "first_missing": missing[:20],
        "first_extra": extra[:20],
    }
    return [by_index[i] for i in indices], indices, stats


def load_model_data(model_key: str, out_dir: Path, refresh_data: bool) -> LoadedModelData:
    spec = MODEL_SPECS[model_key]
    input_dir = out_dir / "cache" / "inputs"
    input_dir.mkdir(parents=True, exist_ok=True)

    paths: dict[str, Path] = {}
    paths["test"] = download_file(TEST_URL, input_dir / "test.json", refresh=refresh_data)
    for key in SOURCE_ORDER:
        if key == "test":
            continue
        paths[key] = download_file(spec[key], input_dir / f"{key}.jsonl", refresh=refresh_data)

    test_outputs, contexts, instructions = extract_test_records(load_json(paths["test"]))
    n_test = len(test_outputs)
    texts: dict[str, list[str]] = {"test": test_outputs}
    source_indices: dict[str, np.ndarray] = {"test": np.arange(n_test, dtype=np.int64)}
    availability: dict[str, dict[str, Any]] = {
        "test": {
            "available": n_test,
            "missing": 0,
            "extra": 0,
            "first_missing": [],
            "first_extra": [],
        }
    }

    for key in SOURCE_ORDER:
        if key == "test":
            continue
        rows = load_jsonl(paths[key])
        pred_texts, pred_indices, stats = extract_predictions(rows, n_test, f"{model_key}/{SOURCE_LABELS[key]}")
        texts[key] = pred_texts
        source_indices[key] = np.array(pred_indices, dtype=np.int64)
        availability[key] = stats

    return LoadedModelData(
        texts=texts,
        source_indices=source_indices,
        contexts=contexts,
        instructions=instructions,
        availability=availability,
        input_paths=paths,
    )


def visible_cuda_index(requested_gpu: int) -> int:
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "").strip()
    if not visible:
        return requested_gpu

    tokens = [token.strip() for token in visible.split(",") if token.strip()]
    if str(requested_gpu) in tokens:
        return tokens.index(str(requested_gpu))
    return requested_gpu


def torch_dtype(dtype_name: str, using_cuda: bool) -> Any:
    import torch

    if dtype_name == "auto":
        return torch.float16 if using_cuda else torch.float32
    return {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }[dtype_name]


def embed_worker(job: EmbeddingJob) -> tuple[str, str]:
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    os.environ.setdefault("HF_HOME", job.model_cache_dir)
    os.environ.setdefault("HF_HUB_CACHE", str(Path(job.model_cache_dir) / "hub"))
    os.environ.setdefault("TRANSFORMERS_CACHE", str(Path(job.model_cache_dir) / "transformers"))

    import torch
    import torch.nn.functional as F
    from transformers import AutoModel, AutoTokenizer

    cache_path = Path(job.cache_path)
    if cache_path.exists() and cache_path.stat().st_size > 0:
        return job.key, str(cache_path)

    using_cuda = torch.cuda.is_available()
    if not using_cuda and not job.allow_cpu:
        raise RuntimeError(
            f"CUDA is not available for {job.display}. Pass --allow-cpu only for a slow smoke test."
        )

    if using_cuda:
        device_ordinal = visible_cuda_index(job.gpu_id)
        if device_ordinal >= torch.cuda.device_count():
            visible = os.environ.get("CUDA_VISIBLE_DEVICES", "<not set>")
            raise RuntimeError(
                f"GPU {job.gpu_id} resolved to cuda:{device_ordinal}, but only "
                f"{torch.cuda.device_count()} CUDA devices are visible. CUDA_VISIBLE_DEVICES={visible}"
            )
        device = torch.device(f"cuda:{device_ordinal}")
        torch.cuda.set_device(device)
    else:
        device = torch.device("cpu")

    tokenizer = AutoTokenizer.from_pretrained(
        job.model_name,
        use_fast=True,
        cache_dir=job.model_cache_dir,
    )
    model = AutoModel.from_pretrained(
        job.model_name,
        torch_dtype=torch_dtype(job.dtype, using_cuda),
        cache_dir=job.model_cache_dir,
    )
    model.to(device)
    model.eval()

    chunks: list[np.ndarray] = []
    iterator = range(0, len(job.texts), job.batch_size)
    desc = f"{job.key} on gpu {job.gpu_id}" if using_cuda else f"{job.key} on cpu"
    for start in tqdm(iterator, desc=desc, position=job.position, leave=True):
        batch_texts = job.texts[start : start + job.batch_size]
        encoded = tokenizer(
            batch_texts,
            padding=True,
            truncation=True,
            max_length=job.max_length,
            return_tensors="pt",
        )
        encoded = {key: value.to(device) for key, value in encoded.items()}
        with torch.inference_mode():
            output = model(**encoded)
            token_embeddings = output.last_hidden_state
            mask = encoded["attention_mask"].unsqueeze(-1).to(token_embeddings.dtype)
            pooled = (token_embeddings * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-6)
            pooled = F.normalize(pooled, p=2, dim=1)
        chunks.append(pooled.float().cpu().numpy())

    embeddings = np.concatenate(chunks, axis=0).astype(np.float32)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(cache_path, embeddings)
    return job.key, str(cache_path)


def pca_2d(matrix: np.ndarray) -> np.ndarray:
    matrix = matrix.astype(np.float32, copy=False)
    centered = matrix - matrix.mean(axis=0, keepdims=True)
    _, _, vt = np.linalg.svd(centered, full_matrices=False)
    return centered @ vt[:2].T


def reduce_2d(matrix: np.ndarray, reducer_name: str, args: argparse.Namespace) -> np.ndarray:
    if reducer_name == "pca":
        return pca_2d(matrix)

    if reducer_name == "tsne":
        try:
            from sklearn.manifold import TSNE
        except ImportError as exc:
            raise RuntimeError("t-SNE requires scikit-learn. Install it with: pip install scikit-learn") from exc

        n_rows = matrix.shape[0]
        perplexity = min(args.tsne_perplexity, max(5.0, (n_rows - 1) / 3.0))
        reducer = TSNE(
            n_components=2,
            perplexity=perplexity,
            metric="cosine",
            init="pca",
            learning_rate="auto",
            random_state=args.seed,
            verbose=0,
        )
        return reducer.fit_transform(matrix).astype(np.float32)

    if reducer_name == "umap":
        try:
            from umap import UMAP
        except ImportError as exc:
            raise RuntimeError("UMAP requires umap-learn. Install it with: pip install umap-learn") from exc

        reducer = UMAP(
            n_components=2,
            n_neighbors=args.umap_neighbors,
            min_dist=args.umap_min_dist,
            metric="cosine",
            random_state=args.seed,
        )
        return reducer.fit_transform(matrix).astype(np.float32)

    raise ValueError(f"Unknown reducer: {reducer_name}")


def index_positions(indices: np.ndarray) -> dict[int, int]:
    return {int(source_idx): row_idx for row_idx, source_idx in enumerate(indices.tolist())}


def choose_pair_indices(
    model_indices: np.ndarray,
    selected_set: set[int],
    reducer_name: str,
    args: argparse.Namespace,
    rng: np.random.Generator,
) -> np.ndarray:
    pair_indices = np.array([int(idx) for idx in model_indices.tolist() if int(idx) in selected_set], dtype=np.int64)
    if reducer_name == "tsne" and args.tsne_max_points and len(pair_indices) > args.tsne_max_points:
        pair_indices = np.sort(rng.choice(pair_indices, size=args.tsne_max_points, replace=False))
    return pair_indices


def symmetric_limits(*arrays: np.ndarray, padding: float = 0.08) -> tuple[tuple[float, float], tuple[float, float]]:
    stacked = np.vstack(arrays)
    xmin, ymin = stacked.min(axis=0)
    xmax, ymax = stacked.max(axis=0)
    xpad = max((xmax - xmin) * padding, 1e-3)
    ypad = max((ymax - ymin) * padding, 1e-3)
    return (float(xmin - xpad), float(xmax + xpad)), (float(ymin - ypad), float(ymax + ypad))


def set_plot_style() -> None:
    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {
            "figure.dpi": 135,
            "savefig.dpi": 340,
            "font.family": "DejaVu Sans",
            "axes.titlesize": 12,
            "axes.labelsize": 9,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "legend.fontsize": 9,
            "axes.facecolor": "#FBFCFE",
            "figure.facecolor": "white",
            "axes.edgecolor": "#475569",
            "axes.linewidth": 0.8,
        }
    )


def plot_pair(
    ax: Any,
    human_2d: np.ndarray,
    model_2d: np.ndarray,
    title: str,
    model_label: str,
    arrow_indices: np.ndarray,
    axis_label: str,
    show_legend: bool = False,
) -> None:
    for idx in arrow_indices:
        ax.plot(
            [human_2d[idx, 0], model_2d[idx, 0]],
            [human_2d[idx, 1], model_2d[idx, 1]],
            color=SHIFT_LINE_COLOR,
            alpha=0.045,
            linewidth=0.55,
            zorder=1,
        )

    ax.scatter(
        human_2d[:, 0],
        human_2d[:, 1],
        s=20,
        c=HUMAN_COLOR,
        edgecolors="#0F172A",
        linewidths=0.28,
        alpha=0.72,
        label="Golden answers",
        zorder=3,
    )
    ax.scatter(
        model_2d[:, 0],
        model_2d[:, 1],
        s=17,
        c=MODEL_COLOR,
        marker="X",
        edgecolors="#2E1065",
        linewidths=0.22,
        alpha=0.54,
        label=model_label,
        zorder=2,
    )

    h_centroid = human_2d.mean(axis=0)
    m_centroid = model_2d.mean(axis=0)
    ax.scatter(
        [h_centroid[0]],
        [h_centroid[1]],
        s=185,
        marker="*",
        c="#2563EB",
        edgecolors="white",
        linewidths=0.85,
        label="Golden centroid",
        zorder=5,
    )
    ax.scatter(
        [m_centroid[0]],
        [m_centroid[1]],
        s=185,
        marker="*",
        c="#F97316",
        edgecolors="#431407",
        linewidths=0.65,
        label="Model centroid",
        zorder=5,
    )

    ax.axhline(0, color="#CBD5E1", linewidth=0.85, alpha=0.75, zorder=0)
    ax.axvline(0, color="#CBD5E1", linewidth=0.85, alpha=0.75, zorder=0)
    ax.grid(True, linestyle="--", linewidth=0.55, color="#D7DEE8", alpha=0.78)
    ax.set_title(title, fontweight="medium", pad=8)
    ax.set_xlabel(f"{axis_label} 1")
    ax.set_ylabel(f"{axis_label} 2")
    xlim, ylim = symmetric_limits(human_2d, model_2d)
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    for spine in ax.spines.values():
        spine.set_color("#475569")
        spine.set_linewidth(0.82)
    if show_legend:
        ax.legend(loc="best", frameon=True, framealpha=0.96, fancybox=False, edgecolor="#CBD5E1")


def axis_label_for_reducer(reducer_name: str) -> str:
    return {
        "pca": "Principal component",
        "tsne": "t-SNE component",
        "umap": "UMAP component",
    }[reducer_name]


def write_embedding_plots(
    model_key: str,
    reducer_name: str,
    embeddings: dict[str, np.ndarray],
    source_indices: dict[str, np.ndarray],
    selected_indices: np.ndarray,
    out_dir: Path,
    args: argparse.Namespace,
) -> list[str]:
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    set_plot_style()
    rng = np.random.default_rng(args.seed + sum(ord(ch) for ch in model_key + reducer_name))
    selected_set = set(int(idx) for idx in selected_indices.tolist())
    test_pos = index_positions(source_indices["test"])
    written: list[str] = []

    reduced_pairs: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
    for source_key, _title in tqdm(COMPARISONS, desc=f"{model_key} {reducer_name} projection"):
        model_pos = index_positions(source_indices[source_key])
        pair_source_indices = choose_pair_indices(source_indices[source_key], selected_set, reducer_name, args, rng)
        pair_source_indices = np.array(
            [int(idx) for idx in pair_source_indices.tolist() if int(idx) in test_pos and int(idx) in model_pos],
            dtype=np.int64,
        )
        if len(pair_source_indices) == 0:
            raise ValueError(f"{model_key}/{SOURCE_LABELS[source_key]} has no available rows for plotting.")

        human_rows = np.array([test_pos[int(idx)] for idx in pair_source_indices], dtype=np.int64)
        model_rows = np.array([model_pos[int(idx)] for idx in pair_source_indices], dtype=np.int64)
        joint = np.vstack([embeddings["test"][human_rows], embeddings[source_key][model_rows]])
        coords = reduce_2d(joint, reducer_name, args)
        n = len(pair_source_indices)
        reduced_pairs[source_key] = (coords[:n], coords[n:], pair_source_indices)

    fig, axes = plt.subplots(2, 3, figsize=(18.5, 10.1), constrained_layout=False)
    axes_flat = axes.ravel()
    legend_handles = [
        Line2D([0], [0], color=SHIFT_LINE_COLOR, alpha=0.30, linewidth=1.3, label="Sample shift"),
        Line2D(
            [0],
            [0],
            marker="o",
            linestyle="",
            markersize=7,
            markerfacecolor=HUMAN_COLOR,
            markeredgecolor="#0F172A",
            label="Golden answers",
        ),
        Line2D(
            [0],
            [0],
            marker="X",
            linestyle="",
            markersize=7,
            markerfacecolor=MODEL_COLOR,
            markeredgecolor="#2E1065",
            label="Model predictions",
        ),
        Line2D(
            [0],
            [0],
            marker="*",
            linestyle="",
            markersize=11,
            markerfacecolor="#2563EB",
            markeredgecolor="white",
            label="Golden centroid",
        ),
        Line2D(
            [0],
            [0],
            marker="*",
            linestyle="",
            markersize=11,
            markerfacecolor="#F97316",
            markeredgecolor="#431407",
            label="Model centroid",
        ),
    ]
    fig.legend(
        handles=legend_handles,
        loc="upper center",
        ncol=5,
        bbox_to_anchor=(0.5, 0.985),
        frameon=True,
        fancybox=False,
        edgecolor="#CBD5E1",
    )
    fig.suptitle(
        f"{MODEL_SPECS[model_key]['title']} Chinese Answer Embedding Shifts ({reducer_name.upper()})",
        y=0.998,
        fontsize=16,
        fontweight="medium",
    )

    axis_label = axis_label_for_reducer(reducer_name)
    for ax, (source_key, title) in zip(axes_flat, COMPARISONS):
        human_2d, model_2d, pair_source_indices = reduced_pairs[source_key]
        max_arrows = min(args.arrow_sample_size, len(pair_source_indices))
        arrow_indices = (
            np.sort(rng.choice(np.arange(len(pair_source_indices)), size=max_arrows, replace=False))
            if max_arrows > 0
            else np.array([], dtype=int)
        )
        plot_pair(
            ax,
            human_2d,
            model_2d,
            title=title,
            model_label=SOURCE_LABELS[source_key],
            arrow_indices=arrow_indices,
            axis_label=axis_label,
            show_legend=False,
        )

    for ax in axes_flat[len(COMPARISONS) :]:
        ax.axis("off")

    fig.subplots_adjust(top=0.90, left=0.055, right=0.985, bottom=0.075, wspace=0.24, hspace=0.30)
    png_path = out_dir / f"embedding_{reducer_name}_comparisons.png"
    pdf_path = out_dir / f"embedding_{reducer_name}_comparisons.pdf"
    fig.savefig(png_path, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)
    written.extend([str(png_path), str(pdf_path)])

    individual_dir = out_dir / "individual"
    individual_dir.mkdir(parents=True, exist_ok=True)
    for source_key, title in COMPARISONS:
        fig_one, ax_one = plt.subplots(1, 1, figsize=(8.5, 6.45), constrained_layout=True)
        human_2d, model_2d, pair_source_indices = reduced_pairs[source_key]
        max_arrows = min(args.arrow_sample_size, len(pair_source_indices))
        arrow_indices = (
            np.sort(rng.choice(np.arange(len(pair_source_indices)), size=max_arrows, replace=False))
            if max_arrows > 0
            else np.array([], dtype=int)
        )
        plot_pair(
            ax_one,
            human_2d,
            model_2d,
            title=title,
            model_label=SOURCE_LABELS[source_key],
            arrow_indices=arrow_indices,
            axis_label=axis_label,
            show_legend=True,
        )
        fig_one.suptitle(
            f"{MODEL_SPECS[model_key]['title']} Chinese Answer Shift ({reducer_name.upper()})",
            fontsize=13.5,
            fontweight="medium",
        )
        out_path = individual_dir / f"{reducer_name}_{title.replace('test-', 'test_vs_').replace('-', '_')}.png"
        fig_one.savefig(out_path, bbox_inches="tight")
        plt.close(fig_one)
        written.append(str(out_path))

    return written


def content_chars(text: str) -> list[str]:
    return [match.group(0).lower() for match in CONTENT_RE.finditer(text)]


def char_entropy(chars: list[str]) -> float:
    if not chars:
        return 0.0
    counts: dict[str, int] = {}
    for char in chars:
        counts[char] = counts.get(char, 0) + 1
    total = float(len(chars))
    return float(-sum((count / total) * math.log2(count / total) for count in counts.values()))


def linguistic_features_for_text(text: str, context: str) -> dict[str, float]:
    raw_chars = [char for char in text if not char.isspace()]
    n_chars = len(raw_chars)
    cjk_count = sum(1 for char in raw_chars if CJK_RE.match(char))
    punct_count = sum(1 for char in raw_chars if char in PUNCT)
    content = content_chars(text)
    context_content = set(content_chars(context))
    content_len = len(content)
    context_overlap = (
        sum(1 for char in content if char in context_content) / content_len if content_len else 0.0
    )
    return {
        "length_chars": float(n_chars),
        "context_overlap": float(context_overlap),
        "char_diversity": float(len(set(content)) / content_len) if content_len else 0.0,
        "char_entropy": char_entropy(content),
        "chinese_ratio": float(cjk_count / n_chars) if n_chars else 0.0,
        "punct_density": float(punct_count / n_chars) if n_chars else 0.0,
    }


def compute_feature_arrays(data: LoadedModelData) -> dict[str, dict[str, np.ndarray]]:
    features: dict[str, dict[str, np.ndarray]] = {}
    for key in SOURCE_ORDER:
        rows: list[dict[str, float]] = []
        for text, source_idx in zip(data.texts[key], data.source_indices[key].tolist()):
            context = data.contexts[int(source_idx)] if 0 <= int(source_idx) < len(data.contexts) else ""
            rows.append(linguistic_features_for_text(text, context))
        features[key] = {
            feature_key: np.array([row[feature_key] for row in rows], dtype=np.float32)
            for feature_key, _label in FEATURE_SPECS
        }
    return features


def smooth_hist_curve(values: np.ndarray, x_min: float, x_max: float, bins: int = 170) -> tuple[np.ndarray, np.ndarray]:
    values = values[np.isfinite(values)]
    if len(values) == 0:
        x = np.linspace(x_min, x_max, bins)
        return x, np.zeros_like(x)
    values = np.clip(values, x_min, x_max)
    hist, edges = np.histogram(values, bins=bins, range=(x_min, x_max), density=True)
    centers = (edges[:-1] + edges[1:]) / 2.0
    kernel_x = np.linspace(-2.5, 2.5, 19)
    kernel = np.exp(-0.5 * kernel_x**2)
    kernel /= kernel.sum()
    smooth = np.convolve(hist, kernel, mode="same")
    return centers, smooth


def robust_range(arrays: list[np.ndarray], feature_key: str) -> tuple[float, float]:
    values = np.concatenate([array[np.isfinite(array)] for array in arrays if len(array)])
    if len(values) == 0:
        return 0.0, 1.0
    if feature_key in {"context_overlap", "char_diversity", "chinese_ratio", "punct_density"}:
        lo, hi = 0.0, min(1.0, max(float(np.percentile(values, 99.5)), 0.01))
        if feature_key != "punct_density":
            hi = 1.0
        return lo, hi
    lo = max(0.0, float(np.percentile(values, 0.5)))
    hi = float(np.percentile(values, 99.2))
    if hi <= lo:
        hi = lo + 1.0
    return lo, hi


def write_linguistic_plots(model_key: str, features: dict[str, dict[str, np.ndarray]], out_dir: Path) -> list[str]:
    import matplotlib.pyplot as plt

    set_plot_style()
    written: list[str] = []

    fig, axes = plt.subplots(2, 3, figsize=(18.4, 9.6), constrained_layout=False)
    axes_flat = axes.ravel()
    for ax, (feature_key, label) in zip(axes_flat, FEATURE_SPECS):
        arrays = [features[source_key][feature_key] for source_key in SOURCE_ORDER]
        x_min, x_max = robust_range(arrays, feature_key)
        for source_key in SOURCE_ORDER:
            x, y = smooth_hist_curve(features[source_key][feature_key], x_min, x_max)
            linewidth = 2.4 if source_key == "test" else 1.75
            alpha = 0.96 if source_key == "test" else 0.86
            ax.plot(
                x,
                y,
                color=SOURCE_COLORS[source_key],
                linewidth=linewidth,
                alpha=alpha,
                label=SOURCE_LABELS[source_key],
            )
        ax.set_title(label, fontweight="medium", pad=8)
        ax.set_xlabel(label)
        ax.set_ylabel("Density")
        ax.grid(True, linestyle="--", linewidth=0.55, color="#D7DEE8", alpha=0.78)
        for spine in ax.spines.values():
            spine.set_color("#475569")
            spine.set_linewidth(0.82)

    handles, labels = axes_flat[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        ncol=6,
        bbox_to_anchor=(0.5, 0.986),
        frameon=True,
        fancybox=False,
        edgecolor="#CBD5E1",
    )
    fig.suptitle(
        f"{MODEL_SPECS[model_key]['title']} Linguistic Feature Distributions",
        y=0.998,
        fontsize=16,
        fontweight="medium",
    )
    fig.subplots_adjust(top=0.88, left=0.055, right=0.985, bottom=0.075, wspace=0.24, hspace=0.34)
    png_path = out_dir / "linguistic_feature_distributions.png"
    pdf_path = out_dir / "linguistic_feature_distributions.pdf"
    fig.savefig(png_path, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)
    written.extend([str(png_path), str(pdf_path)])

    heat = np.zeros((len(FEATURE_SPECS), len(SOURCE_ORDER) - 1), dtype=np.float32)
    golden = features["test"]
    for row_idx, (feature_key, _label) in enumerate(FEATURE_SPECS):
        base = golden[feature_key]
        base_std = float(np.nanstd(base)) or 1.0
        base_mean = float(np.nanmean(base)) if len(base) else 0.0
        for col_idx, source_key in enumerate(SOURCE_ORDER[1:]):
            values = features[source_key][feature_key]
            model_mean = float(np.nanmean(values)) if len(values) else 0.0
            heat[row_idx, col_idx] = (model_mean - base_mean) / base_std

    fig_h, ax_h = plt.subplots(1, 1, figsize=(9.6, 5.6), constrained_layout=True)
    clipped = np.clip(heat, -2.5, 2.5)
    image = ax_h.imshow(clipped, cmap="coolwarm", vmin=-2.5, vmax=2.5, aspect="auto")
    ax_h.set_xticks(np.arange(len(SOURCE_ORDER) - 1))
    ax_h.set_xticklabels([SOURCE_LABELS[key] for key in SOURCE_ORDER[1:]], rotation=20, ha="right")
    ax_h.set_yticks(np.arange(len(FEATURE_SPECS)))
    ax_h.set_yticklabels([label for _key, label in FEATURE_SPECS])
    ax_h.set_title(
        f"{MODEL_SPECS[model_key]['title']} Mean Feature Shift vs Golden",
        fontweight="medium",
        pad=12,
    )
    for row_idx in range(heat.shape[0]):
        for col_idx in range(heat.shape[1]):
            value = heat[row_idx, col_idx]
            color = "white" if abs(clipped[row_idx, col_idx]) > 1.35 else "#111827"
            ax_h.text(col_idx, row_idx, f"{value:+.2f}", ha="center", va="center", color=color, fontsize=8.5)
    cbar = fig_h.colorbar(image, ax=ax_h, fraction=0.046, pad=0.04)
    cbar.set_label("Mean shift in golden standard deviations")
    heat_png = out_dir / "linguistic_feature_shift_heatmap.png"
    heat_pdf = out_dir / "linguistic_feature_shift_heatmap.pdf"
    fig_h.savefig(heat_png, bbox_inches="tight")
    fig_h.savefig(heat_pdf, bbox_inches="tight")
    plt.close(fig_h)
    written.extend([str(heat_png), str(heat_pdf)])

    return written


def selected_source_indices(total_rows: int, args: argparse.Namespace) -> np.ndarray:
    rng = np.random.default_rng(args.seed)
    if args.point_sample_size and args.point_sample_size < total_rows:
        return np.sort(rng.choice(np.arange(total_rows), size=args.point_sample_size, replace=False))
    return np.arange(total_rows)


def maybe_sample_for_embedding(data: LoadedModelData, selected_indices: np.ndarray) -> LoadedModelData:
    selected_set = set(int(idx) for idx in selected_indices.tolist())
    sampled_texts: dict[str, list[str]] = {}
    sampled_indices: dict[str, np.ndarray] = {}
    for key, texts in data.texts.items():
        keep_rows = [
            row_idx
            for row_idx, source_idx in enumerate(data.source_indices[key].tolist())
            if int(source_idx) in selected_set
        ]
        sampled_texts[key] = [texts[row_idx] for row_idx in keep_rows]
        sampled_indices[key] = data.source_indices[key][keep_rows]
    return LoadedModelData(
        texts=sampled_texts,
        source_indices=sampled_indices,
        contexts=data.contexts,
        instructions=data.instructions,
        availability=data.availability,
        input_paths=data.input_paths,
    )


def compute_embeddings_for_model(
    model_key: str,
    data: LoadedModelData,
    out_dir: Path,
    model_cache_dir: Path,
    args: argparse.Namespace,
) -> dict[str, Path]:
    embed_dir = out_dir / "cache" / "embeddings"
    embed_dir.mkdir(parents=True, exist_ok=True)

    gpus = parse_gpus(args.gpus)
    model_slug = slugify(args.embedding_model)
    jobs: list[EmbeddingJob] = []
    embedding_paths: dict[str, Path] = {}
    for position, (source_key, gpu_id) in enumerate(zip(SOURCE_ORDER, gpus)):
        fingerprint = sha1_texts(
            data.texts[source_key],
            extra=f"{args.embedding_model}|{args.max_length}|{model_key}|{source_key}",
        )
        cache_path = embed_dir / f"{source_key}_{model_slug}_{len(data.texts[source_key])}_{fingerprint}.npy"
        if args.refresh_embeddings and cache_path.exists():
            cache_path.unlink()
        embedding_paths[source_key] = cache_path
        jobs.append(
            EmbeddingJob(
                key=source_key,
                display=f"{model_key}/{SOURCE_LABELS[source_key]}",
                texts=data.texts[source_key],
                model_name=args.embedding_model,
                batch_size=args.batch_size,
                max_length=args.max_length,
                gpu_id=gpu_id,
                position=position,
                cache_path=str(cache_path),
                dtype=args.dtype,
                allow_cpu=args.allow_cpu,
                model_cache_dir=str(model_cache_dir),
            )
        )

    pending_jobs = [job for job in jobs if not Path(job.cache_path).exists()]
    if pending_jobs:
        print(f"\nEmbedding groups for {model_key}:")
        for job in jobs:
            status = "cached" if Path(job.cache_path).exists() else f"gpu {job.gpu_id}"
            print(f"  {job.key:10s} -> {status}")
        ctx = mp.get_context("spawn")
        with ctx.Pool(processes=len(pending_jobs)) as pool:
            for key, path in pool.imap_unordered(embed_worker, pending_jobs):
                embedding_paths[key] = Path(path)
    else:
        print(f"\n{model_key}: all embeddings are cached; skipping GPU embedding.")

    return embedding_paths


def load_embedding_arrays(paths: dict[str, Path], data: LoadedModelData) -> dict[str, np.ndarray]:
    embeddings = {key: np.load(path) for key, path in paths.items()}
    for key, matrix in embeddings.items():
        if matrix.ndim != 2:
            raise ValueError(f"{key}: expected 2D embedding matrix, found shape {matrix.shape}")
        if matrix.shape[0] != len(data.texts[key]):
            raise ValueError(f"{key}: expected {len(data.texts[key])} embeddings, found {matrix.shape[0]}")
    return embeddings


def write_metadata(
    model_key: str,
    out_dir: Path,
    data: LoadedModelData,
    selected_indices: np.ndarray,
    reducers: list[str],
    embedding_paths: dict[str, Path],
    written: list[str],
    args: argparse.Namespace,
) -> Path:
    metadata = {
        "model_key": model_key,
        "model_title": MODEL_SPECS[model_key]["title"],
        "n_rows": len(data.contexts),
        "selected_rows": int(len(selected_indices)),
        "embedded_rows": {key: len(value) for key, value in data.texts.items()},
        "availability": data.availability,
        "embedding_model": args.embedding_model,
        "reducers": reducers,
        "gpus": {source_key: gpu for source_key, gpu in zip(SOURCE_ORDER, parse_gpus(args.gpus))},
        "sources": {"test": TEST_URL, **{key: MODEL_SPECS[model_key][key] for key in SOURCE_ORDER if key != "test"}},
        "input_paths": {key: str(path) for key, path in data.input_paths.items()},
        "source_index_ranges": {
            key: {
                "min": int(values.min()) if len(values) else None,
                "max": int(values.max()) if len(values) else None,
                "count": int(len(values)),
            }
            for key, values in data.source_indices.items()
        },
        "embedding_paths": {key: str(path) for key, path in embedding_paths.items()},
        "written": written,
    }
    metadata_path = out_dir / "cache" / "metadata.json"
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    return metadata_path


def process_model(
    model_key: str,
    reducers: list[str],
    model_cache_dir: Path,
    args: argparse.Namespace,
) -> list[str]:
    out_dir = args.output_root / model_key
    out_dir.mkdir(parents=True, exist_ok=True)
    data = load_model_data(model_key, out_dir, args.refresh_data)
    total_rows = len(data.contexts)
    selected_indices = selected_source_indices(total_rows, args)

    print(f"\nAvailable aligned rows for {model_key}:")
    for key in SOURCE_ORDER:
        stats = data.availability[key]
        print(f"  {key:10s}: {stats['available']:5d}/{total_rows} available, {stats['missing']:5d} missing")

    embed_data = maybe_sample_for_embedding(data, selected_indices) if args.embed_sample_only else data
    embedding_paths = compute_embeddings_for_model(model_key, embed_data, out_dir, model_cache_dir, args)
    embeddings = load_embedding_arrays(embedding_paths, embed_data)

    written: list[str] = []
    for reducer_name in reducers:
        written.extend(
            write_embedding_plots(
                model_key=model_key,
                reducer_name=reducer_name,
                embeddings=embeddings,
                source_indices=embed_data.source_indices,
                selected_indices=selected_indices,
                out_dir=out_dir,
                args=args,
            )
        )

    features = compute_feature_arrays(data)
    written.extend(write_linguistic_plots(model_key, features, out_dir))
    metadata_path = write_metadata(model_key, out_dir, data, selected_indices, reducers, embedding_paths, written, args)
    written.append(str(metadata_path))
    return written


def remove_tree_safely(path: Path, allowed_parent: Path) -> None:
    resolved = path.resolve()
    parent = allowed_parent.resolve()
    if not resolved.exists():
        return
    if resolved == parent or parent not in resolved.parents:
        raise RuntimeError(f"Refusing to delete unexpected path: {resolved}")
    shutil.rmtree(resolved)


def main() -> None:
    args = parse_args()
    args.output_root = args.output_root.resolve()
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

    models = parse_models(args.models)
    reducers = parse_reducers(args.reducers)
    args.output_root.mkdir(parents=True, exist_ok=True)

    model_cache_base = args.output_root / "_tmp_model_cache"
    model_cache_dir = model_cache_base / f"{slugify(args.embedding_model)}_{os.getpid()}"
    model_cache_dir.mkdir(parents=True, exist_ok=True)

    all_written: list[str] = []
    try:
        for model_key in models:
            all_written.extend(process_model(model_key, reducers, model_cache_dir, args))
    finally:
        if args.keep_model_cache:
            print(f"\nKeeping temporary model cache: {model_cache_dir}")
        else:
            remove_tree_safely(model_cache_dir, model_cache_base)
            print(f"\nDeleted temporary model cache: {model_cache_dir}")

    print("\nWrote:")
    for path in all_written:
        print(f"  {path}")


if __name__ == "__main__":
    mp.freeze_support()
    main()
