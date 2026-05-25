#!/usr/bin/env python3
"""Embed Chinese human answers and model predictions, then plot answer shifts.

Default output directory:
    embed-plots/chinese/qwen25-1-5b

The script downloads the PolyAlign files from Hugging Face, embeds six aligned
text groups on GPUs 2-7 by default, caches embeddings, and writes:
    - embedding_comparisons.png
    - embedding_comparisons.pdf
    - individual/test_vs_*.png
    - cache/metadata.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import multiprocessing as mp
import os
import re
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from tqdm.auto import tqdm


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_DIR = REPO_ROOT / "embed-plots" / "chinese" / "qwen25-1-5b"

HF_BASE = "https://huggingface.co/datasets/saiteja33/PolyAlign-All/resolve/main"

SOURCES: list[dict[str, str]] = [
    {
        "key": "test",
        "display": "Golden answers",
        "url": f"{HF_BASE}/chinese/merged_sft_dedup/llamafactory/test.json",
        "kind": "test_json",
    },
    {
        "key": "baselm",
        "display": "BaseLM",
        "url": f"{HF_BASE}/chinese/merged_sft_dedup/runs/qwen25_1_5b_zh/predictions.jsonl",
        "kind": "prediction_jsonl",
    },
    {
        "key": "sft",
        "display": "SFT",
        "url": f"{HF_BASE}/chinese/merged_sft_dedup/runs/qwen25_1_5_sft-zh/predictions.jsonl",
        "kind": "prediction_jsonl",
    },
    {
        "key": "dpo",
        "display": "DPO",
        "url": f"{HF_BASE}/chinese/merged_sft_dedup/runs/qwen25-15b-dpo-zh/predictions.jsonl",
        "kind": "prediction_jsonl",
    },
    {
        "key": "bucket_sft",
        "display": "Bucket SFT",
        "url": f"{HF_BASE}/chinese/merged_sft_dedup/runs/qwen25_1_5b_dist_sft_zh_test/predictions.jsonl",
        "kind": "prediction_jsonl",
    },
    {
        "key": "hdpo",
        "display": "HDPO",
        "url": f"{HF_BASE}/chinese/merged_sft_dedup/runs/qwen25-3b-hdpo-zh-ref-conditioned/predictions.jsonl",
        "kind": "prediction_jsonl",
    },
]

COMPARISONS = [
    ("baselm", "test-baselm"),
    ("sft", "test-sft"),
    ("dpo", "test-dpo"),
    ("bucket_sft", "test-bucket-sft"),
    ("hdpo", "test-hdpo"),
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create embedding plots for Chinese qwen25-1.5b PolyAlign outputs."
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--embedding-model",
        default="BAAI/bge-m3",
        help="Hugging Face encoder model used for sentence embeddings.",
    )
    parser.add_argument(
        "--gpus",
        default="2,3,4,5,6,7",
        help="Comma-separated GPU ids. Defaults map test, BaseLM, SFT, DPO, Bucket SFT, HDPO to 2-7.",
    )
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument(
        "--dtype",
        choices=("auto", "float16", "bfloat16", "float32"),
        default="auto",
        help="Embedding model dtype. auto uses float16 on CUDA and float32 on CPU.",
    )
    parser.add_argument(
        "--reducer",
        choices=("pca", "umap"),
        default="pca",
        help="2D reducer for plotting. UMAP requires `pip install umap-learn`.",
    )
    parser.add_argument("--umap-neighbors", type=int, default=30)
    parser.add_argument("--umap-min-dist", type=float, default=0.08)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--arrow-sample-size",
        type=int,
        default=700,
        help="Number of aligned examples to connect with shift lines in each panel.",
    )
    parser.add_argument(
        "--point-sample-size",
        type=int,
        default=0,
        help="Optionally sample this many aligned examples for plotting/reduction. 0 means use all rows.",
    )
    parser.add_argument(
        "--embed-sample-only",
        action="store_true",
        help="Embed only --point-sample-size rows. By default all rows are embedded and only plotting may be sampled.",
    )
    parser.add_argument("--refresh-data", action="store_true", help="Redownload HF input files.")
    parser.add_argument("--refresh-embeddings", action="store_true", help="Recompute cached embeddings.")
    parser.add_argument("--allow-cpu", action="store_true", help="Fall back to CPU if CUDA is unavailable.")
    return parser.parse_args()


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
    request = urllib.request.Request(url, headers={"User-Agent": "polyalign-embedding-plot/1.0"})
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


def extract_test_outputs(rows: Any) -> list[str]:
    if not isinstance(rows, list):
        raise ValueError("test.json must be a JSON list.")
    texts: list[str] = []
    for idx, row in enumerate(rows):
        if not isinstance(row, dict):
            raise ValueError(f"test.json row {idx} is not an object.")
        texts.append(as_text(row.get("output", row.get("reference_output", ""))))
    return texts


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


def load_all_texts(
    data_dir: Path, refresh_data: bool
) -> tuple[dict[str, list[str]], dict[str, np.ndarray], dict[str, dict[str, Any]], dict[str, Path]]:
    data_dir.mkdir(parents=True, exist_ok=True)
    paths: dict[str, Path] = {}

    for source in SOURCES:
        suffix = ".json" if source["kind"] == "test_json" else ".jsonl"
        path = data_dir / f"{source['key']}{suffix}"
        paths[source["key"]] = download_file(source["url"], path, refresh=refresh_data)

    test_outputs = extract_test_outputs(load_json(paths["test"]))
    all_texts: dict[str, list[str]] = {"test": test_outputs}
    source_indices: dict[str, np.ndarray] = {"test": np.arange(len(test_outputs), dtype=np.int64)}
    availability: dict[str, dict[str, Any]] = {
        "test": {
            "available": len(test_outputs),
            "missing": 0,
            "extra": 0,
            "first_missing": [],
            "first_extra": [],
        }
    }
    for source in SOURCES:
        if source["kind"] != "prediction_jsonl":
            continue
        rows = load_jsonl(paths[source["key"]])
        texts, indices, stats = extract_predictions(rows, len(test_outputs), source["display"])
        all_texts[source["key"]] = texts
        source_indices[source["key"]] = np.array(indices, dtype=np.int64)
        availability[source["key"]] = stats

    return all_texts, source_indices, availability, paths


def parse_gpus(raw: str) -> list[int]:
    try:
        gpus = [int(part.strip()) for part in raw.split(",") if part.strip()]
    except ValueError as exc:
        raise ValueError(f"Could not parse --gpus={raw!r}") from exc
    if len(gpus) != len(SOURCES):
        raise ValueError(f"--gpus needs {len(SOURCES)} ids, got {len(gpus)}: {gpus}")
    return gpus


def visible_cuda_index(requested_gpu: int) -> int:
    """Map a physical id to a visible CUDA ordinal when CUDA_VISIBLE_DEVICES is set."""
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

    tokenizer = AutoTokenizer.from_pretrained(job.model_name, use_fast=True)
    model = AutoModel.from_pretrained(
        job.model_name,
        torch_dtype=torch_dtype(job.dtype, using_cuda),
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


def reduce_2d(matrix: np.ndarray, args: argparse.Namespace) -> np.ndarray:
    if args.reducer == "pca":
        return pca_2d(matrix)

    try:
        from umap import UMAP
    except ImportError as exc:
        raise RuntimeError("UMAP reducer requested. Install it with: pip install umap-learn") from exc

    reducer = UMAP(
        n_components=2,
        n_neighbors=args.umap_neighbors,
        min_dist=args.umap_min_dist,
        metric="cosine",
        random_state=args.seed,
    )
    return reducer.fit_transform(matrix).astype(np.float32)


def symmetric_limits(*arrays: np.ndarray, padding: float = 0.08) -> tuple[tuple[float, float], tuple[float, float]]:
    stacked = np.vstack(arrays)
    xmin, ymin = stacked.min(axis=0)
    xmax, ymax = stacked.max(axis=0)
    xpad = max((xmax - xmin) * padding, 1e-3)
    ypad = max((ymax - ymin) * padding, 1e-3)
    return (float(xmin - xpad), float(xmax + xpad)), (float(ymin - ypad), float(ymax + ypad))


def index_positions(indices: np.ndarray) -> dict[int, int]:
    return {int(source_idx): row_idx for row_idx, source_idx in enumerate(indices.tolist())}


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
    human_color = "#20C997"
    model_color = "#7B2CBF"
    line_color = "#111827"

    for idx in arrow_indices:
        ax.plot(
            [human_2d[idx, 0], model_2d[idx, 0]],
            [human_2d[idx, 1], model_2d[idx, 1]],
            color=line_color,
            alpha=0.045,
            linewidth=0.55,
            zorder=1,
        )

    ax.scatter(
        human_2d[:, 0],
        human_2d[:, 1],
        s=18,
        c=human_color,
        edgecolors="#053B2C",
        linewidths=0.35,
        alpha=0.72,
        label="Golden answers",
        zorder=3,
    )
    ax.scatter(
        model_2d[:, 0],
        model_2d[:, 1],
        s=16,
        c=model_color,
        marker="X",
        edgecolors="#1F102B",
        linewidths=0.25,
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
        c="#0D6EFD",
        edgecolors="white",
        linewidths=0.9,
        label="Golden centroid",
        zorder=5,
    )
    ax.scatter(
        [m_centroid[0]],
        [m_centroid[1]],
        s=185,
        marker="*",
        c="#FFB000",
        edgecolors="#2B2118",
        linewidths=0.65,
        label="Model centroid",
        zorder=5,
    )

    ax.axhline(0, color="#94A3B8", linewidth=0.8, alpha=0.55, zorder=0)
    ax.axvline(0, color="#94A3B8", linewidth=0.8, alpha=0.55, zorder=0)
    ax.grid(True, linestyle="--", linewidth=0.5, color="#CBD5E1", alpha=0.65)
    ax.set_title(title, fontweight="semibold", pad=9)
    ax.set_xlabel(f"{axis_label} 1")
    ax.set_ylabel(f"{axis_label} 2")
    xlim, ylim = symmetric_limits(human_2d, model_2d)
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    for spine in ax.spines.values():
        spine.set_color("#475569")
        spine.set_linewidth(0.85)
    if show_legend:
        ax.legend(loc="best", frameon=True, framealpha=0.95, fontsize=9)


def write_plots(
    embeddings: dict[str, np.ndarray],
    source_indices: dict[str, np.ndarray],
    args: argparse.Namespace,
    model_labels: dict[str, str],
    selected_indices: np.ndarray,
) -> list[str]:
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    plt.rcParams.update(
        {
            "figure.dpi": 130,
            "savefig.dpi": 320,
            "font.family": "DejaVu Sans",
            "axes.titlesize": 12,
            "axes.labelsize": 9,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "legend.fontsize": 9,
        }
    )

    out_dir = args.output_dir
    individual_dir = out_dir / "individual"
    individual_dir.mkdir(parents=True, exist_ok=True)
    written: list[str] = []

    rng = np.random.default_rng(args.seed)
    selected_set = set(int(idx) for idx in selected_indices.tolist())
    test_pos = index_positions(source_indices["test"])

    reduced_pairs: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
    for model_key, _title in tqdm(COMPARISONS, desc=f"{args.reducer} projection"):
        model_pos = index_positions(source_indices[model_key])
        pair_source_indices = np.array(
            [
                int(idx)
                for idx in source_indices[model_key].tolist()
                if int(idx) in selected_set and int(idx) in test_pos
            ],
            dtype=np.int64,
        )
        if len(pair_source_indices) == 0:
            raise ValueError(f"{model_labels[model_key]} has no available rows for the selected sample.")

        human_rows = np.array([test_pos[int(idx)] for idx in pair_source_indices], dtype=np.int64)
        model_rows = np.array([model_pos[int(idx)] for idx in pair_source_indices], dtype=np.int64)
        joint = np.vstack([embeddings["test"][human_rows], embeddings[model_key][model_rows]])
        coords = reduce_2d(joint, args)
        n = len(pair_source_indices)
        reduced_pairs[model_key] = (coords[:n], coords[n:], pair_source_indices)

    fig, axes = plt.subplots(2, 3, figsize=(18.5, 10.2), constrained_layout=False)
    fig.patch.set_facecolor("white")
    axes_flat = axes.ravel()

    legend_handles = [
        Line2D([0], [0], color="#111827", alpha=0.28, linewidth=1.3, label="Sample shift"),
        Line2D(
            [0],
            [0],
            marker="o",
            linestyle="",
            markersize=7,
            markerfacecolor="#20C997",
            markeredgecolor="#053B2C",
            label="Golden answers",
        ),
        Line2D(
            [0],
            [0],
            marker="X",
            linestyle="",
            markersize=7,
            markerfacecolor="#7B2CBF",
            markeredgecolor="#1F102B",
            label="Model predictions",
        ),
        Line2D(
            [0],
            [0],
            marker="*",
            linestyle="",
            markersize=11,
            markerfacecolor="#0D6EFD",
            markeredgecolor="white",
            label="Golden centroid",
        ),
        Line2D(
            [0],
            [0],
            marker="*",
            linestyle="",
            markersize=11,
            markerfacecolor="#FFB000",
            markeredgecolor="#2B2118",
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
        edgecolor="#475569",
    )
    fig.suptitle(
        "Chinese Qwen2.5-1.5B Answer Embedding Shifts",
        y=0.998,
        fontsize=17,
        fontweight="bold",
    )

    axis_label = "Principal component" if args.reducer == "pca" else "UMAP component"
    for ax, (model_key, title) in zip(axes_flat, COMPARISONS):
        human_2d, model_2d, pair_source_indices = reduced_pairs[model_key]
        max_arrows = min(args.arrow_sample_size, len(pair_source_indices))
        local_arrow_indices = (
            np.sort(rng.choice(np.arange(len(pair_source_indices)), size=max_arrows, replace=False))
            if max_arrows > 0
            else np.array([], dtype=int)
        )
        plot_pair(
            ax,
            human_2d,
            model_2d,
            title=f"{title} (n={len(pair_source_indices)})",
            model_label=model_labels[model_key],
            arrow_indices=local_arrow_indices,
            axis_label=axis_label,
            show_legend=False,
        )

    for ax in axes_flat[len(COMPARISONS) :]:
        ax.axis("off")

    fig.subplots_adjust(top=0.90, left=0.055, right=0.985, bottom=0.075, wspace=0.24, hspace=0.30)
    png_path = out_dir / "embedding_comparisons.png"
    pdf_path = out_dir / "embedding_comparisons.pdf"
    fig.savefig(png_path, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)
    written.extend([str(png_path), str(pdf_path)])

    for model_key, title in COMPARISONS:
        fig_one, ax_one = plt.subplots(1, 1, figsize=(8.4, 6.4), constrained_layout=True)
        human_2d, model_2d, pair_source_indices = reduced_pairs[model_key]
        max_arrows = min(args.arrow_sample_size, len(pair_source_indices))
        local_arrow_indices = (
            np.sort(rng.choice(np.arange(len(pair_source_indices)), size=max_arrows, replace=False))
            if max_arrows > 0
            else np.array([], dtype=int)
        )
        plot_pair(
            ax_one,
            human_2d,
            model_2d,
            title=f"{title} (n={len(pair_source_indices)})",
            model_label=model_labels[model_key],
            arrow_indices=local_arrow_indices,
            axis_label=axis_label,
            show_legend=True,
        )
        fig_one.suptitle("Chinese Qwen2.5-1.5B Answer Embedding Shift", fontsize=13.5, fontweight="bold")
        path = individual_dir / f"{title.replace('test-', 'test_vs_').replace('-', '_')}.png"
        fig_one.savefig(path, bbox_inches="tight")
        plt.close(fig_one)
        written.append(str(path))

    return written


def main() -> None:
    args = parse_args()
    args.output_dir = args.output_dir.resolve()
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

    data_dir = args.output_dir / "cache" / "inputs"
    embed_dir = args.output_dir / "cache" / "embeddings"
    args.output_dir.mkdir(parents=True, exist_ok=True)
    embed_dir.mkdir(parents=True, exist_ok=True)

    all_texts, source_indices, availability, input_paths = load_all_texts(data_dir, args.refresh_data)
    total_rows = len(source_indices["test"])

    rng = np.random.default_rng(args.seed)
    if args.point_sample_size and args.point_sample_size < total_rows:
        selected_indices = np.sort(rng.choice(np.arange(total_rows), size=args.point_sample_size, replace=False))
    else:
        selected_indices = np.arange(total_rows)

    if args.embed_sample_only:
        if not args.point_sample_size:
            raise ValueError("--embed-sample-only requires --point-sample-size.")
        selected_set = set(int(idx) for idx in selected_indices.tolist())
        sampled_texts: dict[str, list[str]] = {}
        sampled_indices: dict[str, np.ndarray] = {}
        for key, texts in all_texts.items():
            keep_rows = [
                row_idx
                for row_idx, source_idx in enumerate(source_indices[key].tolist())
                if int(source_idx) in selected_set
            ]
            sampled_texts[key] = [texts[row_idx] for row_idx in keep_rows]
            sampled_indices[key] = source_indices[key][keep_rows]
        all_texts = sampled_texts
        source_indices = sampled_indices

    gpus = parse_gpus(args.gpus)
    model_labels = {source["key"]: source["display"] for source in SOURCES}

    print("Available aligned rows:")
    for source in SOURCES:
        key = source["key"]
        stats = availability[key]
        print(
            f"  {key:10s}: {stats['available']:5d}/{total_rows} available, "
            f"{stats['missing']:5d} missing"
        )

    jobs: list[EmbeddingJob] = []
    embedding_paths: dict[str, Path] = {}
    model_slug = slugify(args.embedding_model)
    source_by_key = {source["key"]: source for source in SOURCES}
    for position, (source, gpu_id) in enumerate(zip(SOURCES, gpus)):
        key = source["key"]
        fingerprint = sha1_texts(
            all_texts[key],
            extra=f"{args.embedding_model}|{args.max_length}|{source['key']}",
        )
        cache_path = embed_dir / f"{key}_{model_slug}_{len(all_texts[key])}_{fingerprint}.npy"
        if args.refresh_embeddings and cache_path.exists():
            cache_path.unlink()
        embedding_paths[key] = cache_path
        jobs.append(
            EmbeddingJob(
                key=key,
                display=source["display"],
                texts=all_texts[key],
                model_name=args.embedding_model,
                batch_size=args.batch_size,
                max_length=args.max_length,
                gpu_id=gpu_id,
                position=position,
                cache_path=str(cache_path),
                dtype=args.dtype,
                allow_cpu=args.allow_cpu,
            )
        )

    pending_jobs = [job for job in jobs if not Path(job.cache_path).exists()]
    if pending_jobs:
        print("Embedding groups:")
        for job in jobs:
            status = "cached" if Path(job.cache_path).exists() else f"gpu {job.gpu_id}"
            print(f"  {job.key:10s} -> {status}")
        ctx = mp.get_context("spawn")
        with ctx.Pool(processes=len(pending_jobs)) as pool:
            for key, path in pool.imap_unordered(embed_worker, pending_jobs):
                embedding_paths[key] = Path(path)
    else:
        print("All embeddings are cached; skipping GPU embedding.")

    embeddings = {key: np.load(path) for key, path in embedding_paths.items()}
    for key, matrix in embeddings.items():
        if matrix.shape[0] != len(all_texts[key]):
            raise ValueError(f"{key}: expected {len(all_texts[key])} embeddings, found {matrix.shape[0]}")
        if matrix.ndim != 2:
            raise ValueError(f"{key}: expected 2D embedding matrix, found shape {matrix.shape}")

    written = write_plots(embeddings, source_indices, args, model_labels, selected_indices)

    metadata = {
        "n_rows": total_rows,
        "selected_rows": int(len(selected_indices)),
        "embedded_rows": {key: len(value) for key, value in all_texts.items()},
        "availability": availability,
        "embedding_model": args.embedding_model,
        "reducer": args.reducer,
        "gpus": {source["key"]: gpu for source, gpu in zip(SOURCES, gpus)},
        "sources": source_by_key,
        "input_paths": {key: str(path) for key, path in input_paths.items()},
        "source_index_ranges": {
            key: {
                "min": int(values.min()) if len(values) else None,
                "max": int(values.max()) if len(values) else None,
                "count": int(len(values)),
            }
            for key, values in source_indices.items()
        },
        "embedding_paths": {key: str(path) for key, path in embedding_paths.items()},
        "written": written,
    }
    metadata_path = args.output_dir / "cache" / "metadata.json"
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")

    print("\nWrote:")
    for path in written:
        print(f"  {path}")
    print(f"  {metadata_path}")


if __name__ == "__main__":
    mp.freeze_support()
    main()
