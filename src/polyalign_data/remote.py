from __future__ import annotations

import gzip
import json
import shutil
import time
import urllib.parse
import urllib.request
import zipfile
from pathlib import Path

from datasets import load_dataset
from huggingface_hub import hf_hub_download

from polyalign_data.io_utils import ensure_dir


def download_url(url: str, destination: Path, *, retries: int = 5, timeout: int = 120) -> Path:
    ensure_dir(destination.parent)
    if destination.exists() and destination.stat().st_size > 0:
        return destination
    temp_path = destination.with_suffix(destination.suffix + ".tmp")
    last_error: Exception | None = None
    for attempt in range(1, retries + 1):
        request = urllib.request.Request(url, headers={"User-Agent": "PolyAlign/0.1"})
        try:
            temp_path.unlink(missing_ok=True)
            with urllib.request.urlopen(request, timeout=timeout) as response, temp_path.open("wb") as handle:
                shutil.copyfileobj(response, handle)
            temp_path.replace(destination)
            return destination
        except Exception as exc:
            last_error = exc
            temp_path.unlink(missing_ok=True)
            if attempt == retries:
                raise
            wait_seconds = min(2 ** (attempt - 1), 8)
            print(
                f"download retry {attempt}/{retries - 1} for {url} after {type(exc).__name__}: {exc}",
                flush=True,
            )
            time.sleep(wait_seconds)
    if last_error is not None:
        raise last_error
    return destination


def load_json(path: Path):
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def load_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                yield json.loads(line)


def load_gzip_json(path: Path):
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        return json.load(handle)


def hf_dataset_file(repo_id: str, filename: str) -> Path:
    return Path(hf_hub_download(repo_id=repo_id, repo_type="dataset", filename=filename))


def hf_parquet_urls(repo_id: str, *, config_name: str | None = None, split: str | None = None) -> list[str]:
    encoded_repo = urllib.parse.quote(repo_id, safe="")
    endpoint = f"https://huggingface.co/api/datasets/{encoded_repo}/parquet"
    if config_name is not None:
        endpoint += f"/{urllib.parse.quote(config_name, safe='')}"
    if split is not None:
        endpoint += f"/{urllib.parse.quote(split, safe='')}"
    request = urllib.request.Request(endpoint, headers={"User-Agent": "PolyAlign/0.1"})
    with urllib.request.urlopen(request, timeout=120) as response:
        payload = json.load(response)
    if isinstance(payload, list):
        return [str(item) for item in payload]
    parquet_files = payload.get("parquet_files", [])
    urls: list[str] = []
    for item in parquet_files:
        if config_name is not None and item.get("config") != config_name:
            continue
        if split is not None and item.get("split") != split:
            continue
        urls.append(item["url"])
    return urls


def load_hf_parquet_split(
    repo_id: str,
    *,
    split: str,
    config_name: str | None = None,
):
    urls = hf_parquet_urls(repo_id, config_name=config_name, split=split)
    if not urls:
        raise FileNotFoundError(
            f"No parquet URLs found for repo_id={repo_id!r}, config_name={config_name!r}, split={split!r}"
        )
    return load_dataset("parquet", data_files={split: urls}, split=split)


def open_zip(path: Path) -> zipfile.ZipFile:
    return zipfile.ZipFile(path)


def load_hf_split_with_parquet_fallback(
    repo_id: str,
    *,
    split: str,
    config_name: str | None = None,
    parquet_filename: str | None = None,
):
    try:
        return load_dataset(repo_id, config_name, split=split)
    except Exception:
        if not parquet_filename:
            raise
        parquet_path = hf_dataset_file(repo_id, parquet_filename)
        return load_dataset("parquet", data_files={split: str(parquet_path)}, split=split)
