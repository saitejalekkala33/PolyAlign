from __future__ import annotations

import gzip
import json
import shutil
import urllib.request
import zipfile
from pathlib import Path

from huggingface_hub import hf_hub_download

from polyalign_data.io_utils import ensure_dir


def download_url(url: str, destination: Path) -> Path:
    ensure_dir(destination.parent)
    if destination.exists():
        return destination
    with urllib.request.urlopen(url) as response, destination.open("wb") as handle:
        shutil.copyfileobj(response, handle)
    return destination


def load_json(path: Path):
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def load_gzip_json(path: Path):
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        return json.load(handle)


def hf_dataset_file(repo_id: str, filename: str) -> Path:
    return Path(hf_hub_download(repo_id=repo_id, repo_type="dataset", filename=filename))


def open_zip(path: Path) -> zipfile.ZipFile:
    return zipfile.ZipFile(path)
