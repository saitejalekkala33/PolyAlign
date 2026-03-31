from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
import shutil
from typing import Any

from polyalign_data.io_utils import ensure_dir, read_json, write_json, write_jsonl


class DatasetFormatter(ABC):
    dataset_name: str
    source_name: str
    language: str = "en"

    def __init__(self, *, seed: int = 42, cache_dir: str | Path = "data/cache") -> None:
        self.seed = seed
        self.cache_dir = Path(cache_dir)

    @abstractmethod
    def build_split_records(self) -> dict[str, list[dict[str, Any]]]:
        raise NotImplementedError

    def build_manifest(self, split_records: dict[str, list[dict[str, Any]]]) -> dict[str, Any]:
        manifest = {
            "dataset": self.dataset_name,
            "source": self.source_name,
            "language": self.language,
            "seed": self.seed,
            "split_counts": {split: len(records) for split, records in split_records.items()},
            "split_policy": self.split_policy(),
        }
        manifest.update(self.extra_manifest())
        return manifest

    @abstractmethod
    def split_policy(self) -> str:
        raise NotImplementedError

    def extra_manifest(self) -> dict[str, Any]:
        return {}

    def write(self, output_root: str | Path, overwrite: bool = False) -> dict[str, Any]:
        dataset_dir = Path(output_root) / self.dataset_name
        manifest_path = dataset_dir / "manifest.json"
        if dataset_dir.exists() and any(dataset_dir.iterdir()) and not overwrite:
            if manifest_path.exists():
                existing_manifest = read_json(manifest_path)
                existing_manifest["status"] = "skipped"
                existing_manifest["output_dir"] = str(dataset_dir)
                return existing_manifest
            shutil.rmtree(dataset_dir)
        if dataset_dir.exists() and overwrite:
            shutil.rmtree(dataset_dir)
        ensure_dir(dataset_dir)
        split_records = self.build_split_records()
        for split, records in split_records.items():
            write_jsonl(dataset_dir / f"{split}.jsonl", records)
        manifest = self.build_manifest(split_records)
        manifest["status"] = "written"
        manifest["output_dir"] = str(dataset_dir)
        write_json(dataset_dir / "manifest.json", manifest)
        return manifest
