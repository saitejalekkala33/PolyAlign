from __future__ import annotations

import io
import json

from polyalign_data.datasets.base import DatasetFormatter
from polyalign_data.datasets.helpers import is_truthy, non_empty_texts
from polyalign_data.remote import hf_dataset_file, open_zip
from polyalign_data.schema import build_record


class DuReaderFormatter(DatasetFormatter):
    dataset_name = "dureader"
    source_name = "luozhouyang/dureader"
    language = "zh"
    subsets = ("checklist", "robust")
    subset_archives = {
        "checklist": "dummy/checklist/1.0.0/dummy_data.zip",
        "robust": "dummy/robust/1.0.0/dummy_data.zip",
    }
    split_aliases = {
        "train": ("train.json", ".train.json", "train.jsonl"),
        "validation": ("dev.json", ".dev.json", "validation.json", "validation.jsonl", "dev.jsonl"),
        "test": ("test.json", ".test.json", "test.jsonl"),
    }

    def split_policy(self) -> str:
        return "checklist+robust loaded from the provided HF zip files; train/train and validation/dev; test ignored; impossible checklist rows removed"

    def _zip_member_for_split(self, subset: str, source_split: str, member_names: list[str]) -> str:
        aliases = self.split_aliases[source_split]
        candidates = [
            name for name in member_names if any(name.endswith(alias) for alias in aliases)
        ]
        if not candidates:
            raise FileNotFoundError(
                f"Could not find a {source_split} file inside {self.subset_archives[subset]}"
            )
        candidates.sort(key=lambda item: (len(item), item))
        return candidates[0]

    def _normalize_row_answers(self, row: dict) -> list[str]:
        answers = row.get("answers", {})
        if isinstance(answers, dict):
            return non_empty_texts(answers.get("text", []))
        if isinstance(answers, list):
            texts: list[str] = []
            for answer in answers:
                if isinstance(answer, dict):
                    texts.extend(non_empty_texts([answer.get("text", "")]))
                else:
                    texts.extend(non_empty_texts([answer]))
            return texts
        return []

    def _iter_payload_rows(self, payload: object):
        if isinstance(payload, list):
            for row in payload:
                if isinstance(row, dict):
                    yield row
            return
        if isinstance(payload, dict) and "data" in payload:
            for article in payload.get("data", []):
                title = article.get("title", "")
                for paragraph in article.get("paragraphs", []):
                    context = paragraph.get("context", "")
                    for qa in paragraph.get("qas", []):
                        yield {
                            "id": qa.get("id", ""),
                            "title": title,
                            "context": context,
                            "question": qa.get("question", ""),
                            "is_impossible": qa.get("is_impossible", False),
                            "type": qa.get("type", ""),
                            "answers": qa.get("answers", []),
                        }
            return
        raise TypeError(f"Unsupported DuReader payload type: {type(payload)!r}")

    def _load_split_rows(self, subset: str, source_split: str):
        archive_path = hf_dataset_file(self.source_name, self.subset_archives[subset])
        with open_zip(archive_path) as zip_file:
            member_names = list(map(str, zip_file.namelist()))
            split_member = self._zip_member_for_split(subset, source_split, member_names)
            with zip_file.open(split_member) as raw_handle:
                payload = json.load(io.TextIOWrapper(raw_handle, encoding="utf-8"))
        for row in self._iter_payload_rows(payload):
            yield row

    def build_split_records(self) -> dict[str, list[dict]]:
        outputs = {"train": [], "dev": []}
        split_map = {"train": "train", "validation": "dev"}
        for subset in self.subsets:
            for source_split, target_split in split_map.items():
                for row in self._load_split_rows(subset, source_split):
                    answer_texts = self._normalize_row_answers(row)
                    if subset == "checklist" and is_truthy(row.get("is_impossible")):
                        continue
                    if not answer_texts:
                        continue
                    source_id = f"dureader-{subset}-{source_split}-{row['id']}"
                    outputs[target_split].append(
                        build_record(
                            example_id=source_id,
                            dataset=self.dataset_name,
                            split=target_split,
                            language=self.language,
                            track="single",
                            family="qa",
                            style_bucket="qa_search",
                            question=row["question"],
                            context=row["context"],
                            dialogue_history=[],
                            human_answer=answer_texts[0],
                            meta={
                                "source_dataset": self.source_name,
                                "source_subset": subset,
                                "source_split": source_split,
                                "source_id": row["id"],
                                "title": row.get("title", ""),
                                "is_impossible": is_truthy(row.get("is_impossible")),
                                "answer_count": len(answer_texts),
                                "type": row.get("type", ""),
                            },
                        )
                    )
        return outputs
