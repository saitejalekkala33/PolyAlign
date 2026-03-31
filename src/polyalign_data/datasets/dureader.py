from __future__ import annotations

from datasets import load_dataset

from polyalign_data.datasets.base import DatasetFormatter
from polyalign_data.datasets.helpers import first_non_empty_text, is_truthy, non_empty_texts
from polyalign_data.schema import build_record


class DuReaderFormatter(DatasetFormatter):
    dataset_name = "dureader"
    source_name = "luozhouyang/dureader"
    language = "zh"
    subsets = ("checklist", "robust")

    def split_policy(self) -> str:
        return "official checklist+robust train/train and validation/dev; test ignored; impossible checklist rows removed"

    def build_split_records(self) -> dict[str, list[dict]]:
        outputs = {"train": [], "dev": []}
        split_map = {"train": "train", "validation": "dev"}
        for subset in self.subsets:
            for source_split, target_split in split_map.items():
                dataset = load_dataset(self.source_name, subset, split=source_split)
                for row in dataset:
                    answers = row.get("answers", {})
                    answer_texts = non_empty_texts(answers.get("text", []))
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
                            },
                        )
                    )
        return outputs
