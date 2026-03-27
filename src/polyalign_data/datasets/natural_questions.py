from __future__ import annotations

from datasets import load_dataset

from polyalign_data.datasets.base import DatasetFormatter
from polyalign_data.schema import build_record
from polyalign_data.splits import choose_split


class NaturalQuestionsFormatter(DatasetFormatter):
    dataset_name = "natural_questions"
    source_name = "sentence-transformers/natural-questions"

    def split_policy(self) -> str:
        return "local deterministic 90/5/5 split over the train-only sentence-transformers release"

    def build_split_records(self) -> dict[str, list[dict]]:
        dataset = load_dataset(self.source_name, split="train")
        outputs = {"train": [], "dev": [], "test": []}
        for row_index, row in enumerate(dataset):
            source_id = f"natural-questions-train-{row_index:06d}"
            split = choose_split(
                f"{row['query']}::{row['answer']}",
                self.seed,
                [("train", 0.90), ("dev", 0.05), ("test", 0.05)],
            )
            outputs[split].append(
                build_record(
                    example_id=source_id,
                    dataset=self.dataset_name,
                    split=split,
                    language=self.language,
                    track="single",
                    family="qa",
                    style_bucket="qa_search",
                    question=row["query"],
                    context="",
                    dialogue_history=[],
                    human_answer=row["answer"],
                    meta={
                        "source_dataset": self.source_name,
                        "source_split": "train",
                        "source_id": source_id,
                        "source_variant": "sentence-transformers",
                    },
                )
            )
        return outputs
