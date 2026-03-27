from __future__ import annotations

from datasets import load_dataset

from polyalign_data.datasets.base import DatasetFormatter
from polyalign_data.schema import build_record
from polyalign_data.splits import choose_split
from polyalign_data.text import normalize_text


class DollyFormatter(DatasetFormatter):
    dataset_name = "dolly"
    source_name = "databricks/databricks-dolly-15k"

    def split_policy(self) -> str:
        return "local deterministic 90/5/5 split over the source train split"

    def build_split_records(self) -> dict[str, list[dict]]:
        dataset = load_dataset(self.source_name, split="train")
        outputs = {"train": [], "dev": [], "test": []}
        for row_index, row in enumerate(dataset):
            source_id = f"dolly-train-{row_index:06d}"
            split = choose_split(
                f"{row['instruction']}::{row['response']}::{row['category']}",
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
                    family="assistant",
                    style_bucket="assistant_like",
                    question=normalize_text(row["instruction"]),
                    context=normalize_text(row["context"]),
                    dialogue_history=[],
                    human_answer=row["response"],
                    meta={
                        "source_dataset": self.source_name,
                        "source_split": "train",
                        "source_id": source_id,
                        "category": row["category"],
                    },
                )
            )
        return outputs
