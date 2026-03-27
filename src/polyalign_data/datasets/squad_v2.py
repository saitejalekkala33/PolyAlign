from __future__ import annotations

from datasets import load_dataset

from polyalign_data.datasets.base import DatasetFormatter
from polyalign_data.schema import build_record
from polyalign_data.splits import choose_split


class SQuADV2Formatter(DatasetFormatter):
    dataset_name = "squad_v2"
    source_name = "rajpurkar/squad_v2"

    def split_policy(self) -> str:
        return "paragraph-level 90/10 split over official train; official validation mapped to test"

    def _append_record(
        self,
        outputs: dict[str, list[dict]],
        row: dict,
        source_split: str,
        target_split: str,
    ) -> None:
        answers = row["answers"]["text"]
        human_answer = answers[0] if answers else ""
        source_id = f"squad-v2-{source_split}-{row['id']}"
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
                human_answer=human_answer,
                meta={
                    "source_dataset": self.source_name,
                    "source_split": source_split,
                    "source_id": row["id"],
                    "title": row["title"],
                    "answer_count": len(answers),
                    "is_unanswerable": len(answers) == 0,
                },
            )
        )

    def build_split_records(self) -> dict[str, list[dict]]:
        outputs = {"train": [], "dev": [], "test": []}
        train_dataset = load_dataset(self.source_name, split="train")
        validation_dataset = load_dataset(self.source_name, split="validation")

        for row in train_dataset:
            split = choose_split(
                f"{row['title']}::{row['context']}",
                self.seed,
                [("train", 0.90), ("dev", 0.10)],
            )
            self._append_record(outputs, row, "train", split)

        for row in validation_dataset:
            self._append_record(outputs, row, "validation", "test")

        return outputs
