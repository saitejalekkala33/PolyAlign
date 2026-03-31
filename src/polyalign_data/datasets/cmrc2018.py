from __future__ import annotations

from datasets import load_dataset

from polyalign_data.datasets.base import DatasetFormatter
from polyalign_data.datasets.helpers import non_empty_texts
from polyalign_data.schema import build_record


class CMRC2018Formatter(DatasetFormatter):
    dataset_name = "cmrc2018"
    source_name = "hfl/cmrc2018"
    language = "zh"

    def split_policy(self) -> str:
        return "official HF splits mapped as train/train, validation/dev, test/test"

    def _append_record(
        self,
        outputs: dict[str, list[dict]],
        row: dict,
        source_split: str,
        target_split: str,
    ) -> None:
        answers = row.get("answers", {})
        answer_texts = non_empty_texts(answers.get("text", []))
        if not answer_texts:
            return
        human_answer = answer_texts[0]
        source_id = f"cmrc2018-{source_split}-{row['id']}"
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
                    "answer_count": len(answer_texts),
                },
            )
        )

    def build_split_records(self) -> dict[str, list[dict]]:
        outputs = {"train": [], "dev": [], "test": []}
        split_map = {"train": "train", "validation": "dev", "test": "test"}
        for source_split, target_split in split_map.items():
            dataset = load_dataset(self.source_name, split=source_split)
            for row in dataset:
                self._append_record(outputs, row, source_split, target_split)
        return outputs
