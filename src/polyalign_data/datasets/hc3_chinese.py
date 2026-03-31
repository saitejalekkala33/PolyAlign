from __future__ import annotations

from datasets import load_dataset

from polyalign_data.datasets.base import DatasetFormatter
from polyalign_data.datasets.helpers import non_empty_texts
from polyalign_data.schema import build_record
from polyalign_data.splits import choose_split


class HC3ChineseFormatter(DatasetFormatter):
    dataset_name = "hc3_chinese"
    source_name = "Hello-SimpleAI/HC3-Chinese"
    config_name = "all"
    language = "zh"

    def split_policy(self) -> str:
        return "local deterministic 90/5/5 split over the train-only HC3-Chinese release with one row per human answer"

    def extra_manifest(self) -> dict[str, object]:
        return {"config": self.config_name}

    def build_split_records(self) -> dict[str, list[dict]]:
        dataset = load_dataset(self.source_name, self.config_name, split="train")
        outputs = {"train": [], "dev": [], "test": []}
        for row in dataset:
            split = choose_split(
                f"{row.get('source', '')}::{row['id']}",
                self.seed,
                [("train", 0.90), ("dev", 0.05), ("test", 0.05)],
            )
            human_answers = non_empty_texts(row.get("human_answers", []))
            for answer_index, human_answer in enumerate(human_answers):
                source_id = f"hc3-chinese-{row.get('source', 'all')}-{row['id']}-answer-{answer_index:02d}"
                outputs[split].append(
                    build_record(
                        example_id=source_id,
                        dataset=self.dataset_name,
                        split=split,
                        language=self.language,
                        track="single",
                        family="qa",
                        style_bucket="longform_qa",
                        question=row["question"],
                        context="",
                        dialogue_history=[],
                        human_answer=human_answer,
                        meta={
                            "source_dataset": self.source_name,
                            "source_config": self.config_name,
                            "source_split": "train",
                            "source_id": row["id"],
                            "source_subset": row.get("source", ""),
                            "human_answer_index": answer_index,
                            "num_human_answers": len(human_answers),
                            "num_chatgpt_answers": len(row.get("chatgpt_answers", [])),
                        },
                    )
                )
        return outputs
