from __future__ import annotations

from polyalign_data.datasets.base import DatasetFormatter
from polyalign_data.datasets.helpers import non_empty_texts
from polyalign_data.remote import hf_dataset_file, load_jsonl
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
        dataset = load_jsonl(hf_dataset_file(self.source_name, "all.jsonl"))
        outputs = {"train": [], "dev": [], "test": []}
        for row_index, row in enumerate(dataset):
            row_id = str(row.get("id", row_index))
            question = row.get("question", "")
            if not question:
                continue
            split = choose_split(
                f"{row.get('source', '')}::{row_id}",
                self.seed,
                [("train", 0.90), ("dev", 0.05), ("test", 0.05)],
            )
            human_answers = non_empty_texts(row.get("human_answers", []))
            for answer_index, human_answer in enumerate(human_answers):
                source_id = f"hc3-chinese-{row.get('source', 'all')}-{row_id}-answer-{answer_index:02d}"
                outputs[split].append(
                    build_record(
                        example_id=source_id,
                        dataset=self.dataset_name,
                        split=split,
                        language=self.language,
                        track="single",
                        family="qa",
                        style_bucket="longform_qa",
                        question=question,
                        context="",
                        dialogue_history=[],
                        human_answer=human_answer,
                        meta={
                            "source_dataset": self.source_name,
                            "source_config": self.config_name,
                            "source_split": "train",
                            "source_id": row_id,
                            "source_subset": row.get("source", ""),
                            "human_answer_index": answer_index,
                            "num_human_answers": len(human_answers),
                            "num_chatgpt_answers": len(row.get("chatgpt_answers", [])),
                        },
                    )
                )
        return outputs
