from __future__ import annotations

from datasets import load_dataset

from polyalign_data.datasets.base import DatasetFormatter
from polyalign_data.datasets.helpers import non_empty_texts
from polyalign_data.schema import build_record


class DRCDFormatter(DatasetFormatter):
    dataset_name = "drcd"
    source_name = "voidful/DRCD"
    language = "zh"

    def split_policy(self) -> str:
        return "official HF splits mapped as train/train, dev/dev, test/test with paragraph-level QA flattening"

    def _append_paragraph_rows(
        self,
        outputs: dict[str, list[dict]],
        row: dict,
        source_split: str,
        target_split: str,
    ) -> None:
        article_id = row.get("id", "")
        title = row.get("title", "")
        for paragraph in row.get("paragraphs", []):
            context = paragraph.get("context", "")
            paragraph_id = paragraph.get("id", "")
            for qa in paragraph.get("qas", []):
                answer_texts = non_empty_texts(answer.get("text") for answer in qa.get("answers", []))
                if not answer_texts:
                    continue
                human_answer = answer_texts[0]
                source_id = f"drcd-{source_split}-{qa['id']}"
                outputs[target_split].append(
                    build_record(
                        example_id=source_id,
                        dataset=self.dataset_name,
                        split=target_split,
                        language=self.language,
                        track="single",
                        family="qa",
                        style_bucket="qa_search",
                        question=qa["question"],
                        context=context,
                        dialogue_history=[],
                        human_answer=human_answer,
                        meta={
                            "source_dataset": self.source_name,
                            "source_split": source_split,
                            "source_id": qa["id"],
                            "article_id": article_id,
                            "paragraph_id": paragraph_id,
                            "title": title,
                            "answer_count": len(answer_texts),
                        },
                    )
                )

    def build_split_records(self) -> dict[str, list[dict]]:
        outputs = {"train": [], "dev": [], "test": []}
        for split_name in outputs:
            dataset = load_dataset(self.source_name, split=split_name)
            for row in dataset:
                self._append_paragraph_rows(outputs, row, split_name, split_name)
        return outputs
