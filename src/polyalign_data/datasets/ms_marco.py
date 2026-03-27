from __future__ import annotations

from datasets import load_dataset

from polyalign_data.datasets.base import DatasetFormatter
from polyalign_data.schema import build_record
from polyalign_data.text import join_non_empty


class MSMARCOFormatter(DatasetFormatter):
    dataset_name = "ms_marco"
    source_name = "microsoft/ms_marco"
    config_name = "v1.1"

    def split_policy(self) -> str:
        return "official HF splits mapped as train/train, validation/dev, test/test"

    def extra_manifest(self) -> dict[str, object]:
        return {"config": self.config_name}

    def _choose_answer(self, row: dict) -> tuple[str, bool]:
        well_formed = [answer for answer in row.get("wellFormedAnswers", []) if answer]
        if well_formed:
            return well_formed[0], True
        answers = [answer for answer in row.get("answers", []) if answer]
        if answers:
            return answers[0], False
        return "", False

    def _build_context(self, passages: dict) -> tuple[str, int]:
        texts = passages.get("passage_text", [])
        selected_flags = passages.get("is_selected", [])
        selected = [text for text, flag in zip(texts, selected_flags, strict=False) if flag]
        if selected:
            return join_non_empty(selected, sep="\n\n"), len(selected)
        fallback = [text for text in texts[:3] if text]
        return join_non_empty(fallback, sep="\n\n"), 0

    def build_split_records(self) -> dict[str, list[dict]]:
        outputs = {"train": [], "dev": [], "test": []}
        split_map = {"train": "train", "validation": "dev", "test": "test"}
        for source_split, target_split in split_map.items():
            dataset = load_dataset(self.source_name, self.config_name, split=source_split)
            for row in dataset:
                answer, used_well_formed = self._choose_answer(row)
                if not answer:
                    continue
                context, selected_passage_count = self._build_context(row["passages"])
                source_id = f"ms-marco-{source_split}-{row['query_id']}"
                outputs[target_split].append(
                    build_record(
                        example_id=source_id,
                        dataset=self.dataset_name,
                        split=target_split,
                        language=self.language,
                        track="single",
                        family="qa",
                        style_bucket="qa_search",
                        question=row["query"],
                        context=context,
                        dialogue_history=[],
                        human_answer=answer,
                        meta={
                            "source_dataset": self.source_name,
                            "source_config": self.config_name,
                            "source_split": source_split,
                            "source_id": source_id,
                            "query_id": row["query_id"],
                            "query_type": row["query_type"],
                            "used_well_formed_answer": used_well_formed,
                            "selected_passage_count": selected_passage_count,
                            "passage_count": len(row["passages"].get("passage_text", [])),
                        },
                    )
                )
        return outputs
