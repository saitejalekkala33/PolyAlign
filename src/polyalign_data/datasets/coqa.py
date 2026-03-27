from __future__ import annotations

from datasets import load_dataset

from polyalign_data.datasets.base import DatasetFormatter
from polyalign_data.schema import build_record
from polyalign_data.splits import choose_split


class CoQAFormatter(DatasetFormatter):
    dataset_name = "coqa"
    source_name = "stanfordnlp/coqa"

    def split_policy(self) -> str:
        return "conversation-level 90/10 split over official train; official validation mapped to test"

    def _append_conversation(
        self,
        outputs: dict[str, list[dict]],
        row: dict,
        source_split: str,
        target_split: str,
        conversation_id: str,
    ) -> None:
        questions = row["questions"]
        answers = row["answers"]["input_text"]
        history: list[dict[str, str]] = []
        for turn_index, (question, answer) in enumerate(zip(questions, answers, strict=False)):
            source_id = f"{conversation_id}-turn-{turn_index:03d}"
            outputs[target_split].append(
                build_record(
                    example_id=source_id,
                    dataset=self.dataset_name,
                    split=target_split,
                    language=self.language,
                    track="multi",
                    family="qa",
                    style_bucket="qa_search",
                    question=question,
                    context=row["story"],
                    dialogue_history=list(history),
                    human_answer=answer,
                    meta={
                        "source_dataset": self.source_name,
                        "source_split": source_split,
                        "source_id": source_id,
                        "conversation_id": conversation_id,
                        "source": row["source"],
                        "turn_index": turn_index,
                        "num_history_turns": len(history),
                        "is_unknown_answer": answer.strip().lower() == "unknown",
                    },
                )
            )
            history.extend(
                [
                    {"role": "user", "text": question},
                    {"role": "assistant", "text": answer},
                ]
            )

    def build_split_records(self) -> dict[str, list[dict]]:
        outputs = {"train": [], "dev": [], "test": []}
        train_dataset = load_dataset(self.source_name, split="train")
        validation_dataset = load_dataset(self.source_name, split="validation")

        for row_index, row in enumerate(train_dataset):
            conversation_id = f"coqa-train-{row_index:05d}"
            split = choose_split(
                row["story"],
                self.seed,
                [("train", 0.90), ("dev", 0.10)],
            )
            self._append_conversation(outputs, row, "train", split, conversation_id)

        for row_index, row in enumerate(validation_dataset):
            conversation_id = f"coqa-validation-{row_index:05d}"
            self._append_conversation(outputs, row, "validation", "test", conversation_id)

        return outputs
