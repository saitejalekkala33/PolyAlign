from __future__ import annotations

import io
from itertools import zip_longest

from polyalign_data.datasets.base import DatasetFormatter
from polyalign_data.remote import hf_dataset_file, open_zip
from polyalign_data.schema import build_record
from polyalign_data.text import normalize_text


class DailyDialogFormatter(DatasetFormatter):
    dataset_name = "dailydialog"
    source_name = "roskoN/dailydialog"

    def split_policy(self) -> str:
        return "official splits mapped as train/train, validation/dev, test/test with parity-based role projection"

    def _load_split_rows(self, source_split: str):
        archive = hf_dataset_file(self.source_name, f"{source_split}.zip")
        with open_zip(archive) as zip_file:
            file_names = list(map(str, zip_file.namelist()))
            acts_path = next(name for name in file_names if "act" in name.lower())
            emotions_path = next(name for name in file_names if "emotion" in name.lower())
            utterances_path = next(
                name
                for name in file_names
                if "dialogues" in name.lower() and "act" not in name.lower() and "emotion" not in name.lower()
            )
            acts_file = io.TextIOWrapper(zip_file.open(acts_path), encoding="utf-8")
            emotions_file = io.TextIOWrapper(zip_file.open(emotions_path), encoding="utf-8")
            utterances_file = io.TextIOWrapper(zip_file.open(utterances_path), encoding="utf-8")
            sentinel = object()
            for row_index, triple in enumerate(
                zip_longest(acts_file, emotions_file, utterances_file, fillvalue=sentinel)
            ):
                if sentinel in triple:
                    raise ValueError(f"Misaligned DailyDialog archive for split {source_split}")
                acts_text, emotions_text, utterances_text = triple
                acts = [int(value.strip()) for value in acts_text.strip().split(" ") if value.strip()]
                emotions = [int(value.strip()) for value in emotions_text.strip().split(" ") if value.strip()]
                utterances = [
                    normalize_text(item)
                    for item in utterances_text.strip().strip("__eou__").split("__eou__")
                    if normalize_text(item)
                ]
                if not utterances:
                    continue
                yield row_index, acts, emotions, utterances

    def build_split_records(self) -> dict[str, list[dict]]:
        outputs = {"train": [], "dev": [], "test": []}
        split_map = {"train": "train", "validation": "dev", "test": "test"}
        for source_split, target_split in split_map.items():
            for row_index, acts, emotions, utterances in self._load_split_rows(source_split):
                history: list[dict[str, str]] = []
                for turn_index, utterance in enumerate(utterances):
                    role = "user" if turn_index % 2 == 0 else "assistant"
                    if role == "assistant" and history:
                        source_id = f"dailydialog-{source_split}-{row_index:05d}-turn-{turn_index:03d}"
                        outputs[target_split].append(
                            build_record(
                                example_id=source_id,
                                dataset=self.dataset_name,
                                split=target_split,
                                language=self.language,
                                track="multi",
                                family="dialogue",
                                style_bucket="open_chat",
                                question=history[-1]["text"],
                                context="",
                                dialogue_history=list(history),
                                human_answer=utterance,
                                meta={
                                    "source_dataset": self.source_name,
                                    "source_split": source_split,
                                    "source_id": source_id,
                                    "dialogue_id": f"{source_split}-{row_index:05d}",
                                    "turn_index": turn_index,
                                    "num_history_turns": len(history),
                                    "target_act": acts[turn_index] if turn_index < len(acts) else None,
                                    "target_emotion": emotions[turn_index] if turn_index < len(emotions) else None,
                                    "role_projection": "alternating_roles_starting_with_user",
                                },
                            )
                        )
                    history.append({"role": role, "text": utterance})
        return outputs
