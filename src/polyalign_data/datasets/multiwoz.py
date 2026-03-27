from __future__ import annotations

from pathlib import Path

from polyalign_data.datasets.base import DatasetFormatter
from polyalign_data.remote import download_url, load_json
from polyalign_data.schema import build_record


class MultiWOZFormatter(DatasetFormatter):
    dataset_name = "multiwoz"
    source_name = "pfb30/multi_woz_v22"
    github_root = "https://github.com/budzianowski/multiwoz/raw/master/data/MultiWOZ_2.2"

    def split_policy(self) -> str:
        return "official MultiWOZ 2.2 splits with only SYSTEM turns used as targets"

    def _cache_path(self, relative_name: str) -> Path:
        return Path(self.cache_dir) / self.dataset_name / relative_name

    def _download_json(self, relative_name: str) -> object:
        destination = self._cache_path(relative_name)
        download_url(f"{self.github_root}/{relative_name}", destination)
        return load_json(destination)

    def _split_files(self, source_split: str) -> list[str]:
        if source_split == "train":
            return [f"train/dialogues_{index:03d}.json" for index in range(1, 18)]
        if source_split == "validation":
            return [f"dev/dialogues_{index:03d}.json" for index in range(1, 3)]
        if source_split == "test":
            return [f"test/dialogues_{index:03d}.json" for index in range(1, 3)]
        raise ValueError(f"Unknown split: {source_split}")

    def _active_intents(self, turn: dict) -> list[str]:
        intents: list[str] = []
        for frame in turn.get("frames", []):
            state = frame.get("state", {})
            intent = state.get("active_intent")
            if intent and intent != "NONE":
                intents.append(intent)
        return intents

    def _dialogue_act_types(self, turn: dict, dialogue_acts: dict) -> list[str]:
        turn_id = turn["turn_id"]
        turn_acts = dialogue_acts.get(turn_id, {}).get("dialog_act", {})
        return list(turn_acts.keys())

    def build_split_records(self) -> dict[str, list[dict]]:
        outputs = {"train": [], "dev": [], "test": []}
        split_map = {"train": "train", "validation": "dev", "test": "test"}
        dialogue_acts = self._download_json("dialog_acts.json")
        for source_split, target_split in split_map.items():
            for relative_name in self._split_files(source_split):
                dialogues = self._download_json(relative_name)
                for dialogue in dialogues:
                    dialogue_id = dialogue["dialogue_id"]
                    history: list[dict[str, str]] = []
                    stored_acts = dialogue_acts.get(dialogue_id, {})
                    for turn_index, turn in enumerate(dialogue["turns"]):
                        role = "user" if turn["speaker"] == "USER" else "assistant"
                        utterance = turn["utterance"]
                        if role == "assistant" and history:
                            latest_user = next(
                                (item["text"] for item in reversed(history) if item["role"] == "user"),
                                "",
                            )
                            source_id = f"multiwoz-{source_split}-{dialogue_id}-turn-{turn_index:03d}"
                            outputs[target_split].append(
                                build_record(
                                    example_id=source_id,
                                    dataset=self.dataset_name,
                                    split=target_split,
                                    language=self.language,
                                    track="multi",
                                    family="dialogue",
                                    style_bucket="task_dialogue",
                                    question=latest_user,
                                    context="",
                                    dialogue_history=list(history),
                                    human_answer=utterance,
                                    meta={
                                        "source_dataset": self.source_name,
                                        "source_split": source_split,
                                        "source_id": source_id,
                                        "dialogue_id": dialogue_id,
                                        "services": dialogue.get("services", []),
                                        "turn_id": turn["turn_id"],
                                        "turn_index": turn_index,
                                        "num_history_turns": len(history),
                                        "active_intents": self._active_intents(turn),
                                        "dialogue_act_types": self._dialogue_act_types(turn, stored_acts),
                                    },
                                )
                            )
                        history.append({"role": role, "text": utterance})
        return outputs
