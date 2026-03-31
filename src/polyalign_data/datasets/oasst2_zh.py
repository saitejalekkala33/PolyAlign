from __future__ import annotations

from datasets import load_dataset

from polyalign_data.datasets.base import DatasetFormatter
from polyalign_data.datasets.helpers import is_chinese_lang, is_truthy
from polyalign_data.schema import build_record
from polyalign_data.splits import choose_split
from polyalign_data.text import normalize_text


ROLE_MAP = {"prompter": "user", "assistant": "assistant"}


class OASST2ChineseFormatter(DatasetFormatter):
    dataset_name = "oasst2_zh"
    source_name = "OpenAssistant/oasst2"
    language = "zh"

    def split_policy(self) -> str:
        return "tree-level 90/10 split over official train and official validation mapped to test; only reviewed Chinese branches kept"

    def _is_row_usable(self, row: dict) -> bool:
        if normalize_text(row.get("tree_state")) != "ready_for_export":
            return False
        if is_truthy(row.get("deleted")):
            return False
        if not is_truthy(row.get("review_result")):
            return False
        if not is_chinese_lang(row.get("lang")):
            return False
        if normalize_text(row.get("role")) not in ROLE_MAP:
            return False
        return bool(normalize_text(row.get("text")))

    def _ancestry(self, rows_by_id: dict[str, dict], row: dict) -> list[dict] | None:
        chain: list[dict] = []
        current = row
        visited: set[str] = set()
        while current is not None:
            message_id = current["message_id"]
            if message_id in visited:
                return None
            visited.add(message_id)
            if not self._is_row_usable(current):
                return None
            chain.append(current)
            parent_id = current.get("parent_id")
            current = rows_by_id.get(parent_id) if parent_id else None
        chain.reverse()
        return chain

    def _append_assistant_turn(
        self,
        outputs: dict[str, list[dict]],
        rows_by_id: dict[str, dict],
        row: dict,
        source_split: str,
        target_split: str,
    ) -> None:
        if normalize_text(row.get("role")) != "assistant":
            return
        ancestry = self._ancestry(rows_by_id, row)
        if not ancestry or len(ancestry) < 2:
            return
        prompt_row = ancestry[-2]
        if normalize_text(prompt_row.get("role")) != "prompter":
            return
        history = [
            {"role": ROLE_MAP[item["role"]], "text": item["text"]}
            for item in ancestry[:-2]
            if item.get("role") in ROLE_MAP
        ]
        source_id = f"oasst2-zh-{source_split}-{row['message_id']}"
        outputs[target_split].append(
            build_record(
                example_id=source_id,
                dataset=self.dataset_name,
                split=target_split,
                language=self.language,
                track="multi",
                family="dialogue",
                style_bucket="open_chat",
                question=prompt_row["text"],
                context="",
                dialogue_history=history,
                human_answer=row["text"],
                meta={
                    "source_dataset": self.source_name,
                    "source_split": source_split,
                    "source_id": row["message_id"],
                    "message_tree_id": row.get("message_tree_id", ""),
                    "parent_id": row.get("parent_id"),
                    "rank": row.get("rank"),
                    "created_date": row.get("created_date"),
                    "review_count": row.get("review_count"),
                    "tree_state": row.get("tree_state"),
                    "num_history_turns": len(history),
                },
            )
        )

    def build_split_records(self) -> dict[str, list[dict]]:
        outputs = {"train": [], "dev": [], "test": []}
        split_map = {"train": None, "validation": "test"}
        for source_split, fixed_target_split in split_map.items():
            dataset = load_dataset(self.source_name, split=source_split)
            rows = [row for row in dataset]
            rows_by_id = {row["message_id"]: row for row in rows}
            for row in rows:
                if fixed_target_split is None:
                    target_split = choose_split(
                        row.get("message_tree_id", row["message_id"]),
                        self.seed,
                        [("train", 0.90), ("dev", 0.10)],
                    )
                else:
                    target_split = fixed_target_split
                self._append_assistant_turn(outputs, rows_by_id, row, source_split, target_split)
        return outputs
