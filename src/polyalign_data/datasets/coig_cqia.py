from __future__ import annotations

from datasets import load_dataset

from polyalign_data.datasets.base import DatasetFormatter
from polyalign_data.datasets.helpers import is_truthy
from polyalign_data.schema import build_record
from polyalign_data.splits import choose_split


class COIGCQIAFormatter(DatasetFormatter):
    dataset_name = "coig_cqia"
    source_name = "m-a-p/COIG-CQIA"
    language = "zh"
    subsets = (
        "chinese_traditional",
        "coig_pc",
        "douban",
        "exam",
        "finance",
        "human_value",
        "logi_qa",
        "ruozhiba",
        "segmentfault",
        "wiki",
        "wikihow",
        "xhs",
        "zhihu",
    )

    def split_policy(self) -> str:
        return "local deterministic 90/5/5 split over all train-only COIG-CQIA subsets"

    def build_split_records(self) -> dict[str, list[dict]]:
        outputs = {"train": [], "dev": [], "test": []}
        for subset in self.subsets:
            dataset = load_dataset(self.source_name, subset, split="train")
            for row_index, row in enumerate(dataset):
                if row.get("answer_from") and str(row.get("answer_from")).lower() != "human":
                    continue
                if "human_verified" in row and not is_truthy(row.get("human_verified")):
                    continue
                if not row.get("output"):
                    continue
                split = choose_split(
                    f"{subset}::{row['instruction']}::{row['output']}",
                    self.seed,
                    [("train", 0.90), ("dev", 0.05), ("test", 0.05)],
                )
                source_id = f"coig-cqia-{subset}-{row_index:06d}"
                outputs[split].append(
                    build_record(
                        example_id=source_id,
                        dataset=self.dataset_name,
                        split=split,
                        language=self.language,
                        track="single",
                        family="assistant",
                        style_bucket="assistant_like",
                        question=row["instruction"],
                        context=row.get("input", ""),
                        dialogue_history=[],
                        human_answer=row["output"],
                        meta={
                            "source_dataset": self.source_name,
                            "source_subset": subset,
                            "source_split": "train",
                            "source_id": source_id,
                            "task_type": row.get("task_type", {}),
                            "domain": row.get("domain", []),
                            "metadata": row.get("metadata", ""),
                            "answer_from": row.get("answer_from", ""),
                            "human_verified": is_truthy(row.get("human_verified")),
                            "copyright": row.get("copyright", ""),
                        },
                    )
                )
        return outputs
