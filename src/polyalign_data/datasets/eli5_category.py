from __future__ import annotations

from pathlib import Path

from polyalign_data.datasets.base import DatasetFormatter
from polyalign_data.remote import download_url, load_gzip_json
from polyalign_data.schema import build_record
from polyalign_data.text import join_non_empty


class ELI5CategoryFormatter(DatasetFormatter):
    dataset_name = "eli5_category"
    source_name = "rexarski/eli5_category"
    base_url = "https://jingshensn2.github.io/eli5c/datasets"
    source_files = {
        "train": "eli5-category-train.json.gz",
        "validation1": "eli5-category-validation-1.json.gz",
        "validation2": "eli5-category-validation-2.json.gz",
        "test": "eli5-category-test.json.gz",
    }
    split_map = {
        "train": "train",
        "validation1": "dev",
        "validation2": "validation2",
        "test": "test",
    }

    def split_policy(self) -> str:
        return "official ELI5-Category splits with validation1->dev and validation2 preserved as auxiliary"

    def _download_split(self, source_split: str) -> list[dict]:
        filename = self.source_files[source_split]
        destination = Path(self.cache_dir) / self.dataset_name / filename
        download_url(f"{self.base_url}/{filename}", destination)
        return load_gzip_json(destination)

    def _best_answer(self, row: dict) -> tuple[str, int | None]:
        answers = row.get("answers", {})
        texts = answers.get("text", [])
        scores = answers.get("score", [])
        best_text = ""
        best_score: int | None = None
        for text, score in zip(texts, scores, strict=False):
            if not text:
                continue
            if best_score is None or score > best_score:
                best_text = text
                best_score = score
        return best_text, best_score

    def build_split_records(self) -> dict[str, list[dict]]:
        outputs = {"train": [], "dev": [], "validation2": [], "test": []}
        for source_split, target_split in self.split_map.items():
            rows = self._download_split(source_split)
            for row_index, row in enumerate(rows):
                answer, best_score = self._best_answer(row)
                if not answer:
                    continue
                source_id = f"eli5-{source_split}-{row_index:06d}"
                outputs[target_split].append(
                    build_record(
                        example_id=source_id,
                        dataset=self.dataset_name,
                        split=target_split,
                        language=self.language,
                        track="single",
                        family="qa",
                        style_bucket="longform_qa",
                        question=join_non_empty([row["title"], row.get("selftext", "")]),
                        context="",
                        dialogue_history=[],
                        human_answer=answer,
                        meta={
                            "source_dataset": self.source_name,
                            "source_split": source_split,
                            "source_id": row.get("q_id", source_id),
                            "subreddit": row.get("subreddit", ""),
                            "category": row.get("category", ""),
                            "num_candidate_answers": len(row.get("answers", {}).get("text", [])),
                            "best_answer_score": best_score,
                        },
                    )
                )
        return outputs
