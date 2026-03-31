from __future__ import annotations

from collections.abc import Iterable
from typing import Any

from polyalign_data.text import normalize_text


CHINESE_LANG_CODES = {
    "zh",
    "zh-cn",
    "zh_cn",
    "zh-hans",
    "zh_hans",
    "zh-sg",
    "zh_sg",
    "zh-tw",
    "zh_tw",
    "zh-hant",
    "zh_hant",
    "zh-hk",
    "zh_hk",
    "zh-mo",
    "zh_mo",
}


def is_truthy(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    if isinstance(value, (int, float)):
        return value != 0
    normalized = normalize_text(value).lower()
    return normalized in {"1", "true", "yes", "y", "t"}


def is_chinese_lang(value: str | None) -> bool:
    normalized = normalize_text(value).lower()
    if not normalized:
        return False
    return normalized in CHINESE_LANG_CODES or normalized.startswith("zh-") or normalized.startswith("zh_")


def non_empty_texts(values: Iterable[Any] | None) -> list[str]:
    if values is None:
        return []
    cleaned: list[str] = []
    for value in values:
        normalized = normalize_text(value)
        if normalized:
            cleaned.append(normalized)
    return cleaned


def first_non_empty_text(values: Iterable[Any] | None) -> str:
    texts = non_empty_texts(values)
    return texts[0] if texts else ""
