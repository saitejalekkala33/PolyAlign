from __future__ import annotations

import hashlib
from collections.abc import Iterable


def stable_hash_fraction(key: str, seed: int) -> float:
    digest = hashlib.md5(f"{seed}:{key}".encode("utf-8")).hexdigest()
    numerator = int(digest[:8], 16)
    return numerator / 0xFFFFFFFF


def choose_split(key: str, seed: int, weights: Iterable[tuple[str, float]]) -> str:
    value = stable_hash_fraction(key, seed)
    cumulative = 0.0
    last_label = ""
    for label, weight in weights:
        cumulative += weight
        last_label = label
        if value <= cumulative:
            return label
    return last_label
