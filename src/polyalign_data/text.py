from __future__ import annotations

import re


_TOKEN_RE = re.compile(
    r"[A-Za-z]+(?:'[A-Za-z]+)?|\d+|[\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff]|[^\w\s]",
    re.UNICODE,
)


def normalize_text(text: str | None) -> str:
    if text is None:
        return ""
    raw = str(text).replace("\r\n", "\n").replace("\r", "\n").strip()
    normalized_lines = [" ".join(line.strip().split()) for line in raw.split("\n")]
    collapsed: list[str] = []
    previous_blank = False
    for line in normalized_lines:
        if line:
            collapsed.append(line)
            previous_blank = False
        elif not previous_blank:
            collapsed.append("")
            previous_blank = True
    return "\n".join(collapsed).strip()


def token_count(text: str | None) -> int:
    normalized = normalize_text(text)
    if not normalized:
        return 0
    return len(_TOKEN_RE.findall(normalized))


def length_bin_from_count(count: int) -> str:
    if count <= 40:
        return "short"
    if count <= 120:
        return "medium"
    if count <= 240:
        return "long"
    return "xlong"


def join_non_empty(parts: list[str | None], sep: str = "\n\n") -> str:
    return sep.join(part for part in (normalize_text(item) for item in parts) if part)
