from __future__ import annotations

import json
from typing import Any


SYSTEM_PROMPT = """\
/no_think
You are a careful bilingual NLP evaluator.
Judge the candidate response against the user request, context, dialogue history,
human reference, and PolyAlign bucket metadata.

Do not reveal chain-of-thought. Return one valid JSON object only.
Use the full input text provided by the caller; do not ignore or summarize long context.
"""


def _clean(value: Any) -> str:
    if value is None:
        return ""
    return str(value).replace("\r\n", "\n").replace("\r", "\n").strip()


def _history_lines(history: Any) -> str:
    if not isinstance(history, list):
        return ""
    lines: list[str] = []
    for turn in history:
        if isinstance(turn, dict):
            role = _clean(turn.get("role", ""))
            text = _clean(turn.get("text", turn.get("content", "")))
            if role and text:
                lines.append(f"{role}: {text}")
        elif isinstance(turn, (list, tuple)) and len(turn) == 2:
            user_text = _clean(turn[0])
            assistant_text = _clean(turn[1])
            if user_text:
                lines.append(f"User: {user_text}")
            if assistant_text:
                lines.append(f"Assistant: {assistant_text}")
    return "\n".join(lines)


def _rubric_text(rubric: dict[str, Any]) -> str:
    parts = [
        f"Score range: integers {rubric.get('score_min', 1)} to {rubric.get('score_max', 5)}.",
        "Use 1 for severe failure, 3 for acceptable, and 5 for excellent.",
        "",
        "Dimensions:",
    ]
    for dimension in rubric.get("dimensions", []):
        dim_id = dimension["id"]
        parts.append(f"- {dim_id}: {dimension.get('definition', '').strip()}")
        anchors = dimension.get("anchors", {})
        if anchors:
            parts.append(
                "  Anchors: "
                + "; ".join(f"{score}={text}" for score, text in sorted(anchors.items(), key=lambda item: int(item[0])))
            )
    parts.extend(
        [
            "",
            "Important scoring guidance:",
            "- For extractive QA or fact-grounded tasks, prioritize correctness against the context and reference.",
            "- For open-ended tasks, treat the reference as a strong human exemplar, not the only valid answer.",
            "- For dialogue, preserve turn continuity and answer as a natural participant in that setting.",
            "- Penalize generic assistant boilerplate when a short direct answer is expected.",
            "- Penalize wrong language unless the user explicitly asks for translation or cross-lingual output.",
            "- Empty, copied prompt text, malformed JSON, or refusal on a benign task should score very low.",
        ]
    )
    return "\n".join(parts)


def build_judge_messages(row: dict[str, Any], rubric: dict[str, Any]) -> list[dict[str, str]]:
    metadata = {
        "language": row.get("language", ""),
        "dataset": row.get("dataset", ""),
        "track": row.get("track", ""),
        "family": row.get("family", ""),
        "style_bucket": row.get("style_bucket", ""),
        "length_bin": row.get("length_bin", ""),
        "bucket_id": row.get("bucket_id", ""),
        "source_id": row.get("source_id", ""),
        "model_key": row.get("model_key", ""),
        "stage": row.get("stage", ""),
    }
    history = _history_lines(row.get("history", []))
    context = _clean(row.get("input", ""))
    instruction = _clean(row.get("instruction", ""))
    reference = _clean(row.get("reference_output", row.get("human_text", "")))
    candidate = _clean(row.get("candidate_text", row.get("text", "")))

    user_prompt = f"""\
/no_think
PolyAlign evaluation objective:
The candidate should preserve task utility while matching the human response
distribution appropriate to the current metadata bucket. This means the right
kind of answer for the right language, interaction track, response family,
style bucket, and length bin.

{_rubric_text(rubric)}

Return JSON with exactly these top-level keys:
- scores: an object containing every rubric dimension id as an integer 1-5
- major_errors: an array of short strings
- rationale: one short sentence, max 35 words

Metadata:
{json.dumps(metadata, ensure_ascii=False, indent=2)}

Conversation history:
<history>
{history if history else "[none]"}
</history>

Context:
<context>
{context if context else "[none]"}
</context>

User request:
<instruction>
{instruction}
</instruction>

Human reference response:
<reference>
{reference}
</reference>

Candidate response to judge:
<candidate>
{candidate}
</candidate>

Now judge only the candidate response. Return valid JSON only.
"""
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ]


def judge_json_schema(rubric: dict[str, Any]) -> dict[str, Any]:
    score_properties = {
        dimension["id"]: {
            "type": "integer",
            "minimum": int(rubric.get("score_min", 1)),
            "maximum": int(rubric.get("score_max", 5)),
        }
        for dimension in rubric.get("dimensions", [])
    }
    return {
        "type": "object",
        "properties": {
            "scores": {
                "type": "object",
                "properties": score_properties,
                "required": list(score_properties),
                "additionalProperties": False,
            },
            "major_errors": {
                "type": "array",
                "items": {"type": "string"},
            },
            "rationale": {"type": "string"},
        },
        "required": ["scores", "major_errors", "rationale"],
        "additionalProperties": False,
    }
