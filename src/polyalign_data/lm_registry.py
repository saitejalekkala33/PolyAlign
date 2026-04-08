from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ResearchModelSpec:
    alias: str
    model_id: str
    family: str
    parameter_scale: str


RESEARCH_MODELS: dict[str, ResearchModelSpec] = {
    "qwen25_1_5b": ResearchModelSpec(
        alias="qwen25_1_5b",
        model_id="Qwen/Qwen2.5-1.5B",
        family="qwen2.5",
        parameter_scale="1.5B",
    ),
    "qwen25_3b": ResearchModelSpec(
        alias="qwen25_3b",
        model_id="Qwen/Qwen2.5-3B",
        family="qwen2.5",
        parameter_scale="3B",
    ),
    "smollm2_1_7b": ResearchModelSpec(
        alias="smollm2_1_7b",
        model_id="HuggingFaceTB/SmolLM2-1.7B",
        family="smollm2",
        parameter_scale="1.7B",
    ),
    "gemma2_2b": ResearchModelSpec(
        alias="gemma2_2b",
        model_id="google/gemma-2-2b",
        family="gemma2",
        parameter_scale="2B",
    ),
    "gemma_2_2b": ResearchModelSpec(
        alias="gemma_2_2b",
        model_id="google/gemma-2-2b",
        family="gemma2",
        parameter_scale="2B",
    ),
    "llama32_3b": ResearchModelSpec(
        alias="llama32_3b",
        model_id="meta-llama/Llama-3.2-3B",
        family="llama3.2",
        parameter_scale="3B",
    ),
    "llama31_8b": ResearchModelSpec(
        alias="llama31_8b",
        model_id="meta-llama/Llama-3.1-8B",
        family="llama3.1",
        parameter_scale="8B",
    ),
    "qwen25_7b": ResearchModelSpec(
        alias="qwen25_7b",
        model_id="Qwen/Qwen2.5-7B",
        family="qwen2.5",
        parameter_scale="7B",
    ),
}


def model_aliases() -> tuple[str, ...]:
    return tuple(RESEARCH_MODELS.keys())


def resolve_model_aliases(requested: list[str] | None) -> list[ResearchModelSpec]:
    if not requested:
        return []
    selected = list(model_aliases()) if "all" in requested else requested
    seen: set[str] = set()
    resolved: list[ResearchModelSpec] = []
    for alias in selected:
        if alias == "all":
            continue
        if alias not in RESEARCH_MODELS:
            raise ValueError(
                f"Unknown model alias `{alias}`. Available aliases: {', '.join(['all', *model_aliases()])}"
            )
        if alias not in seen:
            resolved.append(RESEARCH_MODELS[alias])
            seen.add(alias)
    return resolved
