from __future__ import annotations

import math
from pathlib import Path
from statistics import pstdev
from typing import Any

from polyalign_data.io_utils import ensure_dir
from polyalign_data.lm_registry import ResearchModelSpec
from polyalign_data.text import normalize_text

DEFAULT_LM_MAX_SEQ_LENGTH = 4096
ROLE_LABELS = {
    "user": "User",
    "assistant": "Assistant",
    "system": "System",
}


def _safe_div(numerator: float, denominator: float) -> float:
    if denominator == 0:
        return 0.0
    return numerator / denominator


def _safe_exp(value: float) -> float:
    return math.exp(min(value, 50.0))


def _round(value: float | int | bool) -> float | int | bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, float):
        return round(value, 6)
    return value


def _canonical_role(role: str | None) -> str:
    normalized = normalize_text(role).lower()
    return ROLE_LABELS.get(normalized, normalized.title() or "Unknown")


def normalized_history(record: dict[str, Any]) -> list[dict[str, str]]:
    history = record.get("dialogue_history") or []
    question = normalize_text(record.get("question") or record.get("prompt") or "")
    normalized: list[dict[str, str]] = []
    for turn in history:
        if not isinstance(turn, dict):
            continue
        text = normalize_text(turn.get("text", ""))
        if not text:
            continue
        role = normalize_text(turn.get("role", "")).lower()
        normalized.append({"role": role, "text": text})
    if normalized and question:
        last_turn = normalized[-1]
        if last_turn["role"] == "user" and last_turn["text"] == question:
            normalized = normalized[:-1]
    return normalized


def build_condition_prefix(record: dict[str, Any]) -> str:
    parts: list[str] = []
    context = normalize_text(record.get("context", ""))
    history = normalized_history(record)
    question = normalize_text(record.get("question") or record.get("prompt") or "")

    if context:
        parts.append(f"Context:\n{context}")
    if history:
        history_lines = [f"{_canonical_role(turn['role'])}: {turn['text']}" for turn in history]
        parts.append("Dialogue History:\n" + "\n".join(history_lines))
    if question:
        parts.append(f"Current User Message:\n{question}")

    if not parts:
        return ""
    return "\n\n".join(parts) + "\n\nAssistant Response:\n"


class HFLMFeatureExtractor:
    def __init__(
        self,
        model_spec: ResearchModelSpec,
        *,
        device: str = "auto",
        dtype: str = "auto",
        max_seq_length: int = DEFAULT_LM_MAX_SEQ_LENGTH,
        trust_remote_code: bool = False,
    ) -> None:
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as exc:
            raise RuntimeError(
                "Model-based feature extraction requires `torch` and `transformers`. "
                "Install them before using --lm-model."
            ) from exc

        self.torch = torch
        self.model_spec = model_spec
        self.device = self._resolve_device(device)
        self.torch_dtype = self._resolve_dtype(dtype)
        self.max_seq_length = max_seq_length
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_spec.model_id,
            trust_remote_code=trust_remote_code,
            use_fast=True,
        )
        if self.tokenizer.pad_token_id is None:
            if self.tokenizer.eos_token is not None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            elif self.tokenizer.bos_token is not None:
                self.tokenizer.pad_token = self.tokenizer.bos_token
        self.model = AutoModelForCausalLM.from_pretrained(
            model_spec.model_id,
            trust_remote_code=trust_remote_code,
            torch_dtype=self.torch_dtype,
        )
        self.model.to(self.device)
        self.model.eval()
        self.start_token_ids = self._resolve_start_token_ids()
        self.max_seq_length = self._resolve_max_seq_length(max_seq_length)

    def _resolve_device(self, device: str) -> str:
        if device != "auto":
            return device
        return "cuda" if self.torch.cuda.is_available() else "cpu"

    def _resolve_dtype(self, dtype: str):
        if dtype == "float16":
            return self.torch.float16
        if dtype == "bfloat16":
            return self.torch.bfloat16
        if dtype == "float32":
            return self.torch.float32
        if self.device.startswith("cuda"):
            return self.torch.float16
        return self.torch.float32

    def _resolve_start_token_ids(self) -> list[int]:
        if self.tokenizer.bos_token_id is not None:
            return [int(self.tokenizer.bos_token_id)]
        if self.tokenizer.eos_token_id is not None:
            return [int(self.tokenizer.eos_token_id)]
        return []

    def _resolve_max_seq_length(self, requested: int) -> int:
        model_limit = getattr(self.tokenizer, "model_max_length", None)
        if isinstance(model_limit, int) and 0 < model_limit < 1_000_000:
            return min(requested, model_limit)
        config_limit = getattr(getattr(self.model, "config", None), "max_position_embeddings", None)
        if isinstance(config_limit, int) and config_limit > 0:
            return min(requested, config_limit)
        return requested

    def _encode(self, text: str) -> list[int]:
        normalized = normalize_text(text)
        if not normalized:
            return []
        return self.tokenizer(normalized, add_special_tokens=False).input_ids

    def _truncate_sequences(self, prefix_ids: list[int], answer_ids: list[int]) -> tuple[list[int], list[int], dict[str, int | bool]]:
        prefix = list(prefix_ids)
        answer = list(answer_ids)
        available_tokens = max(1, self.max_seq_length - len(self.start_token_ids))
        total_tokens = len(prefix) + len(answer)
        if total_tokens <= available_tokens:
            return prefix, answer, {"lm_truncated": False, "lm_prefix_tokens_dropped": 0, "lm_answer_tokens_dropped": 0}

        overflow = total_tokens - available_tokens
        prefix_drop = min(len(prefix), overflow)
        if prefix_drop:
            prefix = prefix[prefix_drop:]
            overflow -= prefix_drop
        answer_drop = min(len(answer), overflow)
        if answer_drop:
            answer = answer[answer_drop:]

        return prefix, answer, {
            "lm_truncated": True,
            "lm_prefix_tokens_dropped": prefix_drop,
            "lm_answer_tokens_dropped": answer_drop,
        }

    def _summarize_log_probs(
        self,
        log_probs: list[float],
        *,
        available: bool,
        answer_token_count: int,
        prefix_token_count: int,
        dropped_stats: dict[str, int | bool],
        condition_text_available: bool,
    ) -> dict[str, Any]:
        scored_token_count = len(log_probs)
        nll_values = [-value for value in log_probs]
        nll_sum = sum(nll_values)
        mean_nll = _safe_div(nll_sum, scored_token_count)
        ppl = _safe_exp(mean_nll) if scored_token_count else 0.0
        bits_per_token = _safe_div(mean_nll, math.log(2)) if scored_token_count else 0.0
        feature_block = {
            "available": int(available),
            "condition_text_available": int(condition_text_available),
            "answer_token_count": answer_token_count,
            "prefix_token_count": prefix_token_count,
            "scored_token_count": scored_token_count,
            "score_coverage": _safe_div(scored_token_count, answer_token_count),
            "logprob_sum": sum(log_probs),
            "logprob_mean": _safe_div(sum(log_probs), scored_token_count),
            "nll_sum": nll_sum,
            "nll_mean": mean_nll,
            "perplexity": ppl,
            "bits_per_token": bits_per_token,
            "surprisal_mean": mean_nll,
            "surprisal_std": pstdev(nll_values) if len(nll_values) > 1 else 0.0,
            "surprisal_min": min(nll_values) if nll_values else 0.0,
            "surprisal_max": max(nll_values) if nll_values else 0.0,
            **dropped_stats,
        }
        return {name: _round(value) for name, value in feature_block.items()}

    def _score_target(self, prefix_text: str, answer_text: str) -> dict[str, Any]:
        answer_ids = self._encode(answer_text)
        prefix_ids = self._encode(prefix_text)
        truncated_prefix, truncated_answer, dropped_stats = self._truncate_sequences(prefix_ids, answer_ids)
        answer_token_count = len(truncated_answer)
        prefix_token_count = len(truncated_prefix)

        if not truncated_answer:
            return self._summarize_log_probs(
                [],
                available=False,
                answer_token_count=0,
                prefix_token_count=prefix_token_count,
                dropped_stats=dropped_stats,
                condition_text_available=bool(prefix_text),
            )

        input_ids = self.start_token_ids + truncated_prefix + truncated_answer
        if len(input_ids) < 2:
            return self._summarize_log_probs(
                [],
                available=False,
                answer_token_count=answer_token_count,
                prefix_token_count=prefix_token_count,
                dropped_stats=dropped_stats,
                condition_text_available=bool(prefix_text),
            )

        answer_start = len(self.start_token_ids) + len(truncated_prefix)
        score_start = max(answer_start - 1, 0)
        input_tensor = self.torch.tensor([input_ids], dtype=self.torch.long, device=self.device)
        with self.torch.inference_mode():
            outputs = self.model(input_ids=input_tensor)
            shifted_logits = outputs.logits[:, :-1, :]
            shifted_labels = input_tensor[:, 1:]
            token_log_probs = self.torch.log_softmax(shifted_logits, dim=-1)
            gathered = token_log_probs.gather(-1, shifted_labels.unsqueeze(-1)).squeeze(-1)[0]
        answer_log_probs = gathered[score_start:].detach().cpu().tolist()

        return self._summarize_log_probs(
            answer_log_probs,
            available=True,
            answer_token_count=answer_token_count,
            prefix_token_count=prefix_token_count,
            dropped_stats=dropped_stats,
            condition_text_available=bool(prefix_text),
        )

    def score_record(self, record: dict[str, Any], *, text_field: str = "human_answer") -> dict[str, Any]:
        answer_text = normalize_text(record.get(text_field, ""))
        prefix_text = build_condition_prefix(record)

        unconditional = self._score_target("", answer_text)
        conditional = self._score_target(prefix_text, answer_text) if prefix_text else self._summarize_log_probs(
            [],
            available=False,
            answer_token_count=unconditional["answer_token_count"],
            prefix_token_count=0,
            dropped_stats={"lm_truncated": False, "lm_prefix_tokens_dropped": 0, "lm_answer_tokens_dropped": 0},
            condition_text_available=False,
        )

        combined = {
            "lm_answer_char_count": len(answer_text),
            "lm_prefix_char_count": len(prefix_text),
            "lm_max_seq_length": self.max_seq_length,
        }
        for prefix, block in (("lm_unconditional", unconditional), ("lm_conditional", conditional)):
            for name, value in block.items():
                combined[f"{prefix}_{name}"] = value

        logprob_gain = 0.0
        ppl_ratio = 0.0
        if unconditional["available"] and conditional["available"]:
            logprob_gain = conditional["logprob_sum"] - unconditional["logprob_sum"]
            ppl_ratio = _safe_div(unconditional["perplexity"], conditional["perplexity"])
        combined["lm_conditional_logprob_gain"] = _round(logprob_gain)
        combined["lm_conditional_perplexity_ratio"] = _round(ppl_ratio)
        return combined


def derive_model_output_paths(
    input_file: Path,
    *,
    output_root: Path,
    model_alias: str,
    text_field: str,
    base_jsonl_name: str | None,
    write_csv: bool,
    base_csv_name: str | None,
) -> tuple[Path, Path | None]:
    model_root = output_root / model_alias
    ensure_dir(model_root)
    stem = f"{input_file.stem}_{text_field}_features"
    jsonl_name = base_jsonl_name or f"{stem}.jsonl"
    csv_name = None
    if write_csv:
        csv_name = base_csv_name or f"{Path(jsonl_name).stem}.csv"
    return model_root / jsonl_name, (model_root / csv_name) if csv_name else None
