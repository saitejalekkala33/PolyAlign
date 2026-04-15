from __future__ import annotations

import argparse
import json
import random
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib import error, request

from tqdm.auto import tqdm

DEFAULT_SYSTEM_PROMPT = ""

STYLE_BUCKET_GUIDANCE = {
    "en": {
        "assistant_like": "Reason clearly, then give the result.",
        "longform_qa": "Give an explicit multi-step explanation before concluding.",
        "open_chat": "Keep the tone natural while still reasoning step by step.",
        "qa_search": "Ground the reasoning in the provided context and avoid unsupported claims.",
        "task_dialogue": "Reason in a task-oriented way with clear constraints and next steps.",
    },
    "zh": {
        "assistant_like": "先清楚推理，再给出结果。",
        "longform_qa": "先给出明确的多步说明，再下结论。",
        "open_chat": "语气自然，但保持逐步推理。",
        "qa_search": "基于给定上下文进行推理，不要补充没有依据的内容。",
        "task_dialogue": "围绕任务目标、约束和下一步进行推理。",
    },
}

LENGTH_BIN_GUIDANCE = {
    "en": {
        "short": "Keep the reasoning compact and the final answer brief.",
        "medium": "Use a concise but complete explanation before the final answer.",
        "long": "Provide a detailed multi-step explanation before the final answer.",
        "xlong": "Provide a thorough multi-step explanation with careful justification before the final answer.",
    },
    "zh": {
        "short": "推理要简洁，最终答案要简短。",
        "medium": "先给出简洁但完整的说明，再给出最终答案。",
        "long": "先给出详细的多步说明，再给出最终答案。",
        "xlong": "给出充分而细致的多步推理，并谨慎说明依据后再作答。",
    },
}

BUCKET_MAX_TOKENS = {
    "short": 256,
    "medium": 512,
    "long": 1024,
    "xlong": 1536,
}


@dataclass(frozen=True)
class PromptLocale:
    task_header: str
    history_header: str
    context_header: str
    question_header: str
    format_header: str
    reasoning_label: str
    final_answer_label: str
    user_label: str
    assistant_label: str
    chat_system_prompt: str
    task_requirements: tuple[str, ...]
    style_prefix: str
    length_prefix: str
    start_now: str


PROMPT_LOCALES = {
    "en": PromptLocale(
        task_header="Task:",
        history_header="Conversation History:",
        context_header="Context:",
        question_header="Question:",
        format_header="Write the response exactly in this format:",
        reasoning_label="Reasoning:",
        final_answer_label="Final Answer:",
        user_label="User:",
        assistant_label="Assistant:",
        chat_system_prompt="Follow the user's format exactly and do not echo the prompt.",
        task_requirements=(
            "Answer the question in the requested style and length.",
            "Use the provided context and dialogue history when relevant.",
            "Stay grounded in the given information and do not copy the instructions.",
        ),
        style_prefix="Answer style: ",
        length_prefix="Answer length: ",
        start_now="Write the answer now.",
    ),
    "zh": PromptLocale(
        task_header="任务要求：",
        history_header="对话历史：",
        context_header="上下文：",
        question_header="问题：",
        format_header="请严格按下面格式作答：",
        reasoning_label="推理：",
        final_answer_label="最终答案：",
        user_label="用户：",
        assistant_label="助手：",
        chat_system_prompt="严格遵守用户给出的格式要求，不要复述提示词。",
        task_requirements=(
            "按要求的风格和长度回答问题。",
            "需要时依据给定上下文和历史对话作答。",
            "只根据给定信息作答，不要复述这些说明。",
        ),
        style_prefix="作答风格：",
        length_prefix="作答长度：",
        start_now="现在开始作答。",
    ),
}


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    ensure_dir(path.parent)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def iter_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                yield json.loads(line)


def load_records(path: Path) -> list[dict[str, Any]]:
    if path.suffix == ".jsonl":
        return list(iter_jsonl(path))
    return json.loads(path.read_text(encoding="utf-8"))


def first_nonempty_string(example: dict[str, Any], *keys: str) -> str:
    for key in keys:
        value = example.get(key, "")
        if isinstance(value, str):
            text = value.strip()
            if text:
                return text
    return ""


def get_question(example: dict[str, Any]) -> str:
    return first_nonempty_string(example, "question", "instruction")


def get_context(example: dict[str, Any]) -> str:
    return first_nonempty_string(example, "context", "input")


def get_reference_answer(example: dict[str, Any]) -> str:
    return first_nonempty_string(example, "human_answer", "output", "reference_output")


def get_dialogue_history(example: dict[str, Any]) -> list[dict[str, str]]:
    dialogue_history = example.get("dialogue_history")
    if isinstance(dialogue_history, list):
        normalized: list[dict[str, str]] = []
        for turn in dialogue_history:
            if not isinstance(turn, dict):
                continue
            role = str(turn.get("role", "")).strip()
            text = str(turn.get("text", "")).strip()
            if role in {"user", "assistant"} and text:
                normalized.append({"role": role, "text": text})
        if normalized:
            return normalized

    history = example.get("history")
    if isinstance(history, list):
        normalized = []
        for turn in history:
            if not isinstance(turn, list) or len(turn) != 2:
                continue
            user_text = str(turn[0]).strip()
            assistant_text = str(turn[1]).strip()
            if user_text:
                normalized.append({"role": "user", "text": user_text})
            if assistant_text:
                normalized.append({"role": "assistant", "text": assistant_text})
        return normalized

    return []


def validate_current_record(example: dict[str, Any], *, input_path: Path) -> None:
    checks = {
        "id": first_nonempty_string(example, "id"),
        "question_or_instruction": get_question(example),
        "human_answer_or_output": get_reference_answer(example),
        "bucket_id": first_nonempty_string(example, "bucket_id"),
        "style_bucket": first_nonempty_string(example, "style_bucket"),
    }
    missing = [field for field, value in checks.items() if not value]
    if missing:
        raise ValueError(
            f"{input_path} does not look like a current-format PolyAlign record with bucket metadata. "
            f"Missing fields: {', '.join(missing)}. Use current/*.jsonl for CoT bucket evaluation."
        )


def flatten_dialogue_history(history: list[dict[str, str]] | None) -> list[list[str]]:
    turns = list(history or [])
    pairs: list[list[str]] = []
    pending_user: str | None = None
    for turn in turns:
        role = str(turn.get("role", "")).strip()
        text = str(turn.get("text", "")).strip()
        if not text:
            continue
        if role == "user":
            pending_user = text
        elif role == "assistant" and pending_user is not None:
            pairs.append([pending_user, text])
            pending_user = None
    return pairs


def history_for_prompt(example: dict[str, Any]) -> list[dict[str, str]]:
    history = list(get_dialogue_history(example))
    question = get_question(example)
    if history:
        last_turn = history[-1]
        if (
            isinstance(last_turn, dict)
            and str(last_turn.get("role", "")).strip() == "user"
            and str(last_turn.get("text", "")).strip() == question
        ):
            history = history[:-1]
    return history


def normalize_language_tag(value: Any) -> str | None:
    text = str(value or "").strip().lower()
    if not text:
        return None
    if text.startswith("zh"):
        return "zh"
    if text.startswith("en"):
        return "en"
    return None


def resolve_prompt_language(example: dict[str, Any], requested_language: str) -> str:
    if requested_language in PROMPT_LOCALES:
        return requested_language

    for candidate in (
        example.get("language"),
        str(example.get("bucket_id", "")).split("|", 1)[0],
    ):
        normalized = normalize_language_tag(candidate)
        if normalized is not None:
            return normalized
    return "en"


def resolve_style_guidance(example: dict[str, Any], language: str) -> str:
    style_bucket = str(example.get("style_bucket", "")).strip()
    return STYLE_BUCKET_GUIDANCE[language].get(
        style_bucket,
        "Match the requested answer style closely and reason explicitly before concluding."
        if language == "en"
        else "尽量贴合要求的回答风格，并在结论前给出清楚推理。",
    )


def resolve_length_guidance(example: dict[str, Any], language: str) -> str:
    length_bin = str(example.get("length_bin", "")).strip()
    return LENGTH_BIN_GUIDANCE[language].get(
        length_bin,
        "Match the requested answer length while keeping the response coherent."
        if language == "en"
        else "在保持连贯的前提下，匹配要求的回答长度。",
    )


def build_task_block(
    example: dict[str, Any],
    *,
    prompt_language: str,
    extra_instruction: str,
) -> tuple[str, str]:
    language = resolve_prompt_language(example, prompt_language)
    locale = PROMPT_LOCALES[language]

    task_lines = list(locale.task_requirements)
    task_lines.append(locale.style_prefix + resolve_style_guidance(example, language))
    task_lines.append(locale.length_prefix + resolve_length_guidance(example, language))

    parts: list[str] = [
        locale.task_header,
        "\n".join(f"- {line}" for line in task_lines),
    ]

    if extra_instruction.strip():
        parts.append(extra_instruction.strip())

    history = history_for_prompt(example)
    if history:
        history_lines: list[str] = []
        for turn in history:
            role = str(turn.get("role", "")).strip()
            text = str(turn.get("text", "")).strip()
            if not text:
                continue
            if role == "user":
                history_lines.append(f"{locale.user_label} {text}")
            elif role == "assistant":
                history_lines.append(f"{locale.assistant_label} {text}")
        if history_lines:
            parts.append(locale.history_header + "\n" + "\n".join(history_lines))

    context = get_context(example)
    if context:
        parts.append(locale.context_header + "\n" + context)

    question = get_question(example)
    if question:
        parts.append(locale.question_header + "\n" + question)
    else:
        raise ValueError("Record is missing both `question` and `instruction`.")

    parts.append(
        "\n".join(
            [
                locale.format_header,
                f"{locale.reasoning_label} <step-by-step reasoning that matches the requested bucket>",
                f"{locale.final_answer_label} <final answer only>",
            ]
        )
    )
    return "\n\n".join(parts), language


def build_plain_prompt(
    example: dict[str, Any],
    *,
    prompt_language: str,
    extra_instruction: str,
) -> tuple[str, str]:
    task_block, language = build_task_block(
        example,
        prompt_language=prompt_language,
        extra_instruction=extra_instruction,
    )
    locale = PROMPT_LOCALES[language]
    return task_block + "\n\n" + locale.reasoning_label + " ", language


def build_chat_messages(
    example: dict[str, Any],
    *,
    prompt_language: str,
    system_prompt: str,
) -> tuple[list[dict[str, str]], str]:
    task_block, language = build_task_block(
        example,
        prompt_language=prompt_language,
        extra_instruction="",
    )
    locale = PROMPT_LOCALES[language]
    messages = [
        {
            "role": "system",
            "content": system_prompt.strip() or locale.chat_system_prompt,
        },
        {
            "role": "user",
            "content": task_block + "\n\n" + locale.start_now,
        },
    ]
    return messages, language


def render_prompt(
    tokenizer: Any | None,
    example: dict[str, Any],
    *,
    prompt_format: str,
    prompt_language: str,
    system_prompt: str,
) -> tuple[str, str]:
    if prompt_format == "plain":
        return build_plain_prompt(
            example,
            prompt_language=prompt_language,
            extra_instruction=system_prompt,
        )

    if tokenizer is None:
        raise ValueError("A tokenizer is required when --prompt-format=chat.")

    messages, language = build_chat_messages(
        example,
        prompt_language=prompt_language,
        system_prompt=system_prompt,
    )
    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    if not isinstance(prompt, str) or not prompt.strip():
        raise ValueError("Tokenizer chat template returned an empty prompt.")
    return prompt, language


def sample_examples(
    records: list[dict[str, Any]],
    *,
    sample_size: int,
    sample_mode: str,
    seed: int,
) -> list[tuple[int, dict[str, Any]]]:
    indexed = list(enumerate(records))
    if sample_size <= 0 or sample_size >= len(indexed):
        return indexed
    if sample_mode == "random":
        rng = random.Random(seed)
        selected = rng.sample(indexed, sample_size)
        selected.sort(key=lambda item: item[0])
        return selected
    return indexed[:sample_size]


def chunked(items: list[tuple[int, dict[str, Any]]], size: int):
    if size <= 0:
        raise ValueError("batch_size must be > 0")
    for index in range(0, len(items), size):
        yield items[index:index + size]


def load_completed_indices(predictions_path: Path) -> set[int]:
    completed: set[int] = set()
    if not predictions_path.exists():
        return completed
    for row in iter_jsonl(predictions_path):
        source_index = row.get("source_index")
        if isinstance(source_index, int):
            completed.add(source_index)
    return completed


def strip_special_markers(text: str) -> str:
    text = text.strip().replace("\r\n", "\n").replace("\r", "\n")
    cut_markers = [
        "<|im_end|>",
        "<|endoftext|>",
        "\n<|im_start|>",
        "\nSystem:",
        "\nUser:",
        "\nAssistant:",
        "\n用户：",
        "\n助手：",
        "\nTask:",
        "\nConversation History:",
        "\nContext:",
        "\nQuestion:",
        "\n任务要求：",
        "\n对话历史：",
        "\n上下文：",
        "\n问题：",
    ]
    cut_positions = [text.find(marker) for marker in cut_markers if text.find(marker) != -1]
    if cut_positions:
        text = text[:min(cut_positions)]
    if text.startswith("Assistant:"):
        text = text[len("Assistant:"):].strip()
    if text.startswith("助手："):
        text = text[len("助手："):].strip()
    fenced = re.search(r"```(?:[a-zA-Z0-9_+-]+)?\n(.*?)```", text, flags=re.DOTALL)
    if fenced:
        text = fenced.group(1).strip()
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def clean_cot_prediction(text: str, *, language: str) -> str:
    if not isinstance(text, str):
        return text
    cleaned = strip_special_markers(text)
    if not cleaned:
        return ""

    lines = [line.rstrip() for line in cleaned.splitlines()]
    filtered_lines: list[str] = []
    for line in lines:
        stripped = line.strip()
        if not stripped:
            if filtered_lines and filtered_lines[-1] != "":
                filtered_lines.append("")
            continue
        if stripped in {"System:", "User:", "Assistant:", "用户：", "助手："}:
            continue
        filtered_lines.append(stripped)

    while filtered_lines and filtered_lines[-1] == "":
        filtered_lines.pop()

    result = "\n".join(filtered_lines).strip()
    if not result:
        return ""

    locale = PROMPT_LOCALES[language]
    label_prefixes = (
        "Reasoning:",
        "Reasoning：",
        "Final Answer:",
        "Final Answer：",
        "推理：",
        "推理:",
        "最终答案：",
        "最终答案:",
    )
    if not result.startswith(label_prefixes):
        result = f"{locale.reasoning_label} {result}"
    return result


def extract_final_answer(text: str) -> str:
    if not text:
        return ""

    patterns = [
        r"(?is)(?:^|\n)final answer\s*[:：]\s*(.+)$",
        r"(?is)(?:^|\n)最终答案\s*[:：]\s*(.+)$",
        r"(?is)(?:^|\n)answer\s*[:：]\s*(.+)$",
        r"(?is)(?:^|\n)答案\s*[:：]\s*(.+)$",
    ]
    for pattern in patterns:
        match = re.search(pattern, text)
        if match:
            return match.group(1).strip()

    lines = [line.strip() for line in text.splitlines() if line.strip()]
    return lines[-1] if lines else ""


def effective_max_tokens(example: dict[str, Any], *, requested_max_tokens: int, length_aware: bool) -> int:
    if requested_max_tokens <= 0:
        raise ValueError("max_tokens must be > 0")
    if not length_aware:
        return requested_max_tokens

    length_bin = str(example.get("length_bin", "")).strip().lower()
    bucket_cap = BUCKET_MAX_TOKENS.get(length_bin)
    if bucket_cap is None:
        return requested_max_tokens
    return min(requested_max_tokens, bucket_cap)


def call_vllm_completion_batch(
    *,
    base_url: str,
    api_key: str | None,
    model_name: str,
    prompts: list[str],
    max_tokens: int,
    temperature: float,
    top_p: float,
    timeout: float,
    repetition_penalty: float,
    frequency_penalty: float,
    presence_penalty: float,
    stop: list[str] | None = None,
    extra_body: dict[str, Any] | None = None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "model": model_name,
        "prompt": prompts,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "top_p": top_p,
        "n": 1,
        "repetition_penalty": repetition_penalty,
        "frequency_penalty": frequency_penalty,
        "presence_penalty": presence_penalty,
        "stop": stop or [],
    }
    if extra_body:
        payload.update(extra_body)

    req = request.Request(
        url=base_url.rstrip("/") + "/v1/completions",
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Content-Type": "application/json",
            **({"Authorization": f"Bearer {api_key}"} if api_key else {}),
        },
        method="POST",
    )
    try:
        with request.urlopen(req, timeout=timeout) as response:
            body = response.read().decode("utf-8")
    except error.HTTPError as exc:
        error_body = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"vLLM HTTP {exc.code}: {error_body}") from exc
    except error.URLError as exc:
        raise RuntimeError(f"Could not reach vLLM server at {base_url}: {exc}") from exc

    return json.loads(body)


def extract_choice_map(response: dict[str, Any], batch_size: int) -> dict[int, dict[str, Any]]:
    mapped: dict[int, dict[str, Any]] = {}
    for ordinal, choice in enumerate(response.get("choices") or []):
        raw_index = choice.get("index", ordinal)
        index = raw_index if isinstance(raw_index, int) and 0 <= raw_index < batch_size else ordinal
        mapped.setdefault(index, choice)
    return mapped


def write_progress(
    progress_path: Path,
    *,
    status: str,
    records_total: int,
    records_scheduled: int,
    records_completed: int,
    elapsed_seconds: float,
    predictions_path: Path,
    batch_size: int,
) -> None:
    write_json(
        progress_path,
        {
            "status": status,
            "records_total": records_total,
            "records_scheduled": records_scheduled,
            "records_completed": records_completed,
            "elapsed_seconds": round(elapsed_seconds, 3),
            "predictions_path": str(predictions_path),
            "batch_size": batch_size,
        },
    )


def load_tokenizer(args: argparse.Namespace):
    if args.prompt_format != "chat":
        return None
    if not args.tokenizer_name_or_path:
        raise ValueError("--tokenizer-name-or-path is required when --prompt-format=chat.")
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        args.tokenizer_name_or_path,
        trust_remote_code=args.trust_remote_code,
    )
    if not getattr(tokenizer, "chat_template", None):
        raise ValueError(
            "The tokenizer does not expose a chat template. "
            "Use a chat/instruct checkpoint for --prompt-format=chat, or switch to --prompt-format=plain."
        )
    return tokenizer


def run_cot_inference(args: argparse.Namespace) -> dict[str, Any]:
    input_path = Path(args.input_path)
    output_dir = Path(args.output_dir)
    ensure_dir(output_dir)

    predictions_path = output_dir / "predictions.jsonl"
    progress_path = output_dir / "progress.json"
    summary_path = output_dir / "summary.json"
    config_path = output_dir / "config.json"

    if output_dir.exists() and any(output_dir.iterdir()) and not args.overwrite and not args.resume:
        raise FileExistsError(
            f"Output directory already exists and is not empty: {output_dir}. "
            "Use --resume to continue or --overwrite to restart."
        )

    if args.overwrite and not args.resume:
        for path in (predictions_path, progress_path, summary_path):
            if path.exists():
                path.unlink()

    tokenizer = load_tokenizer(args)
    all_records = load_records(input_path)
    if not all_records:
        raise ValueError(f"No records found in {input_path}")
    for example in all_records[: min(4, len(all_records))]:
        validate_current_record(example, input_path=input_path)

    sampled = sample_examples(
        all_records,
        sample_size=args.sample_size,
        sample_mode=args.sample_mode,
        seed=args.seed,
    )

    completed_indices = load_completed_indices(predictions_path) if args.resume else set()
    remaining = [(source_index, example) for source_index, example in sampled if source_index not in completed_indices]

    config = {
        "input_path": str(input_path),
        "output_dir": str(output_dir),
        "base_url": args.base_url,
        "model_name": args.model_name,
        "tokenizer_name_or_path": args.tokenizer_name_or_path,
        "trust_remote_code": args.trust_remote_code,
        "system_prompt": args.system_prompt,
        "prediction_mode": args.prediction_mode,
        "sample_size": args.sample_size,
        "sample_mode": args.sample_mode,
        "seed": args.seed,
        "max_tokens": args.max_tokens,
        "length_aware_max_tokens": not args.disable_length_aware_max_tokens,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "repetition_penalty": args.repetition_penalty,
        "frequency_penalty": args.frequency_penalty,
        "presence_penalty": args.presence_penalty,
        "timeout": args.timeout,
        "batch_size": args.batch_size,
        "prompt_format": args.prompt_format,
        "prompt_language": args.prompt_language,
        "resume": args.resume,
    }
    write_json(config_path, config)

    stop_sequences = (
        [
            "<|im_end|>",
            "<|endoftext|>",
            "\n<|im_start|>",
        ]
        if args.prompt_format == "chat"
        else None
    )

    started = time.time()
    already_completed = len(sampled) - len(remaining)
    completed_count = already_completed
    write_progress(
        progress_path,
        status="running",
        records_total=len(all_records),
        records_scheduled=len(sampled),
        records_completed=completed_count,
        elapsed_seconds=0.0,
        predictions_path=predictions_path,
        batch_size=args.batch_size,
    )

    mode = "a" if args.resume and predictions_path.exists() else "w"
    total_batches = (len(remaining) + args.batch_size - 1) // args.batch_size if remaining else 0
    with predictions_path.open(mode, encoding="utf-8", buffering=1) as handle:
        for batch in tqdm(
            chunked(remaining, args.batch_size),
            total=total_batches,
            desc="cot-vllm-batch",
            unit="batch",
        ):
            valid_items: list[tuple[int, dict[str, Any], str, str, int]] = []
            prompts: list[str] = []
            prompt_languages: list[str] = []
            per_example_max_tokens: list[int] = []

            for original_batch_index, (source_index, example) in enumerate(batch):
                try:
                    prompt, language = render_prompt(
                        tokenizer,
                        example,
                        prompt_format=args.prompt_format,
                        prompt_language=args.prompt_language,
                        system_prompt=args.system_prompt,
                    )
                    example_max_tokens = effective_max_tokens(
                        example,
                        requested_max_tokens=args.max_tokens,
                        length_aware=not args.disable_length_aware_max_tokens,
                    )
                except Exception as exc:
                    row = {
                        "source_index": source_index,
                        "id": example.get("id", ""),
                        "instruction": get_question(example),
                        "input": get_context(example),
                        "history": flatten_dialogue_history(get_dialogue_history(example)),
                        "reference_output": get_reference_answer(example),
                        "prompt": "",
                        "model_name": args.model_name,
                        "prediction": "",
                        "raw_prediction": "",
                        "cot_prediction": "",
                        "final_answer": "",
                        "bucket_id": example.get("bucket_id", ""),
                        "style_bucket": example.get("style_bucket", ""),
                        "family": example.get("family", ""),
                        "track": example.get("track", ""),
                        "length_bin": example.get("length_bin", ""),
                        "prompt_language": resolve_prompt_language(example, args.prompt_language),
                        "prompt_format": args.prompt_format,
                        "requested_max_tokens": args.max_tokens,
                        "effective_max_tokens": None,
                        "batch_max_tokens": None,
                        "finish_reason": "prompt_error",
                        "usage": {},
                        "batch_index": original_batch_index,
                        "error": f"{type(exc).__name__}: {exc}",
                    }
                    handle.write(json.dumps(row, ensure_ascii=False) + "\n")
                    completed_count += 1
                    continue

                valid_items.append((original_batch_index, example, prompt, language, example_max_tokens))
                prompts.append(prompt)
                prompt_languages.append(language)
                per_example_max_tokens.append(example_max_tokens)

            if not valid_items:
                handle.flush()
                write_progress(
                    progress_path,
                    status="running",
                    records_total=len(all_records),
                    records_scheduled=len(sampled),
                    records_completed=completed_count,
                    elapsed_seconds=time.time() - started,
                    predictions_path=predictions_path,
                    batch_size=args.batch_size,
                )
                continue

            batch_max_tokens = max(per_example_max_tokens, default=args.max_tokens)
            response = call_vllm_completion_batch(
                base_url=args.base_url,
                api_key=args.api_key,
                model_name=args.model_name,
                prompts=prompts,
                max_tokens=batch_max_tokens,
                temperature=args.temperature,
                top_p=args.top_p,
                timeout=args.timeout,
                repetition_penalty=args.repetition_penalty,
                frequency_penalty=args.frequency_penalty,
                presence_penalty=args.presence_penalty,
                stop=stop_sequences,
            )
            choice_map = extract_choice_map(response, len(valid_items))
            batch_usage = response.get("usage", {})

            for valid_index, (original_batch_index, example, prompt, row_language, example_max_tokens) in enumerate(valid_items):
                source_index = batch[original_batch_index][0]
                row_error = None
                raw_prediction = ""
                cot_prediction = ""
                final_answer = ""
                prediction = ""
                choice = choice_map.get(valid_index, {})

                try:
                    raw_prediction = choice.get("text") or ""
                    cot_prediction = clean_cot_prediction(
                        raw_prediction,
                        language=row_language,
                    )
                    final_answer = extract_final_answer(cot_prediction)
                    prediction = cot_prediction if args.prediction_mode == "cot" else final_answer
                except Exception as exc:
                    row_error = f"{type(exc).__name__}: {exc}"

                row = {
                    "source_index": source_index,
                    "id": example.get("id", ""),
                    "instruction": get_question(example),
                    "input": get_context(example),
                    "history": flatten_dialogue_history(get_dialogue_history(example)),
                    "reference_output": get_reference_answer(example),
                    "prompt": prompt,
                    "model_name": args.model_name,
                    "prediction": prediction,
                    "raw_prediction": raw_prediction,
                    "cot_prediction": cot_prediction,
                    "final_answer": final_answer,
                    "bucket_id": example.get("bucket_id", ""),
                    "style_bucket": example.get("style_bucket", ""),
                    "family": example.get("family", ""),
                    "track": example.get("track", ""),
                    "length_bin": example.get("length_bin", ""),
                    "prompt_language": row_language,
                    "prompt_format": args.prompt_format,
                    "requested_max_tokens": args.max_tokens,
                    "effective_max_tokens": example_max_tokens,
                    "batch_max_tokens": batch_max_tokens,
                    "finish_reason": choice.get("finish_reason"),
                    "usage": batch_usage,
                    "batch_index": original_batch_index,
                }
                if row_error is not None:
                    row["error"] = row_error
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
                completed_count += 1

            handle.flush()
            write_progress(
                progress_path,
                status="running",
                records_total=len(all_records),
                records_scheduled=len(sampled),
                records_completed=completed_count,
                elapsed_seconds=time.time() - started,
                predictions_path=predictions_path,
                batch_size=args.batch_size,
            )

    summary = {
        "input_path": str(input_path),
        "output_dir": str(output_dir),
        "records_total": len(all_records),
        "records_scheduled": len(sampled),
        "records_completed": completed_count,
        "records_remaining": len(sampled) - completed_count,
        "model_name": args.model_name,
        "tokenizer_name_or_path": args.tokenizer_name_or_path,
        "base_url": args.base_url,
        "predictions_path": str(predictions_path),
        "elapsed_seconds": round(time.time() - started, 3),
        "batch_size": args.batch_size,
        "prediction_mode": args.prediction_mode,
        "prompt_format": args.prompt_format,
        "prompt_language": args.prompt_language,
        "length_aware_max_tokens": not args.disable_length_aware_max_tokens,
        "requested_max_tokens": args.max_tokens,
        "resume": args.resume,
    }
    write_json(summary_path, summary)
    write_progress(
        progress_path,
        status="completed",
        records_total=len(all_records),
        records_scheduled=len(sampled),
        records_completed=completed_count,
        elapsed_seconds=time.time() - started,
        predictions_path=predictions_path,
        batch_size=args.batch_size,
    )
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run batched bucket-conditioned CoT inference against a vLLM completions endpoint."
    )
    parser.add_argument(
        "--input-path",
        required=True,
        help="Path to a current-format PolyAlign JSON or JSONL split file. Use current/*.jsonl so bucket metadata is available.",
    )
    parser.add_argument("--output-dir", required=True, help="Output directory for predictions and run metadata.")
    parser.add_argument("--model-name", required=True, help="Served model name expected by the vLLM endpoint.")
    parser.add_argument("--base-url", required=True, help="Base vLLM URL, e.g. http://127.0.0.1:8000")
    parser.add_argument("--api-key", help="Optional API key if the vLLM server was launched with --api-key.")
    parser.add_argument(
        "--tokenizer-name-or-path",
        help="Tokenizer/checkpoint used to render chat-template prompts. Required only when --prompt-format=chat.",
    )
    parser.add_argument("--trust-remote-code", action="store_true", help="Pass trust_remote_code=True to AutoTokenizer.")
    parser.add_argument(
        "--system-prompt",
        default=DEFAULT_SYSTEM_PROMPT,
        help="Optional extra instruction for plain prompts, or system prompt override for chat prompts.",
    )
    parser.add_argument(
        "--prompt-format",
        choices=("plain", "chat"),
        default="plain",
        help="Prompt rendering mode. Use plain for base LMs and chat for instruct/chat checkpoints.",
    )
    parser.add_argument(
        "--prompt-language",
        choices=("auto", "en", "zh"),
        default="auto",
        help="Prompt language. Use auto to follow each example's language.",
    )
    parser.add_argument(
        "--prediction-mode",
        choices=("cot", "final_answer"),
        default="cot",
        help="Write either the full visible CoT response or only the extracted final answer to `prediction`.",
    )
    parser.add_argument("--sample-size", type=int, default=32, help="Number of examples to run. Use <= 0 for all.")
    parser.add_argument(
        "--sample-mode",
        choices=("first", "random"),
        default="first",
        help="Take the first N examples or a deterministic random sample.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Sampling seed used when --sample-mode=random.")
    parser.add_argument("--max-tokens", type=int, default=512, help="Upper bound on new tokens per completion request.")
    parser.add_argument(
        "--disable-length-aware-max-tokens",
        action="store_true",
        help="Do not cap max_tokens by the example's bucket length.",
    )
    parser.add_argument("--temperature", type=float, default=0.3, help="Sampling temperature sent to vLLM.")
    parser.add_argument("--top-p", type=float, default=0.95, help="top_p sent to vLLM.")
    parser.add_argument("--repetition-penalty", type=float, default=1.08, help="Penalize repeated tokens.")
    parser.add_argument("--frequency-penalty", type=float, default=0.0, help="Penalize repeated words.")
    parser.add_argument("--presence-penalty", type=float, default=0.0, help="Penalty for already-seen tokens.")
    parser.add_argument("--timeout", type=float, default=300.0, help="HTTP timeout in seconds per request.")
    parser.add_argument("--batch-size", type=int, default=4, help="Number of prompts per vLLM completion request.")
    parser.add_argument("--resume", action="store_true", help="Resume from an existing predictions.jsonl in output-dir.")
    parser.add_argument("--overwrite", action="store_true", help="Allow replacing prior outputs when not resuming.")
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    summary = run_cot_inference(args)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
