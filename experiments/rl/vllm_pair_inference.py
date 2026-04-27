from __future__ import annotations

import argparse
import json
import os
import random
import re
import time
from pathlib import Path
from typing import Any
from urllib import error, request

from tqdm.auto import tqdm
from transformers import AutoTokenizer


QWEN_DEFAULT_SYSTEM_PROMPT = "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."

PAIR_METADATA_KEYS = (
    "id",
    "dataset",
    "split",
    "language",
    "track",
    "family",
    "style_bucket",
    "length_bin",
    "bucket_id",
    "pair_type",
    "rejected_model_name",
    "rejected_finish_reason",
    "hdpo_weight",
    "baseline_bucket_gap",
    "chosen_dist_score",
    "rejected_dist_score",
    "lang_weight",
    "critic_bucket_id",
)


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
    if path.suffix.lower() == ".jsonl":
        return list(iter_jsonl(path))

    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError(f"Expected a JSON array or JSONL file at {path}.")
    return [dict(item) for item in payload]


def first_nonempty_string(example: dict[str, Any], *keys: str) -> str:
    for key in keys:
        value = example.get(key)
        if value is None:
            continue
        if isinstance(value, str):
            text = value.strip()
        else:
            text = str(value).strip()
        if text:
            return text
    return ""


def get_instruction(example: dict[str, Any]) -> str:
    return first_nonempty_string(example, "instruction", "question", "prompt")


def get_input_text(example: dict[str, Any]) -> str:
    return first_nonempty_string(example, "input", "context", "query")


def get_chosen_output(example: dict[str, Any]) -> str:
    return first_nonempty_string(example, "chosen", "reference_output", "output", "human_answer")


def get_rejected_output(example: dict[str, Any]) -> str:
    return first_nonempty_string(example, "rejected", "rejected_output", "model_rejected", "prediction")


def get_history(example: dict[str, Any]) -> list[list[str]]:
    history = example.get("history")
    if isinstance(history, list):
        normalized: list[list[str]] = []
        for turn in history:
            if isinstance(turn, dict):
                role = str(turn.get("role", "")).strip()
                text = str(turn.get("text", turn.get("content", ""))).strip()
                if role == "user" and text:
                    normalized.append([text, ""])
                elif role == "assistant" and text:
                    if normalized and not normalized[-1][1]:
                        normalized[-1][1] = text
                    else:
                        normalized.append(["", text])
                continue

            if not isinstance(turn, (list, tuple)) or len(turn) != 2:
                continue
            user_text = str(turn[0]).strip()
            assistant_text = str(turn[1]).strip()
            if user_text or assistant_text:
                normalized.append([user_text, assistant_text])

        if normalized:
            return normalized

    dialogue_history = example.get("dialogue_history")
    if isinstance(dialogue_history, list):
        normalized_dialogue: list[dict[str, str]] = []
        for turn in dialogue_history:
            if not isinstance(turn, dict):
                continue
            role = str(turn.get("role", "")).strip()
            text = str(turn.get("text", turn.get("content", ""))).strip()
            if role in {"user", "assistant"} and text:
                normalized_dialogue.append({"role": role, "text": text})

        question = get_instruction(example)
        if normalized_dialogue:
            last_turn = normalized_dialogue[-1]
            if last_turn["role"] == "user" and last_turn["text"] == question:
                normalized_dialogue = normalized_dialogue[:-1]

        pairs: list[list[str]] = []
        pending_user: str | None = None
        for turn in normalized_dialogue:
            if turn["role"] == "user":
                pending_user = turn["text"]
            elif turn["role"] == "assistant" and pending_user is not None:
                pairs.append([pending_user, turn["text"]])
                pending_user = None
        return pairs

    return []


def build_profile_system_prompt(example: dict[str, Any], *, default_system_prompt: str) -> str:
    prompt_parts = [
        ("family", first_nonempty_string(example, "family")),
        ("track", first_nonempty_string(example, "track")),
        ("style", first_nonempty_string(example, "style_bucket")),
        ("length", first_nonempty_string(example, "length_bin")),
    ]
    profile = "; ".join(f"{name}={value}" for name, value in prompt_parts if value)
    if not profile:
        return default_system_prompt
    return f"You are a helpful assistant. Follow the target response profile when answering. {profile}."


def get_system_message(
    example: dict[str, Any],
    *,
    system_mode: str,
    default_system_prompt: str,
) -> str:
    record_system = first_nonempty_string(example, "system")
    if system_mode == "record-only":
        return record_system
    if system_mode == "profile":
        return record_system or build_profile_system_prompt(example, default_system_prompt=default_system_prompt)
    if system_mode == "record-or-default":
        return record_system or default_system_prompt
    raise ValueError(f"Unsupported system mode: {system_mode}")


def build_user_message(example: dict[str, Any]) -> str:
    parts: list[str] = []
    context = get_input_text(example)
    instruction = get_instruction(example)

    if context:
        parts.append("Context:\n" + context)
    if instruction:
        parts.append("Question:\n" + instruction)
    if not parts:
        raise ValueError("Example must contain at least one of `instruction`/`question` or `input`/`context`.")

    return "\n\n".join(parts)


def build_messages(
    example: dict[str, Any],
    *,
    system_mode: str,
    default_system_prompt: str,
) -> list[dict[str, str]]:
    messages: list[dict[str, str]] = []

    system_text = get_system_message(
        example,
        system_mode=system_mode,
        default_system_prompt=default_system_prompt,
    )
    if system_text:
        messages.append({"role": "system", "content": system_text})

    for user_text, assistant_text in get_history(example):
        if user_text:
            messages.append({"role": "user", "content": user_text})
        if assistant_text:
            messages.append({"role": "assistant", "content": assistant_text})

    messages.append({"role": "user", "content": build_user_message(example)})
    return messages


def render_chat_prompt(
    tokenizer: Any,
    example: dict[str, Any],
    *,
    system_mode: str,
    default_system_prompt: str,
) -> str:
    messages = build_messages(
        example,
        system_mode=system_mode,
        default_system_prompt=default_system_prompt,
    )
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    if not isinstance(prompt, str) or not prompt.strip():
        raise ValueError("Tokenizer chat template returned an empty prompt.")
    return prompt


def maybe_truncate_prompt(
    tokenizer: Any,
    prompt: str,
    *,
    requested_max_tokens: int,
    max_model_len: int,
    safety_margin: int,
) -> tuple[str, int, bool]:
    input_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
    prompt_token_count = len(input_ids)
    if max_model_len <= 0:
        return prompt, prompt_token_count, False

    max_input_tokens = max_model_len - max(1, requested_max_tokens) - safety_margin
    if max_input_tokens <= 0:
        raise ValueError(
            f"max_input_tokens became non-positive. max_model_len={max_model_len}, "
            f"requested_max_tokens={requested_max_tokens}, safety_margin={safety_margin}"
        )
    if prompt_token_count <= max_input_tokens:
        return prompt, prompt_token_count, False

    truncated_ids = input_ids[-max_input_tokens:]
    truncated_prompt = tokenizer.decode(truncated_ids, skip_special_tokens=False)
    return truncated_prompt, len(truncated_ids), True


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
    if sample_mode == "first":
        return indexed[:sample_size]
    raise ValueError(f"Unsupported sample_mode: {sample_mode}")


def chunked(items: list[tuple[int, dict[str, Any]]], size: int):
    if size <= 0:
        raise ValueError("batch_size must be > 0")
    for index in range(0, len(items), size):
        yield items[index : index + size]


def load_completed_indices(predictions_path: Path) -> set[int]:
    completed: set[int] = set()
    if not predictions_path.exists():
        return completed
    for row in iter_jsonl(predictions_path):
        source_index = row.get("source_index")
        if isinstance(source_index, int):
            completed.add(source_index)
    return completed


def clean_prediction(text: str) -> str:
    if not isinstance(text, str):
        return text

    text = text.strip().replace("\r\n", "\n").replace("\r", "\n")
    cut_markers = [
        "<|im_end|>",
        "<|endoftext|>",
        "\n<|im_start|>",
        "\nQuestion:",
        "\nUser:",
        "\nAssistant:",
    ]
    cut_positions = [text.find(marker) for marker in cut_markers if text.find(marker) != -1]
    if cut_positions:
        text = text[: min(cut_positions)]

    text = text.strip()
    for prefix in ("Answer:", "Assistant:"):
        if text.startswith(prefix):
            text = text[len(prefix) :].strip()

    fenced = re.search(r"```(?:[a-zA-Z0-9_+-]+)?\n(.*?)```", text, flags=re.DOTALL)
    if fenced:
        text = fenced.group(1).strip()

    lines = [line.strip() for line in text.splitlines() if line.strip()]
    filtered_lines: list[str] = []
    bad_starts = (
        "<|im_start|>",
        "<|im_end|>",
        "Question:",
        "User:",
        "Assistant:",
    )
    for line in lines:
        if line.startswith(bad_starts):
            continue
        filtered_lines.append(line)

    return "\n".join(filtered_lines).strip()


def parse_extra_body(raw_value: str | None) -> dict[str, Any] | None:
    if not raw_value:
        return None
    payload = json.loads(raw_value)
    if not isinstance(payload, dict):
        raise ValueError("--extra-body-json must decode to a JSON object.")
    return payload


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
    run_kind: str,
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
            "run_kind": run_kind,
            "records_total": records_total,
            "records_scheduled": records_scheduled,
            "records_completed": records_completed,
            "elapsed_seconds": round(elapsed_seconds, 3),
            "predictions_path": str(predictions_path),
            "batch_size": batch_size,
        },
    )


def pair_metadata(example: dict[str, Any]) -> dict[str, Any]:
    metadata = {key: example[key] for key in PAIR_METADATA_KEYS if key in example}
    if "source_index" in example:
        metadata["preference_source_index"] = example["source_index"]
    return metadata


def build_output_row(
    *,
    source_index: int,
    batch_index: int,
    example: dict[str, Any],
    prompt: str,
    prompt_tokens: int,
    prompt_truncated: bool,
    system_mode: str,
    default_system_prompt: str,
    model_name: str,
    raw_prediction: str,
    prediction: str,
    finish_reason: Any,
    usage: dict[str, Any],
    run_kind: str,
    requested_max_tokens: int,
    batch_max_tokens: int,
    error_message: str | None = None,
) -> dict[str, Any]:
    row: dict[str, Any] = {
        "source_index": source_index,
        **pair_metadata(example),
        "instruction": get_instruction(example),
        "input": get_input_text(example),
        "history": get_history(example),
        "system": get_system_message(
            example,
            system_mode=system_mode,
            default_system_prompt=default_system_prompt,
        ),
        "chosen": get_chosen_output(example),
        "rejected": get_rejected_output(example),
        "reference_output": get_chosen_output(example),
        "prompt": prompt,
        "model_name": model_name,
        "run_kind": run_kind,
        "prediction": prediction,
        "raw_prediction": raw_prediction,
        "finish_reason": finish_reason,
        "usage": usage,
        "batch_index": batch_index,
        "prompt_tokens": prompt_tokens,
        "prompt_truncated": prompt_truncated,
        "requested_max_tokens": requested_max_tokens,
        "batch_max_tokens": batch_max_tokens,
    }
    if error_message is not None:
        row["error"] = error_message
    return row


def run_pair_inference(
    args: argparse.Namespace,
    *,
    run_kind: str,
    progress_desc: str,
) -> dict[str, Any]:
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
        for path in (predictions_path, progress_path, summary_path, config_path):
            if path.exists():
                path.unlink()

    tokenizer = AutoTokenizer.from_pretrained(
        args.tokenizer_name_or_path,
        trust_remote_code=args.trust_remote_code,
    )
    if not getattr(tokenizer, "chat_template", None):
        raise ValueError(
            "The tokenizer does not expose a chat template. "
            "Pass the exact Qwen tokenizer/checkpoint used for preference-model inference."
        )

    all_records = load_records(input_path)
    sampled = sample_examples(
        all_records,
        sample_size=args.sample_size,
        sample_mode=args.sample_mode,
        seed=args.seed,
    )
    completed_indices = load_completed_indices(predictions_path) if args.resume else set()
    remaining = [(source_index, example) for source_index, example in sampled if source_index not in completed_indices]

    extra_body = parse_extra_body(args.extra_body_json)
    config = {
        "run_kind": run_kind,
        "input_path": str(input_path),
        "output_dir": str(output_dir),
        "base_url": args.base_url,
        "model_name": args.model_name,
        "tokenizer_name_or_path": args.tokenizer_name_or_path,
        "trust_remote_code": args.trust_remote_code,
        "system_mode": args.system_mode,
        "default_system_prompt": args.default_system_prompt,
        "sample_size": args.sample_size,
        "sample_mode": args.sample_mode,
        "seed": args.seed,
        "max_tokens": args.max_tokens,
        "max_model_len": args.max_model_len,
        "prompt_safety_margin": args.prompt_safety_margin,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "repetition_penalty": args.repetition_penalty,
        "frequency_penalty": args.frequency_penalty,
        "presence_penalty": args.presence_penalty,
        "timeout": args.timeout,
        "batch_size": args.batch_size,
        "extra_body": extra_body,
        "resume": args.resume,
    }
    write_json(config_path, config)

    stop_sequences = [
        "<|im_end|>",
        "<|endoftext|>",
        "\n<|im_start|>",
        "\nQuestion:",
        "\nUser:",
        "\nAssistant:",
    ]

    started = time.time()
    already_completed = len(sampled) - len(remaining)
    completed_count = already_completed
    write_progress(
        progress_path,
        status="running",
        run_kind=run_kind,
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
            desc=progress_desc,
            unit="batch",
        ):
            valid_items: list[tuple[int, int, dict[str, Any], str, int, bool]] = []
            prompts: list[str] = []
            prompt_token_counts: list[int] = []
            prompt_truncation_flags: list[bool] = []
            request_max_tokens = args.max_tokens

            for batch_index, (source_index, example) in enumerate(batch):
                try:
                    prompt = render_chat_prompt(
                        tokenizer,
                        example,
                        system_mode=args.system_mode,
                        default_system_prompt=args.default_system_prompt,
                    )
                    prompt, prompt_tokens, prompt_truncated = maybe_truncate_prompt(
                        tokenizer,
                        prompt,
                        requested_max_tokens=args.max_tokens,
                        max_model_len=args.max_model_len,
                        safety_margin=args.prompt_safety_margin,
                    )
                    if args.max_model_len > 0:
                        allowed_max_tokens = args.max_model_len - prompt_tokens - args.prompt_safety_margin
                        request_max_tokens = min(request_max_tokens, max(1, allowed_max_tokens))
                except Exception as exc:
                    row = build_output_row(
                        source_index=source_index,
                        batch_index=batch_index,
                        example=example,
                        prompt="",
                        prompt_tokens=0,
                        prompt_truncated=False,
                        system_mode=args.system_mode,
                        default_system_prompt=args.default_system_prompt,
                        model_name=args.model_name,
                        raw_prediction="",
                        prediction="",
                        finish_reason="prompt_error",
                        usage={},
                        run_kind=run_kind,
                        requested_max_tokens=args.max_tokens,
                        batch_max_tokens=0,
                        error_message=f"{type(exc).__name__}: {exc}",
                    )
                    handle.write(json.dumps(row, ensure_ascii=False) + "\n")
                    completed_count += 1
                    continue

                valid_items.append((batch_index, source_index, example, prompt, prompt_tokens, prompt_truncated))
                prompts.append(prompt)
                prompt_token_counts.append(prompt_tokens)
                prompt_truncation_flags.append(prompt_truncated)

            if not valid_items:
                handle.flush()
                write_progress(
                    progress_path,
                    status="running",
                    run_kind=run_kind,
                    records_total=len(all_records),
                    records_scheduled=len(sampled),
                    records_completed=completed_count,
                    elapsed_seconds=time.time() - started,
                    predictions_path=predictions_path,
                    batch_size=args.batch_size,
                )
                continue

            response = call_vllm_completion_batch(
                base_url=args.base_url,
                api_key=args.api_key,
                model_name=args.model_name,
                prompts=prompts,
                max_tokens=request_max_tokens,
                temperature=args.temperature,
                top_p=args.top_p,
                timeout=args.timeout,
                repetition_penalty=args.repetition_penalty,
                frequency_penalty=args.frequency_penalty,
                presence_penalty=args.presence_penalty,
                stop=stop_sequences,
                extra_body=extra_body,
            )
            choice_map = extract_choice_map(response, len(valid_items))
            batch_usage = response.get("usage", {})

            for valid_index, (batch_index, source_index, example, prompt, prompt_tokens, prompt_truncated) in enumerate(valid_items):
                choice = choice_map.get(valid_index, {})
                raw_prediction = choice.get("text") or ""
                try:
                    prediction = clean_prediction(raw_prediction)
                    row_error = None
                except Exception as exc:
                    prediction = ""
                    row_error = f"{type(exc).__name__}: {exc}"

                row = build_output_row(
                    source_index=source_index,
                    batch_index=batch_index,
                    example=example,
                    prompt=prompt,
                    prompt_tokens=prompt_tokens,
                    prompt_truncated=prompt_truncated,
                    system_mode=args.system_mode,
                    default_system_prompt=args.default_system_prompt,
                    model_name=args.model_name,
                    raw_prediction=raw_prediction,
                    prediction=prediction,
                    finish_reason=choice.get("finish_reason"),
                    usage=batch_usage,
                    run_kind=run_kind,
                    requested_max_tokens=args.max_tokens,
                    batch_max_tokens=request_max_tokens,
                    error_message=row_error,
                )
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
                completed_count += 1

            handle.flush()
            os.fsync(handle.fileno())
            write_progress(
                progress_path,
                status="running",
                run_kind=run_kind,
                records_total=len(all_records),
                records_scheduled=len(sampled),
                records_completed=completed_count,
                elapsed_seconds=time.time() - started,
                predictions_path=predictions_path,
                batch_size=args.batch_size,
            )

    summary = {
        "run_kind": run_kind,
        "input_path": str(input_path),
        "output_dir": str(output_dir),
        "records_total": len(all_records),
        "records_scheduled": len(sampled),
        "records_completed": completed_count,
        "records_remaining": len(sampled) - completed_count,
        "model_name": args.model_name,
        "tokenizer_name_or_path": args.tokenizer_name_or_path,
        "system_mode": args.system_mode,
        "base_url": args.base_url,
        "predictions_path": str(predictions_path),
        "elapsed_seconds": round(time.time() - started, 3),
        "batch_size": args.batch_size,
        "resume": args.resume,
    }
    write_json(summary_path, summary)
    write_progress(
        progress_path,
        status="completed",
        run_kind=run_kind,
        records_total=len(all_records),
        records_scheduled=len(sampled),
        records_completed=completed_count,
        elapsed_seconds=time.time() - started,
        predictions_path=predictions_path,
        batch_size=args.batch_size,
    )
    return summary


def build_pair_parser(
    *,
    description: str,
    default_system_mode: str,
    default_system_prompt: str,
) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--input-path", required=True, help="Path to a DPO/HDPO JSON or JSONL preference test file.")
    parser.add_argument("--output-dir", required=True, help="Output directory for predictions and run metadata.")
    parser.add_argument("--model-name", required=True, help="Served model name expected by the vLLM endpoint.")
    parser.add_argument(
        "--tokenizer-name-or-path",
        required=True,
        help="Local checkpoint path or model id used to load the tokenizer and chat template.",
    )
    parser.add_argument("--base-url", required=True, help="Base vLLM URL, e.g. http://127.0.0.1:8000")
    parser.add_argument("--api-key", help="Optional API key if the vLLM server was launched with --api-key.")
    parser.add_argument(
        "--system-mode",
        choices=("record-or-default", "profile", "record-only"),
        default=default_system_mode,
        help=(
            "How to build the system message. `record-or-default` uses the record system when present, "
            "`profile` falls back to bucket profile metadata, and `record-only` omits fallback system prompts."
        ),
    )
    parser.add_argument(
        "--default-system-prompt",
        default=default_system_prompt,
        help="Fallback system prompt used when the record has no `system` field content.",
    )
    parser.add_argument("--sample-size", type=int, default=32, help="Number of examples to run. Use <= 0 for all.")
    parser.add_argument(
        "--sample-mode",
        choices=("first", "random"),
        default="first",
        help="Take the first N examples or a deterministic random sample.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Sampling seed used when --sample-mode=random.")
    parser.add_argument("--max-tokens", type=int, default=128, help="Maximum new tokens per completion request.")
    parser.add_argument(
        "--max-model-len",
        type=int,
        default=8192,
        help="Context length budget used for prompt truncation. Use <= 0 to disable local truncation.",
    )
    parser.add_argument(
        "--prompt-safety-margin",
        type=int,
        default=16,
        help="Token margin reserved below max_model_len.",
    )
    parser.add_argument("--temperature", type=float, default=0.0, help="Sampling temperature sent to vLLM.")
    parser.add_argument("--top-p", type=float, default=1.0, help="top_p sent to vLLM.")
    parser.add_argument("--repetition-penalty", type=float, default=1.05, help="Penalize repeated tokens.")
    parser.add_argument("--frequency-penalty", type=float, default=0.0, help="Penalize frequent tokens.")
    parser.add_argument("--presence-penalty", type=float, default=0.0, help="Penalize already-seen tokens.")
    parser.add_argument("--timeout", type=float, default=300.0, help="HTTP timeout in seconds per request.")
    parser.add_argument("--batch-size", type=int, default=4, help="Number of prompts per vLLM completion request.")
    parser.add_argument(
        "--extra-body-json",
        help="Optional JSON object merged into the vLLM /v1/completions request body.",
    )
    parser.add_argument("--trust-remote-code", action="store_true", help="Pass trust_remote_code=True to AutoTokenizer.")
    parser.add_argument("--resume", action="store_true", help="Resume from an existing predictions.jsonl in output-dir.")
    parser.add_argument("--overwrite", action="store_true", help="Allow replacing prior outputs when not resuming.")
    return parser
