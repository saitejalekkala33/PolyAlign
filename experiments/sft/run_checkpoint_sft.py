from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import Any

import torch
from tqdm.auto import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer


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


def build_user_message(example: dict[str, Any]) -> str:
    instruction = str(example.get("instruction", "")).strip()
    context = str(example.get("input", "")).strip()
    parts: list[str] = []

    if instruction:
        parts.append(instruction)
    if context:
        parts.append(context)
    if not parts:
        raise ValueError("Example must contain at least one of `instruction` or `input`.")

    return "\n".join(parts)


def build_messages(example: dict[str, Any]) -> list[dict[str, str]]:
    messages: list[dict[str, str]] = []

    system_text = str(example.get("system", "")).strip()
    if system_text:
        messages.append({"role": "system", "content": system_text})

    history = example.get("history") or []
    for turn in history:
        if not isinstance(turn, list) or len(turn) != 2:
            continue
        user_text = str(turn[0]).strip()
        assistant_text = str(turn[1]).strip()
        if user_text:
            messages.append({"role": "user", "content": user_text})
        if assistant_text:
            messages.append({"role": "assistant", "content": assistant_text})

    messages.append({"role": "user", "content": build_user_message(example)})
    return messages


def render_chat_prompt(tokenizer: Any, example: dict[str, Any]) -> str:
    messages = build_messages(example)
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    if not isinstance(prompt, str) or not prompt.strip():
        raise ValueError("Tokenizer chat template returned an empty prompt.")
    return prompt


def sample_examples(
    records: list[dict[str, Any]],
    *,
    sample_size: int,
    sample_mode: str,
) -> list[tuple[int, dict[str, Any]]]:
    indexed = list(enumerate(records))
    if sample_size <= 0 or sample_size >= len(indexed):
        return indexed
    if sample_mode != "first":
        raise ValueError(f"Unsupported sample_mode: {sample_mode}. Only `first` is allowed for SFT evaluation.")
    return indexed[:sample_size]


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


def resolve_dtype(dtype_name: str) -> torch.dtype | None:
    name = dtype_name.lower()
    if name == "auto":
        return None
    mapping: dict[str, torch.dtype] = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }
    if name not in mapping:
        raise ValueError(f"Unsupported dtype: {dtype_name}")
    return mapping[name]


def resolve_model_name(args: argparse.Namespace) -> str:
    if args.model_name:
        return args.model_name
    return Path(args.checkpoint_path).name or str(args.checkpoint_path)


def get_input_device(model: Any) -> torch.device:
    try:
        return next(model.parameters()).device
    except StopIteration as exc:
        raise ValueError("Could not resolve a model device from the loaded checkpoint.") from exc


def load_tokenizer(args: argparse.Namespace) -> Any:
    tokenizer_name_or_path = args.tokenizer_name_or_path or args.checkpoint_path
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_name_or_path,
        trust_remote_code=args.trust_remote_code,
    )
    if not getattr(tokenizer, "chat_template", None):
        raise ValueError(
            "The tokenizer does not expose a chat template. "
            "Pass the exact Qwen tokenizer/checkpoint used for SFT inference."
        )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    return tokenizer


def load_model(args: argparse.Namespace) -> Any:
    dtype = resolve_dtype(args.dtype)
    load_kwargs: dict[str, Any] = {
        "device_map": args.device_map,
        "trust_remote_code": args.trust_remote_code,
    }
    if dtype is not None:
        load_kwargs["torch_dtype"] = dtype
    if args.attn_implementation != "auto":
        load_kwargs["attn_implementation"] = args.attn_implementation

    model = AutoModelForCausalLM.from_pretrained(args.checkpoint_path, **load_kwargs)
    model.eval()
    return model


def generate_batch(
    *,
    model: Any,
    tokenizer: Any,
    prompts: list[str],
    max_input_length: int,
    max_tokens: int,
    temperature: float,
    top_p: float,
) -> list[dict[str, Any]]:
    model_inputs = tokenizer(
        prompts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=max_input_length,
    )
    input_device = get_input_device(model)
    model_inputs = {name: tensor.to(input_device) for name, tensor in model_inputs.items()}
    prompt_lengths = model_inputs["attention_mask"].sum(dim=1).tolist()
    padded_input_length = int(model_inputs["input_ids"].shape[1])

    generation_kwargs: dict[str, Any] = {
        "max_new_tokens": max_tokens,
        "pad_token_id": tokenizer.pad_token_id,
        "eos_token_id": tokenizer.eos_token_id,
        "use_cache": True,
    }
    if temperature > 0.0:
        generation_kwargs["do_sample"] = True
        generation_kwargs["temperature"] = temperature
        generation_kwargs["top_p"] = top_p
    else:
        generation_kwargs["do_sample"] = False

    with torch.inference_mode():
        generated = model.generate(**model_inputs, **generation_kwargs)

    rows: list[dict[str, Any]] = []
    for sequence, prompt_length in zip(generated, prompt_lengths, strict=True):
        # generate() returns the full padded input prefix followed by new tokens.
        # With left padding, slicing by the non-pad prompt length leaks part of the prompt.
        completion_ids = sequence[padded_input_length:]
        completion_text = tokenizer.decode(completion_ids, skip_special_tokens=True).strip()
        rows.append(
            {
                "prediction": completion_text,
                "usage": {
                    "prompt_tokens": int(prompt_length),
                    "completion_tokens": int(completion_ids.shape[0]),
                    "total_tokens": int(prompt_length + completion_ids.shape[0]),
                },
            }
        )
    return rows


def run_checkpoint_inference(args: argparse.Namespace) -> dict[str, Any]:
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
    model = load_model(args)
    model_name = resolve_model_name(args)

    all_records = load_records(input_path)
    sampled = sample_examples(
        all_records,
        sample_size=args.sample_size,
        sample_mode=args.sample_mode,
    )

    completed_indices = load_completed_indices(predictions_path) if args.resume else set()
    remaining = [(source_index, example) for source_index, example in sampled if source_index not in completed_indices]

    config = {
        "input_path": str(input_path),
        "output_dir": str(output_dir),
        "checkpoint_path": args.checkpoint_path,
        "model_name": model_name,
        "tokenizer_name_or_path": args.tokenizer_name_or_path or args.checkpoint_path,
        "hip_visible_devices": os.environ.get("HIP_VISIBLE_DEVICES", ""),
        "device_map": args.device_map,
        "dtype": args.dtype,
        "attn_implementation": args.attn_implementation,
        "sample_size": args.sample_size,
        "sample_mode": args.sample_mode,
        "max_input_length": args.max_input_length,
        "max_tokens": args.max_tokens,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "batch_size": args.batch_size,
        "resume": args.resume,
    }
    write_json(config_path, config)

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
            desc="sft-checkpoint-batch",
            unit="batch",
        ):
            prompts = [render_chat_prompt(tokenizer, example) for _source_index, example in batch]
            generations = generate_batch(
                model=model,
                tokenizer=tokenizer,
                prompts=prompts,
                max_input_length=args.max_input_length,
                max_tokens=args.max_tokens,
                temperature=args.temperature,
                top_p=args.top_p,
            )

            for batch_index, ((source_index, example), generation) in enumerate(zip(batch, generations, strict=True)):
                row = {
                    "source_index": source_index,
                    "instruction": example.get("instruction", ""),
                    "input": example.get("input", ""),
                    "history": example.get("history", []),
                    "reference_output": example.get("output", ""),
                    "prompt": prompts[batch_index],
                    "model_name": model_name,
                    "prediction": generation["prediction"],
                    "finish_reason": "completed",
                    "usage": generation["usage"],
                    "batch_index": batch_index,
                }
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
                handle.flush()
                os.fsync(handle.fileno())
                completed_count += 1

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
        "checkpoint_path": args.checkpoint_path,
        "records_total": len(all_records),
        "records_scheduled": len(sampled),
        "records_completed": completed_count,
        "records_remaining": len(sampled) - completed_count,
        "model_name": model_name,
        "tokenizer_name_or_path": args.tokenizer_name_or_path or args.checkpoint_path,
        "hip_visible_devices": os.environ.get("HIP_VISIBLE_DEVICES", ""),
        "device_map": args.device_map,
        "dtype": args.dtype,
        "predictions_path": str(predictions_path),
        "elapsed_seconds": round(time.time() - started, 3),
        "batch_size": args.batch_size,
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
        description="Run batched SFT-checkpoint inference directly from a local checkpoint using chat-template prompts."
    )
    parser.add_argument("--input-path", required=True, help="Path to a LlamaFactory JSON or JSONL split file.")
    parser.add_argument("--output-dir", required=True, help="Output directory for predictions and run metadata.")
    parser.add_argument("--checkpoint-path", required=True, help="Local SFT checkpoint path passed to from_pretrained.")
    parser.add_argument("--model-name", help="Optional alias written into predictions.jsonl. Defaults to checkpoint name.")
    parser.add_argument(
        "--tokenizer-name-or-path",
        help="Optional tokenizer path or model id. Defaults to checkpoint-path.",
    )
    parser.add_argument("--sample-size", type=int, default=32, help="Number of examples to run. Use <= 0 for all.")
    parser.add_argument(
        "--sample-mode",
        choices=("first",),
        default="first",
        help="Sampling mode for scheduled examples. SFT evaluation only supports `first`.",
    )
    parser.add_argument("--max-input-length", type=int, default=4096, help="Maximum prompt length passed to the tokenizer.")
    parser.add_argument("--max-tokens", type=int, default=128, help="Maximum new tokens per generation.")
    parser.add_argument("--temperature", type=float, default=0.0, help="Sampling temperature. Use 0.0 for greedy decoding.")
    parser.add_argument("--top-p", type=float, default=1.0, help="top_p used when temperature > 0.")
    parser.add_argument("--batch-size", type=int, default=4, help="Number of prompts per generation batch.")
    parser.add_argument("--dtype", default="auto", choices=("auto", "bfloat16", "float16", "float32"))
    parser.add_argument("--device-map", default="auto", help="device_map passed to from_pretrained, e.g. auto.")
    parser.add_argument(
        "--attn-implementation",
        default="auto",
        choices=("auto", "eager", "sdpa", "flash_attention_2"),
        help="Attention implementation passed to from_pretrained when not `auto`.",
    )
    parser.add_argument("--trust-remote-code", action="store_true", help="Pass trust_remote_code=True to Transformers.")
    parser.add_argument("--resume", action="store_true", help="Resume from an existing predictions.jsonl in output-dir.")
    parser.add_argument("--overwrite", action="store_true", help="Allow replacing prior outputs when not resuming.")
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    summary = run_checkpoint_inference(args)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
