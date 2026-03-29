from __future__ import annotations

import argparse
import json
import os
import random
import time
from pathlib import Path
from typing import Any
from urllib import error, request

from tqdm.auto import tqdm


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


def build_prompt(example: dict[str, Any]) -> str:
    parts: list[str] = []
    history = example.get("history") or []
    if history:
        history_lines = []
        for turn in history:
            if not isinstance(turn, list) or len(turn) != 2:
                continue
            user_text = str(turn[0]).strip()
            assistant_text = str(turn[1]).strip()
            if user_text:
                history_lines.append(f"User: {user_text}")
            if assistant_text:
                history_lines.append(f"Assistant: {assistant_text}")
        if history_lines:
            parts.append("Conversation History:\n" + "\n".join(history_lines))

    context = str(example.get("input", "")).strip()
    if context:
        parts.append("Context:\n" + context)

    instruction = str(example.get("instruction", "")).strip()
    if instruction:
        parts.append("Question:\n" + instruction)

    parts.append("Answer:\n")
    return "\n\n".join(parts)


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
    extra_body: dict[str, Any] | None = None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "model": model_name,
        "prompt": prompts,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "top_p": top_p,
        "n": 1,
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


def run_baseline_inference(args: argparse.Namespace) -> dict[str, Any]:
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

    all_records = load_records(input_path)
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
        "sample_size": args.sample_size,
        "sample_mode": args.sample_mode,
        "seed": args.seed,
        "max_tokens": args.max_tokens,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "timeout": args.timeout,
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
            desc="baseline-vllm-batch",
            unit="batch",
        ):
            prompts = [build_prompt(example) for _source_index, example in batch]
            response = call_vllm_completion_batch(
                base_url=args.base_url,
                api_key=args.api_key,
                model_name=args.model_name,
                prompts=prompts,
                max_tokens=args.max_tokens,
                temperature=args.temperature,
                top_p=args.top_p,
                timeout=args.timeout,
            )
            choice_map = extract_choice_map(response, len(batch))
            batch_usage = response.get("usage", {})

            for batch_index, (source_index, example) in enumerate(batch):
                choice = choice_map.get(batch_index, {})
                prediction = (choice.get("text") or "").strip()
                row = {
                    "source_index": source_index,
                    "instruction": example.get("instruction", ""),
                    "input": example.get("input", ""),
                    "history": example.get("history", []),
                    "reference_output": example.get("output", ""),
                    "prompt": prompts[batch_index],
                    "model_name": args.model_name,
                    "prediction": prediction,
                    "finish_reason": choice.get("finish_reason"),
                    "usage": batch_usage,
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
        "records_total": len(all_records),
        "records_scheduled": len(sampled),
        "records_completed": completed_count,
        "records_remaining": len(sampled) - completed_count,
        "model_name": args.model_name,
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
        description="Run batched base-LM inference against a vLLM OpenAI-compatible completions endpoint."
    )
    parser.add_argument("--input-path", required=True, help="Path to a LlamaFactory JSON or JSONL split file.")
    parser.add_argument("--output-dir", required=True, help="Output directory for predictions and run metadata.")
    parser.add_argument("--model-name", required=True, help="Served model name expected by the vLLM endpoint.")
    parser.add_argument("--base-url", required=True, help="Base vLLM URL, e.g. http://127.0.0.1:8000")
    parser.add_argument("--api-key", help="Optional API key if the vLLM server was launched with --api-key.")
    parser.add_argument("--sample-size", type=int, default=32, help="Number of examples to run. Use <= 0 for all.")
    parser.add_argument(
        "--sample-mode",
        choices=("first", "random"),
        default="first",
        help="Take the first N examples or a deterministic random sample.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Sampling seed used when --sample-mode=random.")
    parser.add_argument("--max-tokens", type=int, default=128, help="Maximum new tokens per completion request.")
    parser.add_argument("--temperature", type=float, default=0.0, help="Sampling temperature sent to vLLM.")
    parser.add_argument("--top-p", type=float, default=1.0, help="top_p sent to vLLM.")
    parser.add_argument("--timeout", type=float, default=300.0, help="HTTP timeout in seconds per request.")
    parser.add_argument("--batch-size", type=int, default=4, help="Number of prompts per vLLM completion request.")
    parser.add_argument("--resume", action="store_true", help="Resume from an existing predictions.jsonl in output-dir.")
    parser.add_argument("--overwrite", action="store_true", help="Allow replacing prior outputs when not resuming.")
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    summary = run_baseline_inference(args)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
