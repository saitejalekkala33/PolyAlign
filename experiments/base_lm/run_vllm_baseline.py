from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path
from typing import Any
from urllib import error, request

from tqdm.auto import tqdm


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


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


def call_vllm_completion(
    *,
    base_url: str,
    api_key: str | None,
    model_name: str,
    prompt: str,
    max_tokens: int,
    temperature: float,
    top_p: float,
    timeout: float,
    extra_body: dict[str, Any] | None = None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "model": model_name,
        "prompt": prompt,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "top_p": top_p,
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


def run_baseline_inference(args: argparse.Namespace) -> dict[str, Any]:
    input_path = Path(args.input_path)
    output_dir = Path(args.output_dir)
    if output_dir.exists() and any(output_dir.iterdir()) and not args.overwrite:
        raise FileExistsError(f"Output directory already exists and is not empty: {output_dir}")
    ensure_dir(output_dir)

    all_records = load_records(input_path)
    sampled = sample_examples(
        all_records,
        sample_size=args.sample_size,
        sample_mode=args.sample_mode,
        seed=args.seed,
    )

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
    }
    (output_dir / "config.json").write_text(json.dumps(config, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    predictions_path = output_dir / "predictions.jsonl"
    started = time.time()
    with predictions_path.open("w", encoding="utf-8") as handle:
        for source_index, example in tqdm(sampled, total=len(sampled), desc="baseline-vllm", unit="example"):
            prompt = build_prompt(example)
            response = call_vllm_completion(
                base_url=args.base_url,
                api_key=args.api_key,
                model_name=args.model_name,
                prompt=prompt,
                max_tokens=args.max_tokens,
                temperature=args.temperature,
                top_p=args.top_p,
                timeout=args.timeout,
            )
            choice = (response.get("choices") or [{}])[0]
            prediction = (choice.get("text") or "").strip()
            row = {
                "source_index": source_index,
                "instruction": example.get("instruction", ""),
                "input": example.get("input", ""),
                "history": example.get("history", []),
                "reference_output": example.get("output", ""),
                "prompt": prompt,
                "model_name": args.model_name,
                "prediction": prediction,
                "finish_reason": choice.get("finish_reason"),
                "usage": response.get("usage", {}),
            }
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    summary = {
        "input_path": str(input_path),
        "output_dir": str(output_dir),
        "records_total": len(all_records),
        "records_inferred": len(sampled),
        "model_name": args.model_name,
        "base_url": args.base_url,
        "predictions_path": str(predictions_path),
        "elapsed_seconds": round(time.time() - started, 3),
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run sample base-LM inference against a vLLM OpenAI-compatible completions endpoint."
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
    parser.add_argument("--overwrite", action="store_true", help="Allow writing into a non-empty output directory.")
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    summary = run_baseline_inference(args)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
