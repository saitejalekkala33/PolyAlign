from __future__ import annotations

import json
import sys
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "rl"))

from vllm_pair_inference import build_pair_parser, run_pair_inference


DEFAULT_SYSTEM_PROMPT = "You are a helpful assistant."


def build_parser():
    return build_pair_parser(
        description="Run batched HDPO-checkpoint inference against a vLLM completions endpoint.",
        default_system_mode="profile",
        default_system_prompt=DEFAULT_SYSTEM_PROMPT,
    )


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    summary = run_pair_inference(args, run_kind="hdpo", progress_desc="hdpo-vllm-batch")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
