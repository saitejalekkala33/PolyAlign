from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "rl"))

import vllm_pair_inference as pair_inference


DEFAULT_SYSTEM_PROMPT = "You are a helpful assistant."
REFERENCE_CONDITIONING_INSTRUCTION = (
    "Use the reference answer only as semantic guidance. "
    "Write an answer that is similar in meaning and appropriate for the question, "
    "but strictly do not output the exact same text as the reference answer."
)


def _get_reference_output(example: dict[str, Any]) -> str:
    if hasattr(pair_inference, "get_reference_output"):
        return pair_inference.get_reference_output(example)
    return pair_inference.get_chosen_output(example)


def _build_reference_conditioned_user_message(
    example: dict[str, Any],
    *,
    reference_conditioning_instruction: str,
) -> str:
    parts: list[str] = []
    context = pair_inference.get_input_text(example)
    instruction = pair_inference.get_instruction(example)
    reference_output = _get_reference_output(example)

    if context:
        parts.append("Context:\n" + context)
    if instruction:
        parts.append("Question:\n" + instruction)
    if not reference_output:
        raise ValueError("Reference-conditioned inference requires a non-empty chosen/reference output.")
    parts.append("Reference answer:\n" + reference_output)
    parts.append("Answering constraint:\n" + reference_conditioning_instruction.strip())

    return "\n\n".join(parts)


def _build_reference_conditioned_messages(
    example: dict[str, Any],
    *,
    system_mode: str,
    default_system_prompt: str,
    reference_conditioning_instruction: str,
) -> list[dict[str, str]]:
    messages: list[dict[str, str]] = []
    system_text = pair_inference.get_system_message(
        example,
        system_mode=system_mode,
        default_system_prompt=default_system_prompt,
    )
    if system_text:
        messages.append({"role": "system", "content": system_text})

    for user_text, assistant_text in pair_inference.get_history(example):
        if user_text:
            messages.append({"role": "user", "content": user_text})
        if assistant_text:
            messages.append({"role": "assistant", "content": assistant_text})

    messages.append(
        {
            "role": "user",
            "content": _build_reference_conditioned_user_message(
                example,
                reference_conditioning_instruction=reference_conditioning_instruction,
            ),
        }
    )
    return messages


def _patch_reference_conditioned_renderer(reference_conditioning_instruction: str) -> None:
    original_build_output_row = pair_inference.build_output_row

    def render_chat_prompt(
        tokenizer: Any,
        example: dict[str, Any],
        *,
        system_mode: str,
        default_system_prompt: str,
        **_ignored: Any,
    ) -> str:
        messages = _build_reference_conditioned_messages(
            example,
            system_mode=system_mode,
            default_system_prompt=default_system_prompt,
            reference_conditioning_instruction=reference_conditioning_instruction,
        )
        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        if not isinstance(prompt, str) or not prompt.strip():
            raise ValueError("Tokenizer chat template returned an empty prompt.")
        return prompt

    def build_output_row(*args: Any, **kwargs: Any) -> dict[str, Any]:
        row = original_build_output_row(*args, **kwargs)
        row["reference_conditioning"] = True
        return row

    pair_inference.render_chat_prompt = render_chat_prompt
    pair_inference.build_output_row = build_output_row


def build_parser() -> argparse.ArgumentParser:
    parser = pair_inference.build_pair_parser(
        description=(
            "Run reference-conditioned HDPO-checkpoint inference against a vLLM completions endpoint. "
            "This intentionally includes `chosen` in the prompt and is only for oracle/leakage analysis."
        ),
        default_system_mode="profile",
        default_system_prompt=DEFAULT_SYSTEM_PROMPT,
    )
    if not any("--conditioning-mode" in action.option_strings for action in parser._actions):
        parser.add_argument(
            "--conditioning-mode",
            choices=("profile",),
            help="Compatibility alias for profile-conditioned inference. Maps to --system-mode profile.",
        )
    if not any("--reference-conditioning-instruction" in action.option_strings for action in parser._actions):
        parser.add_argument(
            "--reference-conditioning-instruction",
            default=REFERENCE_CONDITIONING_INSTRUCTION,
            help="Instruction appended after the reference answer.",
        )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    if getattr(args, "conditioning_mode", None) == "profile":
        args.system_mode = "profile"
    if hasattr(args, "reference_conditioning"):
        args.reference_conditioning = True

    reference_conditioning_instruction = getattr(
        args,
        "reference_conditioning_instruction",
        REFERENCE_CONDITIONING_INSTRUCTION,
    )
    _patch_reference_conditioned_renderer(reference_conditioning_instruction)

    summary = pair_inference.run_pair_inference(
        args,
        run_kind="hdpo_ref_conditioned",
        progress_desc="hdpo-ref-conditioned-vllm-batch",
    )
    summary["reference_conditioning"] = True
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
