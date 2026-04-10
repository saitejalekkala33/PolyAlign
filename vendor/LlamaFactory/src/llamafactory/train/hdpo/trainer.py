# Copyright 2025 HuggingFace Inc. and the LlamaFactory team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import TYPE_CHECKING, Literal

import torch

from .critic import load_hdpo_critic_bundle, predict_hdpo_critic_scores
from ..dpo.trainer import CustomDPOTrainer


if TYPE_CHECKING:
    from transformers import PreTrainedModel


HDPO_METADATA_KEYS = {
    "hdpo_weight",
    "critic_bucket_id",
    "chosen_dist_score",
    "rejected_dist_score",
}


class CustomHDPOTrainer(CustomDPOTrainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.hdpo_online_bundle = None
        self.hdpo_online_reg_lambda = self.finetuning_args.hdpo_online_reg_lambda
        if self.hdpo_online_reg_lambda > 0:
            self.hdpo_online_bundle = load_hdpo_critic_bundle(
                self.finetuning_args.hdpo_critic_path,
                device=self.finetuning_args.hdpo_critic_device,
            )

    def _sanitize_batch(self, batch: dict[str, "torch.Tensor"]) -> dict[str, "torch.Tensor"]:
        return {key: value for key, value in batch.items() if key not in HDPO_METADATA_KEYS}

    def _compute_distribution_regularizer(
        self,
        policy_chosen_logps: "torch.Tensor",
        policy_rejected_logps: "torch.Tensor",
        chosen_dist_scores: "torch.Tensor",
        rejected_dist_scores: "torch.Tensor",
    ) -> "torch.Tensor":
        pair_pref = torch.sigmoid(self.beta * (policy_chosen_logps - policy_rejected_logps))
        return pair_pref * chosen_dist_scores + (1.0 - pair_pref) * rejected_dist_scores

    def _get_tokenizer(self):
        tokenizer = getattr(self, "processing_class", None)
        if tokenizer is None:
            tokenizer = getattr(self, "tokenizer", None)
        if tokenizer is None:
            raise ValueError("HDPO online regularization requires an attached tokenizer.")
        return tokenizer

    def _extract_prompt_sequences(self, batch: dict[str, "torch.Tensor"]) -> tuple[list["torch.Tensor"], list[int]]:
        batch_size = batch["input_ids"].size(0) // 2
        chosen_input_ids = batch["input_ids"][:batch_size]
        chosen_attention_mask = batch["attention_mask"][:batch_size]
        chosen_labels = batch["labels"][:batch_size]

        prompt_sequences: list[torch.Tensor] = []
        kept_indices: list[int] = []
        for sample_index, (input_ids, attention_mask, labels) in enumerate(
            zip(chosen_input_ids, chosen_attention_mask, chosen_labels, strict=True)
        ):
            valid_positions = torch.nonzero(attention_mask != 0, as_tuple=False).flatten()
            if valid_positions.numel() == 0:
                continue

            start = int(valid_positions[0].item())
            end = int(valid_positions[-1].item()) + 1
            valid_input_ids = input_ids[start:end]
            valid_labels = labels[start:end]
            target_positions = torch.nonzero(valid_labels != -100, as_tuple=False).flatten()
            if target_positions.numel() == 0:
                prompt_end = valid_input_ids.size(0)
            else:
                prompt_end = int(target_positions[0].item())
            prompt_ids = valid_input_ids[:prompt_end]
            if prompt_ids.numel() == 0:
                prompt_ids = valid_input_ids[:1]
            prompt_sequences.append(prompt_ids.detach())
            kept_indices.append(sample_index)

        return prompt_sequences, kept_indices

    def _generate_online_samples(
        self,
        model: "PreTrainedModel",
        prompt_sequences: list["torch.Tensor"],
    ) -> list[str]:
        tokenizer = self._get_tokenizer()
        unwrapped_model = self.accelerator.unwrap_model(model)
        was_training = unwrapped_model.training
        model_device = next(unwrapped_model.parameters()).device
        prompt_texts: list[str] = []
        with torch.no_grad():
            unwrapped_model.eval()
            for prompt_ids in prompt_sequences:
                prompt_ids = prompt_ids.to(model_device)
                attention_mask = torch.ones_like(prompt_ids).unsqueeze(0)
                generate_kwargs = dict(
                    input_ids=prompt_ids.unsqueeze(0),
                    attention_mask=attention_mask,
                    max_new_tokens=self.finetuning_args.hdpo_online_max_new_tokens,
                    do_sample=self.finetuning_args.hdpo_online_temperature > 0,
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                    use_cache=True,
                )
                if self.finetuning_args.hdpo_online_temperature > 0:
                    generate_kwargs["temperature"] = self.finetuning_args.hdpo_online_temperature
                    generate_kwargs["top_p"] = self.finetuning_args.hdpo_online_top_p
                generated = unwrapped_model.generate(**generate_kwargs)
                completion_ids = generated[0, prompt_ids.size(0) :]
                prompt_texts.append(tokenizer.decode(completion_ids, skip_special_tokens=True).strip())

        if was_training:
            unwrapped_model.train()
        return prompt_texts

    def _compute_online_distribution_regularizer(
        self,
        model: "PreTrainedModel",
        batch: dict[str, "torch.Tensor"],
        reference_tensor: "torch.Tensor",
    ) -> "torch.Tensor":
        if self.hdpo_online_bundle is None or self.hdpo_online_reg_lambda <= 0:
            return torch.zeros((), device=reference_tensor.device, dtype=reference_tensor.dtype)

        if (self.state.global_step % self.finetuning_args.hdpo_online_reg_interval) != 0:
            return torch.zeros((), device=reference_tensor.device, dtype=reference_tensor.dtype)

        critic_bucket_ids = batch.get("critic_bucket_id")
        if critic_bucket_ids is None:
            return torch.zeros((), device=reference_tensor.device, dtype=reference_tensor.dtype)

        prompt_sequences, kept_indices = self._extract_prompt_sequences(batch)
        if not prompt_sequences:
            return torch.zeros((), device=reference_tensor.device, dtype=reference_tensor.dtype)

        sampled_texts = self._generate_online_samples(model, prompt_sequences)
        if not sampled_texts:
            return torch.zeros((), device=reference_tensor.device, dtype=reference_tensor.dtype)

        bucket_id_list = critic_bucket_ids.detach().cpu().tolist()
        sampled_scores = predict_hdpo_critic_scores(
            self.hdpo_online_bundle,
            sampled_texts,
            [bucket_id_list[index] for index in kept_indices],
            batch_size=min(len(prompt_sequences), 8),
        )
        if sampled_scores.numel() == 0:
            return torch.zeros((), device=reference_tensor.device, dtype=reference_tensor.dtype)

        return sampled_scores.to(device=reference_tensor.device, dtype=reference_tensor.dtype).mean()

    def concatenated_forward(
        self, model: "PreTrainedModel", batch: dict[str, "torch.Tensor"], is_ref_model: bool = False
    ) -> dict[str, "torch.Tensor"]:
        return super().concatenated_forward(model, self._sanitize_batch(batch), is_ref_model=is_ref_model)

    def compute_reference_log_probs(
        self, model: "PreTrainedModel", batch: dict[str, "torch.Tensor"]
    ) -> tuple["torch.Tensor | None", "torch.Tensor | None"]:
        return super().compute_reference_log_probs(model, self._sanitize_batch(batch))

    def get_batch_loss_metrics(
        self,
        model: "PreTrainedModel",
        batch: dict[str, "torch.Tensor"],
        train_eval: Literal["train", "eval"] = "train",
    ) -> tuple["torch.Tensor", dict[str, "torch.Tensor"]]:
        metrics = {}

        hdpo_weight = batch.get("hdpo_weight")
        chosen_dist_scores = batch.get("chosen_dist_score")
        rejected_dist_scores = batch.get("rejected_dist_score")

        model_output = self.concatenated_forward(model, batch)
        policy_chosen_logps = model_output["chosen_logps"]
        policy_rejected_logps = model_output["rejected_logps"]
        policy_chosen_logits = model_output["chosen_logits"]
        policy_rejected_logits = model_output["rejected_logits"]
        policy_chosen_logps_avg = model_output["chosen_logps_avg"]
        if hdpo_weight is None:
            hdpo_weight = torch.ones_like(policy_chosen_logps)
        if chosen_dist_scores is None:
            chosen_dist_scores = torch.zeros_like(policy_chosen_logps)
        if rejected_dist_scores is None:
            rejected_dist_scores = torch.zeros_like(policy_chosen_logps)

        reference_chosen_logps, reference_rejected_logps = self.compute_reference_log_probs(model, batch)
        pref_losses, chosen_rewards, rejected_rewards = self.compute_preference_loss(
            policy_chosen_logps,
            policy_rejected_logps,
            reference_chosen_logps,
            reference_rejected_logps,
        )
        hdpo_weight = hdpo_weight.to(device=pref_losses.device, dtype=pref_losses.dtype)
        chosen_dist_scores = chosen_dist_scores.to(device=pref_losses.device, dtype=pref_losses.dtype)
        rejected_dist_scores = rejected_dist_scores.to(device=pref_losses.device, dtype=pref_losses.dtype)

        sft_loss = -policy_chosen_logps_avg
        dist_reg = self._compute_distribution_regularizer(
            policy_chosen_logps,
            policy_rejected_logps,
            chosen_dist_scores,
            rejected_dist_scores,
        )
        weighted_pref_loss = (pref_losses * hdpo_weight).sum() / hdpo_weight.sum().clamp_min(1e-8)
        dist_reg_loss = dist_reg.mean()
        loss = weighted_pref_loss + self.finetuning_args.hdpo_reg_lambda * dist_reg_loss
        if train_eval == "train":
            online_reg_loss = self._compute_online_distribution_regularizer(model, batch, pref_losses)
        else:
            online_reg_loss = torch.zeros((), device=pref_losses.device, dtype=pref_losses.dtype)
        if self.hdpo_online_reg_lambda > 0:
            loss += self.hdpo_online_reg_lambda * online_reg_loss
        if self.ftx_gamma > 1e-6:
            loss += self.ftx_gamma * sft_loss.mean()

        prefix = "eval_" if train_eval == "eval" else ""
        metrics[f"{prefix}rewards/chosen"] = chosen_rewards.mean().item()
        metrics[f"{prefix}rewards/rejected"] = rejected_rewards.mean().item()
        metrics[f"{prefix}rewards/accuracies"] = (chosen_rewards > rejected_rewards).float().mean().item()
        metrics[f"{prefix}rewards/margins"] = (chosen_rewards - rejected_rewards).mean().item()
        metrics[f"{prefix}logps/chosen"] = policy_chosen_logps.mean().item()
        metrics[f"{prefix}logps/rejected"] = policy_rejected_logps.mean().item()
        metrics[f"{prefix}logits/chosen"] = policy_chosen_logits.mean().item()
        metrics[f"{prefix}logits/rejected"] = policy_rejected_logits.mean().item()
        metrics[f"{prefix}hdpo/weight_mean"] = hdpo_weight.mean().item()
        metrics[f"{prefix}hdpo/chosen_dist_score"] = chosen_dist_scores.mean().item()
        metrics[f"{prefix}hdpo/rejected_dist_score"] = rejected_dist_scores.mean().item()
        metrics[f"{prefix}hdpo/dist_gap"] = (rejected_dist_scores - chosen_dist_scores).mean().item()
        metrics[f"{prefix}hdpo/dist_reg"] = dist_reg_loss.item()
        if self.hdpo_online_reg_lambda > 0:
            metrics[f"{prefix}hdpo/online_reg"] = online_reg_loss.item()
        if self.ftx_gamma > 1e-6:
            metrics[f"{prefix}hdpo/sft_anchor"] = sft_loss.mean().item()

        return loss, metrics
