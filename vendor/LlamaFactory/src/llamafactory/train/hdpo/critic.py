# Copyright 2025 the LlamaFactory team.
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

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Optional, Sequence

import torch
from torch import nn


if TYPE_CHECKING:
    from transformers import PreTrainedModel, PreTrainedTokenizerBase


HDPO_CRITIC_CONFIG = "critic_config.json"
HDPO_BUCKET_VOCAB = "bucket_vocab.json"
HDPO_CRITIC_WEIGHTS = "critic.pt"
UNKNOWN_BUCKET = "__unknown_bucket__"


@dataclass
class HDPOCriticLossOutput:
    loss: "torch.Tensor"
    reg_loss: "torch.Tensor"
    rank_loss: "torch.Tensor"


@dataclass
class HDPOCriticBundleConfig:
    encoder_name_or_path: str
    response_dim: int
    num_buckets: int
    max_length: int = 512
    bucket_dim: int = 64
    hidden_dim: int = 256
    dropout: float = 0.1
    trust_remote_code: bool = False
    pooling: str = "mean"


@dataclass
class HDPOCriticBundle:
    critic: "BucketConditionedDistributionCritic"
    encoder_model: "PreTrainedModel"
    tokenizer: "PreTrainedTokenizerBase"
    bucket_to_id: dict[str, int]
    config: HDPOCriticBundleConfig
    device: "torch.device"


class BucketConditionedDistributionCritic(nn.Module):
    r"""A small bucket-conditioned critic used to predict distance-to-support scores offline."""

    def __init__(
        self,
        response_dim: int,
        num_buckets: int,
        bucket_dim: int = 64,
        hidden_dim: int = 256,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.bucket_embedding = nn.Embedding(num_buckets, bucket_dim)
        self.mlp = nn.Sequential(
            nn.Linear(response_dim + bucket_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, response_embedding: "torch.Tensor", bucket_ids: "torch.Tensor") -> "torch.Tensor":
        bucket_embedding = self.bucket_embedding(bucket_ids)
        critic_input = torch.cat([response_embedding, bucket_embedding], dim=-1)
        return self.mlp(critic_input).squeeze(-1)


def _resolve_hdpo_device(device: str = "auto") -> "torch.device":
    if device != "auto":
        return torch.device(device)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch, "npu") and torch.npu.is_available():
        return torch.device("npu")
    if hasattr(torch, "xpu") and torch.xpu.is_available():
        return torch.device("xpu")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def mean_pool_last_hidden(last_hidden_state: "torch.Tensor", attention_mask: "torch.Tensor") -> "torch.Tensor":
    mask = attention_mask.unsqueeze(-1).to(dtype=last_hidden_state.dtype)
    pooled = (last_hidden_state * mask).sum(dim=1)
    denom = mask.sum(dim=1).clamp_min(1.0)
    return pooled / denom


def encode_response_texts(
    encoder_model: "PreTrainedModel",
    tokenizer: "PreTrainedTokenizerBase",
    texts: Sequence[str],
    *,
    device: "torch.device | str" = "auto",
    max_length: int = 512,
    pooling: str = "mean",
) -> "torch.Tensor":
    if pooling != "mean":
        raise ValueError(f"Unsupported HDPO critic pooling mode: {pooling}.")

    resolved_device = _resolve_hdpo_device(str(device)) if not isinstance(device, torch.device) else device
    safe_texts = [text if str(text).strip() else (tokenizer.eos_token or tokenizer.pad_token or " ") for text in texts]
    tokenized = tokenizer(
        safe_texts,
        padding=True,
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
    )
    tokenized = {key: value.to(resolved_device) for key, value in tokenized.items()}
    with torch.no_grad():
        model_output = encoder_model(**tokenized, output_hidden_states=False, return_dict=True)
    return mean_pool_last_hidden(model_output.last_hidden_state, tokenized["attention_mask"])


def save_hdpo_critic_bundle(
    output_dir: str | Path,
    *,
    critic: "BucketConditionedDistributionCritic",
    bucket_to_id: dict[str, int],
    config: HDPOCriticBundleConfig,
) -> None:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    (output_path / HDPO_CRITIC_CONFIG).write_text(
        json.dumps(asdict(config), indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    (output_path / HDPO_BUCKET_VOCAB).write_text(
        json.dumps(bucket_to_id, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    torch.save(critic.state_dict(), output_path / HDPO_CRITIC_WEIGHTS)


def load_hdpo_critic_bundle(output_dir: str | Path, *, device: str = "auto") -> HDPOCriticBundle:
    from transformers import AutoModel, AutoTokenizer

    output_path = Path(output_dir)
    config = HDPOCriticBundleConfig(
        **json.loads((output_path / HDPO_CRITIC_CONFIG).read_text(encoding="utf-8"))
    )
    bucket_to_id = json.loads((output_path / HDPO_BUCKET_VOCAB).read_text(encoding="utf-8"))
    resolved_device = _resolve_hdpo_device(device)

    tokenizer = AutoTokenizer.from_pretrained(
        config.encoder_name_or_path,
        trust_remote_code=config.trust_remote_code,
        use_fast=True,
    )
    if tokenizer.pad_token_id is None:
        if tokenizer.eos_token is not None:
            tokenizer.pad_token = tokenizer.eos_token
        elif tokenizer.bos_token is not None:
            tokenizer.pad_token = tokenizer.bos_token

    encoder_model = AutoModel.from_pretrained(
        config.encoder_name_or_path,
        trust_remote_code=config.trust_remote_code,
    )
    encoder_model.to(resolved_device)
    encoder_model.eval()

    critic = BucketConditionedDistributionCritic(
        response_dim=config.response_dim,
        num_buckets=config.num_buckets,
        bucket_dim=config.bucket_dim,
        hidden_dim=config.hidden_dim,
        dropout=config.dropout,
    )
    critic.load_state_dict(torch.load(output_path / HDPO_CRITIC_WEIGHTS, map_location=resolved_device))
    critic.to(resolved_device)
    critic.eval()

    return HDPOCriticBundle(
        critic=critic,
        encoder_model=encoder_model,
        tokenizer=tokenizer,
        bucket_to_id=bucket_to_id,
        config=config,
        device=resolved_device,
    )


def bucket_ids_from_names(bucket_to_id: dict[str, int], bucket_names: Sequence[str]) -> list[int]:
    unknown_bucket_id = bucket_to_id.get(UNKNOWN_BUCKET, 0)
    return [int(bucket_to_id.get(bucket_name, unknown_bucket_id)) for bucket_name in bucket_names]


def predict_hdpo_critic_scores(
    bundle: HDPOCriticBundle,
    texts: Sequence[str],
    bucket_ids: Sequence[int],
    *,
    batch_size: int = 8,
) -> "torch.Tensor":
    if len(texts) != len(bucket_ids):
        raise ValueError("texts and bucket_ids must have the same length.")

    score_chunks: list[torch.Tensor] = []
    for start in range(0, len(texts), batch_size):
        end = start + batch_size
        text_chunk = list(texts[start:end])
        bucket_chunk = torch.tensor(bucket_ids[start:end], dtype=torch.long, device=bundle.device)
        response_embeddings = encode_response_texts(
            bundle.encoder_model,
            bundle.tokenizer,
            text_chunk,
            device=bundle.device,
            max_length=bundle.config.max_length,
            pooling=bundle.config.pooling,
        )
        with torch.no_grad():
            score_chunks.append(bundle.critic(response_embeddings, bucket_chunk).detach().cpu())

    if not score_chunks:
        return torch.empty(0, dtype=torch.float32)
    return torch.cat(score_chunks, dim=0)


def compute_hdpo_critic_loss(
    pred_scores: "torch.Tensor",
    target_scores: "torch.Tensor",
    *,
    chosen_pred_scores: Optional["torch.Tensor"] = None,
    rejected_pred_scores: Optional["torch.Tensor"] = None,
    margin: float = 0.1,
    reg_lambda: float = 1.0,
    rank_lambda: float = 1.0,
) -> HDPOCriticLossOutput:
    reg_loss = torch.abs(pred_scores - target_scores).mean()
    if chosen_pred_scores is None or rejected_pred_scores is None:
        rank_loss = torch.zeros_like(reg_loss)
    else:
        rank_loss = torch.relu(margin - rejected_pred_scores + chosen_pred_scores).mean()

    loss = reg_lambda * reg_loss + rank_lambda * rank_loss
    return HDPOCriticLossOutput(loss=loss, reg_loss=reg_loss, rank_loss=rank_loss)
