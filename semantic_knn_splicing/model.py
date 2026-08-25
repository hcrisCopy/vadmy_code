"""Whole-layer semantic localizer adapted from video AnomalyCLIP.

Source mechanisms:
- CoOp prompt learner: rely/AnomalyCLIP/src/models/components/coop.py
- normal-centroid recentering and text top-k selection:
  rely/AnomalyCLIP/src/models/components/selector_model.py
- 768->512 adapter: vendor/dsanet/src/utils/adapter_modules.py

The CLIP visual and text backbones are frozen.  Only shared context tokens,
whole-layer projectors and layer weights are trained.
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn

from .common import state_dict_from_checkpoint
from .prompts import PROMPT_BANKS, abnormal_class_names


def load_frozen_clip(weight_path: str, device: torch.device):
    source = Path(__file__).resolve().parent / "vendor" / "dsanet" / "src"
    sys.path.insert(0, str(source))
    try:
        from clip.clip import tokenize
        from clip.model import build_model
    finally:
        sys.path.pop(0)
    state = state_dict_from_checkpoint(weight_path)
    candidates = ("clipmodel.", "base.clipmodel.", "model.clipmodel.")
    clip_state = {}
    for prefix in candidates:
        value = {
            str(key)[len(prefix):]: tensor
            for key, tensor in state.items()
            if str(key).startswith(prefix) and "prompt_learner" not in str(key)
        }
        if "visual.proj" in value:
            clip_state = value
            break
    if not clip_state:
        raise KeyError("reference checkpoint has no reusable clipmodel.* weights")
    model = build_model(clip_state).float().to(device).eval()
    model.requires_grad_(False)
    return model, tokenize


class DescriptionPromptLearner(nn.Module):
    """Shared CoOp context inserted before every LaGoVAD description."""

    def __init__(self, clip_model: nn.Module, tokenize, dataset: str, context_length: int = 8) -> None:
        super().__init__()
        if context_length <= 0:
            raise ValueError("context_length must be positive")
        descriptions: list[str] = []
        owners: list[int] = []
        names = abnormal_class_names(dataset)
        for class_index, name in enumerate(names):
            for description in PROMPT_BANKS[dataset][name]:
                descriptions.append(description)
                owners.append(class_index)
        prefix = " ".join(["X"] * context_length)
        tokens = torch.cat([tokenize(f"{prefix} {value}.") for value in descriptions])
        tokens = tokens.to(clip_model.token_embedding.weight.device)
        with torch.no_grad():
            embedding = clip_model.token_embedding(tokens).float()
        width = int(embedding.shape[-1])
        context = torch.empty(context_length, width, dtype=torch.float32)
        nn.init.normal_(context, std=0.02)
        self.context = nn.Parameter(context)
        self.register_buffer("token_prefix", embedding[:, :1])
        self.register_buffer("token_suffix", embedding[:, 1 + context_length:])
        self.register_buffer("tokens", tokens)
        self.register_buffer("owners", torch.tensor(owners, dtype=torch.long))
        self.num_classes = len(names)

    def forward(self, clip_model: nn.Module) -> torch.Tensor:
        context = self.context.unsqueeze(0).expand(self.token_prefix.shape[0], -1, -1)
        embeddings = torch.cat([self.token_prefix, context, self.token_suffix], dim=1)
        encoded = clip_model.encode_text(embeddings, self.tokens)
        encoded = F.normalize(encoded.float(), dim=-1)
        prototypes = []
        for class_index in range(self.num_classes):
            value = encoded[self.owners == class_index].mean(dim=0)
            prototypes.append(F.normalize(value, dim=0))
        return torch.stack(prototypes)


class WholeLayerSemanticLocalizer(nn.Module):
    method_name = "whole_layer_anomalyclip_localizer_v1"

    def __init__(
        self,
        selected_layers: list[int],
        clip_model: nn.Module,
        tokenize,
        dataset: str,
        context_length: int = 8,
    ) -> None:
        super().__init__()
        if not selected_layers:
            raise ValueError("selected_layers cannot be empty")
        self.selected_layers = [int(value) for value in selected_layers]
        self.clip_model = clip_model
        self.clip_model.requires_grad_(False)
        self.prompt_learner = DescriptionPromptLearner(
            clip_model, tokenize, dataset, context_length=context_length
        )
        self.layer_norms = nn.ModuleList([nn.LayerNorm(768) for _ in selected_layers])
        self.layer_projectors = nn.ModuleList([
            nn.Sequential(nn.Linear(768, 512, bias=False), nn.LeakyReLU())
            for _ in selected_layers
        ])
        self.layer_logits = nn.Parameter(torch.zeros(len(selected_layers)))
        self.residual_scale = nn.Parameter(torch.tensor(-2.1972246))  # sigmoid=0.1
        self.semantic_bn = nn.BatchNorm1d(len(abnormal_class_names(dataset)), affine=False)
        self.register_buffer("normal_centroid", torch.zeros(512))

    @property
    def residual_weight(self) -> torch.Tensor:
        return torch.sigmoid(self.residual_scale)

    def set_normal_centroid(self, value: torch.Tensor) -> None:
        if value.shape != (512,):
            raise ValueError(f"normal centroid must be [512], got {tuple(value.shape)}")
        self.normal_centroid.copy_(F.normalize(value.float(), dim=0))

    def fused_features(self, clip: torch.Tensor, hidden: torch.Tensor) -> torch.Tensor:
        if clip.ndim != 3 or clip.shape[-1] != 512:
            raise ValueError(f"clip must be [B,T,512], got {tuple(clip.shape)}")
        if hidden.ndim != 4 or hidden.shape[2] != len(self.selected_layers) or hidden.shape[-1] != 768:
            raise ValueError(
                f"hidden must be [B,T,{len(self.selected_layers)},768], got {tuple(hidden.shape)}"
            )
        values = []
        for index, (norm, projector) in enumerate(zip(self.layer_norms, self.layer_projectors)):
            projected = projector(norm(hidden[:, :, index].float()))
            projected = projected * clip.norm(dim=-1, keepdim=True) / projected.norm(dim=-1, keepdim=True).clamp_min(1e-6)
            values.append(projected)
        weights = F.softmax(self.layer_logits, dim=0)
        layer_value = sum(weight * value for weight, value in zip(weights, values))
        weight = self.residual_weight
        return F.normalize((1.0 - weight) * clip.float() + weight * layer_value, dim=-1)

    def forward(self, clip: torch.Tensor, hidden: torch.Tensor) -> dict[str, torch.Tensor]:
        features = self.fused_features(clip, hidden)
        text = self.prompt_learner(self.clip_model)
        centered_features = features - self.normal_centroid
        centered_text = F.normalize(text - self.normal_centroid.unsqueeze(0), dim=-1)
        class_logits = centered_features @ centered_text.T
        shape = class_logits.shape
        class_logits = self.semantic_bn(class_logits.reshape(-1, shape[-1])).reshape(shape)
        semantic_score = class_logits.max(dim=-1).values
        return {
            "features": features,
            "text_features": text,
            "class_logits": class_logits,
            "anomaly_logits": semantic_score,
            "semantic_score": semantic_score,
        }

    def config(self) -> dict:
        return {
            "method": self.method_name,
            "selected_layers_zero_based": self.selected_layers,
            "num_layers": len(self.selected_layers),
            "clip_trainable": False,
        }


def localizer_loss(
    record: dict[str, torch.Tensor],
    target_mask: torch.Tensor,
    binary_label: torch.Tensor,
    lengths: torch.Tensor,
    topk_ratio: int = 16,
    smooth_weight: float = 8e-4,
    sparse_weight: float = 8e-3,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """AnomalyCLIP-style top/bottom MIL with multi-label XD support."""
    class_logits = record["class_logits"]
    anomaly_logits = record["anomaly_logits"]
    positive_terms, normal_terms, bottom_terms, smooth_terms, sparse_terms = [], [], [], [], []
    for index, length_value in enumerate(lengths.tolist()):
        length = max(1, int(length_value))
        k = max(1, length // topk_ratio)
        valid_class = class_logits[index, :length]
        valid_anomaly = anomaly_logits[index, :length]
        active = torch.where(target_mask[index] > 0)[0]
        if binary_label[index] > 0.5 and len(active):
            target_score = valid_class.index_select(1, active).max(dim=1).values
            positive_terms.append(F.softplus(-target_score.topk(k).values).mean())
            bottom_terms.append(F.softplus(target_score.topk(k, largest=False).values).mean())
        else:
            normal_terms.append(F.softplus(valid_class.max(dim=1).values.topk(k).values).mean())
        probability = torch.sigmoid(valid_anomaly)
        if length > 1:
            smooth_terms.append((probability[1:] - probability[:-1]).square().mean())
        sparse_terms.append(probability.mean())
    zero = class_logits.sum() * 0.0
    mean = lambda values: torch.stack(values).mean() if values else zero
    pieces = {
        "positive": mean(positive_terms),
        "normal": mean(normal_terms),
        "bottom": mean(bottom_terms),
        "smooth": mean(smooth_terms),
        "sparse": mean(sparse_terms),
    }
    total = pieces["positive"] + pieces["normal"] + pieces["bottom"]
    total = total + smooth_weight * pieces["smooth"] + sparse_weight * pieces["sparse"]
    return total, pieces
