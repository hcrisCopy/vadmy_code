"""Responsibility-selected whole-layer semantic expert.

The frozen CLIP LayerNorm/projection follows CLIP and the earlier semantic
lens.  The residual bottleneck follows the adapter pattern used by DSANet.
Only the small adapters and layer gate are trainable; neither CLIP backbone is
kept in the training graph.
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from .common import state_dict_from_checkpoint
from .prompts import PROMPT_BANKS, abnormal_class_names, normal_class_name


@dataclass
class ClipResources:
    ln_weight: torch.Tensor
    ln_bias: torch.Tensor
    projection: torch.Tensor
    event_text: torch.Tensor
    event_owner: torch.Tensor
    normal_text: torch.Tensor


def selected_layer_spec(atlas_path: str) -> tuple[list[int], list[float]]:
    atlas = json.loads(Path(atlas_path).read_text(encoding="utf-8"))
    blocks = atlas.get("blocks", [])
    if not blocks:
        raise ValueError(f"{atlas_path}: no selected layer blocks")
    layers = [int(block["layer_zero_based"]) for block in blocks]
    evidence = [
        float(block["width"]) * float(block.get("direction_stability", 1.0))
        for block in blocks
    ]
    total = sum(evidence)
    weights = [value / total for value in evidence] if total > 0 else [1 / len(layers)] * len(layers)
    return layers, weights


def load_clip_resources(weight_path: str, dataset: str, device: torch.device) -> ClipResources:
    source = Path(__file__).resolve().parent / "vendor" / "dsanet" / "src"
    sys.path.insert(0, str(source))
    try:
        from clip.clip import tokenize
        from clip.model import build_model
    finally:
        sys.path.pop(0)
    state = state_dict_from_checkpoint(weight_path)
    clip_state = {}
    for prefix in ("clipmodel.", "base.clipmodel.", "model.clipmodel."):
        candidate = {
            str(key)[len(prefix):]: value
            for key, value in state.items()
            if str(key).startswith(prefix) and "prompt_learner" not in str(key)
        }
        if "visual.proj" in candidate:
            clip_state = candidate
            break
    if not clip_state:
        raise KeyError("checkpoint has no reusable clipmodel.* weights")
    clip_model = build_model(clip_state).float().to(device).eval()
    clip_model.requires_grad_(False)
    events, owners = [], []
    for class_index, class_name in enumerate(abnormal_class_names(dataset)):
        for prompt in PROMPT_BANKS[dataset][class_name]:
            events.append(prompt)
            owners.append(class_index)
    normal_prompts = PROMPT_BANKS[dataset][normal_class_name(dataset)]
    with torch.no_grad():
        event_tokens = torch.cat([tokenize(text) for text in events]).to(device)
        normal_tokens = torch.cat([tokenize(text) for text in normal_prompts]).to(device)
        event_text = clip_model.encode_text(clip_model.encode_token(event_tokens), event_tokens)
        normal_text = clip_model.encode_text(clip_model.encode_token(normal_tokens), normal_tokens)
    return ClipResources(
        ln_weight=clip_model.visual.ln_post.weight.detach().float().cpu(),
        ln_bias=clip_model.visual.ln_post.bias.detach().float().cpu(),
        projection=clip_model.visual.proj.detach().float().cpu(),
        event_text=F.normalize(event_text.float(), dim=-1).cpu(),
        event_owner=torch.tensor(owners, dtype=torch.long),
        normal_text=F.normalize(normal_text.float(), dim=-1).cpu(),
    )


def project_hidden(
    hidden: torch.Tensor,
    ln_weight: torch.Tensor,
    ln_bias: torch.Tensor,
    projection: torch.Tensor,
) -> torch.Tensor:
    value = F.layer_norm(hidden.float(), (hidden.shape[-1],), ln_weight, ln_bias)
    return F.normalize(value @ projection, dim=-1)


class ResidualBottleneck(nn.Module):
    """Zero-start residual adapter; adapted from DSANet's adapter design."""

    def __init__(self, width: int = 512, bottleneck: int = 64) -> None:
        super().__init__()
        self.down = nn.Linear(width, bottleneck, bias=False)
        self.up = nn.Linear(bottleneck, width, bias=False)
        nn.init.kaiming_uniform_(self.down.weight, a=np.sqrt(5))
        nn.init.zeros_(self.up.weight)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return value + self.up(F.gelu(self.down(value)))


class ResponsibilitySemanticExpert(nn.Module):
    method_name = "responsibility_whole_layer_normality_semantic_expert_v1"

    def __init__(
        self,
        layers: list[int],
        initial_layer_weights: list[float],
        resources: ClipResources,
        normal_prototype: torch.Tensor,
        bottleneck: int = 64,
    ) -> None:
        super().__init__()
        if normal_prototype.shape != (len(layers), 512):
            raise ValueError(
                f"normal prototype must be [{len(layers)},512], got {tuple(normal_prototype.shape)}"
            )
        self.layers = [int(value) for value in layers]
        self.adapters = nn.ModuleList(
            [ResidualBottleneck(512, bottleneck) for _ in self.layers]
        )
        initial = torch.tensor(initial_layer_weights, dtype=torch.float32).clamp_min(1e-8)
        self.layer_logits = nn.Parameter(initial.log())
        self.register_buffer("ln_weight", resources.ln_weight)
        self.register_buffer("ln_bias", resources.ln_bias)
        self.register_buffer("projection", resources.projection)
        self.register_buffer("event_text", resources.event_text)
        self.register_buffer("event_owner", resources.event_owner)
        self.register_buffer("normal_text", resources.normal_text)
        self.register_buffer("normal_prototype", F.normalize(normal_prototype.float(), dim=-1))
        self.class_count = int(resources.event_owner.max().item()) + 1
        self.bottleneck = int(bottleneck)

    @property
    def layer_weights(self) -> torch.Tensor:
        return F.softmax(self.layer_logits, dim=0)

    def forward(self, hidden: torch.Tensor) -> dict[str, torch.Tensor]:
        if hidden.ndim != 4 or hidden.shape[2] != len(self.layers) or hidden.shape[-1] != 768:
            raise ValueError(
                f"hidden must be [B,T,{len(self.layers)},768], got {tuple(hidden.shape)}"
            )
        projected = project_hidden(hidden, self.ln_weight, self.ln_bias, self.projection)
        adapted = torch.stack(
            [F.normalize(adapter(projected[:, :, index]), dim=-1) for index, adapter in enumerate(self.adapters)],
            dim=2,
        )
        prompt_similarity = torch.einsum("btld,pd->btlp", adapted, self.event_text)
        class_similarity = []
        for class_index in range(self.class_count):
            class_similarity.append(
                prompt_similarity[..., self.event_owner == class_index].amax(dim=-1)
            )
        class_similarity = torch.stack(class_similarity, dim=-1)
        normal_text_similarity = torch.einsum("btld,nd->btln", adapted, self.normal_text).amax(dim=-1)
        normal_visual_similarity = torch.einsum("btld,ld->btl", adapted, self.normal_prototype)
        normal_similarity = torch.maximum(normal_text_similarity, normal_visual_similarity)
        layer_margin = class_similarity - normal_similarity.unsqueeze(-1)
        class_margin = torch.einsum("btlc,l->btc", layer_margin, self.layer_weights)
        anomaly_logit = class_margin.amax(dim=-1)
        return {
            "anomaly_logit": anomaly_logit,
            "class_margin": class_margin,
            "layer_margin": layer_margin,
            "adapted": adapted,
        }

    def metadata(self) -> dict:
        return {
            "method": self.method_name,
            "selected_layers_zero_based": self.layers,
            "layer_weights": self.layer_weights.detach().cpu().tolist(),
            "adapter_bottleneck": self.bottleneck,
            "trainable_parameters": sum(p.numel() for p in self.parameters() if p.requires_grad),
            "clip_trainable_parameters": 0,
            "normality_sources": ["published normal text", "pure-normal visual prototype"],
        }


def build_semantic_expert(
    atlas_path: str,
    clip_weight: str,
    dataset: str,
    normal_prototype_path: str,
    bottleneck: int,
    device: torch.device,
) -> ResponsibilitySemanticExpert:
    layers, weights = selected_layer_spec(atlas_path)
    stored = np.load(normal_prototype_path)
    if not np.array_equal(stored["layers"], np.asarray(layers)):
        raise ValueError("normal prototype layer identities differ from layer atlas")
    resources = load_clip_resources(clip_weight, dataset, device)
    return ResponsibilitySemanticExpert(
        layers,
        weights,
        resources,
        torch.from_numpy(stored["prototype"].astype(np.float32)),
        bottleneck,
    ).to(device)
