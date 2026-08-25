"""Frozen CLIP whole-layer text lens with neuron-auditable scores.

The text direction follows LAP's event-prompt maximum. Intermediate CLS
states are mapped with CLIP's own final LayerNorm and visual projection; no
projector, prompt token, layer weight, or anomaly head is learned here.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn

from .common import state_dict_from_checkpoint
from .prompts import (
    PROMPT_BANKS,
    abnormal_class_names,
    normal_class_name,
    prompt_provenance,
)


def load_frozen_clip(weight_path: str, device: torch.device):
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
        raise KeyError("reference checkpoint has no reusable clipmodel.* weights")
    model = build_model(clip_state).float().to(device).eval()
    model.requires_grad_(False)
    return model, tokenize


def selected_layer_spec(atlas_path: str) -> tuple[list[int], list[float]]:
    atlas = json.loads(Path(atlas_path).read_text(encoding="utf-8"))
    blocks = atlas.get("blocks", [])
    if not blocks:
        raise ValueError(f"{atlas_path}: no selected layer blocks")
    layers = [int(block["layer_zero_based"]) for block in blocks]
    # Effective responsible-neuron count: selected union width discounted by
    # inconsistent positive/negative direction. Both values are already
    # produced by the score-free circuit discovery stage.
    evidence = [
        float(block["width"]) * float(block.get("direction_stability", 1.0))
        for block in blocks
    ]
    total = sum(evidence)
    if total <= 0:
        weights = [1.0 / len(evidence)] * len(evidence)
    else:
        weights = [value / total for value in evidence]
    return layers, weights


class FrozenWholeLayerSemanticLens(nn.Module):
    method_name = "frozen_clip_whole_layer_lap_margin_v2"

    def __init__(
        self,
        selected_layers: list[int],
        layer_weights: list[float],
        clip_model: nn.Module,
        tokenize,
        dataset: str,
    ) -> None:
        super().__init__()
        if len(selected_layers) != len(layer_weights) or not selected_layers:
            raise ValueError("selected layers and weights must have the same non-zero length")
        self.selected_layers = [int(value) for value in selected_layers]
        self.dataset = dataset
        self.clip_model = clip_model
        self.clip_model.requires_grad_(False)
        self.register_buffer("layer_weights", torch.tensor(layer_weights, dtype=torch.float32))

        class_names = abnormal_class_names(dataset)
        texts: list[str] = []
        owners: list[int] = []
        for class_index, name in enumerate(class_names):
            for prompt in PROMPT_BANKS[dataset][name]:
                texts.append(prompt)
                owners.append(class_index)
        normal_texts = PROMPT_BANKS[dataset][normal_class_name(dataset)]
        device = next(clip_model.parameters()).device
        with torch.no_grad():
            event = clip_model.encode_text(torch.cat([tokenize(value) for value in texts]).to(device))
            normal = clip_model.encode_text(
                torch.cat([tokenize(value) for value in normal_texts]).to(device)
            )
        self.register_buffer("event_text", F.normalize(event.float(), dim=-1))
        self.register_buffer("normal_text", F.normalize(normal.float(), dim=-1))
        self.register_buffer("event_owners", torch.tensor(owners, dtype=torch.long))
        self.class_names = class_names

    @classmethod
    def from_files(
        cls,
        atlas_path: str,
        clip_weight: str,
        dataset: str,
        device: torch.device,
    ) -> "FrozenWholeLayerSemanticLens":
        layers, weights = selected_layer_spec(atlas_path)
        clip_model, tokenize = load_frozen_clip(clip_weight, device)
        return cls(layers, weights, clip_model, tokenize, dataset).to(device).eval()

    def _project(self, hidden: torch.Tensor) -> torch.Tensor:
        if hidden.ndim != 4 or hidden.shape[2] != len(self.selected_layers):
            raise ValueError(
                f"hidden must be [B,T,{len(self.selected_layers)},768], got {tuple(hidden.shape)}"
            )
        values = []
        for layer_index in range(hidden.shape[2]):
            value = self.clip_model.visual.ln_post(hidden[:, :, layer_index].float())
            value = value @ self.clip_model.visual.proj
            values.append(F.normalize(value, dim=-1))
        return torch.stack(values, dim=2)

    def forward(self, hidden: torch.Tensor) -> dict[str, torch.Tensor]:
        projected = self._project(hidden)
        # [B,T,L,P], then LAP-style maximum over atomic prompts owned by a class.
        prompt_similarity = torch.einsum("btld,pd->btlp", projected, self.event_text)
        normal_similarity = torch.einsum("btld,nd->btln", projected, self.normal_text).amax(dim=-1)
        class_values = []
        for class_index in range(len(self.class_names)):
            mask = self.event_owners == class_index
            class_values.append(prompt_similarity[..., mask].amax(dim=-1))
        class_similarity = torch.stack(class_values, dim=-1)
        layer_margin = class_similarity - normal_similarity.unsqueeze(-1)
        class_margin = torch.einsum("btlc,l->btc", layer_margin, self.layer_weights)
        return {
            "class_margin": class_margin,
            "layer_margin": layer_margin,
            "projected": projected,
        }

    def config(self) -> dict:
        return {
            "method": self.method_name,
            "selected_layers_zero_based": self.selected_layers,
            "layer_weights": self.layer_weights.detach().cpu().tolist(),
            "layer_weight_source": "responsible neuron count multiplied by direction stability",
            "prompt_source": prompt_provenance(),
            "trainable_parameters": 0,
            "baseline_score_dependency": False,
        }
