"""Text-conditioned normal-prototype probe over raw CLIP CLS neurons.

The coordinates consumed here are unmodified hidden-state dimensions selected
by ``discover_definition_circuits.py``.  Every class connection is masked by a
frozen CLIP text-gradient circuit, so learned weights cannot silently change a
neuron's semantic ownership.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn


class TextConditionedNeuronProbe(nn.Module):
    method_name = "text_conditioned_normal_prototype_neuron_probe_v1"

    def __init__(self, atlas: dict) -> None:
        super().__init__()
        self.atlas = atlas
        self.class_names = list(atlas["class_names"])
        blocks = atlas["blocks"]
        self.widths = [int(block["width"]) for block in blocks]
        self.width = sum(self.widths)
        centers, scales = [], []
        self.weight_logits = nn.ParameterList()
        self.registered_masks: list[str] = []
        self.registered_directions: list[str] = []
        for index, block in enumerate(blocks):
            centers.append(np.asarray(block["center"], dtype=np.float32))
            scales.append(np.asarray(block["scale"], dtype=np.float32))
            mask = torch.tensor(block["class_mask"], dtype=torch.float32)
            direction = torch.tensor(block["directions"], dtype=torch.float32)
            raw_weight = torch.tensor(block["weights"], dtype=torch.float32) * mask
            raw_weight = raw_weight / raw_weight.sum(-1, keepdim=True).clamp_min(1e-8)
            name_mask, name_direction = f"mask_{index}", f"direction_{index}"
            self.register_buffer(name_mask, mask)
            self.register_buffer(name_direction, direction)
            self.registered_masks.append(name_mask); self.registered_directions.append(name_direction)
            initial = torch.log(torch.expm1((raw_weight * 8.0 + 0.05).clamp_min(1e-4)))
            self.weight_logits.append(nn.Parameter(initial))
        self.register_buffer("center", torch.from_numpy(np.concatenate(centers)))
        self.register_buffer("scale", torch.from_numpy(np.concatenate(scales)).clamp_min(1e-6))
        self.layer_temperature = nn.Parameter(torch.zeros(len(blocks), len(self.class_names)))
        self.layer_bias = nn.Parameter(torch.full((len(blocks), len(self.class_names)), -1.0))
        self.fusion_logits = nn.Parameter(torch.zeros(len(blocks), len(self.class_names)))

    @classmethod
    def from_atlas(cls, path: str | Path) -> "TextConditionedNeuronProbe":
        return cls(json.loads(Path(path).read_text(encoding="utf-8")))

    def forward(self, compact: torch.Tensor) -> dict[str, torch.Tensor]:
        if compact.shape[-1] != self.width:
            raise ValueError(f"expected compact width {self.width}, got {compact.shape[-1]}")
        standardized = (compact.float() - self.center) / self.scale
        pieces = torch.split(standardized, self.widths, dim=-1)
        layer_logits = []
        for index, values in enumerate(pieces):
            mask = getattr(self, self.registered_masks[index])
            direction = getattr(self, self.registered_directions[index])
            activation = F.relu(values.unsqueeze(-2) * direction)
            weights = F.softplus(self.weight_logits[index]) * mask
            weights = weights / weights.sum(-1, keepdim=True).clamp_min(1e-8)
            evidence = (activation * weights).sum(-1)
            temperature = F.softplus(self.layer_temperature[index]) + 0.25
            layer_logits.append(temperature * evidence + self.layer_bias[index])
        layers = torch.stack(layer_logits, dim=-2)
        fusion = F.softmax(self.fusion_logits, dim=0)
        fused = (layers * fusion).sum(dim=-2)
        return {"logits": fused, "layer_logits": layers, "standardized": standardized}

    def config(self) -> dict:
        return {
            "method": self.method_name,
            "atlas": self.atlas,
            "width": self.width,
            "layers_zero_based": [int(value["layer_zero_based"]) for value in self.atlas["blocks"]],
            "raw_neuron_definition": "one coordinate of one CLIP ViT-B/16 CLS hidden state",
            "baseline_score_dependency": False,
        }


def load_probe(path: str | Path, device: torch.device | str = "cpu") -> tuple[TextConditionedNeuronProbe, dict]:
    checkpoint = torch.load(path, map_location="cpu")
    model = TextConditionedNeuronProbe(checkpoint["probe_config"]["atlas"])
    model.load_state_dict(checkpoint["probe_state_dict"], strict=True)
    return model.to(device), checkpoint
