"""Definition-sensitive sparse neuron circuits used only during training.

The implementation follows the score-free circuit representation introduced in
``build_circuit_atlas.py`` and keeps DSANet/DeSC/LaGoVAD inference untouched.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from .baselines import BaselineOutput


class DefinitionEvidence(nn.Module):
    """Convert compact multi-layer CLS dimensions into per-class evidence."""

    def __init__(self, atlas_path: str) -> None:
        super().__init__()
        atlas = json.loads(Path(atlas_path).read_text(encoding="utf-8"))
        self.atlas = atlas
        self.class_names = list(atlas["class_names"])
        blocks = atlas["blocks"]
        self.width = int(sum(int(block["width"]) for block in blocks))
        centers, scales, masks, directions, weights = [], [], [], [], []
        for block in blocks:
            centers.append(np.asarray(block["center"], dtype=np.float32))
            scales.append(np.asarray(block["scale"], dtype=np.float32))
            masks.append(np.asarray(block["class_mask"], dtype=np.float32))
            directions.append(np.asarray(block["directions"], dtype=np.float32))
            weights.append(np.asarray(block["weights"], dtype=np.float32))
        mask = np.concatenate(masks, axis=1)
        direction = np.concatenate(directions, axis=1)
        weight = np.concatenate(weights, axis=1) * mask
        weight = weight / np.maximum(weight.sum(axis=1, keepdims=True), 1e-8)
        self.register_buffer("center", torch.from_numpy(np.concatenate(centers)))
        self.register_buffer("scale", torch.from_numpy(np.concatenate(scales)).clamp_min(1e-6))
        self.register_buffer("direction", torch.from_numpy(direction))
        self.register_buffer("weight", torch.from_numpy(weight))

    def forward(self, compact: torch.Tensor) -> torch.Tensor:
        if compact.shape[-1] != self.width:
            raise ValueError(f"expected compact width {self.width}, got {compact.shape[-1]}")
        z = (compact - self.center) / self.scale
        activation = F.relu(z.unsqueeze(-2) * self.direction)
        return (activation * self.weight).sum(dim=-1)


def _valid_top_bottom(values: torch.Tensor, length: int, fraction: float) -> tuple[torch.Tensor, torch.Tensor]:
    count = max(1, min(length // 2, int(np.ceil(length * fraction))))
    order = torch.argsort(values[:length])
    return order[-count:], order[:count]


def definition_losses(
    output: BaselineOutput,
    evidence: torch.Tensor,
    class_targets: torch.Tensor,
    binary_labels: torch.Tensor,
    lengths: torch.Tensor,
    top_fraction: float,
    binary_margin: float,
    semantic_margin: float,
) -> dict[str, torch.Tensor]:
    """Relative evidence constraints; no baseline score enters circuit discovery."""

    binary_terms, semantic_terms, normal_terms, reconstruction_terms = [], [], [], []
    for row, length_tensor in enumerate(lengths):
        length = max(1, int(length_tensor.item()))
        abnormal_target = class_targets[row, 1:]
        if float(binary_labels[row]) > 0.5:
            target_ids = torch.nonzero(abnormal_target > 0, as_tuple=False).flatten()
            if not len(target_ids):
                continue
            target_evidence = evidence[row, :length].index_select(-1, target_ids).max(-1).values
            other_mask = torch.ones(evidence.shape[-1], dtype=torch.bool, device=evidence.device)
            other_mask[target_ids] = False
            competitor = evidence[row, :length, other_mask].max(-1).values
            responsibility = target_evidence - competitor
            high, low = _valid_top_bottom(responsibility, length, top_fraction)
            binary_terms.append(F.relu(binary_margin - output.binary_logits[row, high].mean() + output.binary_logits[row, low].mean()))

            semantic = output.semantic_logits[row, high].mean(0)
            target_semantic = semantic[1:].index_select(0, target_ids).max()
            semantic_mask = torch.ones_like(semantic, dtype=torch.bool)
            semantic_mask[0] = False
            semantic_mask[1 + target_ids] = False
            competitor_semantic = semantic[semantic_mask].max()
            semantic_terms.append(F.relu(semantic_margin - target_semantic + competitor_semantic))
        else:
            normal_evidence = evidence[row, :length].max(-1).values
            high, _ = _valid_top_bottom(normal_evidence, length, top_fraction)
            normal_terms.append(F.softplus(output.binary_logits[row, high]).mean())
            semantic = output.semantic_logits[row, high].mean(0)
            normal_terms.append(F.relu(semantic_margin - semantic[0] + semantic[1:].max()))

    if isinstance(output.raw, tuple) and len(output.raw) >= 6 and isinstance(output.raw[5], dict):
        dnp = output.raw[5]
        reconstruction = (1.0 - F.cosine_similarity(
            dnp["original_features"], dnp["reconstructed_features"], dim=-1
        )) / 2.0
        for row, length_tensor in enumerate(lengths):
            if float(binary_labels[row]) <= 0.5:
                continue
            length = max(1, int(length_tensor.item()))
            target_ids = torch.nonzero(class_targets[row, 1:] > 0, as_tuple=False).flatten()
            if not len(target_ids):
                continue
            target = evidence[row, :length].index_select(-1, target_ids).max(-1).values
            high, low = _valid_top_bottom(target, length, top_fraction)
            reconstruction_terms.append(F.relu(
                binary_margin - reconstruction[row, high].mean() + reconstruction[row, low].mean()
            ))

    zero = output.binary_logits.sum() * 0.0
    mean = lambda values: torch.stack(values).mean() if values else zero
    return {
        "binary_rank": mean(binary_terms),
        "semantic_hard_negative": mean(semantic_terms),
        "normal_suppression": mean(normal_terms),
        "dnp_rank": mean(reconstruction_terms),
    }
