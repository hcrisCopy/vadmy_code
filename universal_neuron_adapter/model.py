from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import nn


def valid_mask(lengths: torch.Tensor, steps: int, dtype: torch.dtype) -> torch.Tensor:
    positions = torch.arange(steps, device=lengths.device).unsqueeze(0)
    return (positions < lengths.unsqueeze(1)).to(dtype)


def topk_bag(probability: torch.Tensor, lengths: torch.Tensor, divisor: int = 16) -> torch.Tensor:
    values = []
    for row, length in zip(probability, lengths):
        count_valid = max(1, int(length.item()))
        count_top = max(1, count_valid // divisor + 1)
        values.append(row[:count_valid].topk(min(count_top, count_valid)).values.mean())
    return torch.stack(values)


class SparseNeuronExpert(nn.Module):
    """A layer-wise sparse expert whose selected inputs remain CLIP neuron coordinates."""

    def __init__(self, active_per_layer: int = 32, temporal_width: int = 64) -> None:
        super().__init__()
        if not 0 < active_per_layer <= 768:
            raise ValueError("active_per_layer must be in [1, 768]")
        self.active_per_layer = int(active_per_layer)
        self.temporal_width = int(temporal_width)
        self.gate_logits = nn.Parameter(torch.zeros(12, 768))
        self.neuron_weights = nn.Parameter(torch.empty(12, 768))
        nn.init.normal_(self.neuron_weights, std=0.02)
        self.layer_logits = nn.Parameter(torch.zeros(12))
        self.temporal = nn.Sequential(
            nn.Conv1d(12, temporal_width, 3, padding=1),
            nn.GELU(),
            nn.Conv1d(temporal_width, temporal_width, 3, padding=2, dilation=2),
            nn.GELU(),
            nn.Conv1d(temporal_width, 1, 1),
        )

    def gates(self) -> torch.Tensor:
        soft = torch.sigmoid(self.gate_logits)
        indices = soft.topk(self.active_per_layer, dim=-1).indices
        hard = torch.zeros_like(soft).scatter_(-1, indices, 1.0)
        return hard + soft - soft.detach() if self.training else hard

    def forward(self, hidden: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        if hidden.ndim != 4 or hidden.shape[-2:] != (12, 768):
            raise ValueError(f"expected [B,T,12,768], got {tuple(hidden.shape)}")
        normalized = F.layer_norm(hidden, (768,))
        weights = self.neuron_weights * self.gates()
        evidence = (normalized * weights).sum(dim=-1) / math.sqrt(self.active_per_layer)
        evidence = evidence * torch.softmax(self.layer_logits, dim=0).view(1, 1, 12) * 12.0
        logits = self.temporal(evidence.transpose(1, 2)).squeeze(1)
        return logits * valid_mask(lengths, logits.shape[1], logits.dtype)

    def sparsity_loss(self) -> torch.Tensor:
        gates = torch.sigmoid(self.gate_logits)
        return (gates * (1.0 - gates)).mean()

    def selection(self) -> dict:
        gates = torch.sigmoid(self.gate_logits.detach())
        weights = self.neuron_weights.detach().abs()
        rows = []
        for layer in range(12):
            indices = gates[layer].topk(self.active_per_layer).indices.tolist()
            for dimension in indices:
                rows.append({
                    "layer": layer + 1,
                    "dimension": int(dimension),
                    "gate": float(gates[layer, dimension]),
                    "absolute_weight": float(weights[layer, dimension]),
                })
        return {
            "definition": "CLIP ViT-B/16 CLS hidden-state coordinate",
            "active_per_layer": self.active_per_layer,
            "layer_weights": torch.softmax(self.layer_logits.detach(), dim=0).tolist(),
            "neurons": rows,
        }


class ScoreCorrectionHead(nn.Module):
    """Identity-initialized correction using only baseline and neuron score curves."""

    def __init__(self, width: int = 32) -> None:
        super().__init__()
        self.width = int(width)
        self.body = nn.Sequential(
            nn.Conv1d(6, width, 3, padding=1),
            nn.GELU(),
            nn.Conv1d(width, width, 3, padding=2, dilation=2),
            nn.GELU(),
            nn.Conv1d(width, 1, 1),
        )
        nn.init.zeros_(self.body[-1].weight)
        nn.init.zeros_(self.body[-1].bias)

    def forward(self, baseline_probability: torch.Tensor, expert_probability: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        eps = 1e-5
        baseline = torch.logit(baseline_probability.clamp(eps, 1.0 - eps))
        expert = torch.logit(expert_probability.clamp(eps, 1.0 - eps))
        base_change = F.pad(baseline[:, 1:] - baseline[:, :-1], (1, 0))
        expert_change = F.pad(expert[:, 1:] - expert[:, :-1], (1, 0))
        features = torch.stack([
            baseline,
            expert,
            baseline - expert,
            baseline * torch.tanh(expert),
            base_change,
            expert_change,
        ], dim=1)
        delta = self.body(features).squeeze(1)
        mask = valid_mask(lengths, baseline.shape[1], baseline.dtype)
        return baseline + delta * mask


def weak_supervision_loss(
    final_logits: torch.Tensor,
    baseline_probability: torch.Tensor,
    labels: torch.Tensor,
    lengths: torch.Tensor,
) -> tuple[torch.Tensor, dict[str, float]]:
    mask = valid_mask(lengths, final_logits.shape[1], final_logits.dtype)
    probability = torch.sigmoid(final_logits)
    bag = topk_bag(probability, lengths)
    bag_loss = F.binary_cross_entropy(bag, labels)
    normal_mask = (1.0 - labels).unsqueeze(1) * mask
    normal_loss = -(
        torch.log1p(-probability.clamp(max=1.0 - 1e-6)) * normal_mask
    ).sum() / normal_mask.sum().clamp_min(1.0)
    abnormal_bags = bag[labels.bool()]
    normal_hard = []
    for row, length, label in zip(probability, lengths, labels):
        if label < 0.5:
            normal_hard.append(row[: int(length.item())].max())
    if abnormal_bags.numel() and normal_hard:
        hard = torch.stack(normal_hard)
        ranking = F.softplus(0.5 - abnormal_bags[:, None] + hard[None, :]).mean()
    else:
        ranking = bag_loss * 0.0
    baseline_logit = torch.logit(baseline_probability.clamp(1e-5, 1.0 - 1e-5))
    anchor = (((final_logits - baseline_logit) * mask).square().sum() / mask.sum().clamp_min(1.0))
    pair_mask = mask[:, 1:] * mask[:, :-1]
    smooth = (((probability[:, 1:] - probability[:, :-1]).square() * pair_mask).sum()
              / pair_mask.sum().clamp_min(1.0))
    loss = bag_loss + 0.5 * normal_loss + 0.5 * ranking + 0.02 * anchor + 0.02 * smooth
    return loss, {
        "bag": float(bag_loss.detach()),
        "normal": float(normal_loss.detach()),
        "ranking": float(ranking.detach()),
        "anchor": float(anchor.detach()),
        "smooth": float(smooth.detach()),
    }


def expert_mil_loss(logits: torch.Tensor, labels: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
    probability = torch.sigmoid(logits)
    mask = valid_mask(lengths, logits.shape[1], logits.dtype)
    bag = topk_bag(probability, lengths)
    bag_loss = F.binary_cross_entropy(bag, labels)
    normal_mask = (1.0 - labels).unsqueeze(1) * mask
    normal_loss = -(
        torch.log1p(-probability.clamp(max=1.0 - 1e-6)) * normal_mask
    ).sum() / normal_mask.sum().clamp_min(1.0)
    abnormal = bag[labels.bool()]
    normal = bag[~labels.bool()]
    ranking = F.softplus(0.5 - abnormal[:, None] + normal[None, :]).mean() if abnormal.numel() and normal.numel() else bag_loss * 0.0
    pair_mask = mask[:, 1:] * mask[:, :-1]
    smooth = (((probability[:, 1:] - probability[:, :-1]).square() * pair_mask).sum()
              / pair_mask.sum().clamp_min(1.0))
    return bag_loss + 0.5 * normal_loss + 0.5 * ranking + 0.02 * smooth

