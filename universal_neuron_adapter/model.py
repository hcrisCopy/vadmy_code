from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn
import math


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
    """Layer-wise sparse expert; each active input is one CLS hidden dimension."""

    def __init__(
        self,
        active_per_layer: int = 64,
        temporal_width: int = 64,
        forbidden_indices: list[list[int]] | None = None,
    ) -> None:
        super().__init__()
        self.active_per_layer = int(active_per_layer)
        self.temporal_width = int(temporal_width)
        forbidden_mask = torch.zeros(12, 768, dtype=torch.bool)
        if forbidden_indices is not None:
            indices = torch.as_tensor(forbidden_indices, dtype=torch.long)
            if indices.ndim != 2 or indices.shape[0] != 12:
                raise ValueError("forbidden_indices must have shape [12,K]")
            forbidden_mask.scatter_(1, indices, True)
        self.register_buffer("forbidden_mask", forbidden_mask, persistent=False)
        self.gate_logits = nn.Parameter(torch.zeros(12, 768))
        self.neuron_weights = nn.Parameter(torch.empty(12, 768))
        nn.init.normal_(self.neuron_weights, std=0.02)
        self.layer_logits = nn.Parameter(torch.zeros(12))
        self.temporal = nn.Sequential(nn.Conv1d(12, temporal_width, 3, padding=1), nn.GELU(), nn.Conv1d(temporal_width, temporal_width, 3, padding=2, dilation=2), nn.GELU(), nn.Conv1d(temporal_width, 1, 1))

    def gates(self) -> torch.Tensor:
        allowed = (~self.forbidden_mask).to(self.gate_logits.dtype)
        soft = torch.sigmoid(self.gate_logits) * allowed
        indices = soft.topk(self.active_per_layer, dim=-1).indices
        hard = torch.zeros_like(soft).scatter_(-1, indices, 1.0)
        return hard + soft - soft.detach() if self.training else hard

    def forward(self, hidden: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        normalized = F.layer_norm(hidden, (768,))
        evidence = (normalized * self.neuron_weights * self.gates()).sum(-1) / math.sqrt(self.active_per_layer)
        evidence = evidence * torch.softmax(self.layer_logits, 0).view(1, 1, 12) * 12.0
        logits = self.temporal(evidence.transpose(1, 2)).squeeze(1)
        return logits * valid_mask(lengths, logits.shape[1], logits.dtype)

    def sparsity_loss(self) -> torch.Tensor:
        allowed = (~self.forbidden_mask).to(self.gate_logits.dtype)
        gates = torch.sigmoid(self.gate_logits) * allowed
        return (gates * (1.0 - gates)).sum() / allowed.sum().clamp_min(1.0)

    def selection(self) -> dict[str, object]:
        gates = torch.sigmoid(self.gate_logits.detach())
        weights = self.neuron_weights.detach().abs()
        rows = []
        for layer in range(12):
            for dimension in gates[layer].topk(self.active_per_layer).indices.tolist():
                rows.append({"layer": layer + 1, "dimension": int(dimension), "gate": float(gates[layer, dimension]), "absolute_weight": float(weights[layer, dimension])})
        return {"definition": "CLIP ViT-B/16 CLS hidden-state coordinate", "active_per_layer": self.active_per_layer, "layer_weights": torch.softmax(self.layer_logits.detach(), 0).tolist(), "neurons": rows}


def expert_mil_loss(logits: torch.Tensor, labels: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
    probability = torch.sigmoid(logits)
    mask = valid_mask(lengths, logits.shape[1], logits.dtype)
    bag = topk_bag(probability, lengths)
    bag_loss = F.binary_cross_entropy(bag, labels)
    normal_mask = (1.0 - labels).unsqueeze(1) * mask
    normal_loss = -(torch.log1p(-probability.clamp(max=1.0 - 1e-6)) * normal_mask).sum() / normal_mask.sum().clamp_min(1.0)
    abnormal, normal = bag[labels.bool()], bag[~labels.bool()]
    ranking = F.softplus(0.5 - abnormal[:, None] + normal[None, :]).mean() if abnormal.numel() and normal.numel() else bag_loss * 0.0
    pair = mask[:, 1:] * mask[:, :-1]
    smooth = ((probability[:, 1:] - probability[:, :-1]).square() * pair).sum() / pair.sum().clamp_min(1.0)
    return bag_loss + 0.5 * normal_loss + 0.5 * ranking + 0.02 * smooth


class ScoreCorrectionHead(nn.Module):
    """Score head used by the first-round CLS-neuron expert."""

    def __init__(self, width: int = 32) -> None:
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv1d(6, width, 3, padding=1),
            nn.GELU(),
            nn.Conv1d(width, width, 3, padding=2, dilation=2),
            nn.GELU(),
            nn.Conv1d(width, 1, 1),
        )

    def forward(self, baseline_probability: torch.Tensor, expert_probability: torch.Tensor) -> torch.Tensor:
        baseline = torch.logit(baseline_probability.clamp(1e-5, 1.0 - 1e-5))
        expert = torch.logit(expert_probability.clamp(1e-5, 1.0 - 1e-5))
        base_change = F.pad(baseline[:, 1:] - baseline[:, :-1], (1, 0))
        expert_change = F.pad(expert[:, 1:] - expert[:, :-1], (1, 0))
        features = torch.stack(
            [baseline, expert, baseline - expert, baseline * torch.tanh(expert), base_change, expert_change],
            dim=1,
        )
        return baseline + self.body(features).squeeze(1)


def calibrated_probability(
    baseline_probability: torch.Tensor,
    expert_probability: torch.Tensor,
    correction_logits: torch.Tensor,
    correction_weight: float = 0.2,
    neuron_weight: float = 0.1,
) -> torch.Tensor:
    """Conservative universal fusion in logit space.

    The neuron evidence is standardized within each video. Thus its coefficient has
    the same meaning for every baseline and dataset, while retaining temporal neuron
    ordering and avoiding dataset-specific score calibration.
    """
    baseline_logits = torch.logit(baseline_probability.clamp(1e-5, 1.0 - 1e-5))
    mean = expert_probability.mean(dim=1, keepdim=True)
    scale = expert_probability.std(dim=1, keepdim=True, unbiased=False).clamp_min(1e-6)
    neuron_evidence = (expert_probability - mean) / scale
    logits = (
        (1.0 - correction_weight) * baseline_logits
        + correction_weight * correction_logits
        + neuron_weight * neuron_evidence
    )
    return torch.sigmoid(logits)

