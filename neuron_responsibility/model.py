from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import nn


def valid_mask(lengths: torch.Tensor, steps: int, dtype: torch.dtype) -> torch.Tensor:
    positions = torch.arange(steps, device=lengths.device).unsqueeze(0)
    return (positions < lengths.long().clamp(0, steps).unsqueeze(1)).to(dtype)


class NeuronResponsibilityProbe(nn.Module):
    """Sparse, lightweight temporal probe over selected CLIP hidden coordinates."""

    def __init__(self, neuron_width: int, hidden_width: int = 128, dropout: float = 0.1) -> None:
        super().__init__()
        if neuron_width <= 0 or hidden_width <= 0:
            raise ValueError("neuron_width and hidden_width must be positive")
        self.neuron_width = int(neuron_width)
        self.hidden_width = int(hidden_width)
        self.feature_gate_logits = nn.Parameter(torch.zeros(self.neuron_width))
        self.norm = nn.LayerNorm(self.neuron_width)
        self.input_projection = nn.Linear(self.neuron_width, self.hidden_width)
        self.temporal_paths = nn.ModuleList([
            nn.Conv1d(self.hidden_width, self.hidden_width, 3, padding=dilation,
                      dilation=dilation, groups=self.hidden_width, bias=False)
            for dilation in (1, 2)
        ])
        self.temporal_mix = nn.Linear(self.hidden_width * 3, self.hidden_width)
        self.dropout = nn.Dropout(dropout)
        self.output = nn.Linear(self.hidden_width, 1)

    def feature_gates(self) -> torch.Tensor:
        return torch.sigmoid(self.feature_gate_logits)

    def sparsity_loss(self) -> torch.Tensor:
        return self.feature_gates().mean()

    def forward(self, neurons: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        if neurons.ndim != 3 or neurons.shape[-1] != self.neuron_width:
            raise ValueError(
                f"expected neurons [B,T,{self.neuron_width}], got {tuple(neurons.shape)}"
            )
        mask = valid_mask(lengths, neurons.shape[1], neurons.dtype).unsqueeze(-1)
        gated = self.norm(neurons) * self.feature_gates().to(neurons.dtype)
        local = F.gelu(self.input_projection(gated)) * mask
        channels = local.transpose(1, 2)
        temporal = [F.gelu(path(channels)).transpose(1, 2) * mask for path in self.temporal_paths]
        mixed = F.gelu(self.temporal_mix(torch.cat([local, *temporal], dim=-1)))
        return self.output(self.dropout(mixed)).squeeze(-1)


def entropy_confidence(probability: torch.Tensor) -> torch.Tensor:
    probability = probability.clamp(1e-6, 1.0 - 1e-6)
    entropy = -(probability * probability.log() + (1.0 - probability) * (1.0 - probability).log())
    return (1.0 - entropy / math.log(2.0)).clamp(0.0, 1.0)


def responsibility_sets(
    baseline_probability: torch.Tensor,
    neuron_probability: torch.Tensor,
    lengths: torch.Tensor,
) -> dict[str, torch.Tensor]:
    """Return positive, normal and uncertain snippet responsibilities.

    Products require agreement, while entropy confidence prevents uncertain
    0.5-like predictions from being treated as evidence.  Inputs are detached
    by callers when responsibilities are used as selectors.
    """
    if baseline_probability.shape != neuron_probability.shape:
        raise ValueError("baseline and neuron probabilities must have identical [B,T] shapes")
    mask = valid_mask(lengths, baseline_probability.shape[1], baseline_probability.dtype)
    base_conf = entropy_confidence(baseline_probability)
    neuron_conf = entropy_confidence(neuron_probability)
    confidence = torch.sqrt((base_conf * neuron_conf).clamp_min(0.0))
    positive = baseline_probability * neuron_probability * confidence * mask
    normal = (1.0 - baseline_probability) * (1.0 - neuron_probability) * confidence * mask
    uncertain = (mask - positive - normal).clamp_min(0.0)
    return {"positive": positive, "normal": normal, "uncertain": uncertain, "mask": mask}


def topk_mil_probability(probability: torch.Tensor, lengths: torch.Tensor, divisor: int = 16) -> torch.Tensor:
    values = []
    for row, length in zip(probability, lengths):
        valid_length = max(1, int(length.item()))
        count = max(1, valid_length // int(divisor) + 1)
        values.append(row[:valid_length].topk(min(count, valid_length)).values.mean())
    return torch.stack(values)


def probe_mil_loss(
    neuron_logits: torch.Tensor,
    labels: torch.Tensor,
    lengths: torch.Tensor,
    normal_instance_weight: float = 0.25,
) -> torch.Tensor:
    probability = torch.sigmoid(neuron_logits)
    bag_probability = topk_mil_probability(probability, lengths)
    bag_loss = F.binary_cross_entropy(bag_probability, labels.float())
    mask = valid_mask(lengths, probability.shape[1], probability.dtype)
    normal_mask = (labels == 0).to(probability.dtype).unsqueeze(1) * mask
    normal_count = normal_mask.sum().clamp_min(1.0)
    normal_loss = -(torch.log1p(-probability.clamp(max=1.0 - 1e-6)) * normal_mask).sum() / normal_count
    return bag_loss + float(normal_instance_weight) * normal_loss


def responsibility_mil_loss(
    baseline_logits: torch.Tensor,
    neuron_probability: torch.Tensor,
    labels: torch.Tensor,
    lengths: torch.Tensor,
) -> torch.Tensor:
    """Cross-view responsibility loss used in formal joint training."""
    baseline_probability = torch.sigmoid(baseline_logits)
    responsibilities = responsibility_sets(
        baseline_probability.detach(), neuron_probability.detach(), lengths
    )
    probability = baseline_probability.clamp(1e-6, 1.0 - 1e-6)
    positive_loss = -torch.log(probability)
    normal_loss = -torch.log1p(-probability)
    anomaly_bag = labels.bool().unsqueeze(1)
    positive_weight = responsibilities["positive"] * anomaly_bag
    # Pure-normal videos provide dense clean negatives.  Inside abnormal bags,
    # only cross-view normal agreement is used as a negative responsibility.
    normal_weight = torch.where(
        anomaly_bag,
        responsibilities["normal"],
        responsibilities["mask"],
    )
    numerator = (positive_loss * positive_weight + normal_loss * normal_weight).sum()
    denominator = (positive_weight + normal_weight).sum().clamp_min(1.0)
    return numerator / denominator
