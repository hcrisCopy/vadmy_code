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

    def __init__(
        self,
        neuron_width: int,
        hidden_width: int = 128,
        dropout: float = 0.1,
        active_neurons: int = 128,
    ) -> None:
        super().__init__()
        if neuron_width <= 0 or hidden_width <= 0:
            raise ValueError("neuron_width and hidden_width must be positive")
        self.neuron_width = int(neuron_width)
        self.hidden_width = int(hidden_width)
        self.active_neurons = min(int(active_neurons), self.neuron_width)
        if self.active_neurons <= 0:
            raise ValueError("active_neurons must be positive")
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
        soft = torch.sigmoid(self.feature_gate_logits)
        indices = soft.topk(self.active_neurons).indices
        hard = torch.zeros_like(soft).scatter_(0, indices, 1.0)
        # Straight-through top-k: the forward pass is exactly sparse while the
        # gate ranking can still learn through the soft sigmoid values.
        return hard + soft - soft.detach() if self.training else hard

    def sparsity_loss(self) -> torch.Tensor:
        soft = torch.sigmoid(self.feature_gate_logits)
        return (soft * (1.0 - soft)).mean()

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
    ranking_weight: float = 0.2,
    smoothness_weight: float = 0.05,
    anomaly_sparsity_weight: float = 0.01,
    ranking_margin: float = 0.5,
) -> torch.Tensor:
    probability = torch.sigmoid(neuron_logits)
    bag_probability = topk_mil_probability(probability, lengths)
    bag_loss = F.binary_cross_entropy(bag_probability, labels.float())
    mask = valid_mask(lengths, probability.shape[1], probability.dtype)
    normal_mask = (labels == 0).to(probability.dtype).unsqueeze(1) * mask
    normal_count = normal_mask.sum().clamp_min(1.0)
    normal_loss = -(torch.log1p(-probability.clamp(max=1.0 - 1e-6)) * normal_mask).sum() / normal_count
    abnormal_bags = bag_probability[labels.bool()]
    normal_bags = bag_probability[~labels.bool()]
    if abnormal_bags.numel() and normal_bags.numel():
        ranking = F.relu(
            float(ranking_margin) - abnormal_bags[:, None] + normal_bags[None, :]
        ).mean()
    else:
        ranking = bag_loss * 0.0
    pair_mask = mask[:, 1:] * mask[:, :-1]
    smoothness = (
        (probability[:, 1:] - probability[:, :-1]).square() * pair_mask
    ).sum() / pair_mask.sum().clamp_min(1.0)
    abnormal_mask = labels.to(probability.dtype).unsqueeze(1) * mask
    anomaly_sparsity = (probability * abnormal_mask).sum() / abnormal_mask.sum().clamp_min(1.0)
    return (
        bag_loss
        + float(normal_instance_weight) * normal_loss
        + float(ranking_weight) * ranking
        + float(smoothness_weight) * smoothness
        + float(anomaly_sparsity_weight) * anomaly_sparsity
    )


class ResponsibilityCorrectionHead(nn.Module):
    """Small baseline-agnostic temporal residual, identity at initialization."""

    def __init__(self, hidden_width: int = 32) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(4, hidden_width, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv1d(hidden_width, 1, kernel_size=1),
        )
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(
        self,
        baseline_logits: torch.Tensor,
        neuron_probability: torch.Tensor,
        lengths: torch.Tensor,
    ) -> torch.Tensor:
        baseline_probability = torch.sigmoid(baseline_logits)
        inputs = torch.stack(
            [
                baseline_logits,
                torch.logit(neuron_probability.clamp(1e-6, 1.0 - 1e-6)),
                baseline_probability - neuron_probability,
                entropy_confidence(neuron_probability),
            ],
            dim=1,
        )
        delta = self.net(inputs).squeeze(1)
        mask = valid_mask(lengths, baseline_logits.shape[1], baseline_logits.dtype)
        return baseline_logits + delta * mask


def persistent_high_mask(
    probability: torch.Tensor,
    threshold: float,
    minimum_length: int,
) -> torch.Tensor:
    high = (probability >= float(threshold)).to(probability.dtype).unsqueeze(1)
    if minimum_length <= 1:
        return high.squeeze(1)
    kernel = torch.ones(1, 1, int(minimum_length), device=probability.device, dtype=probability.dtype)
    if probability.shape[1] < minimum_length:
        return torch.zeros_like(probability)
    counts = F.conv1d(high, kernel)
    starts = counts >= minimum_length
    # Expand a qualifying run back to all snippets belonging to that run.
    expanded = F.conv_transpose1d(starts.to(probability.dtype), kernel)
    return (expanded > 0).squeeze(1).to(probability.dtype)


def partition_responsibility_loss(
    final_logits: torch.Tensor,
    reference_logits: torch.Tensor,
    neuron_probability: torch.Tensor,
    labels: torch.Tensor,
    lengths: torch.Tensor,
    neuron_threshold: float,
    persistence: int = 3,
    baseline_high: float = 0.8,
    baseline_low: float = 0.2,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Four-way responsibility without treating the baseline as an oracle."""
    reference_probability = torch.sigmoid(reference_logits.detach())
    neuron_probability = neuron_probability.detach()
    mask = valid_mask(lengths, final_logits.shape[1], final_logits.dtype)
    anomaly = labels.bool().unsqueeze(1)
    base_high = reference_probability >= float(baseline_high)
    base_low = reference_probability <= float(baseline_low)
    neuron_high = persistent_high_mask(neuron_probability, neuron_threshold, persistence).bool()
    neuron_low = neuron_probability <= 0.5

    agreement_high = anomaly & base_high & neuron_high & mask.bool()
    baseline_only = anomaly & base_high & ~neuron_high & mask.bool()
    neuron_only = anomaly & ~base_high & neuron_high & mask.bool()
    agreement_low = anomaly & base_low & neuron_low & mask.bool()
    pure_normal = ~anomaly & mask.bool()

    probability = torch.sigmoid(final_logits).clamp(1e-6, 1.0 - 1e-6)

    def mean_selected(values: torch.Tensor, selected: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        weights = selected.to(values.dtype)
        count = weights.sum()
        return (values * weights).sum() / count.clamp_min(1.0), (count > 0).to(values.dtype)

    positive_loss = -torch.log(probability)
    negative_loss = -torch.log1p(-probability)
    # Normalize partitions independently so dense normal snippets cannot
    # overwhelm the rarer neuron-only evidence.
    components = (
        mean_selected(positive_loss, agreement_high),
        mean_selected(positive_loss, baseline_only),
        mean_selected(positive_loss, neuron_only),
        mean_selected(negative_loss, agreement_low),
        mean_selected(negative_loss, pure_normal),
    )
    loss = sum(value * present for value, present in components) / sum(
        present for _, present in components
    ).clamp_min(1.0)
    partitions = {
        "agreement_high": agreement_high.sum(),
        "baseline_only": baseline_only.sum(),
        "neuron_only": neuron_only.sum(),
        "agreement_low": agreement_low.sum(),
        "pure_normal": pure_normal.sum(),
    }
    return loss, partitions


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
