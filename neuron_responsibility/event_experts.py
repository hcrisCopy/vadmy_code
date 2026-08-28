"""Neuron-routed low-rank event experts for weakly supervised VAD.

The module reuses the audited, baseline-score-free circuit atlas.  Neurons
select an event expert; they never replace the released baseline anomaly
score.  The zero-initialized output projection makes the initial forward pass
exactly equal to the released model.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from .model import valid_mask


def _normalise_name(value: str) -> str:
    return "".join(character for character in value.lower() if character.isalnum())


class NeuronRoutedEventExperts(nn.Module):
    """Route post-temporal features through class and time-scale experts."""

    method_name = "neuron_routed_event_experts_v1"

    def __init__(
        self,
        atlas_path: str,
        feature_width: int = 512,
        rank: int = 32,
        slow_dilation: int = 4,
        route_top_fraction: float = 0.10,
    ) -> None:
        super().__init__()
        atlas = json.loads(Path(atlas_path).read_text(encoding="utf-8"))
        blocks = atlas["blocks"]
        self.class_names = [str(value) for value in atlas["class_names"]]
        self.neuron_width = int(sum(int(block["width"]) for block in blocks))
        self.feature_width = int(feature_width)
        self.rank = int(rank)
        self.slow_dilation = int(slow_dilation)
        self.route_top_fraction = float(route_top_fraction)
        if self.feature_width <= 0 or self.rank <= 0:
            raise ValueError("feature_width and rank must be positive")
        if self.slow_dilation <= 0:
            raise ValueError("slow_dilation must be positive")
        if not 0 < self.route_top_fraction <= 1:
            raise ValueError("route_top_fraction must be in (0, 1]")

        centers, scales, masks, directions, weights = [], [], [], [], []
        for block in blocks:
            centers.append(np.asarray(block["center"], dtype=np.float32))
            scales.append(np.asarray(block["scale"], dtype=np.float32))
            masks.append(np.asarray(block["class_mask"], dtype=np.float32))
            directions.append(np.asarray(block["directions"], dtype=np.float32))
            weights.append(np.asarray(block["weights"], dtype=np.float32))
        class_mask = np.concatenate(masks, axis=1)
        direction = np.concatenate(directions, axis=1)
        weight = np.concatenate(weights, axis=1) * class_mask
        weight /= np.maximum(weight.sum(axis=1, keepdims=True), 1e-8)
        self.register_buffer("center", torch.from_numpy(np.concatenate(centers)))
        self.register_buffer(
            "scale", torch.from_numpy(np.concatenate(scales)).clamp_min(1e-6)
        )
        self.register_buffer("direction", torch.from_numpy(direction))
        self.register_buffer("circuit_weight", torch.from_numpy(weight))

        class_count = len(self.class_names)
        self.feature_norm = nn.LayerNorm(self.feature_width)
        self.down = nn.Linear(self.feature_width, self.rank, bias=False)
        self.fast = nn.Conv1d(self.rank, self.rank, kernel_size=3, padding=1)
        self.slow = nn.Conv1d(
            self.rank,
            self.rank,
            kernel_size=3,
            padding=self.slow_dilation,
            dilation=self.slow_dilation,
        )
        self.normal_expert = nn.Linear(self.rank, self.rank)
        self.class_experts = nn.Parameter(
            torch.empty(class_count, self.rank, self.rank)
        )
        self.up = nn.Linear(self.rank, self.feature_width, bias=False)

        # These calibration parameters preserve the audited neuron-to-class
        # mapping while learning only its scale and normal rejection boundary.
        self.class_log_scale = nn.Parameter(torch.zeros(class_count))
        self.class_bias = nn.Parameter(torch.zeros(class_count))
        self.normal_threshold = nn.Parameter(torch.tensor(0.5))
        self.normal_scale = nn.Parameter(torch.tensor(1.0))
        self.fast_class_bias = nn.Parameter(torch.zeros(class_count))
        self.dynamic_scale = nn.Parameter(torch.tensor(1.0))
        self.dynamic_threshold = nn.Parameter(torch.tensor(0.25))
        self.reset_parameters()

    @property
    def class_count(self) -> int:
        return len(self.class_names)

    def reset_parameters(self) -> None:
        nn.init.xavier_uniform_(self.down.weight)
        nn.init.kaiming_uniform_(self.fast.weight, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.slow.weight, a=math.sqrt(5))
        nn.init.xavier_uniform_(self.normal_expert.weight)
        nn.init.zeros_(self.normal_expert.bias)
        nn.init.normal_(self.class_experts, std=0.02)
        # Exact baseline preservation, with a direct gradient on step one.
        nn.init.zeros_(self.up.weight)

    def config(self) -> dict[str, Any]:
        return {
            "method": self.method_name,
            "class_names": self.class_names,
            "neuron_width": self.neuron_width,
            "feature_width": self.feature_width,
            "rank": self.rank,
            "slow_dilation": self.slow_dilation,
            "route_top_fraction": self.route_top_fraction,
        }

    def circuit_evidence(self, neurons: torch.Tensor) -> torch.Tensor:
        if neurons.ndim != 3 or neurons.shape[-1] != self.neuron_width:
            raise ValueError(
                f"expected neurons [B,T,{self.neuron_width}], got {tuple(neurons.shape)}"
            )
        z_score = (neurons - self.center) / self.scale
        activation = F.relu(z_score.unsqueeze(-2) * self.direction)
        return (activation * self.circuit_weight).sum(dim=-1)

    def _video_route(
        self, evidence: torch.Tensor, lengths: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        pooled = []
        for row, length_value in enumerate(lengths.tolist()):
            length = max(1, int(length_value))
            count = max(1, int(math.ceil(length * self.route_top_fraction)))
            pooled.append(evidence[row, :length].topk(count, dim=0).values.mean(dim=0))
        pooled_evidence = torch.stack(pooled)
        stable = torch.log1p(pooled_evidence.clamp_min(0.0))
        class_logits = stable * self.class_log_scale.exp() + self.class_bias
        maximum = stable.max(dim=-1).values
        normal_logit = self.normal_scale * (self.normal_threshold - maximum)
        route_logits = torch.cat([normal_logit.unsqueeze(-1), class_logits], dim=-1)
        return route_logits, F.softmax(route_logits, dim=-1)

    def forward(
        self,
        features: torch.Tensor,
        neurons: torch.Tensor,
        lengths: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        if features.ndim != 3 or features.shape[-1] != self.feature_width:
            raise ValueError(
                f"expected features [B,T,{self.feature_width}], got {tuple(features.shape)}"
            )
        if features.shape[:2] != neurons.shape[:2]:
            raise ValueError("baseline and neuron features are not temporally aligned")

        mask = valid_mask(lengths, features.shape[1], features.dtype).unsqueeze(-1)
        evidence = self.circuit_evidence(neurons) * mask
        route_logits, route = self._video_route(evidence, lengths)
        abnormal_route = route[:, 1:]

        latent = F.gelu(self.down(self.feature_norm(features))) * mask
        channel_first = latent.transpose(1, 2)
        fast = F.gelu(self.fast(channel_first)).transpose(1, 2)
        slow = F.gelu(self.slow(channel_first)).transpose(1, 2)

        difference = torch.zeros_like(evidence)
        difference[:, 1:] = (evidence[:, 1:] - evidence[:, :-1]).abs()
        dynamic = torch.log1p(difference.mean(dim=-1, keepdim=True))
        class_fast_prior = abnormal_route @ self.fast_class_bias
        fast_gate = torch.sigmoid(
            self.dynamic_scale.exp() * (dynamic - self.dynamic_threshold)
            + class_fast_prior[:, None, None]
        ) * mask
        temporal_state = fast_gate * fast + (1.0 - fast_gate) * slow

        class_state = torch.einsum(
            "btr,crs,bc->bts", latent, self.class_experts, abnormal_route
        )
        normal_state = F.gelu(self.normal_expert(latent)) * route[:, :1, None]
        anomaly_strength = abnormal_route.sum(dim=-1, keepdim=True)[:, None]
        expert_state = (
            anomaly_strength * (temporal_state + class_state) / 2.0 + normal_state
        ) * mask
        delta = self.up(expert_state) * mask
        return features + delta, {
            "route_logits": route_logits,
            "route": route,
            "circuit_evidence": evidence,
            "fast_gate": fast_gate.squeeze(-1),
            "delta": delta,
            "mask": mask.squeeze(-1),
        }


def route_targets(
    label_texts: list[str], class_names: list[str], device: torch.device
) -> torch.Tensor:
    """Create normal-plus-class targets without consulting baseline scores."""
    lookup = {_normalise_name(name): index for index, name in enumerate(class_names)}
    targets = torch.zeros(len(label_texts), len(class_names) + 1, device=device)
    for row, text in enumerate(label_texts):
        values = [value for value in str(text).split("-") if value]
        matched = [lookup[_normalise_name(value)] for value in values if _normalise_name(value) in lookup]
        if not matched:
            targets[row, 0] = 1.0
        else:
            targets[row, torch.tensor(matched, device=device) + 1] = 1.0
    return targets


def event_complete_targets(
    logits: torch.Tensor,
    labels: torch.Tensor,
    lengths: torch.Tensor,
    threshold_scale: float = 0.25,
    minimum_sigma: float = 1.5,
    maximum_width_fraction: float = 0.25,
) -> torch.Tensor:
    """Grow detached model peaks into Gaussian event-complete soft targets."""
    probabilities = torch.sigmoid(logits.detach())
    targets = torch.zeros_like(probabilities)
    for row, length_value in enumerate(lengths.tolist()):
        length = max(1, int(length_value))
        if float(labels[row]) <= 0.5:
            continue
        values = F.avg_pool1d(
            probabilities[row, :length][None, None], kernel_size=3, stride=1, padding=1
        ).flatten()
        peak = int(values.argmax().item())
        threshold = values.mean() + threshold_scale * values.std(unbiased=False)
        threshold = torch.minimum(threshold, values[peak] * 0.8)
        left = peak
        right = peak
        while left > 0 and values[left - 1] >= threshold:
            left -= 1
        while right + 1 < length and values[right + 1] >= threshold:
            right += 1
        maximum_width = max(1.0, length * maximum_width_fraction)
        sigma = min(maximum_width, max(minimum_sigma, (right - left + 1) / 2.0))
        positions = torch.arange(length, device=logits.device, dtype=logits.dtype)
        targets[row, :length] = torch.exp(-0.5 * ((positions - peak) / sigma) ** 2)
    return targets


def event_expert_losses(
    records: list[dict[str, torch.Tensor | str]],
    binary_logits: torch.Tensor,
    labels: torch.Tensor,
    label_texts: list[str],
    class_names: list[str],
    lengths: torch.Tensor,
) -> dict[str, torch.Tensor]:
    if not records:
        raise ValueError("at least one event-expert record is required")
    route_logits = torch.stack([
        record["route_logits"] for record in records
        if isinstance(record["route_logits"], torch.Tensor)
    ]).mean(dim=0)
    targets = route_targets(label_texts, class_names, binary_logits.device)
    targets = targets / targets.sum(dim=-1, keepdim=True).clamp_min(1.0)
    route_loss = -(targets * F.log_softmax(route_logits, dim=-1)).sum(dim=-1).mean()

    event_targets = event_complete_targets(binary_logits, labels, lengths)
    mask = valid_mask(lengths, binary_logits.shape[1], binary_logits.dtype)
    event_loss = F.binary_cross_entropy_with_logits(
        binary_logits, event_targets, reduction="none"
    )
    event_loss = (event_loss * mask).sum() / mask.sum().clamp_min(1.0)

    normal_rows = labels < 0.5
    normal_terms, smooth_terms = [], []
    for record in records:
        delta = record["delta"]
        record_mask = record["mask"]
        if not isinstance(delta, torch.Tensor) or not isinstance(record_mask, torch.Tensor):
            raise TypeError("invalid event-expert record")
        if bool(normal_rows.any()):
            normal_mask = record_mask[normal_rows].unsqueeze(-1)
            denominator = normal_mask.sum().clamp_min(1.0) * delta.shape[-1]
            normal_terms.append(
                (delta[normal_rows].square() * normal_mask).sum() / denominator
            )
        else:
            normal_terms.append(delta.sum() * 0.0)
        pair_mask = (record_mask[:, 1:] * record_mask[:, :-1]).unsqueeze(-1)
        difference = delta[:, 1:] - delta[:, :-1]
        denominator = pair_mask.sum().clamp_min(1.0) * delta.shape[-1]
        smooth_terms.append((difference.square() * pair_mask).sum() / denominator)
    return {
        "route": route_loss,
        "event": event_loss,
        "normal": torch.stack(normal_terms).mean(),
        "smooth": torch.stack(smooth_terms).mean(),
    }
