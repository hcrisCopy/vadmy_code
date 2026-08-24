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


class IndependentNeuronLocalizer(nn.Module):
    """Baseline-independent temporal localizer over selected CLIP neurons."""

    method_name = "independent_neuron_boundary_localization_v1"

    def __init__(
        self,
        neuron_width: int,
        active_indices: torch.Tensor,
        thresholds: torch.Tensor,
        hidden_width: int = 64,
        active_neurons: int = 64,
        evidence_cap: float = 6.0,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        indices = torch.as_tensor(active_indices, dtype=torch.long).flatten()
        thresholds = torch.as_tensor(thresholds, dtype=torch.float32).flatten()
        if neuron_width <= 0 or hidden_width <= 0:
            raise ValueError("neuron_width and hidden_width must be positive")
        if not len(indices) or len(indices) != len(thresholds):
            raise ValueError("active indices and thresholds must have equal positive length")
        if int(indices.min()) < 0 or int(indices.max()) >= int(neuron_width):
            raise ValueError("active index is outside neuron_width")
        if not 0 < active_neurons <= len(indices):
            raise ValueError("active_neurons must be within the calibrated evidence width")
        if evidence_cap <= 0:
            raise ValueError("evidence_cap must be positive")

        self.neuron_width = int(neuron_width)
        self.hidden_width = int(hidden_width)
        self.active_neurons = int(active_neurons)
        self.evidence_cap = float(evidence_cap)
        self.register_buffer("active_indices", indices)
        self.register_buffer("thresholds", thresholds)
        self.gate_logits = nn.Parameter(torch.zeros(len(indices)))
        self.input_norm = nn.LayerNorm(len(indices) * 2)
        self.input_projection = nn.Linear(len(indices) * 2, self.hidden_width)
        self.temporal_paths = nn.ModuleList([
            nn.Conv1d(
                self.hidden_width,
                self.hidden_width,
                kernel_size=3,
                padding=dilation,
                dilation=dilation,
                groups=self.hidden_width,
                bias=False,
            )
            for dilation in (1, 2, 4)
        ])
        self.temporal_mix = nn.Linear(self.hidden_width * 4, self.hidden_width)
        self.dropout = nn.Dropout(dropout)
        self.output = nn.Linear(self.hidden_width, 1)

    @property
    def evidence_width(self) -> int:
        return int(self.active_indices.numel())

    def config(self) -> dict[str, Any]:
        return {
            "method": self.method_name,
            "neuron_width": self.neuron_width,
            "evidence_width": self.evidence_width,
            "hidden_width": self.hidden_width,
            "active_neurons": self.active_neurons,
            "evidence_cap": self.evidence_cap,
        }

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> "IndependentNeuronLocalizer":
        width = int(config["evidence_width"])
        return cls(
            neuron_width=int(config["neuron_width"]),
            active_indices=torch.arange(width),
            thresholds=torch.zeros(width),
            hidden_width=int(config["hidden_width"]),
            active_neurons=int(config["active_neurons"]),
            evidence_cap=float(config.get("evidence_cap", 6.0)),
            dropout=0.0,
        )

    def sparse_gates(self) -> torch.Tensor:
        soft = torch.sigmoid(self.gate_logits)
        indices = soft.topk(self.active_neurons).indices
        hard = torch.zeros_like(soft).scatter_(0, indices, 1.0)
        return hard + soft - soft.detach() if self.training else hard

    def sparsity_loss(self) -> torch.Tensor:
        soft = torch.sigmoid(self.gate_logits)
        return (soft * (1.0 - soft)).mean()

    def neuron_inputs(
        self,
        neurons: torch.Tensor,
        lengths: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if neurons.ndim != 3 or neurons.shape[-1] != self.neuron_width:
            raise ValueError(
                f"expected neurons [B,T,{self.neuron_width}], got {tuple(neurons.shape)}"
            )
        mask = valid_mask(lengths, neurons.shape[1], neurons.dtype).unsqueeze(-1)
        selected = neurons.index_select(-1, self.active_indices)
        exceedance = F.softplus(selected - self.thresholds).clamp_max(self.evidence_cap)
        exceedance = exceedance / self.evidence_cap
        signed = torch.tanh(selected / 3.0)
        gate = self.sparse_gates().to(neurons.dtype)
        inputs = torch.cat([exceedance * gate, signed * gate], dim=-1) * mask
        raw_score = (exceedance * gate).sum(dim=-1) / gate.sum().clamp_min(1.0)
        return inputs, raw_score, mask

    def forward_features(
        self,
        neurons: torch.Tensor,
        lengths: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        inputs, raw_score, mask = self.neuron_inputs(neurons, lengths)
        local = F.gelu(self.input_projection(self.input_norm(inputs))) * mask
        channels = local.transpose(1, 2)
        paths = [
            F.gelu(layer(channels)).transpose(1, 2) * mask
            for layer in self.temporal_paths
        ]
        embedding = F.gelu(self.temporal_mix(torch.cat([local, *paths], dim=-1))) * mask
        logits = self.output(self.dropout(embedding)).squeeze(-1)
        return logits, embedding, raw_score

    def forward(self, neurons: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        return self.forward_features(neurons, lengths)[0]

    @torch.no_grad()
    def selection_score(self, neurons: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        was_training = self.training
        self.eval()
        try:
            logits, _, raw = self.forward_features(neurons, lengths)
            learned = torch.sigmoid(logits)
            raw_normalized = raw / raw.amax(dim=1, keepdim=True).clamp_min(1e-6)
            return 0.5 * learned + 0.5 * raw_normalized
        finally:
            self.train(was_training)


class NeuronBoundaryConditioner(nn.Module):
    """Zero-initialized low-rank neuron injection before temporal encoding."""

    method_name = "neuron_boundary_pre_temporal_conditioning_v1"

    def __init__(
        self,
        localizer: IndependentNeuronLocalizer,
        feature_width: int = 512,
        adapter_width: int = 32,
        max_scale: float = 0.25,
    ) -> None:
        super().__init__()
        if feature_width <= 0 or adapter_width <= 0 or max_scale <= 0:
            raise ValueError("feature_width, adapter_width and max_scale must be positive")
        self.localizer = localizer
        self.feature_width = int(feature_width)
        self.adapter_width = int(adapter_width)
        self.max_scale = float(max_scale)
        self.down = nn.Linear(localizer.hidden_width, self.adapter_width, bias=False)
        self.up = nn.Linear(self.adapter_width, self.feature_width, bias=False)
        # The output projection is zero, so the released baseline is still
        # preserved exactly.  A non-zero scale lets that projection receive a
        # gradient on the first update instead of creating a dead branch.
        self.residual_logit = nn.Parameter(torch.tensor(0.2))
        nn.init.xavier_uniform_(self.down.weight)
        nn.init.zeros_(self.up.weight)

    def config(self) -> dict[str, Any]:
        return {
            "method": self.method_name,
            "feature_width": self.feature_width,
            "adapter_width": self.adapter_width,
            "max_scale": self.max_scale,
            "localizer": self.localizer.config(),
        }

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> "NeuronBoundaryConditioner":
        return cls(
            IndependentNeuronLocalizer.from_config(config["localizer"]),
            feature_width=int(config.get("feature_width", 512)),
            adapter_width=int(config.get("adapter_width", 32)),
            max_scale=float(config.get("max_scale", 0.25)),
        )

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
        logits, embedding, raw_score = self.localizer.forward_features(neurons, lengths)
        mask = valid_mask(lengths, features.shape[1], features.dtype)
        probability = torch.sigmoid(logits)
        confidence = (2.0 * (probability - 0.5).abs()).detach()
        delta = self.up(F.gelu(self.down(embedding))) / math.sqrt(self.adapter_width)
        scale = self.max_scale * torch.tanh(self.residual_logit)
        applied = scale * confidence.unsqueeze(-1) * delta * mask.unsqueeze(-1)
        return features + applied, {
            "localizer_logits": logits,
            "raw_neuron_score": raw_score,
            "confidence": confidence,
            "delta": delta,
            "applied_delta": applied,
            "mask": mask,
            "scale": scale,
        }


def load_evidence_localizer(
    evidence_config: str,
    hidden_width: int,
    active_neurons: int,
    evidence_cap: float,
    dropout: float,
) -> IndependentNeuronLocalizer:
    path = Path(evidence_config)
    metadata_path = path.with_suffix(".json")
    if not metadata_path.exists():
        metadata_path = path.parent / "evidence_config.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    arrays = np.load(path)
    return IndependentNeuronLocalizer(
        neuron_width=int(metadata["neuron_width"]),
        active_indices=torch.from_numpy(arrays["active_indices"].astype(np.int64)),
        thresholds=torch.from_numpy(arrays["thresholds"].astype(np.float32)),
        hidden_width=hidden_width,
        active_neurons=min(active_neurons, len(arrays["active_indices"])),
        evidence_cap=evidence_cap,
        dropout=dropout,
    )


def _best_window(score: torch.Tensor, length: int, width: int) -> int:
    valid = score[:length]
    if width >= length:
        return 0
    means = F.avg_pool1d(valid[None, None], kernel_size=width, stride=1).flatten()
    return int(means.argmax().item())


@torch.no_grad()
def synthesize_boundary_batch(
    localizer: IndependentNeuronLocalizer,
    normal_clip: torch.Tensor,
    normal_neurons: torch.Tensor,
    normal_lengths: torch.Tensor,
    abnormal_clip: torch.Tensor,
    abnormal_neurons: torch.Tensor,
    abnormal_lengths: torch.Tensor,
    min_segment: int,
    max_segment: int,
) -> dict[str, torch.Tensor]:
    """Create positive insertions and normal-normal boundary controls."""
    if normal_clip.shape[0] != abnormal_clip.shape[0]:
        raise ValueError("normal and abnormal synthesis batches must have equal size")
    batch, steps = normal_clip.shape[:2]
    abnormal_score = localizer.selection_score(abnormal_neurons, abnormal_lengths)
    normal_means = []
    abnormal_means = []
    for index in range(batch):
        normal_means.append(normal_clip[index, : int(normal_lengths[index])].mean(dim=0))
        abnormal_means.append(abnormal_clip[index, : int(abnormal_lengths[index])].mean(dim=0))
    similarity = F.normalize(torch.stack(abnormal_means), dim=-1) @ F.normalize(
        torch.stack(normal_means), dim=-1
    ).T
    matched_normal = similarity.argmax(dim=1)

    positive_clip = normal_clip.index_select(0, matched_normal).clone()
    positive_neurons = normal_neurons.index_select(0, matched_normal).clone()
    positive_lengths = normal_lengths.index_select(0, matched_normal).clone()
    positive_target = torch.zeros(batch, steps, device=normal_clip.device)
    confidence = torch.ones(batch, device=normal_clip.device)
    for index in range(batch):
        source_length = int(abnormal_lengths[index].item())
        target_length = int(positive_lengths[index].item())
        maximum = min(max_segment, source_length, target_length)
        minimum = min(min_segment, maximum)
        if maximum <= 0:
            continue
        width = max(minimum, min(maximum, max(minimum, source_length // 8)))
        source_start = _best_window(abnormal_score[index], source_length, width)
        available = max(1, target_length - width + 1)
        target_start = int(torch.randint(available, (1,), device=normal_clip.device).item())
        source_slice = slice(source_start, source_start + width)
        target_slice = slice(target_start, target_start + width)
        positive_clip[index, target_slice] = abnormal_clip[index, source_slice]
        positive_neurons[index, target_slice] = abnormal_neurons[index, source_slice]
        positive_target[index, target_slice] = 1.0
        confidence[index] = abnormal_score[index, source_slice].mean().clamp(0.25, 1.0)

    donor = torch.roll(torch.arange(batch, device=normal_clip.device), shifts=1)
    control_clip = normal_clip.clone()
    control_neurons = normal_neurons.clone()
    for index in range(batch):
        target_length = int(normal_lengths[index].item())
        donor_length = int(normal_lengths[donor[index]].item())
        maximum = min(max_segment, target_length, donor_length)
        minimum = min(min_segment, maximum)
        if maximum <= 0:
            continue
        width = max(minimum, min(maximum, max(minimum, donor_length // 8)))
        source_available = max(1, donor_length - width + 1)
        target_available = max(1, target_length - width + 1)
        source_start = int(torch.randint(source_available, (1,), device=normal_clip.device).item())
        target_start = int(torch.randint(target_available, (1,), device=normal_clip.device).item())
        control_clip[index, target_start:target_start + width] = normal_clip[
            donor[index], source_start:source_start + width
        ]
        control_neurons[index, target_start:target_start + width] = normal_neurons[
            donor[index], source_start:source_start + width
        ]

    return {
        "clip": torch.cat([positive_clip, control_clip], dim=0),
        "neurons": torch.cat([positive_neurons, control_neurons], dim=0),
        "lengths": torch.cat([positive_lengths, normal_lengths], dim=0),
        "targets": torch.cat([positive_target, torch.zeros_like(positive_target)], dim=0),
        "confidence": torch.cat([confidence, torch.ones_like(confidence)], dim=0),
    }


def boundary_supervision_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    lengths: torch.Tensor,
    confidence: torch.Tensor,
) -> dict[str, torch.Tensor]:
    mask = valid_mask(lengths, logits.shape[1], logits.dtype)
    sample_weight = confidence.to(logits.dtype).unsqueeze(1)
    denominator = (mask * sample_weight).sum().clamp_min(1.0)
    bce = (
        F.binary_cross_entropy_with_logits(logits, targets.to(logits.dtype), reduction="none")
        * mask
        * sample_weight
    ).sum() / denominator
    probability = torch.sigmoid(logits) * mask
    positive_rows = targets.sum(dim=1) > 0
    if bool(positive_rows.any()):
        prediction = probability[positive_rows]
        target = targets[positive_rows].to(prediction.dtype)
        weights = confidence[positive_rows].to(prediction.dtype)
        intersection = (prediction * target).sum(dim=1)
        dice = 1.0 - (
            (2.0 * intersection + 1.0)
            / (prediction.sum(dim=1) + target.sum(dim=1) + 1.0)
        )
        dice = (dice * weights).sum() / weights.sum().clamp_min(1.0)
    else:
        dice = logits.sum() * 0.0
    pair_mask = mask[:, 1:] * mask[:, :-1]
    predicted_change = (probability[:, 1:] - probability[:, :-1]).abs()
    target_change = (targets[:, 1:] - targets[:, :-1]).abs().to(logits.dtype)
    boundary = (
        F.binary_cross_entropy(
            predicted_change.clamp(1e-6, 1.0 - 1e-6), target_change, reduction="none"
        )
        * pair_mask
        * sample_weight
    ).sum() / (pair_mask * sample_weight).sum().clamp_min(1.0)
    return {"bce": bce, "dice": dice, "boundary": boundary}
