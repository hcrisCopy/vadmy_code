"""Concept-conditioned counterfactual routing over sparse CLIP neurons."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn


@dataclass
class CircuitViews:
    original: torch.Tensor
    enhanced: torch.Tensor
    suppressed: torch.Tensor
    evidence: torch.Tensor
    anomaly_gate: torch.Tensor
    concept_weights: torch.Tensor
    target_text_effect: torch.Tensor


class ConceptCircuitRouter(nn.Module):
    """Create exact post-LayerNorm counterfactuals from compact hidden statistics.

    The compact tensor stores the selected hidden coordinates followed by the
    full 768-D hidden mean and variance.  The released 512-D CLIP feature is
    used as the unmodified projection anchor, so a zero intervention reproduces
    the author's input exactly even when frame sampling differs slightly.
    """

    def __init__(
        self,
        union_indices: torch.Tensor,
        class_mask: torch.Tensor,
        directions: torch.Tensor,
        center: torch.Tensor,
        scale: torch.Tensor,
        ln_weight: torch.Tensor,
        ln_bias: torch.Tensor,
        projection: torch.Tensor,
        normal_text: torch.Tensor,
        abnormal_text: torch.Tensor,
        normal_margin_threshold: float,
        gate_temperature: float = 0.05,
        max_gain: float = 0.50,
        initial_gain: float = 0.10,
        layer_norm_eps: float = 1e-5,
    ) -> None:
        super().__init__()
        if class_mask.shape != directions.shape:
            raise ValueError("class mask and directions must have identical shapes")
        if class_mask.shape[1] != union_indices.numel():
            raise ValueError("class masks do not match the union neuron width")
        self.hidden_width = int(ln_weight.numel())
        self.neuron_width = int(union_indices.numel())
        self.gate_temperature = float(gate_temperature)
        self.max_gain = float(max_gain)
        self.layer_norm_eps = float(layer_norm_eps)
        self.register_buffer("union_indices", union_indices.long())
        self.register_buffer("class_mask", class_mask.float())
        self.register_buffer("directions", directions.float())
        self.register_buffer("center", center.float())
        self.register_buffer("scale", scale.float().clamp_min(1e-6))
        self.register_buffer("ln_weight", ln_weight.float())
        self.register_buffer("ln_bias", ln_bias.float())
        self.register_buffer("projection", projection.float())
        self.register_buffer("normal_text", F.normalize(normal_text.float(), dim=-1))
        self.register_buffer("abnormal_text", F.normalize(abnormal_text.float(), dim=-1))
        self.register_buffer(
            "normal_margin_threshold", torch.tensor(float(normal_margin_threshold))
        )
        initial_ratio = min(max(float(initial_gain) / self.max_gain, 1e-4), 1.0 - 1e-4)
        initial_logit = float(np.log(initial_ratio / (1.0 - initial_ratio)))
        self.gain_logits = nn.Parameter(torch.full(class_mask.shape, initial_logit))

        selected_weight = self.ln_weight.index_select(0, self.union_indices)
        selected_projection = self.projection.index_select(0, self.union_indices)
        self.register_buffer("selected_weighted_projection", selected_weight[:, None] * selected_projection)
        self.register_buffer("weighted_projection_sum", self.ln_weight @ self.projection)
        self.register_buffer("bias_projection", self.ln_bias @ self.projection)

    @property
    def compact_width(self) -> int:
        return self.neuron_width + 2

    def _concept_route(
        self,
        clip: torch.Tensor,
        concept_targets: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        normalized = F.normalize(clip.float(), dim=-1)
        abnormal_similarity = normalized @ self.abnormal_text.T
        normal_similarity = normalized @ self.normal_text.T
        predicted = F.softmax(abnormal_similarity / 0.07, dim=-1)
        if concept_targets is None:
            weights = predicted
        else:
            targets = concept_targets.float()
            has_target = targets.sum(dim=-1, keepdim=True) > 0
            targets = targets / targets.sum(dim=-1, keepdim=True).clamp_min(1.0)
            weights = torch.where(has_target[:, None, :], targets[:, None, :], predicted)
        target_similarity = (weights * abnormal_similarity).sum(dim=-1)
        margin = target_similarity - normal_similarity.max(dim=-1).values
        gate = torch.sigmoid(
            (margin - self.normal_margin_threshold) / max(self.gate_temperature, 1e-6)
        )
        return weights, gate

    def _project_counterfactual(
        self,
        clip: torch.Tensor,
        selected_hidden: torch.Tensor,
        hidden_mean: torch.Tensor,
        hidden_variance: torch.Tensor,
        delta: torch.Tensor,
    ) -> torch.Tensor:
        sigma = torch.sqrt(hidden_variance.clamp_min(0.0) + self.layer_norm_eps)
        weighted_hidden_projection = (
            sigma * (clip - self.bias_projection) + hidden_mean * self.weighted_projection_sum
        )
        shifted_weighted_projection = (
            weighted_hidden_projection + delta @ self.selected_weighted_projection
        )
        mean_delta = delta.sum(dim=-1, keepdim=True) / float(self.hidden_width)
        shifted_mean = hidden_mean + mean_delta
        centered_selected = selected_hidden - hidden_mean
        variance_delta = (
            2.0 * (centered_selected * delta).sum(dim=-1, keepdim=True)
            + delta.square().sum(dim=-1, keepdim=True)
        ) / float(self.hidden_width) - mean_delta.square()
        shifted_variance = (hidden_variance + variance_delta).clamp_min(0.0)
        shifted_sigma = torch.sqrt(shifted_variance + self.layer_norm_eps)
        return (
            shifted_weighted_projection - shifted_mean * self.weighted_projection_sum
        ) / shifted_sigma + self.bias_projection

    def forward(
        self,
        clip: torch.Tensor,
        compact_hidden: torch.Tensor,
        concept_targets: torch.Tensor | None = None,
    ) -> CircuitViews:
        if compact_hidden.shape[-1] != self.compact_width:
            raise ValueError(
                f"expected compact hidden width {self.compact_width}, got {compact_hidden.shape[-1]}"
            )
        selected = compact_hidden[..., : self.neuron_width].float()
        hidden_mean = compact_hidden[..., self.neuron_width:self.neuron_width + 1].float()
        hidden_variance = compact_hidden[..., self.neuron_width + 1:self.neuron_width + 2].float()
        weights, anomaly_gate = self._concept_route(clip, concept_targets)
        membership = weights @ self.class_mask
        signed_membership = weights @ (self.class_mask * self.directions)
        sign = torch.sign(signed_membership)
        signed_evidence = F.relu(sign * (selected - self.center) / self.scale)
        evidence = (membership * signed_evidence).sum(dim=-1) / membership.sum(dim=-1).clamp_min(1e-6)

        learned_gain = self.max_gain * torch.sigmoid(self.gain_logits) * self.class_mask
        coordinate_gain = weights @ learned_gain
        base_delta = anomaly_gate.unsqueeze(-1) * coordinate_gain * sign * self.scale * signed_evidence
        suppression_delta = -anomaly_gate.unsqueeze(-1) * membership * sign * self.scale * signed_evidence
        enhanced = self._project_counterfactual(
            clip.float(), selected, hidden_mean, hidden_variance, base_delta
        )
        suppressed = self._project_counterfactual(
            clip.float(), selected, hidden_mean, hidden_variance, suppression_delta
        )
        target_text_effect = (
            weights * (
                F.normalize(enhanced, dim=-1) @ self.abnormal_text.T
                - F.normalize(suppressed, dim=-1) @ self.abnormal_text.T
            )
        ).sum(dim=-1)
        return CircuitViews(
            original=clip,
            enhanced=enhanced.to(clip.dtype),
            suppressed=suppressed.to(clip.dtype),
            evidence=evidence,
            anomaly_gate=anomaly_gate,
            concept_weights=weights,
            target_text_effect=target_text_effect,
        )


def load_circuit_router(
    atlas_path: str | Path,
    gate_temperature: float = 0.05,
    max_gain: float = 0.50,
    initial_gain: float = 0.10,
) -> tuple[ConceptCircuitRouter, dict]:
    atlas_file = Path(atlas_path)
    atlas = json.loads(atlas_file.read_text(encoding="utf-8"))
    weights_path = Path(atlas["weights_path"])
    if not weights_path.is_absolute():
        weights_path = atlas_file.resolve().parent / weights_path
    values = np.load(weights_path)
    router = ConceptCircuitRouter(
        union_indices=torch.from_numpy(values["union_indices"]),
        class_mask=torch.from_numpy(values["class_mask"]),
        directions=torch.from_numpy(values["directions"]),
        center=torch.from_numpy(values["center"]),
        scale=torch.from_numpy(values["scale"]),
        ln_weight=torch.from_numpy(values["ln_weight"]),
        ln_bias=torch.from_numpy(values["ln_bias"]),
        projection=torch.from_numpy(values["projection"]),
        normal_text=torch.from_numpy(values["normal_text"]),
        abnormal_text=torch.from_numpy(values["abnormal_text"]),
        normal_margin_threshold=float(atlas["normal_margin_threshold"]),
        gate_temperature=gate_temperature,
        max_gain=max_gain,
        initial_gain=initial_gain,
        layer_norm_eps=float(atlas.get("layer_norm_eps", 1e-5)),
    )
    return router, atlas
