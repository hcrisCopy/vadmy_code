"""Cross-layer anomaly concept circuits for score-free VAD conditioning."""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import nn

from .losses import binary_topk_mil
from .model import valid_mask


class CrossLayerAnomalyConceptCircuit(nn.Module):
    """Fuse all CLS layers and gate a small residual by deviation and semantics."""

    method_name = "cross_layer_anomaly_concept_circuit_v1"

    def __init__(
        self,
        center: torch.Tensor,
        scale: torch.Tensor,
        normal_anchors: torch.Tensor,
        abnormal_anchors: torch.Tensor,
        concept_width: int = 64,
        temporal_kernel: int = 5,
        semantic_temperature: float = 0.07,
        max_residual_scale: float = 0.25,
    ) -> None:
        super().__init__()
        if center.ndim != 2 or center.shape != scale.shape:
            raise ValueError("center and scale must have shape [layers, hidden_width]")
        if normal_anchors.ndim != 2 or abnormal_anchors.ndim != 2:
            raise ValueError("text anchors must be matrices")
        if normal_anchors.shape[1] != abnormal_anchors.shape[1]:
            raise ValueError("normal and abnormal anchors must share a feature width")
        if temporal_kernel < 3 or temporal_kernel % 2 == 0:
            raise ValueError("temporal_kernel must be an odd integer >= 3")
        self.layers, self.hidden_width = map(int, center.shape)
        self.feature_width = int(normal_anchors.shape[1])
        self.concept_width = int(concept_width)
        self.temporal_kernel = int(temporal_kernel)
        self.semantic_temperature = float(semantic_temperature)
        self.max_residual_scale = float(max_residual_scale)
        self.register_buffer("center", center.float())
        self.register_buffer("scale", scale.float().clamp_min(1e-6))
        self.register_buffer("normal_anchor_init", F.normalize(normal_anchors.float(), dim=-1))
        self.register_buffer("abnormal_anchor_init", F.normalize(abnormal_anchors.float(), dim=-1))
        self.layer_projection = nn.Parameter(
            torch.empty(self.layers, self.hidden_width, self.concept_width)
        )
        self.layer_bias = nn.Parameter(torch.zeros(self.layers, self.concept_width))
        for layer in range(self.layers):
            nn.init.xavier_uniform_(self.layer_projection[layer])
        self.layer_norm = nn.LayerNorm(self.concept_width)
        self.layer_query = nn.Parameter(torch.randn(self.concept_width) * 0.02)
        self.deviation_head = nn.Linear(self.concept_width * 2, 1)
        self.output_projection = nn.Linear(self.concept_width, self.feature_width, bias=False)
        nn.init.normal_(self.output_projection.weight, std=0.01)
        self.normal_anchor_delta = nn.Parameter(torch.zeros_like(self.normal_anchor_init))
        self.abnormal_anchor_delta = nn.Parameter(torch.zeros_like(self.abnormal_anchor_init))
        self.gain_logit = nn.Parameter(torch.zeros(()))

    @classmethod
    def from_artifact(
        cls,
        arrays: dict[str, torch.Tensor],
        concept_width: int,
        temporal_kernel: int,
        semantic_temperature: float,
        max_residual_scale: float,
    ) -> "CrossLayerAnomalyConceptCircuit":
        return cls(
            arrays["center"], arrays["scale"], arrays["normal_anchors"],
            arrays["abnormal_anchors"], concept_width, temporal_kernel,
            semantic_temperature, max_residual_scale,
        )

    @classmethod
    def from_config(cls, config: dict) -> "CrossLayerAnomalyConceptCircuit":
        return cls(
            torch.tensor(config["center"]), torch.tensor(config["scale"]),
            torch.tensor(config["normal_anchors"]), torch.tensor(config["abnormal_anchors"]),
            int(config["concept_width"]), int(config["temporal_kernel"]),
            float(config["semantic_temperature"]), float(config["max_residual_scale"]),
        )

    def config(self) -> dict:
        return {
            "method": self.method_name,
            "center": self.center.cpu().tolist(),
            "scale": self.scale.cpu().tolist(),
            "normal_anchors": self.normal_anchor_init.cpu().tolist(),
            "abnormal_anchors": self.abnormal_anchor_init.cpu().tolist(),
            "concept_width": self.concept_width,
            "temporal_kernel": self.temporal_kernel,
            "semantic_temperature": self.semantic_temperature,
            "max_residual_scale": self.max_residual_scale,
        }

    def anchors(self) -> tuple[torch.Tensor, torch.Tensor]:
        normal = F.normalize(self.normal_anchor_init + self.normal_anchor_delta, dim=-1)
        abnormal = F.normalize(self.abnormal_anchor_init + self.abnormal_anchor_delta, dim=-1)
        return normal, abnormal

    def _temporal_context(self, concept: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        padding = self.temporal_kernel // 2
        values = (concept * mask.unsqueeze(-1)).transpose(1, 2)
        total = F.avg_pool1d(
            values, self.temporal_kernel, stride=1, padding=padding,
            count_include_pad=False,
        )
        counts = F.avg_pool1d(
            mask.unsqueeze(1), self.temporal_kernel, stride=1, padding=padding,
            count_include_pad=False,
        ).clamp_min(1e-6)
        return (total / counts).transpose(1, 2)

    def forward(
        self,
        features: torch.Tensor,
        hidden: torch.Tensor,
        lengths: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        if hidden.ndim != 4 or hidden.shape[-2:] != (self.layers, self.hidden_width):
            raise ValueError(
                f"expected hidden [B,T,{self.layers},{self.hidden_width}], got {tuple(hidden.shape)}"
            )
        if features.shape[:2] != hidden.shape[:2] or features.shape[-1] != self.feature_width:
            raise ValueError("feature and hidden shapes are incompatible")
        mask = valid_mask(lengths, hidden.shape[1], hidden.dtype)
        standardized = (hidden.float() - self.center) / self.scale
        projected = F.gelu(
            torch.einsum("btld,ldc->btlc", standardized, self.layer_projection)
            + self.layer_bias
        )
        normalized_projected = self.layer_norm(projected)
        attention_logits = torch.einsum("btlc,c->btl", normalized_projected, self.layer_query)
        attention_logits = attention_logits / math.sqrt(self.concept_width)
        layer_weights = F.softmax(attention_logits, dim=-1)
        concept = (projected * layer_weights.unsqueeze(-1)).sum(dim=-2)
        context = self._temporal_context(concept, mask)
        innovation = concept - context
        deviation_logit = self.deviation_head(
            torch.cat([concept.abs(), innovation.abs()], dim=-1)
        ).squeeze(-1)

        normal_anchor, abnormal_anchor = self.anchors()
        normalized_feature = F.normalize(features.float(), dim=-1)
        normal_similarity = normalized_feature @ normal_anchor.T
        abnormal_similarity = normalized_feature @ abnormal_anchor.T
        semantic_margin = (
            torch.logsumexp(abnormal_similarity / self.semantic_temperature, dim=-1)
            - math.log(abnormal_similarity.shape[-1])
            - normal_similarity.mean(dim=-1) / self.semantic_temperature
        )
        deviation_gate = torch.sigmoid(deviation_logit)
        semantic_gate = torch.sigmoid(semantic_margin)
        gate = deviation_gate * semantic_gate * mask
        gain = self.max_residual_scale * torch.tanh(self.gain_logit)
        delta = self.output_projection(concept)
        modulated = features + gain * gate.unsqueeze(-1) * delta
        return modulated, {
            "concept": concept,
            "innovation": innovation,
            "layer_weights": layer_weights,
            "deviation_logits": deviation_logit,
            "semantic_margin": semantic_margin,
            "deviation_gate": deviation_gate,
            "semantic_gate": semantic_gate,
            "gate": gate,
            "gain": gain,
        }


def cacc_losses(
    records: list[dict[str, torch.Tensor | str]],
    labels: torch.Tensor,
    lengths: torch.Tensor,
) -> dict[str, torch.Tensor]:
    """Weak MIL plus normal-only and temporal regularizers, without baseline scores."""
    values = [record for record in records if isinstance(record.get("gate"), torch.Tensor)]
    if not values:
        raise RuntimeError("CACC forward produced no tensor records")
    losses = {name: [] for name in ("mil", "normal", "compact", "smooth", "layer_sparse", "anchor")}
    for record in values:
        deviation = record["deviation_logits"]
        semantic = record["semantic_margin"]
        evidence_logit = deviation + semantic
        mask = valid_mask(lengths, deviation.shape[1], deviation.dtype)
        losses["mil"].append(binary_topk_mil(evidence_logit, labels, lengths))
        normal_rows = labels < 0.5
        if normal_rows.any():
            normal_mask = mask[normal_rows]
            normal_denominator = normal_mask.sum().clamp_min(1.0)
            losses["normal"].append(
                (F.softplus(evidence_logit[normal_rows]) * normal_mask).sum() / normal_denominator
            )
            concept = record["concept"][normal_rows]
            losses["compact"].append(
                (concept.square() * normal_mask.unsqueeze(-1)).sum()
                / (normal_denominator * concept.shape[-1])
            )
        gate = record["gate"]
        pair_mask = mask[:, 1:] * mask[:, :-1]
        losses["smooth"].append(
            ((gate[:, 1:] - gate[:, :-1]).square() * pair_mask).sum()
            / pair_mask.sum().clamp_min(1.0)
        )
        weights = record["layer_weights"].clamp_min(1e-8)
        entropy = -(weights * weights.log()).sum(dim=-1)
        losses["layer_sparse"].append((entropy * mask).sum() / mask.sum().clamp_min(1.0))
    # Anchor deltas are added by the trainer because records intentionally do
    # not hold a module reference. Keep a differentiable zero for a fixed API.
    reference = values[0]["gate"]
    losses["anchor"].append(reference.sum() * 0.0)
    return {name: torch.stack(items).mean() for name, items in losses.items() if items}
