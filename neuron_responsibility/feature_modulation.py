from __future__ import annotations

import math
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn

from .losses import binary_topk_mil
from .model import valid_mask


class SparseNeuronFeatureModulator(nn.Module):
    """Inject baseline-independent neuron evidence into post-temporal features.

    Selected CLIP coordinates remain individually traceable.  The baseline
    feature supplies context for a gate, but no baseline anomaly score is used.
    """

    method_name = "sparse_neuron_feature_modulation_v1"

    def __init__(
        self,
        neuron_width: int,
        active_indices: torch.Tensor,
        thresholds: torch.Tensor,
        feature_width: int = 512,
        context_width: int = 32,
        temporal_kernel: int = 5,
        evidence_cap: float = 6.0,
    ) -> None:
        super().__init__()
        active_indices = torch.as_tensor(active_indices, dtype=torch.long).flatten()
        thresholds = torch.as_tensor(thresholds, dtype=torch.float32).flatten()
        if neuron_width <= 0 or feature_width <= 0 or context_width <= 0:
            raise ValueError("neuron, feature and context widths must be positive")
        if not len(active_indices) or len(active_indices) != len(thresholds):
            raise ValueError("active_indices and thresholds must have the same positive length")
        if int(active_indices.min()) < 0 or int(active_indices.max()) >= int(neuron_width):
            raise ValueError("active neuron index is outside neuron_width")
        if temporal_kernel <= 0 or temporal_kernel % 2 == 0:
            raise ValueError("temporal_kernel must be a positive odd number")
        if evidence_cap <= 0:
            raise ValueError("evidence_cap must be positive")

        self.neuron_width = int(neuron_width)
        self.feature_width = int(feature_width)
        self.context_width = int(context_width)
        self.temporal_kernel = int(temporal_kernel)
        self.evidence_cap = float(evidence_cap)
        self.register_buffer("active_indices", active_indices)
        self.register_buffer("thresholds", thresholds)

        active_width = len(active_indices)
        self.feature_norm = nn.LayerNorm(self.feature_width)
        self.context_projection = nn.Linear(self.feature_width, self.context_width, bias=False)
        self.gate_queries = nn.Parameter(torch.empty(active_width, self.context_width))
        self.neuron_directions = nn.Parameter(torch.empty(active_width, self.feature_width))
        self.auxiliary_head = nn.Linear(active_width, 1)
        self.residual_scale = nn.Parameter(torch.zeros(()))
        self.reset_parameters()

    @property
    def active_width(self) -> int:
        return int(self.active_indices.numel())

    def reset_parameters(self) -> None:
        nn.init.xavier_uniform_(self.context_projection.weight)
        nn.init.normal_(self.gate_queries, std=0.02)
        # A zero output dictionary preserves the released baseline exactly,
        # while receiving a direct gradient on the first optimization step.
        nn.init.zeros_(self.neuron_directions)
        nn.init.zeros_(self.auxiliary_head.bias)
        nn.init.constant_(self.auxiliary_head.weight, 1.0 / self.active_width)
        nn.init.ones_(self.residual_scale)

    def config(self) -> dict[str, Any]:
        return {
            "method": self.method_name,
            "neuron_width": self.neuron_width,
            "active_width": self.active_width,
            "feature_width": self.feature_width,
            "context_width": self.context_width,
            "temporal_kernel": self.temporal_kernel,
            "evidence_cap": self.evidence_cap,
        }

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> "SparseNeuronFeatureModulator":
        active_width = int(config["active_width"])
        return cls(
            neuron_width=int(config["neuron_width"]),
            active_indices=torch.arange(active_width),
            thresholds=torch.zeros(active_width),
            feature_width=int(config.get("feature_width", 512)),
            context_width=int(config.get("context_width", 32)),
            temporal_kernel=int(config.get("temporal_kernel", 5)),
            evidence_cap=float(config.get("evidence_cap", 6.0)),
        )

    def neuron_evidence(
        self,
        neurons: torch.Tensor,
        lengths: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if neurons.ndim != 3 or neurons.shape[-1] != self.neuron_width:
            raise ValueError(
                f"expected neurons [B,T,{self.neuron_width}], got {tuple(neurons.shape)}"
            )
        mask = valid_mask(lengths, neurons.shape[1], neurons.dtype).unsqueeze(-1)
        selected = neurons.index_select(-1, self.active_indices)
        # Softplus is a smooth exceedance above the normal quantile.  Unlike a
        # hard ReLU, it retains weak but rank-preserving evidence below q99.
        short = F.softplus(selected - self.thresholds).clamp_max(self.evidence_cap)
        channels = short.transpose(1, 2)
        long = F.avg_pool1d(
            channels,
            kernel_size=self.temporal_kernel,
            stride=1,
            padding=self.temporal_kernel // 2,
        ).transpose(1, 2)
        evidence = 0.5 * (short + long)
        return evidence * mask / self.evidence_cap, mask

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
            raise ValueError("baseline features and neuron evidence are not temporally aligned")
        evidence, mask = self.neuron_evidence(neurons, lengths)
        context = F.gelu(self.context_projection(self.feature_norm(features)))
        gate_logits = torch.einsum("btc,kc->btk", context, self.gate_queries)
        gates = torch.sigmoid(gate_logits / math.sqrt(self.context_width)) * mask
        gated_evidence = evidence * gates
        delta = torch.einsum("btk,kd->btd", gated_evidence, self.neuron_directions)
        delta = delta / math.sqrt(self.active_width)
        scale = self.residual_scale
        modulated = features + scale * delta
        auxiliary_logits = self.auxiliary_head(gated_evidence).squeeze(-1)
        return modulated, {
            "auxiliary_logits": auxiliary_logits,
            "delta": delta,
            "applied_delta": scale * delta,
            "gates": gates,
            "evidence": evidence,
            "mask": mask.squeeze(-1),
            "scale": scale,
        }


def score_free_modulation_losses(
    records: list[dict[str, torch.Tensor | str]],
    labels: torch.Tensor,
    lengths: torch.Tensor,
) -> dict[str, torch.Tensor]:
    """Weakly supervised losses that never consume baseline anomaly scores."""
    if not records:
        raise ValueError("at least one modulation record is required")
    auxiliary_logits = torch.stack([
        record["auxiliary_logits"] for record in records
        if isinstance(record["auxiliary_logits"], torch.Tensor)
    ]).mean(dim=0)
    auxiliary = binary_topk_mil(auxiliary_logits, labels, lengths)

    normal_rows = labels < 0.5
    normal_terms = []
    smooth_terms = []
    sparse_terms = []
    for record in records:
        delta = record["applied_delta"]
        gates = record["gates"]
        evidence = record["evidence"]
        mask = record["mask"]
        if not all(isinstance(value, torch.Tensor) for value in (delta, gates, evidence, mask)):
            raise TypeError("invalid modulation record")
        if bool(normal_rows.any()):
            normal_mask = mask[normal_rows].unsqueeze(-1)
            normal_delta = delta[normal_rows] * normal_mask
            denominator = normal_mask.sum().clamp_min(1.0) * delta.shape[-1]
            normal_terms.append(normal_delta.square().sum() / denominator)
        else:
            normal_terms.append(delta.sum() * 0.0)
        pair_mask = (mask[:, 1:] * mask[:, :-1]).unsqueeze(-1)
        difference = delta[:, 1:] - delta[:, :-1]
        smooth_denominator = pair_mask.sum().clamp_min(1.0) * delta.shape[-1]
        smooth_terms.append((difference.square() * pair_mask).sum() / smooth_denominator)
        sparse_denominator = mask.sum().clamp_min(1.0) * gates.shape[-1]
        sparse_terms.append((gates * evidence).sum() / sparse_denominator)

    return {
        "auxiliary": auxiliary,
        "normal": torch.stack(normal_terms).mean(),
        "smooth": torch.stack(smooth_terms).mean(),
        "sparse": torch.stack(sparse_terms).mean(),
    }
