from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

from vin_vad.witness_neurons import SignedTopKWitnessNeurons
from vin_vad.witness_router import (
    WitnessRouter,
    inverse_softplus,
    masked_mean,
    masked_standardize,
    masked_summary,
    masked_topk_anchor,
)
from vin_vad.witness_temporal import WitnessTemporalReadout


class WitnessExpert(nn.Module):
    """Neuron-only path: its API intentionally has no host-score argument."""

    def __init__(self, active: int = 32, temporal_width: int = 64) -> None:
        super().__init__()
        self.neurons = SignedTopKWitnessNeurons(active=active)
        self.temporal = WitnessTemporalReadout(width=temporal_width)

    def forward(
        self,
        hidden: torch.Tensor,
        validity: torch.Tensor,
        neuron_keep_mask: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        neuron = self.neurons(hidden, validity, neuron_keep_mask)
        logits, temporal_features = self.temporal(
            neuron["temporal_input"], validity, return_features=True
        )
        evidence = torch.sigmoid(logits).masked_fill(~validity, 0.0)
        return {
            **neuron,
            "evidence_logits": logits,
            "evidence": evidence,
            "temporal_features": temporal_features,
        }


class WitnessVAD(nn.Module):
    def __init__(
        self,
        active: int = 32,
        temporal_width: int = 64,
        eta_normal: float = 1.0,
        eta_anomaly: float = 0.25,
    ) -> None:
        super().__init__()
        self.expert = WitnessExpert(active=active, temporal_width=temporal_width)
        self.router = WitnessRouter(
            eta_normal=eta_normal,
            eta_anomaly=eta_anomaly,
            witness_width=temporal_width,
        )

    def forward(
        self,
        hidden: torch.Tensor,
        host_score: torch.Tensor,
        validity: torch.Tensor,
        neuron_keep_mask: torch.Tensor | None = None,
        eta_normal_override: float | None = None,
        eta_anomaly_override: float | None = None,
    ) -> dict[str, torch.Tensor]:
        expert = self.expert(hidden, validity, neuron_keep_mask)
        routed = self.router(
            host_score,
            expert["evidence"],
            validity,
            witness_features=expert["temporal_features"],
            eta_normal_override=eta_normal_override,
            eta_anomaly_override=eta_anomaly_override,
        )
        return {**expert, **routed}


class HostVideoOnlyVAD(nn.Module):
    """W1: host-level state may only apply a uniform non-positive shift."""

    def __init__(self, eta_normal: float = 1.0) -> None:
        super().__init__()
        self.video_head = nn.Linear(4, 1)
        self.raw_eta_normal = nn.Parameter(torch.tensor(inverse_softplus(eta_normal)))

    def forward(self, host_score: torch.Tensor, validity: torch.Tensor) -> dict[str, torch.Tensor]:
        summary = masked_summary(host_score, validity)
        video_logit = self.video_head(summary).squeeze(1)
        video_probability = torch.sigmoid(video_logit)
        eta_normal = F.softplus(self.raw_eta_normal)
        delta = eta_normal * torch.minimum(video_logit, torch.zeros_like(video_logit))
        delta = delta.unsqueeze(1).expand_as(host_score).masked_fill(~validity, 0.0)
        clipped = host_score.clamp(1e-6, 1.0 - 1e-6)
        corrected = host_score + torch.sigmoid(torch.logit(clipped) + delta) - clipped
        return {
            "summary": summary,
            "video_logit": video_logit,
            "video_probability": video_probability,
            "delta_normal": delta,
            "delta_anomaly": torch.zeros_like(delta),
            "corrected_score": corrected.clamp(0.0, 1.0).masked_fill(~validity, 0.0),
        }


class NeuronOnlyRouter(nn.Module):
    """W2: no video state; signed neuron evidence supplies local correction."""

    def __init__(
        self, eta_anomaly: float = 0.25, local_width: int = 16, witness_width: int = 64
    ) -> None:
        super().__init__()
        self.witness_width = int(witness_width)
        self.local_head = nn.Sequential(
            nn.Conv1d(4 + self.witness_width, local_width, kernel_size=1),
            nn.GELU(),
            nn.Conv1d(local_width, 1, kernel_size=1),
        )
        self.raw_eta_anomaly = nn.Parameter(torch.tensor(inverse_softplus(eta_anomaly)))

    def forward(
        self,
        host_score: torch.Tensor,
        evidence: torch.Tensor,
        validity: torch.Tensor,
        eta_anomaly_override: float | None = None,
        witness_features: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        host_clipped = host_score.clamp(1e-6, 1.0 - 1e-6)
        evidence_clipped = evidence.clamp(1e-6, 1.0 - 1e-6)
        score_features = torch.stack(
            [host_clipped, evidence_clipped, host_clipped - evidence_clipped, host_clipped * evidence_clipped],
            dim=1,
        )
        if witness_features is None:
            witness_features = host_score.new_zeros(
                *host_score.shape, self.witness_width
            )
        if witness_features.shape != (*host_score.shape, self.witness_width):
            raise ValueError("witness_features must have shape [B,T,witness_width]")
        features = torch.cat([score_features, witness_features.transpose(1, 2)], dim=1)
        direct_witness = masked_standardize(evidence_clipped, validity).clamp(-3.0, 3.0)
        raw = torch.tanh(
            self.local_head(features).squeeze(1) + direct_witness
        ).masked_fill(~validity, 0.0)
        witness_support = torch.relu(raw)
        veto_support = torch.relu(-raw)
        event_anchor = masked_topk_anchor(host_clipped, validity)
        event_gap = torch.relu(
            torch.logit(event_anchor.clamp(1e-6, 1.0 - 1e-6)).unsqueeze(1)
            - torch.logit(host_clipped)
        ).masked_fill(~validity, 0.0)
        local_shape = witness_support * event_gap - veto_support
        eta_anomaly = (
            F.softplus(self.raw_eta_anomaly)
            if eta_anomaly_override is None
            else host_score.new_tensor(eta_anomaly_override)
        )
        delta = eta_anomaly * local_shape
        clipped = host_score.clamp(1e-6, 1.0 - 1e-6)
        corrected = host_score + torch.sigmoid(torch.logit(clipped) + delta) - clipped
        return {
            "video_probability": torch.ones(host_score.shape[0], device=host_score.device),
            "eta_anomaly": eta_anomaly,
            "delta_normal": torch.zeros_like(delta),
            "delta_anomaly": delta,
            "local_shape": local_shape,
            "witness_support": witness_support,
            "veto_support": veto_support,
            "event_anchor": event_anchor,
            "event_gap": event_gap,
            "corrected_score": corrected.clamp(0.0, 1.0).masked_fill(~validity, 0.0),
        }


class NeuronOnlyWitnessVAD(nn.Module):
    def __init__(self, active: int = 32, temporal_width: int = 64, eta_anomaly: float = 0.25) -> None:
        super().__init__()
        self.expert = WitnessExpert(active=active, temporal_width=temporal_width)
        self.router = NeuronOnlyRouter(
            eta_anomaly=eta_anomaly, witness_width=temporal_width
        )

    def forward(
        self,
        hidden: torch.Tensor,
        host_score: torch.Tensor,
        validity: torch.Tensor,
        eta_anomaly_override: float | None = None,
    ) -> dict[str, torch.Tensor]:
        expert = self.expert(hidden, validity)
        return {
            **expert,
            **self.router(
                host_score,
                expert["evidence"],
                validity,
                eta_anomaly_override=eta_anomaly_override,
                witness_features=expert["temporal_features"],
            ),
        }


def build_witness_variant(
    variant: str,
    active: int = 32,
    temporal_width: int = 64,
    eta_normal: float = 1.0,
    eta_anomaly: float = 0.25,
) -> nn.Module:
    if variant == "w1":
        return HostVideoOnlyVAD(eta_normal=eta_normal)
    if variant == "w2":
        return NeuronOnlyWitnessVAD(
            active=active, temporal_width=temporal_width, eta_anomaly=eta_anomaly
        )
    if variant == "w6":
        return WitnessVAD(
            active=active,
            temporal_width=temporal_width,
            eta_normal=eta_normal,
            eta_anomaly=eta_anomaly,
        )
    raise ValueError("variant must be w1, w2 or w6")
