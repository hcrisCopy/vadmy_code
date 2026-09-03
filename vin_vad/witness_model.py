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


def masked_temporal_mean(
    values: torch.Tensor, validity: torch.Tensor, width: int
) -> torch.Tensor:
    if values.shape != validity.shape or width <= 0 or width % 2 == 0:
        raise ValueError("values/validity must share [B,T] and width must be positive odd")
    mask = validity.unsqueeze(1).to(values.dtype)
    kernel = torch.ones(1, 1, width, dtype=values.dtype, device=values.device)
    numerator = F.conv1d(
        (values * validity.to(values.dtype)).unsqueeze(1),
        kernel,
        padding=width // 2,
    )
    denominator = F.conv1d(mask, kernel, padding=width // 2).clamp_min(1.0)
    return (numerator / denominator).squeeze(1).masked_fill(~validity, 0.0)


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
        primary_logits = self.temporal(neuron["temporal_input"], validity)
        layer_weight = self.neurons.layers * neuron["layer_probability"].view(1, 1, -1)
        normality_raw = (neuron["normality_layer_evidence"] * layer_weight).sum(dim=-1)
        normality_view = masked_standardize(normality_raw, validity).clamp(-3.0, 3.0)
        short_context = masked_temporal_mean(normality_view, validity, width=13)
        long_context = masked_temporal_mean(normality_view, validity, width=25)
        context_view = masked_standardize(
            0.5 * (short_context + long_context), validity
        ).clamp(-3.0, 3.0)
        logits = primary_logits + normality_view + context_view
        evidence = torch.sigmoid(logits).masked_fill(~validity, 0.0)
        return {
            **neuron,
            "primary_evidence": torch.sigmoid(primary_logits).masked_fill(~validity, 0.0),
            "normality_evidence": torch.sigmoid(normality_view).masked_fill(~validity, 0.0),
            "context_evidence": torch.sigmoid(context_view).masked_fill(~validity, 0.0),
            "evidence_logits": logits,
            "evidence": evidence,
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
        self.router = WitnessRouter(eta_normal=eta_normal, eta_anomaly=eta_anomaly)

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

    def __init__(self, eta_anomaly: float = 0.25, local_width: int = 16) -> None:
        super().__init__()
        self.local_head = nn.Sequential(
            nn.Conv1d(4, local_width, kernel_size=1),
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
    ) -> dict[str, torch.Tensor]:
        host_clipped = host_score.clamp(1e-6, 1.0 - 1e-6)
        evidence_clipped = evidence.clamp(1e-6, 1.0 - 1e-6)
        features = torch.stack(
            [host_clipped, evidence_clipped, host_clipped - evidence_clipped, host_clipped * evidence_clipped],
            dim=1,
        )
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
        self.router = NeuronOnlyRouter(eta_anomaly=eta_anomaly)

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
