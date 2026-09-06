from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

from vin_vad.witness_neurons import SignedTopKWitnessNeurons
from vin_vad.witness_router import (
    WitnessRouter,
    inverse_softplus,
    masked_standardize,
    masked_summary,
    masked_topk_anchor,
)
from vin_vad.witness_temporal import WitnessTemporalReadout


def masked_temporal_mean(
    values: torch.Tensor, validity: torch.Tensor, width: int
) -> torch.Tensor:
    if values.ndim != 3 or validity.shape != values.shape[:2] or width % 2 != 1:
        raise ValueError("values must be [B,T,C], validity [B,T], and width odd")
    channels = values.shape[-1]
    mask = validity[:, None].to(values.dtype)
    kernel = torch.ones(channels, 1, width, dtype=values.dtype, device=values.device)
    numerator = F.conv1d(
        values.transpose(1, 2) * mask, kernel, padding=width // 2, groups=channels
    )
    denominator = F.conv1d(mask, kernel[:1], padding=width // 2).clamp_min(1.0)
    return (numerator / denominator).transpose(1, 2).masked_fill(
        ~validity.unsqueeze(-1), 0.0
    )


class WitnessExpert(nn.Module):
    """Neuron-only path: its API intentionally has no host-score argument."""

    def __init__(
        self, active: int = 32, temporal_width: int = 64, normal_contexts: int = 4
    ) -> None:
        super().__init__()
        self.neurons = SignedTopKWitnessNeurons(
            active=active, contexts=normal_contexts
        )
        self.context_temporal = WitnessTemporalReadout(input_channels=24, width=temporal_width)

    def forward(
        self,
        hidden: torch.Tensor,
        validity: torch.Tensor,
        neuron_keep_mask: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        neuron = self.neurons(hidden, validity, neuron_keep_mask)
        signed_layers = neuron["layer_evidence"]
        signed_raw = signed_layers.mean(dim=-1)
        signed_absolute = (
            signed_raw - self.neurons.signed_score_mean
        ) / self.neurons.signed_score_std
        signed_absolute = signed_absolute.clamp(-3.0, 3.0).masked_fill(~validity, 0.0)
        rectified_layers = neuron["rectified_layer_evidence"]
        context_input = torch.cat(
            [
                masked_temporal_mean(rectified_layers, validity, width=9),
                masked_temporal_mean(rectified_layers, validity, width=25),
            ],
            dim=-1,
        )
        context_logits = self.context_temporal(context_input, validity)
        signed_role = masked_standardize(signed_raw, validity).clamp(-3.0, 3.0)
        context_role = masked_standardize(context_logits, validity).clamp(-3.0, 3.0)
        # One witness chain, not a jury of loosely motivated roles. Its signed
        # projection determines whether the host should move up or down; the
        # rectified magnitude supplies non-cancelling temporal support from the
        # same coordinates. Their symmetric mean is relative localization;
        # normal calibration separately preserves cross-video ordering.
        logits = 0.5 * (signed_role + context_role)
        evidence = torch.sigmoid(logits).masked_fill(~validity, 0.0)
        return {
            **neuron,
            "signed_evidence": torch.sigmoid(signed_role).masked_fill(~validity, 0.0),
            "context_evidence": torch.sigmoid(context_logits).masked_fill(~validity, 0.0),
            "absolute_evidence_logits": signed_absolute,
            "evidence_logits": logits.masked_fill(~validity, 0.0),
            "evidence": evidence,
        }


class WitnessVAD(nn.Module):
    def __init__(
        self,
        active: int = 32,
        temporal_width: int = 64,
        eta_normal: float = 1.0,
        eta_anomaly: float = 0.25,
        normal_contexts: int = 4,
    ) -> None:
        super().__init__()
        self.expert = WitnessExpert(
            active=active,
            temporal_width=temporal_width,
            normal_contexts=normal_contexts,
        )
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
            absolute_evidence_logits=expert["absolute_evidence_logits"],
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
    def __init__(
        self,
        active: int = 32,
        temporal_width: int = 64,
        eta_anomaly: float = 0.25,
        normal_contexts: int = 4,
    ) -> None:
        super().__init__()
        self.expert = WitnessExpert(
            active=active,
            temporal_width=temporal_width,
            normal_contexts=normal_contexts,
        )
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
    normal_contexts: int = 4,
) -> nn.Module:
    if variant == "w1":
        return HostVideoOnlyVAD(eta_normal=eta_normal)
    if variant == "w2":
        return NeuronOnlyWitnessVAD(
            active=active,
            temporal_width=temporal_width,
            eta_anomaly=eta_anomaly,
            normal_contexts=normal_contexts,
        )
    if variant == "w6":
        return WitnessVAD(
            active=active,
            temporal_width=temporal_width,
            eta_normal=eta_normal,
            eta_anomaly=eta_anomaly,
            normal_contexts=normal_contexts,
        )
    raise ValueError("variant must be w1, w2 or w6")
