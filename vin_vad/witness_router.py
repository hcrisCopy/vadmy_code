from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F


def masked_mean(values: torch.Tensor, validity: torch.Tensor) -> torch.Tensor:
    denominator = validity.sum(dim=1).clamp_min(1).to(values.dtype)
    return (values * validity.to(values.dtype)).sum(dim=1) / denominator


def masked_standardize(values: torch.Tensor, validity: torch.Tensor) -> torch.Tensor:
    mean = masked_mean(values, validity).unsqueeze(1)
    centered = (values - mean).masked_fill(~validity, 0.0)
    variance = masked_mean(centered.square(), validity).unsqueeze(1)
    return (centered / torch.sqrt(variance + 1e-6)).masked_fill(~validity, 0.0)


def masked_summary(values: torch.Tensor, validity: torch.Tensor) -> torch.Tensor:
    rows = []
    for value, mask in zip(values, validity):
        valid = value[mask]
        if valid.numel() == 0:
            raise ValueError("every video needs at least one valid snippet")
        count = min(valid.numel(), max(1, math.ceil(0.1 * valid.numel())))
        rows.append(
            torch.stack(
                [valid.mean(), valid.std(unbiased=False), torch.topk(valid, count).values.mean(), valid.max()]
            )
        )
    return torch.stack(rows)


def masked_topk_anchor(values: torch.Tensor, validity: torch.Tensor) -> torch.Tensor:
    """Use the host's standard weak-MIL top-k score as an event anchor."""
    anchors = []
    for value, mask in zip(values, validity):
        valid = value[mask]
        if valid.numel() == 0:
            raise ValueError("every video needs at least one valid snippet")
        count = min(valid.numel(), int(valid.numel() / 16 + 1))
        anchors.append(torch.topk(valid, count).values.mean())
    return torch.stack(anchors)


def masked_local_max(
    values: torch.Tensor, validity: torch.Tensor, width: int = 41
) -> torch.Tensor:
    if values.shape != validity.shape or width <= 0 or width % 2 == 0:
        raise ValueError("values/validity must share [B,T] and width must be positive odd")
    masked = values.masked_fill(~validity, -torch.inf)
    pooled = F.max_pool1d(
        masked.unsqueeze(1), kernel_size=width, stride=1, padding=width // 2
    ).squeeze(1)
    return pooled.masked_fill(~validity, 0.0)


def masked_correlation(first: torch.Tensor, second: torch.Tensor, validity: torch.Tensor) -> torch.Tensor:
    outputs = []
    for left, right, mask in zip(first, second, validity):
        left = left[mask]
        right = right[mask]
        left_centered = left - left.mean()
        right_centered = right - right.mean()
        denominator = torch.sqrt(
            left_centered.square().sum() * right_centered.square().sum()
        ).clamp_min(1e-6)
        outputs.append((left_centered * right_centered).sum() / denominator)
    return torch.stack(outputs)


def video_summary(host_score: torch.Tensor, evidence: torch.Tensor, validity: torch.Tensor) -> torch.Tensor:
    if host_score.shape != evidence.shape or validity.shape != host_score.shape:
        raise ValueError("host_score, evidence and validity must share [B,T]")
    host = masked_summary(host_score, validity)
    neuron = masked_summary(evidence, validity)
    correlation = masked_correlation(host_score, evidence, validity).unsqueeze(1)
    disagreement = masked_mean((host_score - evidence).abs(), validity).unsqueeze(1)
    return torch.cat([host, neuron, correlation, disagreement], dim=1)


def inverse_softplus(value: float) -> float:
    if value <= 0.0:
        raise ValueError("softplus target must be positive")
    return math.log(math.expm1(value))


class WitnessRouter(nn.Module):
    """One video state routes global suppression and local witness correction."""

    def __init__(self, eta_normal: float = 1.0, eta_anomaly: float = 0.25, local_width: int = 16) -> None:
        super().__init__()
        self.video_head = nn.Linear(10, 1)
        self.raw_eta_normal = nn.Parameter(torch.tensor(inverse_softplus(eta_normal)))
        self.raw_eta_anomaly = nn.Parameter(torch.tensor(inverse_softplus(eta_anomaly)))

    def forward(
        self,
        host_score: torch.Tensor,
        evidence: torch.Tensor,
        validity: torch.Tensor,
        eta_normal_override: float | None = None,
        eta_anomaly_override: float | None = None,
        positive_consensus: torch.Tensor | None = None,
        negative_consensus: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        if positive_consensus is not None and positive_consensus.shape != host_score.shape:
            raise ValueError("positive_consensus must share the [B,T] host-score shape")
        if negative_consensus is not None and negative_consensus.shape != host_score.shape:
            raise ValueError("negative_consensus must share the [B,T] host-score shape")
        summary = video_summary(host_score, evidence, validity)
        video_logit = self.video_head(summary).squeeze(1)
        video_probability = torch.sigmoid(video_logit)
        hard_authorization = (video_probability >= 0.5).to(video_probability.dtype)
        anomaly_authorized = (
            hard_authorization
            + video_probability
            - video_probability.detach()
        )
        normal_authorized = 1.0 - anomaly_authorized
        # The hard state decides whether correction is allowed.  Its bounded
        # positive confidence only controls the strength of already-localized
        # anomaly correction, so it cannot create a whole-video score offset.
        anomaly_confidence_gain = 1.0 + torch.tanh(torch.relu(video_logit))
        eta_normal = (
            torch.nn.functional.softplus(self.raw_eta_normal)
            if eta_normal_override is None
            else host_score.new_tensor(eta_normal_override)
        )
        eta_anomaly = (
            torch.nn.functional.softplus(self.raw_eta_anomaly)
            if eta_anomaly_override is None
            else host_score.new_tensor(eta_anomaly_override)
        )
        delta_normal_video = eta_normal * torch.minimum(video_logit, torch.zeros_like(video_logit))
        # A video-level normal decision is weak supervision, so it cannot safely
        # erase a location where every independent witness role agrees on an
        # anomaly.  Positive consensus only protects the frozen host here; it
        # does not create anomaly score by itself.
        positive_normal_protection = (
            torch.zeros_like(host_score)
            if positive_consensus is None
            else positive_consensus.clamp(0.0, 1.0)
        ).masked_fill(~validity, 0.0)
        candidate_delta_normal = (
            delta_normal_video.unsqueeze(1).expand_as(host_score)
            * (1.0 - positive_normal_protection)
        )

        host_clipped = host_score.clamp(1e-6, 1.0 - 1e-6)
        evidence_clipped = evidence.clamp(1e-6, 1.0 - 1e-6)
        direct_witness = masked_standardize(evidence_clipped, validity).clamp(-3.0, 3.0)
        direct_host = masked_standardize(host_clipped, validity).clamp(-3.0, 3.0)
        witness_support = torch.relu(direct_witness)
        veto_support = torch.relu(-direct_witness)
        # The global normal route fills only locations not already handled by
        # the local veto.  Their supports are complementary, preventing two
        # negative corrections from being stacked on the same snippet.
        local_veto_coverage = veto_support.clamp(max=1.0)
        delta_normal = candidate_delta_normal * (1.0 - local_veto_coverage)
        host_support = torch.relu(direct_host)
        consensus_conflict_veto = (
            torch.zeros_like(host_support)
            if negative_consensus is None
            else torch.minimum(
                host_support,
                negative_consensus.clamp_min(0.0),
            )
        ).masked_fill(~validity, 0.0)
        host_miss_support = torch.relu(-direct_host)
        complementary_support = torch.minimum(
            witness_support, host_miss_support
        )
        witness_event_support = masked_local_max(witness_support, validity)
        event_anchor = masked_local_max(host_clipped, validity)
        event_gap = torch.relu(
            torch.logit(event_anchor.clamp(1e-6, 1.0 - 1e-6))
            - torch.logit(host_clipped)
        ).masked_fill(~validity, 0.0)
        local_shape = (
            anomaly_authorized.unsqueeze(1)
            * (
                witness_support
                + complementary_support
                + witness_event_support * event_gap
                - consensus_conflict_veto
            )
            - normal_authorized.unsqueeze(1) * veto_support
        ).masked_fill(~validity, 0.0)
        # q decides the correction direction; neuron evidence decides its support.
        # A non-zero mean is required to repair cross-video ranking, which dominates
        # frame AUC/AP, while the support remains temporally localized.
        delta_anomaly = (
            eta_anomaly * anomaly_confidence_gain.unsqueeze(1) * local_shape
        )
        delta_anomaly = delta_anomaly.masked_fill(~validity, 0.0)
        delta_normal = delta_normal.masked_fill(~validity, 0.0)

        host_logit = torch.logit(host_clipped)
        base = torch.sigmoid(host_logit)
        shifted = torch.sigmoid(host_logit + delta_normal + delta_anomaly)
        # The consensus residual seeds missed event positions.  A second,
        # point-authorized convex step completes only locations that carry their
        # own witness support and can never overshoot the local event peak.
        completion_anchor = masked_local_max(shifted, validity)
        completion_gate = (
            anomaly_authorized.unsqueeze(1) * witness_support.clamp(max=1.0)
        ).masked_fill(~validity, 0.0)
        completed = shifted + completion_gate * (completion_anchor - shifted)
        corrected = host_score + completed - base
        corrected = corrected.clamp(0.0, 1.0).masked_fill(~validity, 0.0)
        if eta_normal_override == 0.0 and eta_anomaly_override == 0.0:
            # The explicit ablation contract is bitwise identity, not merely
            # numerical closeness after logit/sigmoid round trips.
            corrected = host_score.masked_fill(~validity, 0.0)
        return {
            "summary": summary,
            "video_logit": video_logit,
            "video_probability": video_probability,
            "anomaly_authorized": anomaly_authorized,
            "normal_authorized": normal_authorized,
            "anomaly_confidence_gain": anomaly_confidence_gain,
            "eta_normal": eta_normal,
            "eta_anomaly": eta_anomaly,
            "delta_normal": delta_normal,
            "delta_anomaly": delta_anomaly,
            "positive_normal_protection": positive_normal_protection,
            "local_shape": local_shape,
            "witness_support": witness_support,
            "host_miss_support": host_miss_support,
            "complementary_support": complementary_support,
            "witness_event_support": witness_event_support,
            "veto_support": veto_support,
            "local_veto_coverage": local_veto_coverage,
            "consensus_conflict_veto": consensus_conflict_veto,
            "event_anchor": event_anchor,
            "event_gap": event_gap,
            "completion_anchor": completion_anchor,
            "completion_gate": completion_gate,
            "corrected_score": corrected,
        }
