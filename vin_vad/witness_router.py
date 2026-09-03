from __future__ import annotations

import math

import torch
from torch import nn


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


def video_summary(
    host_score: torch.Tensor,
    evidence: torch.Tensor,
    validity: torch.Tensor,
    veto_evidence: torch.Tensor | None = None,
) -> torch.Tensor:
    if host_score.shape != evidence.shape or validity.shape != host_score.shape:
        raise ValueError("host_score, evidence and validity must share [B,T]")
    host = masked_summary(host_score, validity)
    neuron = masked_summary(evidence, validity)
    veto_evidence = 1.0 - evidence if veto_evidence is None else veto_evidence
    veto = masked_summary(veto_evidence, validity)
    witness_correlation = masked_correlation(host_score, evidence, validity).unsqueeze(1)
    veto_correlation = masked_correlation(host_score, veto_evidence, validity).unsqueeze(1)
    witness_disagreement = masked_mean((host_score - evidence).abs(), validity).unsqueeze(1)
    veto_disagreement = masked_mean((host_score - veto_evidence).abs(), validity).unsqueeze(1)
    return torch.cat(
        [host, neuron, veto, witness_correlation, veto_correlation,
         witness_disagreement, veto_disagreement],
        dim=1,
    )


def inverse_softplus(value: float) -> float:
    if value <= 0.0:
        raise ValueError("softplus target must be positive")
    return math.log(math.expm1(value))


class WitnessRouter(nn.Module):
    """One video state routes global suppression and local witness correction."""

    def __init__(self, eta_normal: float = 1.0, eta_anomaly: float = 0.25, local_width: int = 16) -> None:
        super().__init__()
        self.video_head = nn.Linear(16, 1)
        self.local_head = nn.Sequential(
            nn.Conv1d(6, local_width, kernel_size=1),
            nn.GELU(),
            nn.Conv1d(local_width, 2, kernel_size=1),
        )
        self.raw_eta_normal = nn.Parameter(torch.tensor(inverse_softplus(eta_normal)))
        self.raw_eta_anomaly = nn.Parameter(torch.tensor(inverse_softplus(eta_anomaly)))

    def forward(
        self,
        host_score: torch.Tensor,
        evidence: torch.Tensor,
        validity: torch.Tensor,
        eta_normal_override: float | None = None,
        eta_anomaly_override: float | None = None,
        veto_evidence: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        veto_evidence = 1.0 - evidence if veto_evidence is None else veto_evidence
        summary = video_summary(host_score, evidence, validity, veto_evidence)
        video_logit = self.video_head(summary).squeeze(1)
        video_probability = torch.sigmoid(video_logit)
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
        delta_normal = delta_normal_video.unsqueeze(1).expand_as(host_score)

        host_clipped = host_score.clamp(1e-6, 1.0 - 1e-6)
        evidence_clipped = evidence.clamp(1e-6, 1.0 - 1e-6)
        veto_clipped = veto_evidence.clamp(1e-6, 1.0 - 1e-6)
        local_input = torch.stack(
            [host_clipped, evidence_clipped, veto_clipped,
             host_clipped - evidence_clipped, host_clipped - veto_clipped,
             evidence_clipped - veto_clipped],
            dim=1,
        )
        direct_witness = masked_standardize(evidence_clipped, validity).clamp(-3.0, 3.0)
        direct_veto = masked_standardize(veto_clipped, validity).clamp(-3.0, 3.0)
        local_residual = self.local_head(local_input)
        witness_support = torch.relu(torch.tanh(local_residual[:, 0] + direct_witness))
        veto_support = torch.relu(torch.tanh(local_residual[:, 1] + direct_veto))
        witness_support = witness_support.masked_fill(~validity, 0.0)
        veto_support = veto_support.masked_fill(~validity, 0.0)
        local_shape = (
            video_probability.unsqueeze(1) * witness_support
            - (1.0 - video_probability).unsqueeze(1) * veto_support
        ).masked_fill(~validity, 0.0)
        # q decides the correction direction; neuron evidence decides its support.
        # A non-zero mean is required to repair cross-video ranking, which dominates
        # frame AUC/AP, while the support remains temporally localized.
        delta_anomaly = eta_anomaly * local_shape
        delta_anomaly = delta_anomaly.masked_fill(~validity, 0.0)
        delta_normal = delta_normal.masked_fill(~validity, 0.0)

        host_logit = torch.logit(host_clipped)
        base = torch.sigmoid(host_logit)
        shifted = torch.sigmoid(host_logit + delta_normal + delta_anomaly)
        corrected = host_score + shifted - base
        corrected = corrected.clamp(0.0, 1.0).masked_fill(~validity, 0.0)
        if eta_normal_override == 0.0 and eta_anomaly_override == 0.0:
            # The explicit ablation contract is bitwise identity, not merely
            # numerical closeness after logit/sigmoid round trips.
            corrected = host_score.masked_fill(~validity, 0.0)
        return {
            "summary": summary,
            "video_logit": video_logit,
            "video_probability": video_probability,
            "eta_normal": eta_normal,
            "eta_anomaly": eta_anomaly,
            "delta_normal": delta_normal,
            "delta_anomaly": delta_anomaly,
            "local_shape": local_shape,
            "witness_support": witness_support,
            "veto_support": veto_support,
            "corrected_score": corrected,
        }
