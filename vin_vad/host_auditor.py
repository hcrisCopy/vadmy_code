from __future__ import annotations

import torch
from torch import nn


def masked_topk_mean(values: torch.Tensor, validity: torch.Tensor) -> torch.Tensor:
    """Apply DSANet's floor(T/16)+1 pooling rule without reading padding."""
    if values.ndim != 2 or values.shape != validity.shape:
        raise ValueError("values and validity must share [B,T]")
    if validity.dtype != torch.bool:
        raise ValueError("validity must be boolean")
    pooled = []
    for row, mask in zip(values, validity):
        valid = row[mask]
        if valid.numel() == 0:
            raise ValueError("every video must contain at least one valid snippet")
        count = min(valid.numel(), int(valid.numel() / 16 + 1))
        pooled.append(torch.topk(valid, k=count).values.mean())
    return torch.stack(pooled)


def masked_mean(values: torch.Tensor, validity: torch.Tensor) -> torch.Tensor:
    if values.shape != validity.shape:
        raise ValueError("values and validity must share [B,T]")
    denominator = validity.sum(dim=1).clamp_min(1).to(values.dtype)
    return (values * validity.to(values.dtype)).sum(dim=1) / denominator


class TwoAxisHostAuditor(nn.Module):
    """Bounded cross-video suppression plus zero-mean within-video reordering."""

    def __init__(
        self,
        alpha_cross: float,
        alpha_within: float,
        normal_q_median: float = 0.0,
        normal_q_mad: float = 1.0,
        tau_normal: float = 0.0,
        epsilon: float = 1e-6,
    ) -> None:
        super().__init__()
        if alpha_cross < 0.0 or alpha_within < 0.0:
            raise ValueError("alpha bounds must be non-negative")
        if normal_q_mad < 0.0 or epsilon <= 0.0:
            raise ValueError("normal MAD must be non-negative and epsilon positive")
        self.alpha_cross = float(alpha_cross)
        self.alpha_within = float(alpha_within)
        self.epsilon = float(epsilon)
        self.kappa_cross = nn.Parameter(torch.tensor(0.0))
        self.kappa_within = nn.Parameter(torch.tensor(0.0))
        self.register_buffer("normal_q_median", torch.tensor(float(normal_q_median)))
        self.register_buffer("normal_q_mad", torch.tensor(float(normal_q_mad)))
        self.register_buffer("tau_normal", torch.tensor(float(tau_normal)))

    @torch.no_grad()
    def project_parameters(self) -> None:
        """Apply after every optimizer step in B4."""
        self.kappa_cross.clamp_(0.0, 1.0)
        self.kappa_within.clamp_(0.0, 1.0)

    @torch.no_grad()
    def set_normal_q_statistics(
        self, median: float, mad: float, tau_normal: float
    ) -> None:
        if mad < 0.0:
            raise ValueError("normal q MAD must be non-negative")
        self.normal_q_median.fill_(float(median))
        self.normal_q_mad.fill_(float(mad))
        self.tau_normal.fill_(float(tau_normal))

    def forward(
        self,
        host_score: torch.Tensor,
        evidence: torch.Tensor,
        validity: torch.Tensor,
        enable_cross: bool = True,
        enable_within: bool = True,
    ) -> dict[str, torch.Tensor]:
        if host_score.ndim != 2 or host_score.shape != evidence.shape:
            raise ValueError("host_score and evidence must share [B,T]")
        if validity.shape != host_score.shape or validity.dtype != torch.bool:
            raise ValueError("validity must be a boolean [B,T] tensor")
        valid_host = host_score[validity]
        if not torch.isfinite(valid_host).all() or torch.any(
            (valid_host < 0.0) | (valid_host > 1.0)
        ):
            raise ValueError("valid host scores must be finite probabilities")
        if not torch.isfinite(evidence[validity]).all():
            raise ValueError("valid evidence must be finite")

        host_video = masked_topk_mean(host_score, validity)
        evidence_video = masked_topk_mean(evidence, validity)
        q_scale = 1.4826 * self.normal_q_mad + self.epsilon
        standardized_q = (evidence_video - self.normal_q_median) / q_scale
        normal_support = torch.sigmoid(self.tau_normal - standardized_q)

        if enable_cross:
            cross_video = (
                -self.alpha_cross
                * self.kappa_cross
                * host_video
                * normal_support
            )
        else:
            cross_video = torch.zeros_like(host_video)
        delta_cross = cross_video.unsqueeze(1).expand_as(host_score)
        delta_cross = delta_cross.masked_fill(~validity, 0.0)

        if enable_within:
            within_raw = self.alpha_within * self.kappa_within * torch.tanh(evidence)
            center = masked_mean(within_raw, validity)
            delta_within = within_raw - center.unsqueeze(1)
            delta_within = delta_within.masked_fill(~validity, 0.0)
        else:
            delta_within = torch.zeros_like(evidence)

        total_delta = delta_cross + delta_within
        clipped_host = host_score.clamp(self.epsilon, 1.0 - self.epsilon)
        host_logit = torch.logit(clipped_host)
        base_probability = torch.sigmoid(host_logit)
        shifted_probability = torch.sigmoid(host_logit + total_delta)
        # Anchoring the numerical delta to the original tensor gives exact
        # identity at kappa=0 while retaining a first-order gradient.
        corrected_score = host_score + (shifted_probability - base_probability)
        corrected_score = corrected_score.clamp(0.0, 1.0).masked_fill(~validity, 0.0)

        within_absolute_mean = masked_mean(delta_within.abs(), validity)
        correction_size = cross_video.abs() + within_absolute_mean
        return {
            "corrected_score": corrected_score,
            "host_video": host_video,
            "evidence_video": evidence_video,
            "standardized_q": standardized_q,
            "normal_support": normal_support,
            "delta_cross_video": cross_video,
            "delta_cross": delta_cross,
            "delta_within": delta_within,
            "total_delta": total_delta,
            "correction_size_per_video": correction_size,
            "correction_size": correction_size.mean(),
        }
