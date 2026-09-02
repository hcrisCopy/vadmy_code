from __future__ import annotations

import torch
from torch import nn


class _Entmax15Function(torch.autograd.Function):
    """Exact alpha=1.5 entmax with the closed-form threshold.

    Adapted from DeepSPIN/entmax (MIT license), kept under ``rely/entmax``.
    The local copy avoids adding a runtime dependency to the formal server.
    """

    @staticmethod
    def forward(ctx: object, logits: torch.Tensor, dim: int) -> torch.Tensor:
        shifted = logits - logits.max(dim=dim, keepdim=True).values
        shifted = shifted / 2.0
        sorted_logits = torch.sort(shifted, dim=dim, descending=True).values
        size = sorted_logits.shape[dim]
        rho = torch.arange(
            1, size + 1, device=logits.device, dtype=logits.dtype
        )
        view = [1] * logits.ndim
        view[dim] = size
        rho = rho.view(view)
        mean = sorted_logits.cumsum(dim) / rho
        mean_square = sorted_logits.square().cumsum(dim) / rho
        variance_sum = rho * (mean_square - mean.square())
        delta = torch.clamp((1.0 - variance_sum) / rho, min=0.0)
        thresholds = mean - torch.sqrt(delta)
        support_size = (thresholds <= sorted_logits).sum(dim, keepdim=True)
        threshold = thresholds.gather(dim, support_size - 1)
        probabilities = torch.clamp(shifted - threshold, min=0.0).square()
        ctx.dim = dim
        ctx.save_for_backward(probabilities)
        return probabilities

    @staticmethod
    def backward(
        ctx: object, gradient: torch.Tensor
    ) -> tuple[torch.Tensor, None]:
        (probabilities,) = ctx.saved_tensors
        inverse_hessian = probabilities.sqrt()
        scaled = gradient * inverse_hessian
        normalizer = inverse_hessian.sum(ctx.dim, keepdim=True).clamp_min(1e-12)
        projection = scaled.sum(ctx.dim, keepdim=True) / normalizer
        return scaled - projection * inverse_hessian, None


def entmax15(logits: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """Map logits to a non-negative, sum-one, potentially sparse field."""
    return _Entmax15Function.apply(logits, dim)


class ViolationField(nn.Module):
    """Convert masked conditional residuals into one directional evidence curve."""

    layers = 12
    dimensions = 768
    directions = 2

    def __init__(
        self,
        delta: float,
        statistics_momentum: float,
        epsilon: float = 1e-6,
        evidence_type: str = "contextual_directional",
        global_mean: torch.Tensor | None = None,
        global_sigma: torch.Tensor | None = None,
    ) -> None:
        super().__init__()
        if delta < 0.0:
            raise ValueError("delta must be non-negative")
        if not 0.0 < statistics_momentum <= 1.0:
            raise ValueError("statistics_momentum must be in (0,1]")
        if epsilon <= 0.0:
            raise ValueError("epsilon must be positive")
        allowed = {
            "raw_directional",
            "global_directional",
            "contextual_absolute",
            "contextual_directional",
        }
        if evidence_type not in allowed:
            raise ValueError(f"evidence_type must be one of {sorted(allowed)}")
        self.delta = float(delta)
        self.statistics_momentum = float(statistics_momentum)
        self.epsilon = float(epsilon)
        self.evidence_type = str(evidence_type)
        # Equal logits are required by the protocol. The correction loss in B4
        # will learn omega jointly; B2 does not preselect coordinates.
        self.omega = nn.Parameter(
            torch.zeros(self.layers, self.dimensions, self.directions)
        )
        self.register_buffer("running_median", torch.tensor(0.0))
        self.register_buffer("running_mad", torch.tensor(1.0))
        self.register_buffer("statistics_initialized", torch.tensor(False))
        self.register_buffer("statistics_updates", torch.tensor(0, dtype=torch.long))
        self.register_buffer("normal_snippets_seen", torch.tensor(0, dtype=torch.long))
        global_shape = (self.layers, self.dimensions)
        if global_mean is None:
            global_mean = torch.zeros(global_shape)
        if global_sigma is None:
            global_sigma = torch.ones(global_shape)
        if tuple(global_mean.shape) != global_shape or tuple(global_sigma.shape) != global_shape:
            raise ValueError("global mean/sigma must have shape [12,768]")
        if torch.any(global_sigma <= 0.0):
            raise ValueError("global sigma must be positive")
        self.register_buffer("global_mean", global_mean.detach().float().clone())
        self.register_buffer("global_sigma", global_sigma.detach().float().clone())

    def probabilities(self) -> torch.Tensor:
        flat = entmax15(self.omega.reshape(-1), dim=0)
        return flat.reshape_as(self.omega)

    @torch.no_grad()
    def update_normal_statistics(
        self,
        activation: torch.Tensor,
        validity: torch.Tensor,
        labels: torch.Tensor,
    ) -> None:
        if activation.shape != validity.shape:
            raise ValueError("activation and validity must share [B,T]")
        if labels.shape != activation.shape[:1]:
            raise ValueError("labels must have shape [B]")
        normal_mask = validity & (labels <= 0.5).unsqueeze(1)
        values = activation.detach()[normal_mask]
        if values.numel() == 0:
            return
        batch_median = values.median()
        batch_mad = (values - batch_median).abs().median()
        if not bool(self.statistics_initialized):
            self.running_median.copy_(batch_median)
            self.running_mad.copy_(batch_mad)
            self.statistics_initialized.fill_(True)
        else:
            momentum = self.statistics_momentum
            self.running_median.lerp_(batch_median, momentum)
            self.running_mad.lerp_(batch_mad, momentum)
        self.statistics_updates.add_(1)
        self.normal_snippets_seen.add_(values.numel())

    def forward(
        self,
        normalized_hidden: torch.Tensor,
        mean: torch.Tensor,
        sigma: torch.Tensor,
        validity: torch.Tensor,
        labels: torch.Tensor | None = None,
        update_statistics: bool = False,
    ) -> dict[str, torch.Tensor]:
        expected = (self.layers, self.dimensions)
        if normalized_hidden.ndim != 4 or normalized_hidden.shape[-2:] != expected:
            raise ValueError("normalized_hidden must have shape [B,T,12,768]")
        if mean.shape != normalized_hidden.shape or sigma.shape != normalized_hidden.shape:
            raise ValueError("mean and sigma must match normalized_hidden")
        if validity.shape != normalized_hidden.shape[:2] or validity.dtype != torch.bool:
            raise ValueError("validity must be a boolean [B,T] tensor")
        if torch.any(sigma <= 0.0):
            raise ValueError("sigma must be strictly positive")
        # Weak labels must never alter either normal reference distribution.
        if self.evidence_type == "raw_directional":
            residual = normalized_hidden
        elif self.evidence_type == "global_directional":
            residual = (normalized_hidden - self.global_mean) / (
                self.global_sigma + self.epsilon
            )
        else:
            residual = (normalized_hidden - mean.detach()) / (
                sigma.detach() + self.epsilon
            )
        if self.evidence_type == "contextual_absolute":
            absolute = torch.relu(residual.abs() - self.delta)
            # Duplicate the sign-free value so C2 keeps the exact same
            # 18,432-parameter entmax readout as the directional controls.
            directional = torch.stack((absolute, absolute), dim=-1)
        else:
            positive = torch.relu(residual - self.delta)
            negative = torch.relu(-residual - self.delta)
            directional = torch.stack((positive, negative), dim=-1)
        probability = self.probabilities()
        activation = torch.einsum("btldq,ldq->bt", directional, probability)
        if update_statistics:
            if labels is None:
                raise ValueError("labels are required when updating normal statistics")
            self.update_normal_statistics(activation, validity, labels)
        scale = 1.4826 * self.running_mad + self.epsilon
        evidence = (activation - self.running_median) / scale
        activation = activation.masked_fill(~validity, 0.0)
        evidence = evidence.masked_fill(~validity, 0.0)
        residual = residual.masked_fill(~validity[..., None, None], 0.0)
        directional = directional.masked_fill(
            ~validity[..., None, None, None], 0.0
        )
        return {
            "residual": residual,
            "directional": directional,
            "probability": probability,
            "activation": activation,
            "evidence": evidence,
        }
