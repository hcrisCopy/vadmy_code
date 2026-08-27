from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


class ScoreCorrectionHead(nn.Module):
    """Score head used by the first-round CLS-neuron expert."""

    def __init__(self, width: int = 32) -> None:
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv1d(6, width, 3, padding=1),
            nn.GELU(),
            nn.Conv1d(width, width, 3, padding=2, dilation=2),
            nn.GELU(),
            nn.Conv1d(width, 1, 1),
        )
        nn.init.zeros_(self.body[-1].weight)
        nn.init.zeros_(self.body[-1].bias)

    def forward(self, baseline_probability: torch.Tensor, expert_probability: torch.Tensor) -> torch.Tensor:
        baseline = torch.logit(baseline_probability.clamp(1e-5, 1.0 - 1e-5))
        expert = torch.logit(expert_probability.clamp(1e-5, 1.0 - 1e-5))
        base_change = F.pad(baseline[:, 1:] - baseline[:, :-1], (1, 0))
        expert_change = F.pad(expert[:, 1:] - expert[:, :-1], (1, 0))
        features = torch.stack(
            [baseline, expert, baseline - expert, baseline * torch.tanh(expert), base_change, expert_change],
            dim=1,
        )
        return baseline + self.body(features).squeeze(1)


def calibrated_probability(
    baseline_probability: torch.Tensor,
    expert_probability: torch.Tensor,
    correction_logits: torch.Tensor,
    correction_weight: float = 0.2,
    neuron_weight: float = 0.1,
) -> torch.Tensor:
    """Conservative universal fusion in logit space.

    The neuron evidence is standardized within each video. Thus its coefficient has
    the same meaning for every baseline and dataset, while retaining temporal neuron
    ordering and avoiding dataset-specific score calibration.
    """
    baseline_logits = torch.logit(baseline_probability.clamp(1e-5, 1.0 - 1e-5))
    mean = expert_probability.mean(dim=1, keepdim=True)
    scale = expert_probability.std(dim=1, keepdim=True, unbiased=False).clamp_min(1e-6)
    neuron_evidence = (expert_probability - mean) / scale
    logits = (
        (1.0 - correction_weight) * baseline_logits
        + correction_weight * correction_logits
        + neuron_weight * neuron_evidence
    )
    return torch.sigmoid(logits)

