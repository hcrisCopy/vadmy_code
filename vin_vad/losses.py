from __future__ import annotations

import torch
from torch.nn import functional as F

from vin_vad.event_chain import EventChain
from vin_vad.host_auditor import masked_mean, masked_topk_mean


def topk_video_probability(logits: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """DSANet binary MIL rule: mean of floor(T/16)+1 largest probabilities."""
    probabilities = torch.sigmoid(logits)
    outputs = []
    for row, valid in zip(probabilities, mask):
        values = row[valid]
        count = min(len(values), int(len(values) / 16 + 1))
        outputs.append(torch.topk(values, k=count).values.mean())
    return torch.stack(outputs)


def bag_loss(
    variant: str,
    logits: torch.Tensor,
    mask: torch.Tensor,
    labels: torch.Tensor,
    chain: EventChain | None,
) -> tuple[torch.Tensor, torch.Tensor]:
    labels = labels.to(logits.dtype)
    if variant == "e0":
        video_probability = topk_video_probability(logits, mask)
        return F.binary_cross_entropy(video_probability, labels), video_probability
    if chain is None:
        raise ValueError(f"{variant} requires an EventChain")
    log_p0, log_p1 = chain.video_log_probs(logits, mask)
    selected = torch.where(labels > 0.5, log_p1, log_p0)
    return -selected.mean(), log_p1.exp()


def asymmetric_mil_loss(
    corrected_score: torch.Tensor,
    mask: torch.Tensor,
    labels: torch.Tensor,
    epsilon: float = 1e-6,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Dense negatives for normal bags and top-k positives for abnormal bags."""
    if corrected_score.ndim != 2 or corrected_score.shape != mask.shape:
        raise ValueError("corrected_score and mask must share [B,T]")
    if labels.shape != corrected_score.shape[:1]:
        raise ValueError("labels must have shape [B]")
    if mask.dtype != torch.bool:
        raise ValueError("mask must be boolean")
    score = corrected_score.clamp(epsilon, 1.0 - epsilon)
    normal_per_video = masked_mean(-torch.log1p(-score), mask)
    abnormal_probability = masked_topk_mean(score, mask).clamp_min(epsilon)
    abnormal_per_video = -torch.log(abnormal_probability)
    per_video = torch.where(labels > 0.5, abnormal_per_video, normal_per_video)
    return per_video.mean(), per_video


def correction_budget_loss(
    correction_size: torch.Tensor,
    budget: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return the one-sided budget penalty and its unweighted violation."""
    if budget < 0.0:
        raise ValueError("correction budget must be non-negative")
    violation = torch.relu(correction_size - budget)
    return violation, correction_size
