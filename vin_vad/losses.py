from __future__ import annotations

import torch
from torch.nn import functional as F

from vin_vad.event_chain import EventChain


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
