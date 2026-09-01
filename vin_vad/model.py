from __future__ import annotations

import torch
from torch import nn

from vin_vad.base_tcn import BaseTCN
from vin_vad.event_chain import EventChain
from vin_vad.losses import topk_video_probability


class EventAblationModel(nn.Module):
    def __init__(self, variant: str, width: int = 512, dropout: float = 0.1) -> None:
        super().__init__()
        if variant not in {"e0", "e1", "e2", "e3"}:
            raise ValueError(f"variant must be e0/e1/e2/e3, got {variant}")
        self.variant = variant
        self.tcn = BaseTCN(width=width, dropout=dropout)
        self.chain = None if variant == "e0" else EventChain(variant)

    def forward(self, features: torch.Tensor, mask: torch.Tensor) -> dict[str, torch.Tensor]:
        emissions = self.tcn(features, mask)
        if self.variant == "e0":
            snippet_probability = torch.sigmoid(emissions).masked_fill(~mask, 0.0)
            video_probability = topk_video_probability(emissions, mask)
        else:
            _, log_p1 = self.chain.video_log_probs(emissions, mask)
            video_probability = log_p1.exp()
            snippet_probability = self.chain.snippet_marginals(emissions, mask)
        return {
            "emissions": emissions,
            "video_prob": video_probability,
            "snippet_prob": snippet_probability,
        }
