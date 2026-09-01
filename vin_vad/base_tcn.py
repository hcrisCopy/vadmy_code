from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


class ResidualTCNBlock(nn.Module):
    def __init__(self, width: int, dilation: int, dropout: float) -> None:
        super().__init__()
        self.conv = nn.Conv1d(
            width,
            width,
            kernel_size=3,
            dilation=dilation,
            padding=dilation,
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, features: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        valid = mask[:, None, :].to(features.dtype)
        residual = features * valid
        update = self.conv(residual)
        update = self.dropout(F.gelu(update))
        return (residual + update) * valid


class BaseTCN(nn.Module):
    """Final-layer CLS encoder fixed for every E0--E3 ablation."""

    def __init__(self, input_dim: int = 768, width: int = 512, dropout: float = 0.1) -> None:
        super().__init__()
        self.normalization = nn.LayerNorm(input_dim, elementwise_affine=False)
        self.input_projection = nn.Linear(input_dim, width)
        self.blocks = nn.ModuleList(
            ResidualTCNBlock(width, dilation, dropout) for dilation in (1, 2, 4)
        )
        self.readout = nn.Linear(width, 1)

    def forward(self, final_layer_cls: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        if final_layer_cls.ndim != 3 or final_layer_cls.shape[-1] != 768:
            raise ValueError(f"expected [B,T,768], got {tuple(final_layer_cls.shape)}")
        if mask.shape != final_layer_cls.shape[:2]:
            raise ValueError(f"mask shape {tuple(mask.shape)} does not match input")
        valid = mask.unsqueeze(-1).to(final_layer_cls.dtype)
        features = self.normalization(final_layer_cls) * valid
        features = self.input_projection(features) * valid
        features = features.transpose(1, 2)
        for block in self.blocks:
            features = block(features, mask)
        logits = self.readout(features.transpose(1, 2)).squeeze(-1)
        return logits.masked_fill(~mask, 0.0)
