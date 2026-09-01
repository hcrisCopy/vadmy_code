from __future__ import annotations

import math

import torch
from torch import nn


def sinusoidal_positions(length: int, width: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    """Return fixed positions so the query never reads visual target content."""
    if width % 2:
        raise ValueError("position width must be even")
    positions = torch.arange(length, device=device, dtype=torch.float32).unsqueeze(1)
    frequencies = torch.exp(
        torch.arange(0, width, 2, device=device, dtype=torch.float32)
        * (-math.log(10000.0) / width)
    )
    encoding = torch.zeros(length, width, device=device, dtype=torch.float32)
    encoding[:, 0::2] = torch.sin(positions * frequencies)
    encoding[:, 1::2] = torch.cos(positions * frequencies)
    return encoding.to(dtype=dtype)


def guarded_attention_mask(
    validity: torch.Tensor,
    guard_radius: int,
    attention_heads: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Mask target neighborhoods and padding for batched cross-attention.

    Each valid target in a sequence of length at least two retains at least one
    context key. Length-one sequences are excluded from the prediction loss.
    """
    if validity.ndim != 2 or validity.dtype != torch.bool:
        raise ValueError("validity must be a boolean [B,T] tensor")
    if guard_radius < 0 or attention_heads < 1:
        raise ValueError("guard_radius must be non-negative and heads positive")
    batch, steps = validity.shape
    mask = torch.ones(batch, steps, steps, dtype=torch.bool, device=validity.device)
    prediction_mask = torch.zeros_like(validity)
    indices = torch.arange(steps, device=validity.device)
    for batch_index in range(batch):
        length = int(validity[batch_index].sum().item())
        if length < 1 or not torch.all(validity[batch_index, :length]):
            raise ValueError("valid positions must form a non-empty left-aligned prefix")
        if torch.any(validity[batch_index, length:]):
            raise ValueError("valid positions must be left aligned")
        if length >= 2:
            radius = min(guard_radius, max(0, (length - 2) // 2))
            distance = torch.abs(indices[:length, None] - indices[None, :length])
            mask[batch_index, :length, :length] = distance <= radius
            prediction_mask[batch_index, :length] = True
        else:
            mask[batch_index, 0, 0] = False
        # Padded query rows do not enter the loss, but need one finite attention
        # path to avoid all-masked softmax rows producing NaN.
        if length < steps:
            mask[batch_index, length:, 0] = False
    expanded = (
        mask[:, None]
        .expand(batch, attention_heads, steps, steps)
        .reshape(batch * attention_heads, steps, steps)
    )
    return expanded, prediction_mask


class MaskedCrossAttentionBlock(nn.Module):
    def __init__(self, width: int, heads: int, dropout: float) -> None:
        super().__init__()
        self.query_norm = nn.LayerNorm(width)
        self.memory_norm = nn.LayerNorm(width)
        self.attention = nn.MultiheadAttention(
            width, heads, dropout=dropout, batch_first=True
        )
        self.attention_dropout = nn.Dropout(dropout)
        self.feed_forward_norm = nn.LayerNorm(width)
        self.feed_forward = nn.Sequential(
            nn.Linear(width, 4 * width),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(4 * width, width),
            nn.Dropout(dropout),
        )

    def forward(
        self,
        query: torch.Tensor,
        memory: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        attended, _ = self.attention(
            self.query_norm(query),
            self.memory_norm(memory),
            memory,
            attn_mask=attention_mask,
            need_weights=False,
        )
        query = query + self.attention_dropout(attended)
        return query + self.feed_forward(self.feed_forward_norm(query))


class MaskedContextPredictor(nn.Module):
    """Predict all 12x768 CLS coordinates from guarded temporal context."""

    layers = 12
    dimensions = 768

    def __init__(
        self,
        model_width: int,
        input_rank: int,
        head_rank: int,
        attention_heads: int,
        attention_layers: int,
        guard_radius: int,
        dropout: float,
        sigma_min: float,
        sigma_max: float,
    ) -> None:
        super().__init__()
        if model_width % attention_heads:
            raise ValueError("model_width must be divisible by attention_heads")
        if attention_layers < 1 or not 0.0 < sigma_min < sigma_max:
            raise ValueError("invalid attention layer count or sigma bounds")
        self.model_width = int(model_width)
        self.attention_heads = int(attention_heads)
        self.guard_radius = int(guard_radius)
        self.sigma_min = float(sigma_min)
        self.sigma_max = float(sigma_max)
        self.fixed_layer_norm = nn.LayerNorm(
            self.dimensions, elementwise_affine=False
        )
        self.coordinate_reduction = nn.Linear(self.dimensions, input_rank)
        self.token_projection = nn.Linear(self.layers * input_rank, model_width)
        self.blocks = nn.ModuleList(
            [
                MaskedCrossAttentionBlock(model_width, attention_heads, dropout)
                for _ in range(attention_layers)
            ]
        )
        self.output_norm = nn.LayerNorm(model_width)
        self.layer_down = nn.Parameter(
            torch.empty(self.layers, model_width, head_rank)
        )
        self.mean_up = nn.Parameter(
            torch.empty(self.layers, head_rank, self.dimensions)
        )
        self.scale_up = nn.Parameter(
            torch.empty(self.layers, head_rank, self.dimensions)
        )
        self.mean_bias = nn.Parameter(torch.zeros(self.layers, self.dimensions))
        initial_probability = (1.0 - sigma_min) / (sigma_max - sigma_min)
        initial_probability = min(1.0 - 1e-4, max(1e-4, initial_probability))
        initial_scale_bias = math.log(
            initial_probability / (1.0 - initial_probability)
        )
        self.scale_bias = nn.Parameter(
            torch.full((self.layers, self.dimensions), initial_scale_bias)
        )
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.xavier_uniform_(self.layer_down)
        nn.init.xavier_uniform_(self.mean_up)
        nn.init.xavier_uniform_(self.scale_up)

    def normalize_hidden(self, hidden: torch.Tensor) -> torch.Tensor:
        if hidden.ndim != 4 or hidden.shape[-2:] != (
            self.layers,
            self.dimensions,
        ):
            raise ValueError(f"expected hidden [B,T,12,768], got {hidden.shape}")
        return self.fixed_layer_norm(hidden)

    def forward(
        self, hidden: torch.Tensor, validity: torch.Tensor
    ) -> dict[str, torch.Tensor]:
        normalized = self.normalize_hidden(hidden)
        batch, steps = normalized.shape[:2]
        reduced = self.coordinate_reduction(normalized).flatten(start_dim=2)
        positions = sinusoidal_positions(
            steps, self.model_width, hidden.device, hidden.dtype
        )
        memory = self.token_projection(reduced) + positions.unsqueeze(0)
        # The query contains positions only. It has no hidden-state input path.
        query = positions.unsqueeze(0).expand(batch, steps, self.model_width)
        attention_mask, prediction_mask = guarded_attention_mask(
            validity, self.guard_radius, self.attention_heads
        )
        for block in self.blocks:
            query = block(query, memory, attention_mask)
        context = self.output_norm(query)
        layer_context = torch.einsum("btc,lcr->btlr", context, self.layer_down)
        mean = (
            torch.einsum("btlr,lrd->btld", layer_context, self.mean_up)
            + self.mean_bias
        )
        raw_scale = (
            torch.einsum("btlr,lrd->btld", layer_context, self.scale_up)
            + self.scale_bias
        )
        sigma = self.sigma_min + (self.sigma_max - self.sigma_min) * torch.sigmoid(
            raw_scale
        )
        return {
            "normalized_hidden": normalized,
            "mean": mean,
            "sigma": sigma,
            "prediction_mask": prediction_mask,
        }


def gaussian_nll(
    target: torch.Tensor,
    mean: torch.Tensor,
    sigma: torch.Tensor,
    prediction_mask: torch.Tensor,
) -> torch.Tensor:
    if target.shape != mean.shape or target.shape != sigma.shape:
        raise ValueError("target, mean, and sigma must share [B,T,12,768]")
    if prediction_mask.shape != target.shape[:2]:
        raise ValueError("prediction_mask must match target [B,T]")
    per_coordinate = 0.5 * ((target - mean) / sigma).square() + torch.log(sigma)
    weights = prediction_mask[..., None, None].to(per_coordinate.dtype)
    denominator = weights.sum() * target.shape[-2] * target.shape[-1]
    if denominator.item() == 0:
        raise ValueError("Gaussian NLL received no context-predictable targets")
    return (per_coordinate * weights).sum() / denominator


def detached_distribution(result: dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
    """Route weak bag losses away from the normal reference predictor."""
    return result["mean"].detach(), result["sigma"].detach()
