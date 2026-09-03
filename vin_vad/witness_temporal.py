from __future__ import annotations

import torch
from torch import nn


class WitnessTemporalReadout(nn.Module):
    """Fixed d=1/d=2 temporal readout inherited from the Universal probe."""

    def __init__(self, input_channels: int = 12, width: int = 64) -> None:
        super().__init__()
        self.first = nn.Conv1d(input_channels, width, kernel_size=3, padding=1, dilation=1)
        self.second = nn.Conv1d(width, width, kernel_size=3, padding=2, dilation=2)
        self.output = nn.Conv1d(width, 1, kernel_size=1)
        self.activation = nn.GELU()

    def forward(self, layer_evidence: torch.Tensor, validity: torch.Tensor) -> torch.Tensor:
        if layer_evidence.ndim != 3 or validity.shape != layer_evidence.shape[:2]:
            raise ValueError("layer_evidence must be [B,T,L] and validity [B,T]")
        mask = validity.unsqueeze(1).to(layer_evidence.dtype)
        value = layer_evidence.transpose(1, 2) * mask
        value = self.activation(self.first(value)) * mask
        value = self.activation(self.second(value)) * mask
        logits = self.output(value).squeeze(1)
        return logits.masked_fill(~validity, 0.0)


class WitnessCoordinateReadout(nn.Module):
    """Keep sparse neuron identities until a shared learned projection."""

    def __init__(
        self, layers: int = 12, dimensions: int = 768, width: int = 64
    ) -> None:
        super().__init__()
        self.layers = int(layers)
        self.dimensions = int(dimensions)
        self.project = nn.Linear(layers * dimensions, width, bias=False)
        self.first = nn.Conv1d(width, width, kernel_size=3, padding=1, dilation=1)
        self.second = nn.Conv1d(width, width, kernel_size=3, padding=2, dilation=2)
        self.output = nn.Conv1d(width, 1, kernel_size=1)
        self.activation = nn.GELU()

    def forward(
        self, coordinate_evidence: torch.Tensor, validity: torch.Tensor
    ) -> torch.Tensor:
        expected = (self.layers, self.dimensions)
        if coordinate_evidence.ndim != 4 or coordinate_evidence.shape[2:] != expected:
            raise ValueError(
                f"coordinate_evidence must be [B,T,{self.layers},{self.dimensions}]"
            )
        if validity.shape != coordinate_evidence.shape[:2]:
            raise ValueError("validity must match coordinate_evidence [B,T]")
        mask = validity.unsqueeze(1).to(coordinate_evidence.dtype)
        value = self.project(coordinate_evidence.flatten(2)).transpose(1, 2) * mask
        value = self.activation(self.first(value)) * mask
        value = self.activation(self.second(value)) * mask
        logits = self.output(value).squeeze(1)
        return logits.masked_fill(~validity, 0.0)
