from __future__ import annotations

import math

import torch
from torch import nn


class SignedTopKWitnessNeurons(nn.Module):
    """One globally sparse signed set over all CLIP CLS coordinates.

    The hard top-k mask defines the auditable neuron set in the forward pass;
    its soft surrogate carries gradients to the gate logits.
    """

    def __init__(self, layers: int = 12, dimensions: int = 768, active: int = 32) -> None:
        super().__init__()
        if not 0 < active <= layers * dimensions:
            raise ValueError("active must be in [1, layers * dimensions]")
        self.layers = int(layers)
        self.dimensions = int(dimensions)
        self.active = int(active)
        self.normalization = nn.LayerNorm(dimensions, elementwise_affine=False)
        self.gate_logits = nn.Parameter(torch.empty(layers, dimensions))
        self.signed_weights = nn.Parameter(torch.empty(layers, dimensions))
        self.layer_logits = nn.Parameter(torch.zeros(layers))
        nn.init.normal_(self.gate_logits, mean=0.0, std=1e-3)
        nn.init.normal_(self.signed_weights, mean=0.0, std=0.02)

    def gates(self, neuron_keep_mask: torch.Tensor | None = None) -> torch.Tensor:
        soft = torch.sigmoid(self.gate_logits)
        flat_indices = torch.topk(self.gate_logits.flatten(), k=self.active).indices
        hard = torch.zeros_like(soft).flatten().scatter_(0, flat_indices, 1.0).view_as(soft)
        straight_through = hard + soft - soft.detach()
        if neuron_keep_mask is not None:
            if neuron_keep_mask.shape != straight_through.shape:
                raise ValueError("neuron_keep_mask must have shape [layers, dimensions]")
            straight_through = straight_through * neuron_keep_mask.to(
                device=straight_through.device, dtype=straight_through.dtype
            )
        return straight_through

    def forward(
        self,
        hidden: torch.Tensor,
        validity: torch.Tensor,
        neuron_keep_mask: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        if hidden.ndim != 4 or hidden.shape[2:] != (self.layers, self.dimensions):
            raise ValueError(
                f"hidden must have shape [B,T,{self.layers},{self.dimensions}]"
            )
        if validity.shape != hidden.shape[:2] or validity.dtype != torch.bool:
            raise ValueError("validity must be a boolean [B,T] tensor")
        normalized = self.normalization(hidden)
        gate = self.gates(neuron_keep_mask)
        coordinate_weights = gate * self.signed_weights
        coordinate_evidence = (
            normalized * coordinate_weights.view(1, 1, self.layers, self.dimensions)
        ) / math.sqrt(self.active)
        coordinate_evidence = coordinate_evidence.masked_fill(
            ~validity.unsqueeze(-1).unsqueeze(-1), 0.0
        )
        layer_evidence = coordinate_evidence.sum(dim=-1)
        layer_evidence = layer_evidence.masked_fill(~validity.unsqueeze(-1), 0.0)
        layer_probability = torch.softmax(self.layer_logits, dim=0)
        temporal_input = coordinate_evidence * (
            self.layers * layer_probability.view(1, 1, -1, 1)
        )
        return {
            "layer_evidence": layer_evidence,
            "coordinate_evidence": coordinate_evidence,
            "temporal_input": temporal_input,
            "gates": gate,
            "coordinate_weights": coordinate_weights,
            "layer_probability": layer_probability,
        }

    def active_counts(self) -> torch.Tensor:
        return (self.gates().detach() > 0.5).sum(dim=-1)

    def sparsity_surrogate(self) -> torch.Tensor:
        """Differentiable L0 proxy; hard support is fixed by top-k."""
        return torch.sigmoid(self.gate_logits).mean()
