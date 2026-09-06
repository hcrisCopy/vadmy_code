from __future__ import annotations

import math

import torch
from torch import nn


class SignedTopKWitnessNeurons(nn.Module):
    """One sparse signed witness set in every CLIP layer.

    The hard top-k mask defines the auditable neuron set in the forward pass;
    its soft surrogate carries gradients to the gate logits.
    """

    def __init__(
        self,
        layers: int = 12,
        dimensions: int = 768,
        active: int = 32,
        contexts: int = 4,
    ) -> None:
        super().__init__()
        if not 0 < active <= layers * dimensions:
            raise ValueError("active must be in [1, layers * dimensions]")
        self.layers = int(layers)
        self.dimensions = int(dimensions)
        self.active = int(active)
        if contexts < 1:
            raise ValueError("contexts must be positive")
        self.contexts = int(contexts)
        self.normalization = nn.LayerNorm(dimensions, elementwise_affine=False)
        self.register_buffer("normal_mean", torch.zeros(layers, dimensions))
        self.register_buffer("normal_std", torch.ones(layers, dimensions))
        self.register_buffer("normal_role_mask", torch.zeros(layers, dimensions))
        self.register_buffer("normal_role_direction", torch.ones(layers, dimensions))
        self.register_buffer("normal_role_weight", torch.zeros(layers, dimensions))
        self.register_buffer("signed_score_mean", torch.tensor(0.0))
        self.register_buffer("signed_score_std", torch.tensor(1.0))
        self.register_buffer("normal_role_ready", torch.tensor(False))
        self.register_buffer(
            "normal_context_centers", torch.zeros(contexts, dimensions)
        )
        self.register_buffer(
            "normal_context_mean", torch.zeros(contexts, layers, dimensions)
        )
        self.register_buffer(
            "normal_context_std", torch.ones(contexts, layers, dimensions)
        )
        self.register_buffer("normal_context_ready", torch.tensor(False))
        self.gate_logits = nn.Parameter(torch.empty(layers, dimensions))
        self.signed_weights = nn.Parameter(torch.empty(layers, dimensions))
        self.layer_logits = nn.Parameter(torch.zeros(layers))
        nn.init.normal_(self.gate_logits, mean=0.0, std=1e-3)
        nn.init.normal_(self.signed_weights, mean=0.0, std=0.02)

    @torch.no_grad()
    def set_normal_context_reference(
        self,
        centers: torch.Tensor,
        mean: torch.Tensor,
        standard_deviation: torch.Tensor,
    ) -> None:
        if centers.shape != self.normal_context_centers.shape:
            raise ValueError("centers must have shape [contexts, dimensions]")
        if mean.shape != self.normal_context_mean.shape:
            raise ValueError("mean must have shape [contexts, layers, dimensions]")
        if standard_deviation.shape != self.normal_context_std.shape:
            raise ValueError(
                "standard_deviation must have shape [contexts, layers, dimensions]"
            )
        self.normal_context_centers.copy_(centers.to(self.normal_context_centers))
        self.normal_context_mean.copy_(mean.to(self.normal_context_mean))
        self.normal_context_std.copy_(
            standard_deviation.to(self.normal_context_std).clamp_min(1e-4)
        )
        self.normal_context_ready.fill_(True)

    def contextual_deviation(
        self, normalized: torch.Tensor, validity: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compare each video with its nearest training-normal context."""
        context_layers = max(1, self.layers // 2)
        descriptors = torch.stack(
            [
                row[mask, :context_layers]
                .mean(dim=1)
                .median(dim=0)
                .values
                for row, mask in zip(normalized, validity)
            ]
        )
        distance = (
            descriptors.unsqueeze(1) - self.normal_context_centers.unsqueeze(0)
        ).square().mean(dim=-1)
        context_index = distance.argmin(dim=1)
        mean = self.normal_context_mean[context_index]
        standard_deviation = self.normal_context_std[context_index]
        deviation = (
            normalized - mean.unsqueeze(1)
        ) / standard_deviation.unsqueeze(1)
        return deviation, context_index

    @torch.no_grad()
    def set_normal_role(
        self,
        mean: torch.Tensor,
        standard_deviation: torch.Tensor,
        mask: torch.Tensor,
        direction: torch.Tensor,
        weight: torch.Tensor,
        score_mean: torch.Tensor,
        score_std: torch.Tensor,
    ) -> None:
        expected = self.normal_mean.shape
        if any(value.shape != expected for value in (mean, standard_deviation, mask, direction, weight)):
            raise ValueError("normal-role tensors must all have shape [layers, dimensions]")
        self.normal_mean.copy_(mean.to(self.normal_mean))
        self.normal_std.copy_(standard_deviation.to(self.normal_std).clamp_min(1e-4))
        self.normal_role_mask.copy_(mask.to(self.normal_role_mask))
        self.normal_role_direction.copy_(direction.to(self.normal_role_direction))
        self.normal_role_weight.copy_(weight.to(self.normal_role_weight))
        self.signed_score_mean.copy_(score_mean.to(self.signed_score_mean))
        self.signed_score_std.copy_(
            score_std.to(self.signed_score_std).clamp_min(1e-4)
        )
        # A witness must start from a weak-label-defined functional role instead
        # of asking a random sparse gate to discover both support and direction.
        # The parameters remain trainable in the single joint optimization.
        selected = mask.to(self.gate_logits) > 0
        self.gate_logits.copy_(torch.where(selected, 4.0, -4.0))
        self.signed_weights.copy_(
            direction.to(self.signed_weights) * weight.to(self.signed_weights)
        )
        self.normal_role_ready.fill_(True)

    @torch.no_grad()
    def set_primary_role(
        self,
        mask: torch.Tensor,
        direction: torch.Tensor,
        weight: torch.Tensor,
    ) -> None:
        expected = self.gate_logits.shape
        if any(value.shape != expected for value in (mask, direction, weight)):
            raise ValueError("primary-role tensors must all have shape [layers, dimensions]")
        selected = mask.to(self.gate_logits) > 0
        self.gate_logits.copy_(torch.where(selected, 4.0, -4.0))
        self.signed_weights.copy_(
            direction.to(self.signed_weights) * weight.to(self.signed_weights)
        )

    def gates(self, neuron_keep_mask: torch.Tensor | None = None) -> torch.Tensor:
        soft = torch.sigmoid(self.gate_logits)
        indices = torch.topk(self.gate_logits, k=self.active, dim=-1).indices
        hard = torch.zeros_like(soft).scatter_(-1, indices, 1.0)
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
        if bool(self.normal_role_ready):
            if bool(self.normal_context_ready):
                deviation, context_index = self.contextual_deviation(
                    normalized, validity
                )
            else:
                deviation = (
                    normalized
                    - self.normal_mean.view(1, 1, self.layers, self.dimensions)
                ) / self.normal_std.view(1, 1, self.layers, self.dimensions)
                context_index = torch.full(
                    (hidden.shape[0],), -1, dtype=torch.long, device=hidden.device
                )
            primary_input = deviation
        else:
            deviation = None
            context_index = torch.full(
                (hidden.shape[0],), -1, dtype=torch.long, device=hidden.device
            )
            primary_input = normalized
        gate = self.gates(neuron_keep_mask)
        coordinate_weights = gate * self.signed_weights
        layer_evidence = torch.einsum(
            "btld,ld->btl", primary_input, coordinate_weights
        ) / math.sqrt(self.active)
        layer_evidence = layer_evidence.masked_fill(~validity.unsqueeze(-1), 0.0)
        layer_probability = torch.softmax(self.layer_logits, dim=0)
        temporal_input = layer_evidence * (self.layers * layer_probability.view(1, 1, -1))
        return {
            "layer_evidence": layer_evidence,
            "temporal_input": temporal_input,
            "gates": gate,
            "coordinate_weights": coordinate_weights,
            "layer_probability": layer_probability,
            "normal_context_index": context_index,
        }

    def active_counts(self) -> torch.Tensor:
        return (self.gates().detach() > 0.5).sum(dim=-1)

    def sparsity_surrogate(self) -> torch.Tensor:
        """Differentiable L0 proxy; hard support is fixed by top-k."""
        return torch.sigmoid(self.gate_logits).mean()
