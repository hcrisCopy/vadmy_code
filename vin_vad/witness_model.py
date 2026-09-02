from __future__ import annotations

import torch
from torch import nn

from vin_vad.witness_neurons import SignedTopKWitnessNeurons
from vin_vad.witness_router import WitnessRouter
from vin_vad.witness_temporal import WitnessTemporalReadout


class WitnessExpert(nn.Module):
    """Neuron-only path: its API intentionally has no host-score argument."""

    def __init__(self, active: int = 32, temporal_width: int = 64) -> None:
        super().__init__()
        self.neurons = SignedTopKWitnessNeurons(active=active)
        self.temporal = WitnessTemporalReadout(width=temporal_width)

    def forward(
        self,
        hidden: torch.Tensor,
        validity: torch.Tensor,
        neuron_keep_mask: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        neuron = self.neurons(hidden, validity, neuron_keep_mask)
        logits = self.temporal(neuron["temporal_input"], validity)
        evidence = torch.sigmoid(logits).masked_fill(~validity, 0.0)
        return {**neuron, "evidence_logits": logits, "evidence": evidence}


class WitnessVAD(nn.Module):
    def __init__(
        self,
        active: int = 32,
        temporal_width: int = 64,
        eta_normal: float = 1.0,
        eta_anomaly: float = 0.25,
    ) -> None:
        super().__init__()
        self.expert = WitnessExpert(active=active, temporal_width=temporal_width)
        self.router = WitnessRouter(eta_normal=eta_normal, eta_anomaly=eta_anomaly)

    def forward(
        self,
        hidden: torch.Tensor,
        host_score: torch.Tensor,
        validity: torch.Tensor,
        neuron_keep_mask: torch.Tensor | None = None,
        eta_normal_override: float | None = None,
        eta_anomaly_override: float | None = None,
    ) -> dict[str, torch.Tensor]:
        expert = self.expert(hidden, validity, neuron_keep_mask)
        routed = self.router(
            host_score,
            expert["evidence"],
            validity,
            eta_normal_override=eta_normal_override,
            eta_anomaly_override=eta_anomaly_override,
        )
        return {**expert, **routed}
