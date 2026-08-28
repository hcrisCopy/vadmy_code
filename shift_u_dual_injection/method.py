from __future__ import annotations

from typing import Any

import torch
from torch import nn


class UDualInjector(nn.Module):
    """One shared neuron trunk with zero-initialised early and late branches."""

    method_name = "shift_global768_u_dual_injection_v1"

    def __init__(
        self,
        neuron_width: int = 768,
        feature_width: int = 512,
        hidden_width: int = 1024,
        trunk_depth: int = 2,
        initial_gate_logit: float = -4.0,
    ) -> None:
        super().__init__()
        if min(neuron_width, feature_width, hidden_width, trunk_depth) <= 0:
            raise ValueError("all widths and trunk_depth must be positive")
        self.neuron_width = int(neuron_width)
        self.feature_width = int(feature_width)
        self.hidden_width = int(hidden_width)
        self.trunk_depth = int(trunk_depth)
        self.initial_gate_logit = float(initial_gate_logit)
        self.neuron_norm = nn.LayerNorm(self.neuron_width)
        widths = [self.neuron_width] + [self.hidden_width] * self.trunk_depth
        layers: list[nn.Module] = []
        for input_width, output_width in zip(widths[:-1], widths[1:]):
            layers.extend([nn.Linear(input_width, output_width), nn.GELU()])
        self.shared_trunk = nn.Sequential(*layers)
        self.early_projection = nn.Linear(self.hidden_width, self.feature_width)
        self.late_projection = nn.Linear(self.hidden_width, self.feature_width)
        nn.init.zeros_(self.early_projection.weight)
        nn.init.zeros_(self.early_projection.bias)
        nn.init.zeros_(self.late_projection.weight)
        nn.init.zeros_(self.late_projection.bias)
        self.early_gate_logit = nn.Parameter(torch.tensor(self.initial_gate_logit))
        self.late_gate_logit = nn.Parameter(torch.tensor(self.initial_gate_logit))

    def config(self) -> dict[str, Any]:
        return {
            "method": self.method_name,
            "neuron_width": self.neuron_width,
            "feature_width": self.feature_width,
            "hidden_width": self.hidden_width,
            "trunk_depth": self.trunk_depth,
            "initial_gate_logit": self.initial_gate_logit,
            "shared_neuron_trunk": True,
            "early_and_late_projection_zero_initialised": True,
        }

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> "UDualInjector":
        return cls(
            neuron_width=int(config["neuron_width"]),
            feature_width=int(config.get("feature_width", 512)),
            hidden_width=int(config.get("hidden_width", 1024)),
            trunk_depth=int(config.get("trunk_depth", 2)),
            initial_gate_logit=float(config.get("initial_gate_logit", -4.0)),
        )

    def early_gate(self) -> torch.Tensor:
        return torch.sigmoid(self.early_gate_logit)

    def late_gate(self) -> torch.Tensor:
        return torch.sigmoid(self.late_gate_logit)

    def _shared(self, neurons: torch.Tensor, lengths: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if neurons.ndim != 3 or neurons.shape[-1] != self.neuron_width:
            raise ValueError(f"expected neurons [B,T,{self.neuron_width}], got {tuple(neurons.shape)}")
        positions = torch.arange(neurons.shape[1], device=neurons.device).unsqueeze(0)
        valid = (positions < lengths.unsqueeze(1)).to(neurons.dtype).unsqueeze(-1)
        return self.shared_trunk(self.neuron_norm(neurons)) * valid, valid

    def inject(
        self,
        branch: str,
        features: torch.Tensor,
        neurons: torch.Tensor,
        lengths: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor | str]]:
        if features.ndim != 3 or features.shape[-1] != self.feature_width:
            raise ValueError(f"expected features [B,T,{self.feature_width}], got {tuple(features.shape)}")
        if features.shape[:2] != neurons.shape[:2]:
            raise ValueError("baseline and neuron features are not temporally aligned")
        shared, valid = self._shared(neurons, lengths)
        if branch == "early":
            delta, gate = self.early_projection(shared), self.early_gate()
        elif branch == "late":
            delta, gate = self.late_projection(shared), self.late_gate()
        else:
            raise ValueError(f"unknown injection branch: {branch}")
        delta = delta * valid
        applied = gate.to(features.dtype) * delta
        return features + applied, {
            "branch": branch,
            "delta": delta,
            "applied_delta": applied,
            "gate": gate,
            "valid_mask": valid.squeeze(-1),
        }


class InjectionView(nn.Module):
    """Expose one branch through the common baseline hook contract."""

    def __init__(self, core: UDualInjector, branch: str) -> None:
        super().__init__()
        if branch not in {"early", "late"}:
            raise ValueError("branch must be early or late")
        self.core = core
        self.branch = branch

    def forward(self, features, neurons, lengths):
        return self.core.inject(self.branch, features, neurons, lengths)


def attach_dual_injector(adapter: nn.Module, injector: UDualInjector) -> None:
    adapter.attach_pre_temporal_conditioner(InjectionView(injector, "early"))
    adapter.attach_feature_modulator(InjectionView(injector, "late"))


def forward_dual(adapter: nn.Module, clip: torch.Tensor, neurons: torch.Tensor, lengths: torch.Tensor):
    """Run both registered hooks and return records from both U branches."""
    adapter._current_neurons = neurons
    adapter._current_lengths = lengths
    adapter._conditioning_records = []
    adapter._modulation_records = []
    try:
        output = adapter.forward_baseline(clip, lengths)
        early = list(adapter._conditioning_records)
        late = list(adapter._modulation_records)
    finally:
        adapter._current_neurons = None
        adapter._current_lengths = None
        adapter._conditioning_records = []
        adapter._modulation_records = []
    if not early or not late:
        raise RuntimeError("baseline forward did not reach both U-shaped injection points")
    return output, {"early": early, "late": late}


def freeze_entire_baseline(adapter: nn.Module, injector: UDualInjector) -> list[str]:
    """Freeze every baseline tensor while leaving only the shared injector trainable."""
    adapter.requires_grad_(False)
    injector.requires_grad_(True)
    baseline_trainable = [
        name for name, parameter in adapter.named_parameters()
        if parameter.requires_grad
        and not name.startswith("pre_temporal_conditioner.")
        and not name.startswith("feature_modulator.")
    ]
    if baseline_trainable:
        raise RuntimeError(f"frozen-baseline audit failed: {baseline_trainable}")
    return baseline_trainable


def frozen_baseline_train_mode(adapter: nn.Module, injector: UDualInjector) -> None:
    adapter.eval()
    injector.train()
