from __future__ import annotations

from typing import Any

import torch
from torch import nn


class ShiftResidualInjector(nn.Module):
    """Map the selected 768D neuron feature to a gated 512D CLIP residual.

    This mirrors the established Shift-Global768 injection: LayerNorm, a deep
    MLP, zero-initialised final projection, and a small learned sigmoid gate.
    The module is installed immediately before each baseline temporal encoder.
    """

    method_name = "shift_global768_residual_score_head_v1"

    def __init__(
        self,
        neuron_width: int = 768,
        feature_width: int = 512,
        hidden_width: int = 1024,
        depth: int = 3,
        initial_gate_logit: float = -4.0,
    ) -> None:
        super().__init__()
        if min(neuron_width, feature_width, hidden_width, depth) <= 0:
            raise ValueError("all residual widths and depth must be positive")
        self.neuron_width = int(neuron_width)
        self.feature_width = int(feature_width)
        self.hidden_width = int(hidden_width)
        self.depth = int(depth)
        self.initial_gate_logit = float(initial_gate_logit)
        self.neuron_norm = nn.LayerNorm(self.neuron_width)
        widths = [self.neuron_width] + [self.hidden_width] * (self.depth - 1) + [self.feature_width]
        layers: list[nn.Module] = []
        for index, (input_width, output_width) in enumerate(zip(widths[:-1], widths[1:])):
            layers.append(nn.Linear(input_width, output_width))
            if index < len(widths) - 2:
                layers.append(nn.GELU())
        self.projection = nn.Sequential(*layers)
        final = next(module for module in reversed(self.projection) if isinstance(module, nn.Linear))
        nn.init.zeros_(final.weight)
        nn.init.zeros_(final.bias)
        self.gate_logit = nn.Parameter(torch.tensor(self.initial_gate_logit))

    def config(self) -> dict[str, Any]:
        return {
            "method": self.method_name,
            "neuron_width": self.neuron_width,
            "feature_width": self.feature_width,
            "hidden_width": self.hidden_width,
            "depth": self.depth,
            "initial_gate_logit": self.initial_gate_logit,
        }

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> "ShiftResidualInjector":
        return cls(
            neuron_width=int(config["neuron_width"]),
            feature_width=int(config.get("feature_width", 512)),
            hidden_width=int(config.get("hidden_width", 1024)),
            depth=int(config.get("depth", 3)),
            initial_gate_logit=float(config.get("initial_gate_logit", -4.0)),
        )

    def gate(self) -> torch.Tensor:
        return torch.sigmoid(self.gate_logit)

    def forward(
        self,
        features: torch.Tensor,
        neurons: torch.Tensor,
        lengths: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        if features.ndim != 3 or features.shape[-1] != self.feature_width:
            raise ValueError(f"expected baseline features [B,T,{self.feature_width}], got {tuple(features.shape)}")
        if neurons.ndim != 3 or neurons.shape[-1] != self.neuron_width:
            raise ValueError(f"expected neuron features [B,T,{self.neuron_width}], got {tuple(neurons.shape)}")
        if features.shape[:2] != neurons.shape[:2]:
            raise ValueError("CLIP and neuron features are not temporally aligned")
        positions = torch.arange(features.shape[1], device=features.device).unsqueeze(0)
        valid = (positions < lengths.unsqueeze(1)).to(features.dtype).unsqueeze(-1)
        delta = self.projection(self.neuron_norm(neurons)) * valid
        applied = self.gate().to(features.dtype) * delta
        return features + applied, {
            "delta": delta,
            "applied_delta": applied,
            "gate": self.gate(),
            "valid_mask": valid.squeeze(-1),
        }


def configure_score_head_only(adapter: nn.Module, baseline: str) -> tuple[list[nn.Parameter], list[str]]:
    """Freeze everything except the baseline's binary snippet scoring layer."""
    adapter.requires_grad_(False)
    if baseline == "dsanet":
        modules = [("base.classifier", adapter.base.classifier)]
    elif baseline == "desc":
        modules = [
            ("sensitivity.classifier", adapter.sensitivity.classifier),
            ("consistency.classifier", adapter.consistency.classifier),
        ]
    elif baseline == "lagovad":
        modules = [("base.bin_head", adapter.base.bin_head)]
    else:
        raise ValueError(f"unknown baseline: {baseline}")
    parameters: list[nn.Parameter] = []
    names: list[str] = []
    for prefix, module in modules:
        module.requires_grad_(True)
        parameters.extend(list(module.parameters()))
        names.extend(f"{prefix}.{name}" for name, _ in module.named_parameters())
    if not parameters:
        raise RuntimeError("binary score-head parameter set is empty")
    return parameters, names


def score_head_train_mode(adapter: nn.Module, injector: nn.Module, baseline: str) -> None:
    """Keep the frozen backbone in eval mode and train only the allowed modules."""
    adapter.eval()
    injector.train()
    if baseline == "dsanet":
        adapter.base.classifier.train()
    elif baseline == "desc":
        adapter.sensitivity.classifier.train()
        adapter.consistency.classifier.train()
    elif baseline == "lagovad":
        adapter.base.bin_head.train()
