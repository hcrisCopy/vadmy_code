from __future__ import annotations

import torch
from torch import nn

from neuron_responsibility.baselines import BaselineOutput
from neuron_responsibility.desc_inference import (
    desc_official_probabilities,
    desc_primary_anomaly_probability,
)
from shift_single_frozen.method import FrozenSingleResidualInjector, freeze_entire_baseline


class DummyAdapter(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.backbone = nn.Linear(4, 4)
        self.pre_temporal_conditioner: nn.Module | None = None

    def attach_pre_temporal_conditioner(self, module: nn.Module) -> None:
        self.pre_temporal_conditioner = module


def test_only_single_residual_remains_trainable() -> None:
    adapter = DummyAdapter()
    injector = FrozenSingleResidualInjector(neuron_width=8, feature_width=4, hidden_width=16, depth=2)
    adapter.attach_pre_temporal_conditioner(injector)
    assert freeze_entire_baseline(adapter, injector) == []
    assert not any(parameter.requires_grad for parameter in adapter.backbone.parameters())
    assert all(parameter.requires_grad for parameter in injector.parameters())


def test_zero_initialization_preserves_input() -> None:
    injector = FrozenSingleResidualInjector(neuron_width=8, feature_width=4, hidden_width=16, depth=2)
    features = torch.randn(2, 3, 4)
    neurons = torch.randn(2, 3, 8)
    output, record = injector(features, neurons, torch.tensor([3, 2]))
    torch.testing.assert_close(output, features)
    torch.testing.assert_close(record["applied_delta"], torch.zeros_like(features))


class DummyDeSC:
    def __init__(self, dataset: str) -> None:
        self.dataset = dataset
        self.visual_length = 4

    @staticmethod
    def _output(clip: torch.Tensor) -> BaselineOutput:
        signal = clip[..., 0]
        semantic = torch.stack((-signal, signal), dim=-1)
        return BaselineOutput(signal, semantic, clip, raw=None)

    def forward_baseline(self, clip: torch.Tensor, lengths: torch.Tensor) -> BaselineOutput:
        return self._output(clip)

    def forward_conditioned(self, clip, neurons, lengths):
        return self._output(clip), [{"stream_name": "sensitivity"}]


def test_desc_ucf_sliding_protocol_preserves_length() -> None:
    adapter = DummyDeSC("ucf")
    clip = torch.arange(7, dtype=torch.float32).unsqueeze(-1)
    result = desc_official_probabilities(adapter, clip, torch.device("cpu"))
    assert result["binary"].shape == (7,)
    assert result["semantic"].shape == (7, 2)
    torch.testing.assert_close(result["binary"], torch.sigmoid(clip[:, 0]))


def test_desc_xd_primary_uses_semantic_normal_probability() -> None:
    adapter = DummyDeSC("xd")
    clip = torch.arange(7, dtype=torch.float32).unsqueeze(-1)
    neurons = torch.zeros(7, 3)
    result = desc_official_probabilities(adapter, clip, torch.device("cpu"), neurons)
    primary = desc_primary_anomaly_probability(result, "xd")
    assert primary.shape == (7,)
    torch.testing.assert_close(primary, 1.0 - result["semantic"][:, 0])
