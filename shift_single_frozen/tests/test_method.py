from __future__ import annotations

import torch
from torch import nn

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
