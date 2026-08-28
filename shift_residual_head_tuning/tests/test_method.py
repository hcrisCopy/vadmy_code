from __future__ import annotations

import torch
from torch import nn

from shift_residual_head_tuning.method import ShiftResidualInjector, configure_score_head_only


class DummyAdapter(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.base = nn.Module()
        self.base.classifier = nn.Linear(512, 1)
        self.base.backbone = nn.Linear(512, 512)


def test_zero_initialised_residual_is_exact_noop() -> None:
    module = ShiftResidualInjector(neuron_width=8, feature_width=4, hidden_width=16, depth=3)
    features = torch.randn(2, 5, 4)
    neurons = torch.randn(2, 5, 8)
    output, record = module(features, neurons, torch.tensor([5, 3]))
    torch.testing.assert_close(output, features, rtol=0.0, atol=0.0)
    torch.testing.assert_close(record["applied_delta"], torch.zeros_like(record["applied_delta"]), rtol=0.0, atol=0.0)


def test_dsanet_scope_exposes_only_binary_classifier() -> None:
    adapter = DummyAdapter()
    parameters, names = configure_score_head_only(adapter, "dsanet")
    assert names == ["base.classifier.weight", "base.classifier.bias"]
    assert [id(parameter) for parameter in parameters] == [
        id(parameter) for parameter in adapter.base.classifier.parameters()
    ]
    assert all(parameter.requires_grad for parameter in adapter.base.classifier.parameters())
    assert not any(parameter.requires_grad for parameter in adapter.base.backbone.parameters())
