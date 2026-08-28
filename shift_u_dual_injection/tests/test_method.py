from __future__ import annotations

import torch

from shift_u_dual_injection.method import UDualInjector


def test_both_zero_initialised_branches_are_exact_noops() -> None:
    module = UDualInjector(neuron_width=8, feature_width=4, hidden_width=16, trunk_depth=2)
    features = torch.randn(2, 5, 4)
    neurons = torch.randn(2, 5, 8)
    lengths = torch.tensor([5, 3])
    early, early_record = module.inject("early", features, neurons, lengths)
    late, late_record = module.inject("late", features, neurons, lengths)
    torch.testing.assert_close(early, features, rtol=0.0, atol=0.0)
    torch.testing.assert_close(late, features, rtol=0.0, atol=0.0)
    torch.testing.assert_close(early_record["applied_delta"], torch.zeros_like(features), rtol=0.0, atol=0.0)
    torch.testing.assert_close(late_record["applied_delta"], torch.zeros_like(features), rtol=0.0, atol=0.0)


def test_early_and_late_heads_share_one_trunk_but_not_outputs() -> None:
    module = UDualInjector(neuron_width=8, feature_width=4, hidden_width=16, trunk_depth=2)
    assert module.early_projection is not module.late_projection
    assert sum(parameter.numel() for parameter in module.shared_trunk.parameters()) > 0
