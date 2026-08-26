from __future__ import annotations

import torch
from torch import nn

from shift_u_dual_head_tuning.heads import (
    clone_score_head_parameters, load_score_head_state, relative_score_head_change,
    score_head_state,
)


class DummyAdapter(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.base = nn.Module()
        self.base.classifier = nn.Linear(4, 1)


def test_score_head_round_trip_and_change() -> None:
    adapter = DummyAdapter()
    initial = clone_score_head_parameters(adapter, "dsanet")
    saved = score_head_state(adapter, "dsanet")
    with torch.no_grad():
        adapter.base.classifier.weight.add_(1.0)
    assert float(relative_score_head_change(adapter, "dsanet", initial)) > 0.0
    load_score_head_state(adapter, "dsanet", saved)
    torch.testing.assert_close(relative_score_head_change(adapter, "dsanet", initial), torch.zeros(()))
