from __future__ import annotations

import numpy as np
import torch

from responsibility_cross_expert.build_consensus_labels import refine_curve
from responsibility_cross_expert.common import temporal_mean_process
from responsibility_cross_expert.semantic_model import ResidualBottleneck
from responsibility_cross_expert.train_binary_head import consensus_loss


def test_zero_start_adapter_preserves_input() -> None:
    adapter = ResidualBottleneck(width=8, bottleneck=2)
    value = torch.randn(2, 3, 8)
    torch.testing.assert_close(adapter(value), value)


def test_temporal_mean_process_keeps_alignment() -> None:
    value = np.arange(24, dtype=np.float32).reshape(6, 4)
    pooled, length = temporal_mean_process(value, 3)
    assert length == 3
    np.testing.assert_allclose(pooled[0], value[:2].mean(axis=0))


def test_consensus_loss_ignores_unknown_positions() -> None:
    logits = [torch.tensor([[0.0, 3.0, -3.0]], requires_grad=True)]
    target = torch.tensor([[-1.0, 1.0, 0.0]])
    lengths = torch.tensor([3])
    loss = consensus_loss(logits, target, lengths)
    assert torch.isfinite(loss)
    loss.backward()
    assert logits[0].grad[0, 0].item() == 0.0


def test_multiscale_refiner_does_not_force_empty_curve() -> None:
    empty = refine_curve(np.zeros(64, dtype=np.float32), 6, 4, 22.0, 0.55)
    assert not empty.any()
    obvious = np.zeros(64, dtype=np.float32)
    obvious[16:48] = 1.0
    refined = refine_curve(obvious, 6, 4, 22.0, 0.55)
    assert refined[24:40].min() > 0
