from __future__ import annotations

import torch

from universal_neuron_adapter.model import ScoreCorrectionHead, topk_bag, valid_mask


def test_valid_mask_respects_each_video_length() -> None:
    result = valid_mask(torch.tensor([1, 3]), steps=4, dtype=torch.float32)
    expected = torch.tensor([[1, 0, 0, 0], [1, 1, 1, 0]], dtype=torch.float32)
    assert torch.equal(result, expected)


def test_topk_bag_ignores_padding() -> None:
    probability = torch.tensor([[0.2, 0.9, 1.0], [0.1, 0.3, 0.8]])
    lengths = torch.tensor([2, 3])
    result = topk_bag(probability, lengths, divisor=16)
    assert torch.allclose(result, torch.tensor([0.9, 0.8]))


def test_identity_initialized_correction_preserves_baseline() -> None:
    model = ScoreCorrectionHead(width=4, identity_init=True)
    baseline = torch.tensor([[0.2, 0.7, 0.4]])
    expert = torch.tensor([[0.8, 0.1, 0.6]])
    corrected = torch.sigmoid(model(baseline, expert, torch.tensor([3])))
    assert torch.allclose(corrected, baseline, atol=1e-6)
