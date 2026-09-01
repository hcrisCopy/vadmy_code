from __future__ import annotations

import torch

from vin_vad.base_tcn import BaseTCN
from vin_vad.data import uniform_temporal_average
from vin_vad.losses import bag_loss
from vin_vad.model import EventAblationModel


def test_tcn_valid_logits_ignore_padding_content_and_batch_padding_length() -> None:
    torch.manual_seed(7)
    model = BaseTCN(width=16, dropout=0.0).eval()
    valid = torch.randn(1, 5, 768)
    first = torch.cat([valid, torch.zeros(1, 2, 768)], dim=1)
    first_mask = torch.tensor([[True] * 5 + [False] * 2])
    second = torch.cat([valid, torch.randn(1, 4, 768) * 1000.0], dim=1)
    second_mask = torch.tensor([[True] * 5 + [False] * 4])
    torch.testing.assert_close(
        model(first, first_mask)[:, :5],
        model(second, second_mask)[:, :5],
        atol=1e-6,
        rtol=1e-6,
    )


def test_every_ablation_has_finite_loss_and_one_optimizer() -> None:
    features = torch.randn(2, 8, 768)
    mask = torch.tensor([[True] * 8, [True] * 6 + [False] * 2])
    labels = torch.tensor([0.0, 1.0])
    for variant in ("e0", "e1", "e2", "e3"):
        model = EventAblationModel(variant, width=16, dropout=0.0)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
        logits = model.tcn(features, mask)
        loss, _ = bag_loss(variant, logits, mask, labels, model.chain)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        assert torch.isfinite(loss)


def test_uniform_temporal_average_matches_dsanet_bins() -> None:
    features = torch.arange(10, dtype=torch.float32).numpy()[:, None]
    reduced = uniform_temporal_average(features, 4).reshape(-1)
    torch.testing.assert_close(
        torch.from_numpy(reduced),
        torch.tensor([0.5, 3.0, 5.5, 8.0]),
    )
