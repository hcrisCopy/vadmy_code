from __future__ import annotations

import torch

from vin_vad.context_predictor import (
    MaskedContextPredictor,
    detached_distribution,
    gaussian_nll,
)


def build_predictor(guard_radius: int = 1) -> MaskedContextPredictor:
    torch.manual_seed(42)
    return MaskedContextPredictor(
        model_width=16,
        input_rank=4,
        head_rank=4,
        attention_heads=4,
        attention_layers=2,
        guard_radius=guard_radius,
        dropout=0.0,
        sigma_min=0.1,
        sigma_max=3.0,
    ).eval()


def test_guarded_target_changes_do_not_leak_into_its_prediction() -> None:
    model = build_predictor(guard_radius=2)
    hidden = torch.randn(1, 10, 12, 768)
    mask = torch.ones(1, 10, dtype=torch.bool)
    first = model(hidden, mask)
    changed = hidden.clone()
    changed[:, 3:8] = changed[:, 3:8] * -7.0 + 13.0
    second = model(changed, mask)
    torch.testing.assert_close(first["mean"][:, 5], second["mean"][:, 5])
    torch.testing.assert_close(first["sigma"][:, 5], second["sigma"][:, 5])


def test_padding_values_do_not_change_valid_predictions() -> None:
    model = build_predictor()
    hidden = torch.randn(1, 7, 12, 768)
    mask = torch.tensor([[True, True, True, True, False, False, False]])
    first = model(hidden, mask)
    changed = hidden.clone()
    changed[:, 4:] = 1000.0 * torch.randn_like(changed[:, 4:])
    second = model(changed, mask)
    torch.testing.assert_close(first["mean"][:, :4], second["mean"][:, :4])
    torch.testing.assert_close(first["sigma"][:, :4], second["sigma"][:, :4])


def test_context_loss_updates_predictor_but_detached_bag_path_does_not() -> None:
    model = build_predictor().train()
    hidden = torch.randn(2, 6, 12, 768)
    mask = torch.ones(2, 6, dtype=torch.bool)
    result = model(hidden, mask)
    context_loss = gaussian_nll(
        result["normalized_hidden"],
        result["mean"],
        result["sigma"],
        result["prediction_mask"],
    )
    context_loss.backward()
    assert sum(
        float(parameter.grad.abs().sum())
        for parameter in model.parameters()
        if parameter.grad is not None
    ) > 0.0

    model.zero_grad(set_to_none=True)
    result = model(hidden, mask)
    mean, sigma = detached_distribution(result)
    field_weight = torch.nn.Parameter(torch.tensor(1.0))
    weak_bag_loss = field_weight * ((result["normalized_hidden"] - mean) / sigma).mean()
    weak_bag_loss.backward()
    assert field_weight.grad is not None
    assert all(parameter.grad is None for parameter in model.parameters())


def test_sigma_is_finite_and_respects_fixed_bounds() -> None:
    model = build_predictor()
    result = model(
        torch.randn(2, 5, 12, 768), torch.ones(2, 5, dtype=torch.bool)
    )
    assert torch.isfinite(result["mean"]).all()
    assert torch.isfinite(result["sigma"]).all()
    assert float(result["sigma"].min()) >= 0.1
    assert float(result["sigma"].max()) <= 3.0


def test_length_two_sequence_keeps_one_context_key() -> None:
    model = build_predictor(guard_radius=10)
    result = model(
        torch.randn(1, 2, 12, 768), torch.ones(1, 2, dtype=torch.bool)
    )
    assert result["prediction_mask"].tolist() == [[True, True]]
    assert torch.isfinite(result["mean"]).all()
