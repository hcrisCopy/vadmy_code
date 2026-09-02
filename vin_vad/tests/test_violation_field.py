from __future__ import annotations

import torch

from vin_vad.violation_field import ViolationField, entmax15


def sample_inputs() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    torch.manual_seed(42)
    hidden = torch.randn(2, 4, 12, 768)
    mean = torch.zeros_like(hidden)
    sigma = torch.ones_like(hidden)
    mask = torch.tensor(
        [[True, True, True, False], [True, True, False, False]]
    )
    return hidden, mean, sigma, mask


def test_directions_are_mutually_exclusive() -> None:
    hidden, mean, sigma, mask = sample_inputs()
    output = ViolationField(delta=0.5, statistics_momentum=0.1)(
        hidden, mean, sigma, mask
    )
    positive = output["directional"][..., 0]
    negative = output["directional"][..., 1]
    assert not torch.any((positive > 0.0) & (negative > 0.0))


def test_entmax_is_non_negative_normalized_and_can_be_sparse() -> None:
    logits = torch.tensor([8.0, 1.0, 0.0, -3.0], requires_grad=True)
    probability = entmax15(logits)
    assert torch.all(probability >= 0.0)
    torch.testing.assert_close(probability.sum(), torch.tensor(1.0))
    assert int((probability > 0.0).sum()) < logits.numel()
    (probability * torch.arange(4.0)).sum().backward()
    assert logits.grad is not None and torch.isfinite(logits.grad).all()


def test_running_statistics_only_read_normal_valid_snippets() -> None:
    hidden, mean, sigma, mask = sample_inputs()
    field = ViolationField(delta=0.5, statistics_momentum=1.0)
    labels = torch.tensor([0.0, 1.0])
    first = field(hidden, mean, sigma, mask, labels, update_statistics=True)
    normal_values = first["activation"][0, :3]
    expected_median = normal_values.median()
    expected_mad = (normal_values - expected_median).abs().median()
    torch.testing.assert_close(field.running_median, expected_median)
    torch.testing.assert_close(field.running_mad, expected_mad)
    assert int(field.normal_snippets_seen) == 3

    changed = hidden.clone()
    changed[0, 3] = 10000.0  # padding
    changed[1] = -10000.0  # abnormal video
    field(changed, mean, sigma, mask, labels, update_statistics=True)
    torch.testing.assert_close(field.running_median, expected_median)
    torch.testing.assert_close(field.running_mad, expected_mad)
    assert int(field.normal_snippets_seen) == 6


def test_saved_terms_recompute_activation_and_evidence() -> None:
    hidden, mean, sigma, mask = sample_inputs()
    field = ViolationField(delta=0.5, statistics_momentum=1.0)
    labels = torch.zeros(2)
    output = field(hidden, mean, sigma, mask, labels, update_statistics=True)
    recomputed_activation = torch.einsum(
        "btldq,ldq->bt", output["directional"], output["probability"]
    ).masked_fill(~mask, 0.0)
    recomputed_evidence = (
        (recomputed_activation - field.running_median)
        / (1.4826 * field.running_mad + field.epsilon)
    ).masked_fill(~mask, 0.0)
    torch.testing.assert_close(output["activation"], recomputed_activation)
    torch.testing.assert_close(output["evidence"], recomputed_evidence)


def test_padding_does_not_change_valid_outputs_or_statistics() -> None:
    hidden, mean, sigma, mask = sample_inputs()
    labels = torch.zeros(2)
    first_field = ViolationField(delta=0.5, statistics_momentum=0.1)
    second_field = ViolationField(delta=0.5, statistics_momentum=0.1)
    first = first_field(hidden, mean, sigma, mask, labels, True)
    changed = hidden.clone()
    changed[~mask] = 1000.0 * torch.randn_like(changed[~mask])
    second = second_field(changed, mean, sigma, mask, labels, True)
    torch.testing.assert_close(first["evidence"][mask], second["evidence"][mask])
    torch.testing.assert_close(first_field.running_median, second_field.running_median)
    torch.testing.assert_close(first_field.running_mad, second_field.running_mad)
