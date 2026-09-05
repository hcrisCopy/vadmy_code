from __future__ import annotations

import torch

from vin_vad.witness_losses import (
    class_balanced_hard_bce,
    temporal_smoothness,
    topk_bag_probability,
    witness_objective,
)
from vin_vad.witness_model import WitnessVAD


def sample() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    torch.manual_seed(17)
    hidden = torch.randn(2, 8, 12, 768)
    host = torch.tensor(
        [[0.75, 0.60, 0.55, 0.40, 0.30, 0.20, 0.0, 0.0],
         [0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45]]
    )
    validity = torch.tensor([[True] * 6 + [False] * 2, [True] * 8])
    labels = torch.tensor([0.0, 1.0])
    return hidden, host, validity, labels


def test_hard_bce_selects_each_class_instead_of_only_the_global_hardest() -> None:
    probability = torch.tensor([0.1, 0.4, 0.9, 0.6])
    labels = torch.tensor([0.0, 0.0, 1.0, 1.0])
    expected = -0.5 * (torch.log(torch.tensor(0.6)) + torch.log(torch.tensor(0.6)))
    torch.testing.assert_close(class_balanced_hard_bce(probability, labels), expected)


def test_padding_does_not_enter_topk_or_smoothness() -> None:
    _, host, validity, _ = sample()
    first_topk = topk_bag_probability(host, validity)
    first_smooth = temporal_smoothness(host, validity)
    changed = host.clone()
    changed[~validity] = 1e6
    torch.testing.assert_close(first_topk, topk_bag_probability(changed, validity))
    torch.testing.assert_close(first_smooth, temporal_smoothness(changed, validity))


def test_every_objective_component_reaches_witness_parameters() -> None:
    hidden, host, validity, labels = sample()
    model = WitnessVAD()
    for name in ("video", "witness_mil", "final_mil", "dense_normal", "sparse"):
        model.zero_grad(set_to_none=True)
        result = model(hidden, host, validity)
        losses = witness_objective(
            result,
            host,
            validity,
            labels,
            model.expert.neurons.sparsity_surrogate(),
        )
        losses[name].backward()
        gradients = [
            parameter.grad.abs().sum()
            for parameter in model.expert.parameters()
            if parameter.grad is not None
        ]
        assert gradients and float(torch.stack(gradients).sum()) > 0.0, name


def test_one_forward_backward_and_one_optimizer_step() -> None:
    hidden, host, validity, labels = sample()
    model = WitnessVAD()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    optimizer.zero_grad(set_to_none=True)
    result = model(hidden, host, validity)
    losses = witness_objective(
        result, host, validity, labels, model.expert.neurons.sparsity_surrogate()
    )
    losses["total"].backward()
    optimizer.step()
    assert torch.isfinite(losses["total"])


def test_witness_mil_orients_primary_and_context_roles() -> None:
    hidden, host, validity, labels = sample()
    model = WitnessVAD()
    result = model(hidden, host, validity)
    losses = witness_objective(
        result, host, validity, labels, model.expert.neurons.sparsity_surrogate()
    )
    losses["witness_mil"].backward()
    assert float(model.expert.temporal.output.weight.grad.abs().sum()) > 0.0
    assert float(model.expert.context_temporal.output.weight.grad.abs().sum()) > 0.0
