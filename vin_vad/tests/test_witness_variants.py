from __future__ import annotations

import torch

from vin_vad.witness_losses import variant_objective
from vin_vad.witness_model import HostVideoOnlyVAD, NeuronOnlyWitnessVAD
from vin_vad.witness_router import masked_mean


def inputs() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    torch.manual_seed(9)
    hidden = torch.randn(2, 6, 12, 768)
    host = torch.tensor(
        [[0.7, 0.6, 0.5, 0.4, 0.0, 0.0], [0.1, 0.2, 0.4, 0.8, 0.7, 0.6]]
    )
    validity = torch.tensor([[True] * 4 + [False] * 2, [True] * 6])
    labels = torch.tensor([0.0, 1.0])
    return hidden, host, validity, labels


def test_w1_has_no_neuron_expert_and_only_nonpositive_uniform_shift() -> None:
    _, host, validity, _ = inputs()
    model = HostVideoOnlyVAD()
    assert not hasattr(model, "expert")
    result = model(host, validity)
    assert torch.all(result["delta_normal"][validity] <= 0.0)
    for row, mask in zip(result["delta_normal"], validity):
        assert torch.unique(row[mask]).numel() == 1
    assert torch.count_nonzero(result["delta_anomaly"]) == 0


def test_w2_neuron_evidence_is_host_independent_and_correction_is_local() -> None:
    hidden, host, validity, _ = inputs()
    model = NeuronOnlyWitnessVAD()
    first = model(hidden, host, validity)
    changed_host = host.clone()
    changed_host[validity] = 1.0 - changed_host[validity]
    second = model(hidden, changed_host, validity)
    torch.testing.assert_close(first["evidence"], second["evidence"])
    assert torch.any(masked_mean(first["delta_anomaly"], validity).abs() > 1e-5)
    assert torch.count_nonzero(first["delta_normal"]) == 0


def test_variant_losses_update_only_present_paths() -> None:
    hidden, host, validity, labels = inputs()
    w1 = HostVideoOnlyVAD()
    result_w1 = w1(host, validity)
    loss_w1 = variant_objective(
        "w1", result_w1, host, validity, labels, sparsity=None
    )
    loss_w1["total"].backward()
    assert any(parameter.grad is not None for parameter in w1.parameters())
    assert float(loss_w1["witness_mil"]) == 0.0

    w2 = NeuronOnlyWitnessVAD()
    result_w2 = w2(hidden, host, validity)
    loss_w2 = variant_objective(
        "w2",
        result_w2,
        host,
        validity,
        labels,
        sparsity=w2.expert.neurons.sparsity_surrogate(),
    )
    loss_w2["total"].backward()
    assert any(parameter.grad is not None for parameter in w2.expert.parameters())
    assert float(loss_w2["video"]) == 0.0
