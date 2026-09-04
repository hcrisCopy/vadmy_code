from __future__ import annotations

import torch

from vin_vad.witness_model import WitnessExpert
from vin_vad.witness_neurons import SignedTopKWitnessNeurons
from vin_vad.witness_temporal import WitnessTemporalReadout


def sample_hidden() -> tuple[torch.Tensor, torch.Tensor]:
    torch.manual_seed(4)
    hidden = torch.randn(2, 7, 12, 768)
    validity = torch.tensor(
        [[True, True, True, True, True, False, False], [True] * 7]
    )
    return hidden, validity


def test_exactly_32_signed_neurons_per_layer_and_gradients() -> None:
    hidden, validity = sample_hidden()
    module = SignedTopKWitnessNeurons(active=32)
    result = module(hidden, validity)
    assert torch.equal(module.active_counts(), torch.full((12,), 32))
    assert torch.any(result["coordinate_weights"] < 0)
    assert torch.any(result["coordinate_weights"] > 0)
    result["layer_evidence"].square().mean().backward()
    assert float(module.gate_logits.grad.abs().sum()) > 0.0
    assert float(module.signed_weights.grad.abs().sum()) > 0.0


def test_neuron_only_api_and_output_do_not_depend_on_host() -> None:
    hidden, validity = sample_hidden()
    expert = WitnessExpert()
    first = expert(hidden, validity)["evidence"]
    second = expert(hidden, validity)["evidence"]
    torch.testing.assert_close(first, second)
    assert "host" not in expert.forward.__annotations__


def test_tag_deletion_changes_only_requested_coordinate_support() -> None:
    hidden, validity = sample_hidden()
    module = SignedTopKWitnessNeurons(active=32)
    original = module(hidden, validity)
    selected = torch.nonzero(original["gates"].detach() > 0.5, as_tuple=False)
    layer, dimension = (int(value) for value in selected[0])
    keep = torch.ones(12, 768)
    keep[layer, dimension] = 0.0
    deleted = module(hidden, validity, keep)
    difference = original["gates"].detach() - deleted["gates"].detach()
    changed = torch.nonzero(difference != 0.0, as_tuple=False)
    assert changed.tolist() == [[layer, dimension]]
    other_layers = torch.arange(12) != layer
    torch.testing.assert_close(
        original["layer_evidence"][:, :, other_layers],
        deleted["layer_evidence"][:, :, other_layers],
    )


def test_temporal_padding_never_changes_valid_output() -> None:
    hidden, validity = sample_hidden()
    layer_evidence = torch.randn(2, 7, 12)
    module = WitnessTemporalReadout()
    first = module(layer_evidence, validity)
    changed = layer_evidence.clone()
    changed[~validity] = 1e6
    second = module(changed, validity)
    torch.testing.assert_close(first[validity], second[validity])
    assert torch.equal(second[~validity], torch.zeros_like(second[~validity]))


def test_role_jury_has_distinct_auditable_views() -> None:
    hidden, validity = sample_hidden()
    expert = WitnessExpert()
    mask = torch.zeros(12, 768)
    mask[:, :32] = 1.0
    expert.neurons.set_normal_role(
        torch.zeros(12, 768),
        torch.ones(12, 768),
        mask,
        torch.ones(12, 768),
        mask,
        torch.arange(12, dtype=torch.float32),
        torch.full((12,), 2.0),
        torch.tensor(0.4),
        torch.tensor(0.2),
    )
    primary_mask = torch.zeros(12, 768)
    primary_mask[:, 32:64] = 1.0
    expert.neurons.set_primary_role(
        primary_mask,
        -torch.ones(12, 768),
        primary_mask,
    )
    torch.testing.assert_close(
        (expert.neurons.gates().detach() > 0.5).to(primary_mask), primary_mask
    )
    torch.testing.assert_close(
        expert.neurons.signed_weights.detach(), -primary_mask
    )
    torch.testing.assert_close(expert.neurons.normal_role_mask, mask)
    result = expert(hidden, validity)
    expected_normality = (
        result["raw_normality_layer_evidence"]
        - torch.arange(12, dtype=torch.float32).view(1, 1, 12)
    ) / 2.0
    torch.testing.assert_close(
        result["normality_layer_evidence"][validity], expected_normality[validity]
    )
    for name in (
        "primary_evidence",
        "normality_evidence",
        "context_evidence",
        "positive_agreement",
        "negative_agreement",
    ):
        assert result[name].shape == validity.shape
        assert torch.equal(result[name][~validity], torch.zeros_like(result[name][~validity]))
