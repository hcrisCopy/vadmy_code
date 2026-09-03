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


def test_exactly_32_signed_neurons_globally_and_gradients() -> None:
    hidden, validity = sample_hidden()
    module = SignedTopKWitnessNeurons(active=32)
    result = module(hidden, validity)
    assert int(module.active_counts().sum()) == 32
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


def test_normal_reference_makes_neuron_response_one_sided() -> None:
    module = SignedTopKWitnessNeurons(layers=1, dimensions=2, active=1)
    module.set_normal_reference(torch.zeros(1, 2), torch.ones(1, 2))
    module.gate_logits.data.copy_(torch.tensor([[2.0, -2.0]]))
    module.signed_weights.data.copy_(torch.tensor([[1.0, 1.0]]))
    hidden = torch.tensor([[[[-1.0, 1.0]], [[1.0, -1.0]]]])
    validity = torch.tensor([[True, True]])
    evidence = module(hidden, validity)["layer_evidence"]
    assert evidence[0, 0, 0] == 0.0
    assert evidence[0, 1, 0] > 0.0


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
