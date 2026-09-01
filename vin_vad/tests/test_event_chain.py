from __future__ import annotations

import itertools

import torch

from vin_vad.event_chain import EventChain


def enumerate_chain(chain: EventChain, emissions: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    length = len(emissions)
    length_arg = torch.tensor([length])
    transition = chain.transition_log_probs(length_arg)[0]
    onset = chain.onset_probability(length_arg)[0]
    scores = []
    states = []
    for sequence in itertools.product((0, 1), repeat=length):
        score = torch.log(onset if sequence[0] else 1.0 - onset) + sequence[0] * emissions[0]
        for time in range(1, length):
            score = score + transition[sequence[time - 1], sequence[time]] + sequence[time] * emissions[time]
        scores.append(score)
        states.append(sequence)
    stacked = torch.stack(scores)
    normal = torch.tensor([not any(sequence) for sequence in states])
    log_z0 = torch.logsumexp(stacked[normal], dim=0)
    log_z1 = torch.logsumexp(stacked[~normal], dim=0)
    total = torch.logsumexp(stacked, dim=0)
    marginals = []
    for time in range(length):
        selected = torch.tensor([bool(sequence[time]) for sequence in states])
        marginals.append(torch.exp(torch.logsumexp(stacked[selected], dim=0) - total))
    return log_z0, log_z1, torch.stack(marginals)


def test_dp_matches_enumeration_for_all_event_variants() -> None:
    emissions = torch.tensor([0.4, -0.7, 1.2, 0.1])
    mask = torch.ones(1, len(emissions), dtype=torch.bool)
    for variant in ("e1", "e2", "e3"):
        chain = EventChain(variant)
        expected_z0, expected_z1, expected_marginals = enumerate_chain(chain, emissions)
        actual_z0, actual_z1 = chain.log_partitions(emissions[None], mask)
        actual_marginals = chain.snippet_marginals(emissions[None], mask)[0]
        torch.testing.assert_close(actual_z0[0], expected_z0, atol=1e-5, rtol=1e-5)
        torch.testing.assert_close(actual_z1[0], expected_z1, atol=1e-5, rtol=1e-5)
        torch.testing.assert_close(actual_marginals, expected_marginals, atol=1e-5, rtol=1e-5)


def test_gradients_are_finite() -> None:
    emissions = torch.randn(2, 7, requires_grad=True)
    mask = torch.tensor([[True] * 7, [True] * 5 + [False] * 2])
    chain = EventChain("e3")
    log_p0, log_p1 = chain.video_log_probs(emissions, mask)
    loss = -(log_p0[0] + log_p1[1])
    loss.backward()
    assert torch.isfinite(emissions.grad).all()
    assert torch.isfinite(chain.onset_raw.grad).all()
    assert torch.isfinite(chain.persistence_raw.grad).all()


def test_padding_values_and_padding_length_do_not_change_chain_output() -> None:
    chain = EventChain("e3")
    short = torch.tensor([[0.2, -0.4, 1.0, 99.0, -99.0]])
    mask = torch.tensor([[True, True, True, False, False]])
    changed = torch.tensor([[0.2, -0.4, 1.0, -500.0, 500.0, 700.0]])
    changed_mask = torch.tensor([[True, True, True, False, False, False]])
    first = chain.video_log_probs(short, mask)
    second = chain.video_log_probs(changed, changed_mask)
    torch.testing.assert_close(first[0], second[0])
    torch.testing.assert_close(first[1], second[1])


def test_length_calibrated_prior_is_invariant_to_sequence_length() -> None:
    chain = EventChain("e3")
    probabilities = []
    for length in (16, 64, 256):
        emissions = torch.zeros(1, length)
        mask = torch.ones_like(emissions, dtype=torch.bool)
        probabilities.append(chain.video_log_probs(emissions, mask)[1].exp())
    torch.testing.assert_close(probabilities[0], probabilities[1], atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(probabilities[0], probabilities[2], atol=1e-5, rtol=1e-5)
