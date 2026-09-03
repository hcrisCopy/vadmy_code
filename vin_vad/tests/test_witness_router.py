from __future__ import annotations

import torch

from vin_vad.witness_model import NeuronOnlyRouter
from vin_vad.witness_router import (
    WitnessRouter,
    masked_local_max,
    masked_mean,
    masked_topk_anchor,
    role_video_summary,
    video_summary,
)


def inputs() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    host = torch.tensor([[0.1, 0.7, 0.4, 0.0], [0.2, 0.8, 0.0, 0.0]])
    evidence = torch.tensor([[0.2, 0.9, 0.3, 0.0], [0.7, 0.4, 0.0, 0.0]])
    validity = torch.tensor([[True, True, True, False], [True, True, False, False]])
    return host, evidence, validity


def test_zero_eta_is_exact_host_identity() -> None:
    host, evidence, validity = inputs()
    router = WitnessRouter()
    optimizer = torch.optim.AdamW(router.parameters(), lr=1e-3)
    optimizer.zero_grad(set_to_none=True)
    router(host, evidence, validity)["corrected_score"].sum().backward()
    optimizer.step()
    result = router(host, evidence, validity, 0.0, 0.0)
    assert torch.equal(result["corrected_score"][validity], host[validity])


def test_video_state_routes_witness_and_veto_support() -> None:
    host, evidence, validity = inputs()
    router = WitnessRouter()
    router.video_head.weight.data.zero_()
    router.video_head.bias.data.fill_(8.0)
    abnormal = router(host, evidence, validity)
    assert torch.equal(
        abnormal["anomaly_authorized"],
        torch.ones_like(abnormal["anomaly_authorized"]),
    )
    router.video_head.bias.data.fill_(-8.0)
    normal = router(host, evidence, validity)
    assert torch.equal(
        normal["anomaly_authorized"],
        torch.zeros_like(normal["anomaly_authorized"]),
    )
    assert torch.all(abnormal["delta_normal"][validity] <= 0.0)
    completion = (abnormal["witness_support"] > 0) & (abnormal["event_gap"] > 0)
    assert torch.all(abnormal["delta_anomaly"][completion] > 0.0)
    neighbor_completion = (
        (abnormal["witness_support"] == 0)
        & (abnormal["witness_event_support"] > 0)
        & (abnormal["event_gap"] > 0)
    )
    assert torch.any(neighbor_completion)
    assert torch.all(abnormal["delta_anomaly"][neighbor_completion] > 0.0)
    assert torch.all(normal["delta_anomaly"][normal["veto_support"] > 0] < 0.0)
    assert torch.any(masked_mean(abnormal["delta_anomaly"], validity).abs() > 1e-5)
    assert torch.all(abnormal["completion_gate"][validity] >= 0.0)
    assert torch.all(abnormal["completion_gate"][validity] <= 1.0)
    shifted = torch.sigmoid(
        torch.logit(host.clamp(1e-6, 1.0 - 1e-6))
        + abnormal["delta_normal"]
        + abnormal["delta_anomaly"]
    )
    completed = shifted + abnormal["completion_gate"] * (
        abnormal["completion_anchor"] - shifted
    )
    assert torch.all(completed[validity] >= shifted[validity])
    assert torch.all(completed[validity] <= abnormal["completion_anchor"][validity])


def test_event_anchor_uses_standard_weak_mil_topk() -> None:
    score = torch.tensor([[0.1, 0.9, 0.7, 0.2, 99.0]])
    validity = torch.tensor([[True, True, True, True, False]])
    anchor = masked_topk_anchor(score, validity)
    torch.testing.assert_close(anchor, torch.tensor([0.9]))


def test_local_event_completion_ignores_padding() -> None:
    score = torch.tensor([[0.1, 0.9, 0.2, 99.0]])
    validity = torch.tensor([[True, True, True, False]])
    completed = masked_local_max(score, validity, width=3)
    torch.testing.assert_close(completed, torch.tensor([[0.9, 0.9, 0.9, 0.0]]))


def test_padding_does_not_enter_video_pooling_or_router() -> None:
    host, evidence, validity = inputs()
    router = WitnessRouter()
    first_summary = video_summary(host, evidence, validity)
    first = router(host, evidence, validity)
    changed_host = host.clone()
    changed_evidence = evidence.clone()
    changed_host[~validity] = 0.99
    changed_evidence[~validity] = 0.01
    second_summary = video_summary(changed_host, changed_evidence, validity)
    second = router(changed_host, changed_evidence, validity)
    torch.testing.assert_close(first_summary, second_summary)
    for name in ("video_probability", "delta_normal", "delta_anomaly"):
        torch.testing.assert_close(first[name], second[name])


def test_role_video_summary_preserves_three_role_statistics() -> None:
    host, evidence, validity = inputs()
    roles = torch.stack([evidence, 1.0 - evidence, host], dim=-1)
    summary = role_video_summary(host, roles, validity)
    assert summary.shape == (2, 10)
    changed = roles.clone()
    changed[~validity] = 1e6
    torch.testing.assert_close(summary, role_video_summary(host, changed, validity))


def test_neuron_only_eta_override_changes_only_correction_strength() -> None:
    host, evidence, validity = inputs()
    router = NeuronOnlyRouter(eta_anomaly=0.25)
    weak = router(host, evidence, validity, eta_anomaly_override=0.25)
    strong = router(host, evidence, validity, eta_anomaly_override=0.60)
    torch.testing.assert_close(weak["local_shape"], strong["local_shape"])
    torch.testing.assert_close(
        strong["delta_anomaly"],
        weak["delta_anomaly"] * (0.60 / 0.25),
        atol=1e-6,
        rtol=1e-6,
    )
