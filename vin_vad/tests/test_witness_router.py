from __future__ import annotations

import torch

from vin_vad.witness_model import NeuronOnlyRouter
from vin_vad.witness_router import (
    WitnessRouter,
    masked_local_max,
    masked_mean,
    masked_topk_anchor,
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
    assert torch.all(
        abnormal["complementary_support"] <= abnormal["witness_support"]
    )
    assert torch.all(
        abnormal["complementary_support"] <= abnormal["host_miss_support"]
    )
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


def test_negative_role_consensus_vetoes_only_host_conflicts() -> None:
    router = WitnessRouter()
    with torch.no_grad():
        router.video_head.weight.zero_()
        router.video_head.bias.fill_(10.0)
    host = torch.tensor([[0.10, 0.90, 0.20]])
    evidence = torch.tensor([[0.90, 0.10, 0.80]])
    validity = torch.ones_like(host, dtype=torch.bool)
    consensus = torch.tensor([[1.00, 0.25, 0.00]])

    result = router(host, evidence, validity, negative_consensus=consensus)

    assert result["consensus_conflict_veto"][0, 0].item() == 0.0
    assert result["consensus_conflict_veto"][0, 1].item() > 0.0
    assert result["consensus_conflict_veto"][0, 1] <= consensus[0, 1]
    assert result["consensus_conflict_veto"][0, 2].item() == 0.0


def test_positive_consensus_only_protects_normal_route_from_suppression() -> None:
    router = WitnessRouter()
    with torch.no_grad():
        router.video_head.weight.zero_()
        router.video_head.bias.fill_(-2.0)
    host = torch.tensor([[0.10, 0.90, 0.20]])
    evidence = torch.tensor([[0.20, 0.80, 0.30]])
    validity = torch.ones_like(host, dtype=torch.bool)
    consensus = torch.tensor([[0.00, 1.00, 0.25]])

    protected = router(host, evidence, validity, positive_consensus=consensus)
    unprotected = router(host, evidence, validity)

    assert protected["delta_normal"][0, 1].item() == 0.0
    torch.testing.assert_close(
        protected["delta_normal"][0, 2],
        unprotected["delta_normal"][0, 2] * 0.75,
    )
    torch.testing.assert_close(
        protected["delta_anomaly"], unprotected["delta_anomaly"]
    )


def test_event_anchor_uses_standard_weak_mil_topk() -> None:
    score = torch.tensor([[0.1, 0.9, 0.7, 0.2, 99.0]])
    validity = torch.tensor([[True, True, True, True, False]])
    anchor = masked_topk_anchor(score, validity)
    torch.testing.assert_close(anchor, torch.tensor([0.9]))


def test_video_summary_appends_calibrated_normality_topk() -> None:
    host, evidence, validity = inputs()
    normality = torch.tensor([[0.1, 0.8, 0.3, 99.0], [0.7, 0.2, 99.0, 99.0]])
    summary = video_summary(host, evidence, validity, normality)
    assert summary.shape == (2, 11)
    torch.testing.assert_close(
        summary[:, -1], masked_topk_anchor(normality, validity)
    )


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
