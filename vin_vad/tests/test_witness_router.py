from __future__ import annotations

import torch

from vin_vad.witness_router import WitnessRouter, masked_mean, video_summary


def inputs() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    host = torch.tensor([[0.1, 0.7, 0.4, 0.0], [0.2, 0.8, 0.0, 0.0]])
    evidence = torch.tensor([[0.2, 0.9, 0.3, 0.0], [0.7, 0.4, 0.0, 0.0]])
    validity = torch.tensor([[True, True, True, False], [True, True, False, False]])
    return host, evidence, validity


def test_zero_eta_is_exact_host_identity() -> None:
    host, evidence, validity = inputs()
    result = WitnessRouter()(host, evidence, validity, 0.0, 0.0)
    assert torch.equal(result["corrected_score"][validity], host[validity])


def test_route_sign_and_local_zero_mean_are_structural() -> None:
    host, evidence, validity = inputs()
    result = WitnessRouter()(host, evidence, validity)
    assert torch.all(result["delta_normal"][validity] <= 0.0)
    torch.testing.assert_close(
        masked_mean(result["delta_anomaly"], validity),
        torch.zeros(2),
        atol=1e-7,
        rtol=0.0,
    )


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
