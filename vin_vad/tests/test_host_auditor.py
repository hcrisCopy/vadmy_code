from __future__ import annotations

import torch

from vin_vad.host_auditor import TwoAxisHostAuditor, masked_mean


def inputs() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    host = torch.tensor(
        [[0.10, 0.75, 0.30, 0.0], [0.25, 0.60, 0.0, 0.0]]
    )
    evidence = torch.tensor(
        [[-1.0, 2.0, 0.5, 99.0], [1.5, -0.5, -99.0, 100.0]]
    )
    mask = torch.tensor(
        [[True, True, True, False], [True, True, False, False]]
    )
    return host, evidence, mask


def build_auditor() -> TwoAxisHostAuditor:
    return TwoAxisHostAuditor(
        alpha_cross=0.4,
        alpha_within=0.3,
        normal_q_median=0.0,
        normal_q_mad=1.0,
        tau_normal=0.0,
    )


def test_zero_kappas_are_exact_identity_with_nonzero_gradients() -> None:
    host, evidence, mask = inputs()
    auditor = build_auditor()
    result = auditor(host, evidence, mask)
    assert torch.equal(result["corrected_score"][mask], host[mask])
    weights = torch.tensor([[1.0, -2.0, 3.0, 0.0], [2.0, -1.0, 0.0, 0.0]])
    loss = (result["corrected_score"] * weights).sum()
    loss.backward()
    assert auditor.kappa_cross.grad is not None
    assert auditor.kappa_within.grad is not None
    assert float(auditor.kappa_cross.grad.abs()) > 0.0
    assert float(auditor.kappa_within.grad.abs()) > 0.0


def test_cross_is_constant_nonpositive_and_bounded() -> None:
    host, evidence, mask = inputs()
    auditor = build_auditor()
    auditor.kappa_cross.data.fill_(0.8)
    result = auditor(host, evidence, mask, enable_within=False)
    assert torch.all(result["delta_cross"][mask] <= 0.0)
    for row, valid in zip(result["delta_cross"], mask):
        assert torch.unique(row[valid]).numel() == 1
    assert float(result["delta_cross"].abs().max()) <= auditor.alpha_cross


def test_within_is_zero_mean_and_bounded() -> None:
    host, evidence, mask = inputs()
    auditor = build_auditor()
    auditor.kappa_within.data.fill_(0.7)
    result = auditor(host, evidence, mask, enable_cross=False)
    torch.testing.assert_close(
        masked_mean(result["delta_within"], mask), torch.zeros(2), atol=1e-7, rtol=0.0
    )
    assert float(result["delta_within"].abs().max()) <= 2 * auditor.alpha_within


def test_branches_are_numerically_independent() -> None:
    host, evidence, mask = inputs()
    auditor = build_auditor()
    auditor.kappa_cross.data.fill_(0.8)
    auditor.kappa_within.data.fill_(0.7)
    full = auditor(host, evidence, mask)
    cross_only = auditor(host, evidence, mask, enable_within=False)
    within_only = auditor(host, evidence, mask, enable_cross=False)
    torch.testing.assert_close(full["delta_cross"], cross_only["delta_cross"])
    torch.testing.assert_close(full["delta_within"], within_only["delta_within"])


def test_padding_does_not_enter_pooling_centering_or_budget() -> None:
    host, evidence, mask = inputs()
    auditor = build_auditor()
    auditor.kappa_cross.data.fill_(0.8)
    auditor.kappa_within.data.fill_(0.7)
    first = auditor(host, evidence, mask)
    changed_host = host.clone()
    changed_evidence = evidence.clone()
    changed_host[~mask] = 1.0
    changed_evidence[~mask] = 1e6
    second = auditor(changed_host, changed_evidence, mask)
    for key in (
        "host_video",
        "evidence_video",
        "delta_cross",
        "delta_within",
        "corrected_score",
        "correction_size",
    ):
        torch.testing.assert_close(first[key], second[key])


def test_projected_update_keeps_kappas_in_unit_interval() -> None:
    auditor = build_auditor()
    auditor.kappa_cross.data.fill_(-2.0)
    auditor.kappa_within.data.fill_(3.0)
    auditor.project_parameters()
    assert float(auditor.kappa_cross) == 0.0
    assert float(auditor.kappa_within) == 1.0
