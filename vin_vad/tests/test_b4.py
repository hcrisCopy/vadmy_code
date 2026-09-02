from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pandas as pd
import torch

from vin_vad.context_predictor import MaskedContextPredictor, gaussian_nll
from vin_vad.data import AuditorTrainingDataset
from vin_vad.host_auditor import NormalQCalibrator, TwoAxisHostAuditor
from vin_vad.losses import asymmetric_mil_loss, correction_budget_loss
from vin_vad.model import CVAVADCorrectionModel
from vin_vad.train import merge_class_batches
from vin_vad.violation_field import ViolationField


def _predictor() -> MaskedContextPredictor:
    return MaskedContextPredictor(
        model_width=8,
        input_rank=2,
        head_rank=2,
        attention_heads=2,
        attention_layers=1,
        guard_radius=1,
        dropout=0.0,
        sigma_min=0.1,
        sigma_max=2.0,
    )


def test_asymmetric_mil_and_budget_match_manual_values() -> None:
    score = torch.tensor([[0.1, 0.2, 0.0], [0.6, 0.9, 0.8]])
    mask = torch.tensor([[True, True, False], [True, True, True]])
    labels = torch.tensor([0.0, 1.0])
    loss, per_video = asymmetric_mil_loss(score, mask, labels)
    expected_normal = (-torch.log1p(-torch.tensor([0.1, 0.2]))).mean()
    # DSANet uses floor(T/16)+1, so a three-snippet bag selects one score.
    expected_abnormal = -torch.log(torch.tensor(0.9))
    torch.testing.assert_close(per_video, torch.stack((expected_normal, expected_abnormal)))
    torch.testing.assert_close(loss, per_video.mean())
    penalty, size = correction_budget_loss(torch.tensor(0.14), 0.1)
    torch.testing.assert_close(size, torch.tensor(0.14))
    torch.testing.assert_close(penalty, torch.tensor(0.04))


def test_normal_q_calibrator_ignores_abnormal_values_and_round_trips_state() -> None:
    calibrator = NormalQCalibrator(capacity=4, normal_quantile=0.75)
    calibrator.update(torch.tensor([1.0, 1000.0, 3.0]), torch.tensor([0.0, 1.0, 0.0]))
    assert int(calibrator.count) == 2
    assert set(calibrator.reservoir[:2].tolist()) == {1.0, 3.0}
    clone = NormalQCalibrator(capacity=4, normal_quantile=0.75)
    clone.load_state_dict(calibrator.state_dict())
    torch.testing.assert_close(clone.reservoir, calibrator.reservoir)
    torch.testing.assert_close(clone.tau_normal, calibrator.tau_normal)


def test_training_dataset_applies_one_shared_uniform_temporal_partition(tmp_path) -> None:
    hidden = np.zeros((4, 12, 768), dtype=np.float32)
    for index in range(4):
        hidden[index].fill(index)
    hidden_path = tmp_path / "hidden.npz"
    score_path = tmp_path / "score.npy"
    np.savez_compressed(hidden_path, hidden=hidden)
    np.save(score_path, np.asarray([0.0, 1.0, 2.0, 3.0], dtype=np.float32))
    manifest_path = tmp_path / "manifest.csv"
    pd.DataFrame(
        [
            {
                "key": "video",
                "binary_label": 1,
                "hidden_path": hidden_path,
                "host_score_path": score_path,
                "valid_snippets": 4,
            }
        ]
    ).to_csv(manifest_path, index=False)
    item = AuditorTrainingDataset(str(manifest_path), maximum_length=2)[0]
    torch.testing.assert_close(item["host_score"], torch.tensor([0.5, 2.5]))
    torch.testing.assert_close(item["hidden"][:, 0, 0], torch.tensor([0.5, 2.5]))


def test_weak_loss_isolated_from_predictor_but_context_loss_updates_it() -> None:
    torch.manual_seed(4)
    model = CVAVADCorrectionModel(
        predictor=_predictor(),
        field=ViolationField(delta=0.5, statistics_momentum=0.1),
        auditor=TwoAxisHostAuditor(alpha_cross=0.5, alpha_within=0.25),
        q_calibrator=NormalQCalibrator(capacity=8, normal_quantile=0.75),
    )
    hidden = torch.randn(2, 5, 12, 768)
    host = torch.tensor(
        [[0.1, 0.2, 0.15, 0.1, 0.2], [0.1, 0.8, 0.2, 0.9, 0.3]]
    )
    mask = torch.ones(2, 5, dtype=torch.bool)
    labels = torch.tensor([0.0, 1.0])
    result = model(hidden, host, mask, labels, update_statistics=True)
    weak_loss, _ = asymmetric_mil_loss(result["corrected_score"], mask, labels)
    weak_loss.backward()
    assert all(
        parameter.grad is None or torch.count_nonzero(parameter.grad) == 0
        for parameter in model.predictor.parameters()
    )
    assert model.auditor.kappa_cross.grad is not None
    assert model.auditor.kappa_within.grad is not None

    model.zero_grad(set_to_none=True)
    result = model(hidden, host, mask, labels, update_statistics=False)
    normal_mask = result["distribution"]["prediction_mask"] & labels.eq(0).unsqueeze(1)
    context_loss = gaussian_nll(
        result["distribution"]["normalized_hidden"],
        result["distribution"]["mean"],
        result["distribution"]["sigma"],
        normal_mask,
    )
    context_loss.backward()
    assert any(
        parameter.grad is not None and torch.count_nonzero(parameter.grad) > 0
        for parameter in model.predictor.parameters()
    )
    assert model.field.omega.grad is None


def test_merge_class_batches_preserves_padding_masks() -> None:
    normal = {
        "hidden": torch.ones(1, 2, 12, 768),
        "host_score": torch.ones(1, 2),
        "mask": torch.ones(1, 2, dtype=torch.bool),
        "labels": torch.zeros(1),
    }
    abnormal = {
        "hidden": torch.ones(1, 3, 12, 768),
        "host_score": torch.ones(1, 3),
        "mask": torch.ones(1, 3, dtype=torch.bool),
        "labels": torch.ones(1),
    }
    merged = merge_class_batches(normal, abnormal)
    assert merged["hidden"].shape == (2, 3, 12, 768)
    assert merged["mask"].tolist() == [[True, True, False], [True, True, True]]
