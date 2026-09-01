from __future__ import annotations

import numpy as np
import pandas as pd

from vin_vad.b0_identity import align_host_manifest, evaluate_identity, has_issues
from vin_vad.evaluate import auc_decomposition, score_curve_metrics


def test_auc_decomposition_reconstructs_pooled_auc() -> None:
    scores = [
        np.asarray([0.1, 0.8, 0.7, 0.2], dtype=np.float32),
        np.asarray([0.3, 0.4], dtype=np.float32),
    ]
    labels = [
        np.asarray([0, 1, 1, 0], dtype=np.int8),
        np.asarray([0, 0], dtype=np.int8),
    ]
    metrics = auc_decomposition(scores, labels)
    assert metrics["within_auc"] == 1.0
    assert metrics["macro_within_auc"] == 1.0
    assert metrics["same_video_pair_share"] == 0.5
    assert metrics["decomposition_abs_error"] < 1e-12


def test_video_constant_control_removes_within_video_ordering() -> None:
    scores = [
        np.asarray([0.1, 0.9], dtype=np.float32),
        np.asarray([0.2, 0.3], dtype=np.float32),
    ]
    labels = [
        np.asarray([0, 1], dtype=np.int8),
        np.asarray([0, 0], dtype=np.int8),
    ]
    metrics = score_curve_metrics(scores, labels, target_tpr=1.0)
    assert metrics["within_auc"] == 1.0
    assert metrics["video_constant_auc"] == 5.0 / 6.0
    assert metrics["normal_fpr"]["achieved_tpr"] == 1.0


def test_host_alignment_rejects_length_mismatch_without_resampling(tmp_path) -> None:
    score_path = tmp_path / "score.npy"
    np.save(score_path, np.asarray([0.1, 0.2], dtype=np.float32))
    audited_path = tmp_path / "audited.csv"
    host_path = tmp_path / "host.csv"
    pd.DataFrame(
        [
            {
                "key": "video",
                "binary_label": 1,
                "valid_snippets": 3,
                "evaluation_frames": 48,
                "hidden_path": "hidden.npz",
            }
        ]
    ).to_csv(audited_path, index=False)
    pd.DataFrame(
        [
            {
                "key": "video",
                "binary_label": 1,
                "baseline_score_path": str(score_path),
                "snippets": 2,
            }
        ]
    ).to_csv(host_path, index=False)
    aligned, issues = align_host_manifest(
        str(audited_path), str(host_path), "test"
    )
    assert aligned.empty
    assert has_issues(issues)
    assert issues["length_mismatches"][0]["key"] == "video"


def test_identity_evaluation_saves_exact_host_curves(tmp_path) -> None:
    rows = []
    for key, scores, label in (
        ("anomaly", [0.1, 0.9], 1),
        ("normal", [0.2, 0.3], 0),
    ):
        hidden_path = tmp_path / f"{key}.npz"
        score_path = tmp_path / f"{key}.npy"
        np.savez_compressed(
            hidden_path,
            hidden=np.zeros((2, 12, 768), dtype=np.float16),
            frame_indices=np.asarray([0, 1], dtype=np.int64),
        )
        np.save(score_path, np.asarray(scores, dtype=np.float32))
        rows.append(
            {
                "key": key,
                "binary_label": label,
                "valid_snippets": 2,
                "evaluation_frames": 2,
                "hidden_path": str(hidden_path),
                "host_score_path": str(score_path),
            }
        )
    gt_path = tmp_path / "gt.npy"
    np.save(gt_path, np.asarray([0, 1, 0, 0], dtype=np.int8))
    metrics, per_video = evaluate_identity(
        pd.DataFrame(rows), str(gt_path), tmp_path / "evaluation", target_tpr=1.0
    )
    assert metrics["status"] == "pass"
    assert metrics["host_identity_max_abs_error"] == 0.0
    assert len(per_video) == 2
    with np.load(tmp_path / "evaluation" / "curves" / "anomaly.npz") as archive:
        np.testing.assert_array_equal(archive["host_score"], archive["corrected_score"])
