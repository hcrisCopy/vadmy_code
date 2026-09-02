from __future__ import annotations

import numpy as np

from vin_vad.universal_autopsy import (
    VARIANTS,
    diagnostic_metrics,
    dominant_source,
    require_data_path,
)


def test_f0_variants_remove_exact_information_groups() -> None:
    assert VARIANTS["u1_full"] == ()
    assert set(VARIANTS["u2_no_video_suppression"]) == {
        "--disable-video-suppression"
    }
    assert set(VARIANTS["u3_no_local_neuron_correction"]) == {
        "--disable-correction",
        "--disable-agreement",
        "--disable-event-gate",
    }
    assert set(VARIANTS["u4_no_temporal_rules"]) == {"--disable-temporal"}
    assert set(VARIANTS["u0_host"]) == {
        "--disable-correction",
        "--disable-agreement",
        "--disable-event-gate",
        "--disable-video-suppression",
        "--disable-temporal",
    }


def test_diagnostics_include_abnormal_only_metrics() -> None:
    scores = [
        np.asarray([0.1, 0.9, 0.8, 0.2], dtype=np.float32),
        np.asarray([0.3, 0.4], dtype=np.float32),
    ]
    labels = [
        np.asarray([0, 1, 1, 0], dtype=np.int8),
        np.asarray([0, 0], dtype=np.int8),
    ]
    metrics = diagnostic_metrics(scores, labels, target_tpr=1.0)
    assert metrics["abnormal_videos"] == 1
    assert metrics["abnormal_only_auc"] == 1.0
    assert metrics["abnormal_only_ap"] == 1.0
    assert "cross_auc" in metrics
    assert "normal_fpr" in metrics


def test_dominant_source_requires_clear_two_to_one_lead() -> None:
    assert dominant_source({"video": 0.5, "local": 0.2, "temporal": 0.1}) == "video"
    assert (
        dominant_source({"video": 0.3, "local": 0.2, "temporal": 0.1})
        == "combination"
    )
    assert dominant_source({"video": -0.1, "local": 0.0}) == "none_positive"


def test_data_path_rejects_absolute_and_repository_paths() -> None:
    for value in ("C:/absolute/output", "vin_vad/output"):
        try:
            require_data_path(value, "test")
        except ValueError:
            pass
        else:
            raise AssertionError(f"unsafe path accepted: {value}")
