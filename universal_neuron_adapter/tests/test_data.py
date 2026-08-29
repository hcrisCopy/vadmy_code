from __future__ import annotations

import numpy as np
import pytest

from universal_neuron_adapter.data import base_key, is_normal_label, resample_curve


@pytest.mark.parametrize(
    ("path", "expected"),
    [
        ("video__0.npy", "video"),
        ("folder/video__12.mp4", "video"),
        ("video_name.avi", "video_name"),
    ],
)
def test_base_key_groups_real_double_underscore_chunks(path: str, expected: str) -> None:
    assert base_key(path) == expected


def test_dataset_specific_normal_labels() -> None:
    assert is_normal_label("ucf", "Normal_Videos")
    assert is_normal_label("xd", "A")
    assert not is_normal_label("ucf", "Abuse")


def test_resample_curve_preserves_endpoints() -> None:
    result = resample_curve(np.asarray([0.0, 1.0], dtype=np.float32), 5)
    np.testing.assert_allclose(result, [0.0, 0.25, 0.5, 0.75, 1.0])


def test_resample_curve_rejects_empty_input() -> None:
    with pytest.raises(ValueError, match="empty"):
        resample_curve(np.asarray([], dtype=np.float32), 3)
