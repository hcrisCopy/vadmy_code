from __future__ import annotations

import pytest

from universal_neuron_adapter.cache_baseline import select_training_views


def test_fixed_training_view_is_deterministic() -> None:
    paths = [f"video__{index}.npy" for index in range(10)]
    assert select_training_views(paths, "fixed") == ["video__5.npy"]


def test_mean_training_view_keeps_every_augmentation() -> None:
    paths = ["video__0.npy", "video__1.npy"]
    assert select_training_views(paths, "mean") == paths


def test_training_view_policy_rejects_unknown_value() -> None:
    with pytest.raises(ValueError, match="unsupported"):
        select_training_views(["video.npy"], "unknown")
