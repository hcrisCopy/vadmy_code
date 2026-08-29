from __future__ import annotations

import numpy as np

from universal_neuron_adapter.evaluate import (
    FusionSettings,
    spectral_consensus_weights,
    standardize,
)


def test_duration_settings_use_training_persistence_endpoints() -> None:
    short = FusionSettings.from_persistence(11)
    long = FusionSettings.from_persistence(15)
    assert short.duration_factor == 0.0
    assert long.duration_factor == 1.0
    assert short.final_dilation_width == 1
    assert long.final_dilation_width == 25
    assert short.correction_weight == 0.0
    assert long.correction_weight == 3.0


def test_duration_settings_clip_outside_observed_range() -> None:
    assert FusionSettings.from_persistence(7).duration_factor == 0.0
    assert FusionSettings.from_persistence(21).duration_factor == 1.0


def test_spectral_consensus_is_finite_and_mean_one() -> None:
    curve = np.linspace(-1.0, 1.0, 9, dtype=np.float32)
    weights = spectral_consensus_weights(curve, curve, np.ones_like(curve))
    assert np.isfinite(weights).all()
    np.testing.assert_allclose(weights.mean(), 1.0, atol=1e-6)


def test_standardize_handles_constant_curve() -> None:
    result = standardize(np.ones(8, dtype=np.float32))
    np.testing.assert_array_equal(result, np.zeros(8, dtype=np.float32))
