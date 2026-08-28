from __future__ import annotations

from pathlib import Path

import numpy as np

from universal_neuron_adapter.normality import layer_normalize


def local_temporal_contrast(normalized: np.ndarray, scale: int) -> np.ndarray:
    """Measure departure from the symmetric temporal neighbourhood at one scale."""
    if scale < 1:
        raise ValueError("temporal scale must be positive")
    padded = np.pad(normalized, ((scale, scale), (0, 0), (0, 0)), mode="edge")
    neighbourhood = 0.5 * (padded[: len(normalized)] + padded[2 * scale :])
    return np.abs(normalized - neighbourhood).astype(np.float32)


def temporal_surprise(hidden: np.ndarray, model: dict[str, np.ndarray]) -> np.ndarray:
    """Return normal-calibrated multi-scale change for every CLS neuron."""
    normalized = layer_normalize(hidden)
    surprise = np.zeros_like(normalized, dtype=np.float32)
    for position, scale in enumerate(model["scales"].astype(np.int64).tolist()):
        contrast = local_temporal_contrast(normalized, int(scale))
        calibrated = (contrast - model["normal_mean"][position]) / model["normal_scale"][position]
        np.maximum(calibrated, 0.0, out=calibrated)
        surprise = np.maximum(surprise, calibrated)
    return surprise


def temporal_information_evidence(hidden: np.ndarray, model: dict[str, np.ndarray]) -> np.ndarray:
    """Aggregate selected change-neuron responses into one snippet evidence curve."""
    surprise = temporal_surprise(hidden, model)
    selected = np.take_along_axis(surprise, model["indices"][None], axis=2)
    weights = model["weights"][None]
    evidence = (selected * weights).sum(axis=(1, 2)) / np.maximum(weights.sum(), 1e-6)
    return evidence.astype(np.float32)


def load_temporal_information_model(path: str | Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as model:
        return {name: np.asarray(model[name]) for name in model.files}

