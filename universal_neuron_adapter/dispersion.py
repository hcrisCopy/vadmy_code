from __future__ import annotations

from pathlib import Path

import numpy as np

from universal_neuron_adapter.normality import layer_normalize


def load_dispersion_model(path: str | Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as model:
        return {name: np.asarray(model[name]) for name in model.files}


def dispersion_evidence(
    hidden: np.ndarray,
    model: dict[str, np.ndarray],
) -> np.ndarray:
    """Measure selected-neuron energy relative to the normal training state."""
    normalized = layer_normalize(hidden)
    z_score = (normalized - model["normal_mean"]) / model["normal_scale"]
    selected = np.take_along_axis(z_score, model["indices"][None], axis=2)
    energy = np.log1p(np.square(selected))
    weights = model["weights"][None]
    evidence = (energy * weights).sum(axis=(1, 2)) / np.maximum(weights.sum(), 1e-6)
    return evidence.astype(np.float32)
