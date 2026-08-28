from __future__ import annotations

from pathlib import Path

import numpy as np

from universal_neuron_adapter.normality import layer_normalize


def selected_directional_responses(
    hidden: np.ndarray,
    normality_model: dict[str, np.ndarray],
) -> np.ndarray:
    normalized = layer_normalize(hidden)
    z_score = (normalized - normality_model["normal_mean"]) / normality_model["normal_scale"]
    selected = np.take_along_axis(z_score, normality_model["indices"][None], axis=2)
    oriented = np.where(normality_model["directions"][None] == 0, selected, -selected)
    return np.maximum(oriented, 0.0).astype(np.float32)


def load_covariance_model(path: str | Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as model:
        return {name: np.asarray(model[name]) for name in model.files}


def covariance_evidence(
    hidden: np.ndarray,
    normality_model: dict[str, np.ndarray],
    covariance_model: dict[str, np.ndarray],
) -> np.ndarray:
    """Return layer-averaged Mahalanobis co-activation energy."""
    responses = selected_directional_responses(hidden, normality_model)
    centered = responses - covariance_model["location"][None]
    energy = np.einsum(
        "tld,ldk,tlk->tl", centered, covariance_model["precision"], centered, optimize=True
    )
    return np.sqrt(np.maximum(energy, 0.0) / responses.shape[-1]).mean(axis=1).astype(np.float32)
