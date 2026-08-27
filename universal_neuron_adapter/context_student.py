from __future__ import annotations

from pathlib import Path

import numpy as np
from scipy.ndimage import gaussian_filter1d

from universal_neuron_adapter.data import load_hidden_array
from universal_neuron_adapter.normality import layer_normalize, load_normality_model


def context_features(hidden_path: str, normality_model: dict[str, np.ndarray]) -> np.ndarray:
    """Return cheap multi-scale features for the selected CLS-neuron coordinates."""
    hidden = layer_normalize(load_hidden_array(hidden_path))
    z_score = (hidden - normality_model["normal_mean"]) / normality_model["normal_scale"]
    selected = np.take_along_axis(z_score, normality_model["indices"][None], axis=2)
    selected = np.where(normality_model["directions"][None] == 0, selected, -selected)
    selected = selected.reshape(len(hidden), -1).astype(np.float32)
    return np.concatenate(
        [
            selected,
            gaussian_filter1d(selected, 1.5, axis=0, mode="nearest"),
            gaussian_filter1d(selected, 4.0, axis=0, mode="nearest"),
        ],
        axis=1,
    ).astype(np.float32)


def load_context_student(path: str | Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as model:
        return {name: np.asarray(model[name]) for name in model.files}


def context_student_scores(
    hidden_path: str,
    normality_model: dict[str, np.ndarray],
    student: dict[str, np.ndarray],
) -> np.ndarray:
    features = context_features(hidden_path, normality_model)
    standardized = (features - student["mean"]) / student["scale"]
    logits = standardized @ student["coef"].reshape(-1) + float(student["intercept"].reshape(-1)[0])
    return (1.0 / (1.0 + np.exp(-np.clip(logits, -30.0, 30.0)))).astype(np.float32)

