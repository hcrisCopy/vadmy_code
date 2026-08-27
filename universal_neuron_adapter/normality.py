from __future__ import annotations

from pathlib import Path

import numpy as np


def layer_normalize(hidden: np.ndarray) -> np.ndarray:
    hidden = np.asarray(hidden, dtype=np.float32)
    mean = hidden.mean(axis=-1, keepdims=True)
    scale = hidden.std(axis=-1, keepdims=True)
    return (hidden - mean) / np.maximum(scale, 1e-6)


def load_normality_model(path: str | Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as model:
        return {name: np.asarray(model[name]) for name in model.files}


def normality_evidence(hidden: np.ndarray, model: dict[str, np.ndarray]) -> np.ndarray:
    normalized = layer_normalize(hidden)
    z_score = np.abs((normalized - model["normal_mean"]) / model["normal_scale"])
    selected = np.take_along_axis(z_score, model["indices"][None], axis=2)
    weights = model["weights"][None]
    evidence = (selected * weights).sum(axis=(1, 2)) / np.maximum(weights.sum(), 1e-6)
    return evidence.astype(np.float32)
