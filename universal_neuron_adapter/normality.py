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
    z_score = (normalized - model["normal_mean"]) / model["normal_scale"]
    selected = np.take_along_axis(z_score, model["indices"][None], axis=2)
    if "directions" in model:
        selected = np.where(model["directions"][None] == 0, selected, -selected)
        selected = np.maximum(selected, 0.0)
    else:
        selected = np.abs(selected)
    weighted = (selected * model["weights"][None]).reshape(len(selected), -1)
    tail_count = max(1, weighted.shape[1] // 4)
    evidence = np.partition(weighted, weighted.shape[1] - tail_count, axis=1)[:, -tail_count:].mean(axis=1)
    return evidence.astype(np.float32)
