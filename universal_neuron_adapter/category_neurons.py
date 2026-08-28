from __future__ import annotations

from pathlib import Path

import numpy as np

from universal_neuron_adapter.normality import layer_normalize


def atomic_categories(label: str, binary_label: int) -> list[str]:
    if not binary_label:
        return []
    parts = [part for part in str(label).split("-") if part and part != "0"]
    return sorted(set(parts or [str(label)]))


def load_category_model(path: str | Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as model:
        return {name: np.asarray(model[name]) for name in model.files}


def category_evidence(hidden: np.ndarray, model: dict[str, np.ndarray]) -> np.ndarray:
    normalized = layer_normalize(hidden)
    z_score = (normalized - model["normal_mean"]) / model["normal_scale"]
    class_scores = []
    for indices, directions, weights in zip(
        model["indices"], model["directions"], model["weights"]
    ):
        selected = np.take_along_axis(z_score, indices[None], axis=2)
        selected = np.where(directions[None] == 0, selected, -selected)
        selected = np.maximum(selected, 0.0)
        score = (selected * weights[None]).sum(axis=(1, 2)) / np.maximum(weights.sum(), 1e-6)
        class_scores.append(score)
    return np.max(np.stack(class_scores, axis=1), axis=1).astype(np.float32)
