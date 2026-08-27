from __future__ import annotations

import numpy as np


def resample_curve(curve: np.ndarray, length: int) -> np.ndarray:
    curve = np.asarray(curve, dtype=np.float32).reshape(-1)
    if not len(curve):
        raise ValueError("cannot resample an empty curve")
    if len(curve) == length:
        return curve
    return np.interp(
        np.linspace(0.0, 1.0, length, dtype=np.float32),
        np.linspace(0.0, 1.0, len(curve), dtype=np.float32),
        curve,
    ).astype(np.float32)


def resample_matrix(values: np.ndarray, length: int) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    if values.ndim != 2 or not len(values):
        raise ValueError("expected a non-empty [time, feature] matrix")
    if len(values) == length:
        return values
    positions = np.linspace(0.0, len(values) - 1, length, dtype=np.float32)
    lower = np.floor(positions).astype(np.int64)
    upper = np.minimum(lower + 1, len(values) - 1)
    alpha = (positions - lower).reshape(-1, 1)
    return (values[lower] * (1.0 - alpha) + values[upper] * alpha).astype(np.float32)


def normalize_selected_layers(values: np.ndarray, layers: int = 12) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    grouped = values.reshape(len(values), layers, -1)
    return ((grouped - grouped.mean(-1, keepdims=True)) / np.maximum(grouped.std(-1, keepdims=True), 1e-6)).reshape(values.shape).astype(np.float32)

