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

