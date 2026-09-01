from __future__ import annotations

import numpy as np
import torch


def expand_snippet_scores(
    snippet_scores: np.ndarray,
    frame_indices: np.ndarray,
    frame_count: int,
) -> np.ndarray:
    """Expand scores according to audited frame boundaries, without smoothing."""
    scores = np.asarray(snippet_scores, dtype=np.float32).reshape(-1)
    indices = np.asarray(frame_indices, dtype=np.int64).reshape(-1)
    if len(scores) != len(indices) or not len(scores):
        raise ValueError("scores and frame_indices must have the same non-zero length")
    if indices[0] != 0 or np.any(np.diff(indices) <= 0):
        raise ValueError("frame_indices must start at zero and be strictly increasing")
    if frame_count <= int(indices[-1]):
        raise ValueError("frame_count must extend beyond the last snippet start")

    boundaries = np.concatenate([indices, np.asarray([frame_count], dtype=np.int64)])
    output = np.empty(frame_count, dtype=np.float32)
    for index, score in enumerate(scores):
        output[boundaries[index] : boundaries[index + 1]] = score
    return output


def masked_mean(values: torch.Tensor, mask: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """Mean over valid snippets only; padded values never affect the result."""
    if values.shape != mask.shape:
        raise ValueError(f"values/mask shape mismatch: {values.shape} vs {mask.shape}")
    weights = mask.to(values.dtype)
    return (values * weights).sum(dim=dim) / weights.sum(dim=dim).clamp_min(1.0)
