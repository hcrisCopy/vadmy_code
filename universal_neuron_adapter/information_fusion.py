from __future__ import annotations

from itertools import combinations

import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter1d

from universal_neuron_adapter.data import resample_curve


def standardize(curve: np.ndarray) -> np.ndarray:
    curve = np.asarray(curve, dtype=np.float32)
    return (curve - curve.mean()) / max(float(curve.std()), 1e-6)


def contextual_standardize(primary: np.ndarray, context: np.ndarray, weight: float) -> np.ndarray:
    context = resample_curve(context, len(primary))
    return standardize(standardize(primary) + weight * standardize(context))


def top_bag(curve: np.ndarray) -> float:
    count = min(len(curve), max(1, len(curve) // 16 + 1))
    return float(np.partition(curve, len(curve) - count)[-count:].mean())


def fit_information_fusion_weights(
    key_manifest: str,
    primary_manifest: str,
    complementary_manifest: str,
    normality_manifest: str,
    context_manifest: str,
    context_complementary_weight: float,
    context_normality_weight: float,
    normality_smoothing_blend: float,
    total_weight: float,
) -> np.ndarray:
    """Fit a non-negative precision-weighted direction on a fixed-sum simplex."""
    if total_weight <= 0.0:
        raise ValueError("total information-fusion weight must be positive")
    keys = pd.read_csv(key_manifest)[["key", "binary_label"]]
    primary = pd.read_csv(primary_manifest)[["key", "expert_score_path"]]
    complementary = pd.read_csv(complementary_manifest)[["key", "expert2_score_path"]]
    normality = pd.read_csv(normality_manifest)[["key", "expert3_score_path"]]
    context = pd.read_csv(context_manifest)[["key", "student_score_path"]]
    frame = keys.merge(primary, on="key", validate="one_to_one").merge(
        complementary, on="key", validate="one_to_one"
    ).merge(normality, on="key", validate="one_to_one").merge(
        context, on="key", validate="one_to_one"
    )

    summaries, labels = [], []
    for row in frame.itertuples(index=False):
        first = np.load(str(row.expert_score_path)).astype(np.float32)
        second = resample_curve(np.load(str(row.expert2_score_path)), len(first))
        third = resample_curve(np.load(str(row.expert3_score_path)), len(first))
        student = resample_curve(np.load(str(row.student_score_path)), len(first))
        first = standardize(first)
        second = contextual_standardize(second, student, context_complementary_weight)
        third = contextual_standardize(third, student, context_normality_weight)
        if normality_smoothing_blend:
            smoothed = gaussian_filter1d(third, 1.0, mode="nearest")
            third = standardize(
                (1.0 - normality_smoothing_blend) * third
                + normality_smoothing_blend * smoothed
            )
        summaries.append([top_bag(first), top_bag(second), top_bag(third)])
        labels.append(int(row.binary_label))

    values = np.asarray(summaries, dtype=np.float64)
    targets = np.asarray(labels, dtype=np.int64)
    normal, abnormal = values[targets == 0], values[targets == 1]
    if len(normal) < 2 or len(abnormal) < 2:
        raise ValueError("information fusion needs at least two videos from each class")
    mean_difference = abnormal.mean(axis=0) - normal.mean(axis=0)
    centered_normal = normal - normal.mean(axis=0)
    centered_abnormal = abnormal - abnormal.mean(axis=0)
    covariance = (
        centered_normal.T @ centered_normal + centered_abnormal.T @ centered_abnormal
    ) / max(len(normal) + len(abnormal) - 2, 1)
    ridge = max(0.1 * float(np.trace(covariance)) / len(mean_difference), 1e-3)
    precision = covariance + ridge * np.eye(len(mean_difference))
    best_weights, best_objective = None, -np.inf
    dimensions = range(len(mean_difference))
    for active_count in range(1, len(mean_difference) + 1):
        for active in combinations(dimensions, active_count):
            active = np.asarray(active, dtype=np.int64)
            matrix = precision[np.ix_(active, active)]
            inverse = np.linalg.inv(matrix)
            ones = np.ones(len(active), dtype=np.float64)
            relevance = mean_difference[active]
            lagrange = (ones @ inverse @ relevance - total_weight) / (ones @ inverse @ ones)
            active_weights = inverse @ (relevance - lagrange * ones)
            if np.any(active_weights < -1e-9):
                continue
            weights = np.zeros(len(mean_difference), dtype=np.float64)
            weights[active] = np.maximum(active_weights, 0.0)
            objective = float(weights @ mean_difference - 0.5 * weights @ precision @ weights)
            if objective > best_objective:
                best_weights, best_objective = weights, objective
    if best_weights is None or not np.isfinite(best_weights).all():
        raise ValueError("non-negative information fusion has no feasible solution")
    return best_weights.astype(np.float32)
