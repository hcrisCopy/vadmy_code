from __future__ import annotations

import shutil
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import expit
from sklearn.metrics import average_precision_score, roc_auc_score

from universal_neuron_adapter.data import resample_curve
from universal_neuron_adapter.evaluate import spectral_consensus_weights, standardize


DATASETS = ("ucf", "xd")
DATASET_NAMES = {"ucf": "UCF-Crime", "xd": "XD-Violence"}
BASELINES = ("lagovad", "desc", "dsanet")


def require_relative(path: Path, argument: str) -> None:
    if path.is_absolute():
        raise ValueError(f"{argument} must be a relative path: {path}")


def prepare_output(output: Path, clean: bool, sentinel: str) -> bool:
    require_relative(output, "--out-dir")
    if clean and output.exists():
        shutil.rmtree(output)
    if (output / sentinel).exists():
        print(f"[resume] reuse {output / sentinel}", flush=True)
        return False
    output.mkdir(parents=True, exist_ok=True)
    return True


def merge_expert_manifests(
    primary_path: Path, context_path: Path, normality_path: Path
) -> pd.DataFrame:
    primary = pd.read_csv(primary_path)[["key", "expert_score_path"]]
    context = pd.read_csv(context_path)[["key", "student_score_path"]]
    normality = pd.read_csv(normality_path)[["key", "expert3_score_path"]]
    return primary.merge(context, on="key", validate="one_to_one").merge(
        normality, on="key", validate="one_to_one"
    )


def expert_curves(row: object) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    primary = np.load(str(row.expert_score_path)).astype(np.float32)
    context = resample_curve(np.load(str(row.student_score_path)), len(primary))
    normality = resample_curve(np.load(str(row.expert3_score_path)), len(primary))
    return standardize(primary), standardize(context), standardize(normality)


def frame_metric(dataset: str, truth: np.ndarray, score: np.ndarray) -> float:
    if dataset == "ucf":
        return float(roc_auc_score(truth, score))
    return float(average_precision_score(truth, score))


def sigmoid_gate(
    curves: tuple[np.ndarray, np.ndarray, np.ndarray], weights: np.ndarray
) -> np.ndarray:
    return expit(sum(float(weight) * curve for weight, curve in zip(weights, curves)))
