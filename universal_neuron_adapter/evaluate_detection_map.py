from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd

from universal_neuron_adapter.data import resample_curve


def load_official_evaluator(path: Path):
    spec = importlib.util.spec_from_file_location("official_detection_map", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load official evaluator: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.getDetectionMAP


def redistribute_semantic_mass(semantic: np.ndarray, anomaly: np.ndarray) -> np.ndarray:
    """Preserve DSANet's abnormal-class ratios while replacing total anomaly mass."""
    anomaly = resample_curve(anomaly, len(semantic)).reshape(-1, 1)
    abnormal = np.maximum(semantic[:, 1:], 0.0)
    conditional = abnormal / np.maximum(abnormal.sum(axis=1, keepdims=True), 1e-8)
    output = np.concatenate([1.0 - anomaly, anomaly * conditional], axis=1)
    return output.astype(np.float32)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the unmodified official CLIPVAD-lineage detection-mAP protocol.")
    parser.add_argument("--dataset", choices=["ucf", "xd"], required=True)
    parser.add_argument("--baseline-manifest", required=True)
    parser.add_argument("--evaluation-manifest", required=True)
    parser.add_argument("--segment-gt", required=True)
    parser.add_argument("--label-gt", required=True)
    parser.add_argument("--baseline-root", default="baseline/DSANet", help="Baseline whose src/utils/{ucf,xd}_detectionMAP.py implements the official evaluator (DSANet and VadCLIP ship identical files).")
    parser.add_argument("--out-path", required=True)
    args = parser.parse_args()
    baseline = pd.read_csv(args.baseline_manifest)
    evaluation = pd.read_csv(args.evaluation_manifest)[["key", "curve_path"]]
    frame = baseline.merge(evaluation, on="key", validate="one_to_one")
    if "semantic_score_path" not in frame.columns:
        raise ValueError("baseline manifest must be regenerated with semantic_score_path")
    original, corrected = [], []
    for row in frame.itertuples(index=False):
        semantic = np.load(str(row.semantic_score_path)).astype(np.float32)
        curves = np.load(str(row.curve_path))
        original.append(np.repeat(semantic, 16, axis=0))
        corrected.append(np.repeat(redistribute_semantic_mass(semantic, curves["corrected"]), 16, axis=0))
    evaluator_path = Path(args.baseline_root) / "src" / "utils" / f"{args.dataset}_detectionMAP.py"
    official = load_official_evaluator(evaluator_path)
    segments = np.load(args.segment_gt, allow_pickle=True)
    labels = np.load(args.label_gt, allow_pickle=True)
    original_map, ious = official(original, segments, labels, excludeNormal=False)
    corrected_map, corrected_ious = official(corrected, segments, labels, excludeNormal=False)
    if list(ious) != list(corrected_ious):
        raise RuntimeError("official evaluator returned inconsistent IoU grids")
    report = {
        "protocol": f"official {Path(args.baseline_root).name} getDetectionMAP, excludeNormal=False",
        "semantic_mapping": "preserve abnormal-class conditional ratios; replace total abnormal mass with adapter score",
        "iou": [float(value) for value in ious],
        "baseline_map": [float(value) for value in original_map],
        "corrected_map": [float(value) for value in corrected_map],
        "baseline_mean": float(np.mean(original_map)),
        "corrected_mean": float(np.mean(corrected_map)),
    }
    target = Path(args.out_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
