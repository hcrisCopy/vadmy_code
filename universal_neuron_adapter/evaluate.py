from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.ndimage import maximum_filter1d
from sklearn.metrics import average_precision_score, roc_auc_score
from tqdm import tqdm

from universal_neuron_adapter.data import resample_curve


def empirical_cdf(reference: np.ndarray, values: np.ndarray) -> np.ndarray:
    return np.searchsorted(reference, values, side="right").astype(np.float32) / float(len(reference) + 1)


def logit(values: np.ndarray) -> np.ndarray:
    clipped = np.clip(values, 1e-5, 1.0 - 1e-5)
    return np.log(clipped / (1.0 - clipped))


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate single-baseline self-calibrated CLS-neuron fusion.")
    parser.add_argument("--baseline-train-manifest", required=True)
    parser.add_argument("--baseline-manifest", required=True)
    parser.add_argument("--expert-manifest", required=True)
    parser.add_argument("--gt-path", required=True)
    parser.add_argument("--baseline", choices=["lagovad", "desc", "dsanet"], required=True)
    parser.add_argument("--dataset", choices=["ucf", "xd"], required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--frames-per-snippet", type=int, default=16)
    parser.add_argument("--rank-weight", type=float, default=0.5)
    parser.add_argument("--event-width", type=int, default=25)
    parser.add_argument("--event-weight", type=float, default=0.5)
    parser.add_argument("--neuron-weight", type=float, default=0.15)
    args = parser.parse_args()

    output = Path(args.out_dir)
    output.mkdir(parents=True, exist_ok=True)
    train = pd.read_csv(args.baseline_train_manifest)
    train_reference = np.sort(
        np.concatenate([np.load(str(row.baseline_score_path)).astype(np.float32) for row in train.itertuples(index=False)])
    )
    baseline = pd.read_csv(args.baseline_manifest)
    expert = pd.read_csv(args.expert_manifest)[["key", "expert_score_path"]]
    frame = baseline.merge(expert, on="key", validate="one_to_one")
    baseline_curves, corrected_curves, rows = [], [], []
    for row in tqdm(frame.itertuples(index=False), total=len(frame), desc=f"evaluate {args.baseline}/{args.dataset}"):
        base = np.load(str(row.baseline_score_path)).astype(np.float32)
        rank = empirical_cdf(train_reference, base)
        calibrated = (1.0 - args.rank_weight) * base + args.rank_weight * rank
        expanded = maximum_filter1d(calibrated, args.event_width, mode="nearest")
        event_score = (1.0 - args.event_weight) * calibrated + args.event_weight * expanded
        neuron = resample_curve(np.load(str(row.expert_score_path)), len(base))
        neuron = (neuron - neuron.mean()) / max(float(neuron.std()), 1e-6)
        corrected = 1.0 / (1.0 + np.exp(-(logit(event_score) + args.neuron_weight * neuron)))
        corrected = corrected.astype(np.float32)
        baseline_curves.append(base)
        corrected_curves.append(corrected)
        rows.append({"key": str(row.key), "snippets": len(base), "corrected_mean": float(corrected.mean())})

    truth = np.load(args.gt_path).astype(np.int64).reshape(-1)
    baseline_frames = np.repeat(np.concatenate(baseline_curves), args.frames_per_snippet)
    corrected_frames = np.repeat(np.concatenate(corrected_curves), args.frames_per_snippet)
    if len(truth) != len(baseline_frames) or len(truth) != len(corrected_frames):
        raise RuntimeError(
            f"strict frame alignment failed: gt={len(truth)} baseline={len(baseline_frames)} corrected={len(corrected_frames)}"
        )
    metrics = {
        "baseline": {
            "auc": float(roc_auc_score(truth, baseline_frames)),
            "ap": float(average_precision_score(truth, baseline_frames)),
        },
        "corrected": {
            "auc": float(roc_auc_score(truth, corrected_frames)),
            "ap": float(average_precision_score(truth, corrected_frames)),
        },
        "configuration": {
            "calibration_source": "current baseline training scores",
            "rank_weight": args.rank_weight,
            "event_width": args.event_width,
            "event_weight": args.event_weight,
            "neuron_weight": args.neuron_weight,
        },
        "frames": len(truth),
    }
    (output / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    pd.DataFrame(rows).to_csv(output / "per_video.csv", index=False)
    print(json.dumps(metrics, indent=2), flush=True)


if __name__ == "__main__":
    main()
