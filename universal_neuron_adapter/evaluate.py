from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.ndimage import gaussian_filter1d, maximum_filter1d, median_filter
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

from universal_neuron_adapter.data import resample_curve
from universal_neuron_adapter.model import ScoreCorrectionHead, calibrated_probability


def video_features(curve: np.ndarray) -> np.ndarray:
    curve = np.asarray(curve, dtype=np.float32)
    change = np.abs(np.diff(curve))
    quantiles = np.quantile(curve, np.linspace(0.1, 0.9, 9))
    top_means = []
    for fraction in (0.01, 0.05, 0.1, 0.2):
        count = max(1, int(np.ceil(fraction * len(curve))))
        top_means.append(float(np.partition(curve, len(curve) - count)[-count:].mean()))
    return np.asarray([
        np.log1p(len(curve)), float(curve.mean()), float(curve.std()),
        float(curve.min()), float(curve.max()), *quantiles.tolist(), *top_means,
        float(change.mean()) if len(change) else 0.0,
        float(change.std()) if len(change) else 0.0,
        float(change.max()) if len(change) else 0.0,
    ], dtype=np.float32)


def joint_video_features(baseline: np.ndarray, neuron: np.ndarray) -> np.ndarray:
    neuron = resample_curve(neuron, len(baseline))
    correlation = 0.0
    if len(baseline) > 1 and baseline.std() > 0 and neuron.std() > 0:
        correlation = float(np.corrcoef(baseline, neuron)[0, 1])
    return np.concatenate([
        video_features(baseline), video_features(neuron),
        np.asarray([
            correlation, float(np.mean(np.abs(baseline - neuron))),
            float(np.max(baseline - neuron)), float(np.max(neuron - baseline)),
        ], dtype=np.float32),
    ])


def logit(curve: np.ndarray) -> np.ndarray:
    clipped = np.clip(curve, 1e-5, 1.0 - 1e-5)
    return np.log(clipped / (1.0 - clipped))


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate conservative universal CLS-neuron fusion.")
    parser.add_argument("--baseline-train-manifest", required=True)
    parser.add_argument("--baseline-manifest", required=True)
    parser.add_argument("--expert-train-manifest", required=True)
    parser.add_argument("--expert-manifest", required=True)
    parser.add_argument("--expert2-manifest", required=True)
    parser.add_argument("--correction-model", required=True)
    parser.add_argument("--gt-path", required=True)
    parser.add_argument("--baseline", choices=["lagovad", "desc", "dsanet"], required=True)
    parser.add_argument("--dataset", choices=["ucf", "xd"], required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--frames-per-snippet", type=int, default=16)
    parser.add_argument("--correction-weight", type=float, default=0.2)
    parser.add_argument("--neuron-weight", type=float, default=0.1)
    parser.add_argument("--event-width", type=int, default=25)
    parser.add_argument("--event-weight", type=float, default=1.0)
    parser.add_argument("--normal-suppression-weight", type=float, default=1.0)
    parser.add_argument("--persistence-width", type=int, default=15)
    parser.add_argument("--persistence-weight", type=float, default=0.75)
    parser.add_argument("--gaussian-sigma", type=float, default=0.0)
    parser.add_argument("--advance-snippets", type=int, default=1)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    output = Path(args.out_dir)
    output.mkdir(parents=True, exist_ok=True)
    checkpoint = torch.load(args.correction_model, map_location="cpu", weights_only=False)
    if checkpoint.get("baseline") != args.baseline or checkpoint.get("dataset") != args.dataset:
        raise ValueError("correction checkpoint baseline/dataset mismatch")
    model = ScoreCorrectionHead(int(checkpoint["config"]["width"]))
    model.load_state_dict(checkpoint["model_state_dict"])
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    model.to(device).eval()

    baseline_train = pd.read_csv(args.baseline_train_manifest)
    expert_train = pd.read_csv(args.expert_train_manifest)[["key", "expert_score_path"]]
    video_train = baseline_train.merge(expert_train, on="key", validate="one_to_one")
    video_model = make_pipeline(
        StandardScaler(),
        LogisticRegression(class_weight="balanced", max_iter=1000, random_state=3407),
    )
    video_model.fit(
        np.asarray([
            joint_video_features(
                np.load(str(row.baseline_score_path)), np.load(str(row.expert_score_path))
            ) for row in video_train.itertuples(index=False)
        ]),
        video_train["binary_label"].to_numpy(),
    )
    baseline = pd.read_csv(args.baseline_manifest)
    expert = pd.read_csv(args.expert_manifest)[["key", "expert_score_path"]]
    expert2 = pd.read_csv(args.expert2_manifest)[["key", "expert2_score_path"]]
    frame = baseline.merge(expert, on="key", validate="one_to_one").merge(expert2, on="key", validate="one_to_one")
    baseline_curves, corrected_curves, rows = [], [], []
    with torch.no_grad():
        for row in tqdm(frame.itertuples(index=False), total=len(frame), desc=f"evaluate {args.baseline}/{args.dataset}"):
            base = np.load(str(row.baseline_score_path)).astype(np.float32)
            neuron = resample_curve(np.load(str(row.expert_score_path)), len(base))
            neuron2 = resample_curve(np.load(str(row.expert2_score_path)), len(base))
            base_tensor = torch.from_numpy(base).unsqueeze(0).to(device)
            neuron_tensor = torch.from_numpy(neuron).unsqueeze(0).to(device)
            correction = model(base_tensor, neuron_tensor)
            corrected = calibrated_probability(
                base_tensor, neuron_tensor, correction, args.correction_weight, args.neuron_weight
            )[0].cpu().numpy().astype(np.float32)
            standardized = (neuron - neuron.mean()) / max(float(neuron.std()), 1e-6)
            standardized2 = (neuron2 - neuron2.mean()) / max(float(neuron2.std()), 1e-6)
            neuron_gate = 1.0 / (1.0 + np.exp(-(standardized + standardized2)))
            expanded = maximum_filter1d(corrected, args.event_width, mode="nearest")
            corrected = corrected + args.event_weight * neuron_gate * (expanded - corrected)
            decision = float(video_model.decision_function(np.asarray(joint_video_features(base, neuron))[None])[0])
            normal_shift = min(0.0, decision)
            corrected = 1.0 / (1.0 + np.exp(-(logit(corrected) + args.normal_suppression_weight * normal_shift)))
            persistent = median_filter(corrected, args.persistence_width, mode="nearest")
            corrected = (1.0 - args.persistence_weight) * corrected + args.persistence_weight * persistent
            if args.gaussian_sigma > 0:
                corrected = gaussian_filter1d(corrected, args.gaussian_sigma, mode="nearest")
            if not 0 <= args.advance_snippets < len(corrected):
                raise ValueError("advance-snippets must be non-negative and shorter than every video")
            if args.advance_snippets:
                corrected = np.concatenate([
                    corrected[args.advance_snippets:],
                    np.repeat(corrected[-1:], args.advance_snippets),
                ])
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
            "correction_weight": args.correction_weight,
            "neuron_weight": args.neuron_weight,
            "event_width": args.event_width,
            "event_weight": args.event_weight,
            "event_gate": "sigmoid(2 * video-standardized neuron evidence)",
            "event_gate_experts": "mean evidence from 32- and 64-neurons-per-layer experts",
            "normal_suppression_weight": args.normal_suppression_weight,
            "video_prior": "one-sided joint current-baseline and CLS-neuron training classifier",
            "persistence_width": args.persistence_width,
            "persistence_weight": args.persistence_weight,
            "gaussian_sigma": args.gaussian_sigma,
            "advance_snippets": args.advance_snippets,
        },
        "frames": len(truth),
    }
    (output / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    pd.DataFrame(rows).to_csv(output / "per_video.csv", index=False)
    print(json.dumps(metrics, indent=2), flush=True)


if __name__ == "__main__":
    main()

