from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.ndimage import gaussian_filter1d, maximum_filter1d, median_filter
from scipy.special import expit
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
    features = np.concatenate([
        video_features(baseline), video_features(neuron),
        np.asarray([
            correlation, float(np.mean(np.abs(baseline - neuron))),
            float(np.max(baseline - neuron)), float(np.max(neuron - baseline)),
        ], dtype=np.float32),
    ])
    return np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)


def normality_video_features(
    baseline: np.ndarray,
    neuron: np.ndarray,
    auxiliary: np.ndarray,
    normality: np.ndarray,
) -> np.ndarray:
    neuron = resample_curve(neuron, len(baseline))
    auxiliary = resample_curve(auxiliary, len(baseline))
    normality = resample_curve(normality, len(baseline))
    streams = (baseline, neuron, auxiliary, normality)
    correlations, disagreements = [], []
    for left_index in range(len(streams)):
        for right_index in range(left_index + 1, len(streams)):
            left, right = streams[left_index], streams[right_index]
            correlation = 0.0
            if len(left) > 1 and left.std() > 0 and right.std() > 0:
                correlation = float(np.corrcoef(left, right)[0, 1])
            correlations.append(correlation)
            disagreements.append(float(np.mean(np.abs(left - right))))
    features = np.concatenate([
        joint_video_features(baseline, neuron),
        video_features(auxiliary),
        video_features(normality),
        np.asarray([*correlations, *disagreements], dtype=np.float32),
    ])
    return np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)


def logit(curve: np.ndarray) -> np.ndarray:
    clipped = np.clip(curve, 1e-5, 1.0 - 1e-5)
    return np.log(clipped / (1.0 - clipped))


def blend_normality(curve: np.ndarray, blend: float) -> np.ndarray:
    if not 0.0 <= blend <= 1.0:
        raise ValueError("normality-smoothing-blend must be in [0, 1]")
    if not blend:
        return curve
    return (1.0 - blend) * curve + blend * gaussian_filter1d(curve, 1.0, mode="nearest")


def fuse_standardized(primary: np.ndarray, auxiliary: np.ndarray, weight: float) -> np.ndarray:
    auxiliary = resample_curve(auxiliary, len(primary))
    primary = (primary - primary.mean()) / max(float(primary.std()), 1e-6)
    auxiliary = (auxiliary - auxiliary.mean()) / max(float(auxiliary.std()), 1e-6)
    return (primary + weight * auxiliary).astype(np.float32)


def longest_positive_run(values: np.ndarray) -> int:
    mask = np.asarray(values) > 0.0
    padded = np.concatenate([np.zeros(1, dtype=bool), mask, np.zeros(1, dtype=bool)])
    changes = np.flatnonzero(padded[1:] != padded[:-1])
    lengths = changes[1::2] - changes[::2]
    return int(lengths.max()) if len(lengths) else 1


def read_second_expert_manifest(path: str) -> pd.DataFrame:
    frame = pd.read_csv(path)
    if "expert2_score_path" not in frame and "student_score_path" in frame:
        frame = frame.rename(columns={"student_score_path": "expert2_score_path"})
    return frame[["key", "expert2_score_path"]]


def spectral_consensus_weights(*curves: np.ndarray) -> np.ndarray:
    """Return mean-one eigenvector-centrality weights from positive agreement."""
    matrix = np.corrcoef(np.stack(curves))
    matrix = np.nan_to_num(matrix, nan=0.0, posinf=0.0, neginf=0.0)
    matrix = np.maximum(matrix, 0.0)
    np.fill_diagonal(matrix, 1.0)
    _, eigenvectors = np.linalg.eigh(matrix)
    weights = np.abs(eigenvectors[:, -1]).astype(np.float32)
    return weights / max(float(weights.mean()), 1e-6)


def estimate_persistence_width(
    expert_manifest: str,
    expert2_manifest: str,
    expert3_manifest: str,
    normality_blend: float,
) -> int:
    expert = pd.read_csv(expert_manifest)[["key", "expert_score_path"]]
    expert2 = read_second_expert_manifest(expert2_manifest)
    expert3 = pd.read_csv(expert3_manifest)[["key", "expert3_score_path"]]
    frame = expert.merge(expert2, on="key", validate="one_to_one").merge(
        expert3, on="key", validate="one_to_one"
    )
    run_lengths = []
    for row in frame.itertuples(index=False):
        neuron = np.load(str(row.expert_score_path))
        neuron2 = resample_curve(np.load(str(row.expert2_score_path)), len(neuron))
        neuron3 = blend_normality(
            resample_curve(np.load(str(row.expert3_score_path)), len(neuron)), normality_blend
        )
        standardized = (neuron - neuron.mean()) / max(float(neuron.std()), 1e-6)
        standardized2 = (neuron2 - neuron2.mean()) / max(float(neuron2.std()), 1e-6)
        standardized3 = (neuron3 - neuron3.mean()) / max(float(neuron3.std()), 1e-6)
        run_lengths.append(longest_positive_run(standardized + standardized2 + 1.5 * standardized3))
    raw_width = 0.35 * float(np.quantile(run_lengths, 0.75))
    odd_width = 2 * round((raw_width - 1.0) / 2.0) + 1
    return int(np.clip(odd_width, 7, 21))


def main() -> None:
    evaluation_start = time.perf_counter()
    parser = argparse.ArgumentParser(description="Evaluate conservative universal CLS-neuron fusion.")
    parser.add_argument("--baseline-train-manifest", required=True)
    parser.add_argument("--baseline-manifest", required=True)
    parser.add_argument("--expert-train-manifest", required=True)
    parser.add_argument("--expert-manifest", required=True)
    parser.add_argument("--expert2-manifest", required=True)
    parser.add_argument("--expert2-train-manifest", required=True)
    parser.add_argument("--expert3-manifest", required=True)
    parser.add_argument("--expert3-train-manifest", required=True)
    parser.add_argument("--student-manifest", required=True)
    parser.add_argument("--student-train-manifest", required=True)
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
    parser.add_argument("--normality-gate-weight", type=float, default=0.5)
    parser.add_argument("--normality-smoothing-blend", type=float, default=0.0)
    parser.add_argument("--agreement-residual-weight", type=float, default=0.0)
    parser.add_argument("--triple-agreement-weight", type=float, default=0.0)
    parser.add_argument("--normal-suppression-weight", type=float, default=1.0)
    parser.add_argument("--persistence-weight", type=float, default=0.75)
    parser.add_argument("--gaussian-sigma", type=float, default=0.0)
    parser.add_argument("--advance-snippets", type=float, default=0.5)
    parser.add_argument("--disable-correction", action="store_true")
    parser.add_argument("--disable-agreement", action="store_true")
    parser.add_argument("--disable-event-gate", action="store_true")
    parser.add_argument("--disable-video-suppression", action="store_true")
    parser.add_argument("--disable-temporal", action="store_true")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    persistence_width = estimate_persistence_width(
        args.expert_train_manifest,
        args.expert2_train_manifest,
        args.expert3_train_manifest,
        args.normality_smoothing_blend,
    )
    duration_factor = float(np.clip((persistence_width - 11.0) / 4.0, 0.0, 1.0))
    correction_weight = 3.0 * duration_factor
    neuron_weight = 0.3 - 0.1 * duration_factor
    normality_gate_weight = 1.0 + 2.0 * duration_factor
    agreement_residual_weight = 0.5 - 0.3 * duration_factor
    triple_agreement_weight = 3.0
    neuron_consensus_weight = 1.0
    neuron_conflict_weight = 1.2
    normal_suppression_weight = 1.0
    context_diverse_weight = 8.0 * duration_factor
    context_normality_weight = duration_factor
    final_dilation_width = 1 + 2 * round(12.0 * duration_factor)
    final_dilation_weight = 0.75 * duration_factor
    effective_gaussian_sigma = args.gaussian_sigma + 0.5 * duration_factor

    output = Path(args.out_dir)
    output.mkdir(parents=True, exist_ok=True)
    curve_dir = output / "curves"
    curve_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = torch.load(args.correction_model, map_location="cpu", weights_only=False)
    if checkpoint.get("baseline") != args.baseline or checkpoint.get("dataset") != args.dataset:
        raise ValueError("correction checkpoint baseline/dataset mismatch")
    model = ScoreCorrectionHead(int(checkpoint["config"]["width"]))
    model.load_state_dict(checkpoint["model_state_dict"])
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    model.to(device).eval()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    baseline_train = pd.read_csv(args.baseline_train_manifest)
    expert_train = pd.read_csv(args.expert_train_manifest)[["key", "expert_score_path"]]
    expert2_train = read_second_expert_manifest(args.expert2_train_manifest)
    expert3_train = pd.read_csv(args.expert3_train_manifest)[["key", "expert3_score_path"]]
    student_train = pd.read_csv(args.student_train_manifest)[["key", "student_score_path"]]
    video_train = baseline_train.merge(expert_train, on="key", validate="one_to_one").merge(
        expert2_train, on="key", validate="one_to_one"
    ).merge(expert3_train, on="key", validate="one_to_one").merge(
        student_train, on="key", validate="one_to_one"
    )
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
    normality_video_model = make_pipeline(
        StandardScaler(),
        LogisticRegression(class_weight="balanced", max_iter=1000, random_state=3407),
    )
    normality_video_model.fit(
        np.asarray([
            normality_video_features(
                np.load(str(row.baseline_score_path)),
                np.load(str(row.expert_score_path)),
                np.load(str(row.expert2_score_path)),
                blend_normality(
                    fuse_standardized(
                        np.load(str(row.expert3_score_path)),
                        np.load(str(row.student_score_path)),
                        context_normality_weight,
                    ),
                    args.normality_smoothing_blend,
                ),
            ) for row in video_train.itertuples(index=False)
        ]),
        video_train["binary_label"].to_numpy(),
    )
    baseline = pd.read_csv(args.baseline_manifest)
    expert = pd.read_csv(args.expert_manifest)[["key", "expert_score_path"]]
    expert2 = read_second_expert_manifest(args.expert2_manifest)
    expert3 = pd.read_csv(args.expert3_manifest)[["key", "expert3_score_path"]]
    student = pd.read_csv(args.student_manifest)[["key", "student_score_path"]]
    expected_train_keys = set(pd.read_csv(args.expert2_train_manifest)["key"].astype(str))
    if set(student_train["key"].astype(str)) != expected_train_keys:
        raise ValueError("student training manifest keys must match the shared diverse-neuron training subset")
    frame = baseline.merge(expert, on="key", validate="one_to_one").merge(
        expert2, on="key", validate="one_to_one"
    ).merge(expert3, on="key", validate="one_to_one").merge(
        student, on="key", validate="one_to_one"
    )
    baseline_curves, corrected_curves, rows = [], [], []
    with torch.no_grad():
        for row in tqdm(frame.itertuples(index=False), total=len(frame), desc=f"evaluate {args.baseline}/{args.dataset}"):
            base = np.load(str(row.baseline_score_path)).astype(np.float32)
            neuron = resample_curve(np.load(str(row.expert_score_path)), len(base))
            neuron2 = resample_curve(np.load(str(row.expert2_score_path)), len(base))
            neuron3 = resample_curve(np.load(str(row.expert3_score_path)), len(base))
            student_curve = resample_curve(np.load(str(row.student_score_path)), len(base))
            base_tensor = torch.from_numpy(base).unsqueeze(0).to(device)
            neuron_tensor = torch.from_numpy(neuron).unsqueeze(0).to(device)
            if args.disable_correction:
                corrected = base.copy()
            else:
                correction = model(base_tensor, neuron_tensor)
                corrected = calibrated_probability(
                    base_tensor, neuron_tensor, correction, correction_weight, neuron_weight
                )[0].cpu().numpy().astype(np.float32)
            standardized = (neuron - neuron.mean()) / max(float(neuron.std()), 1e-6)
            standardized2 = fuse_standardized(neuron2, student_curve, context_diverse_weight)
            standardized2 = (standardized2 - standardized2.mean()) / max(float(standardized2.std()), 1e-6)
            neuron3_context = fuse_standardized(neuron3, student_curve, context_normality_weight)
            standardized3 = neuron3_context
            if not 0.0 <= args.normality_smoothing_blend <= 1.0:
                raise ValueError("normality-smoothing-blend must be in [0, 1]")
            if args.normality_smoothing_blend:
                smooth_neuron3 = gaussian_filter1d(standardized3, 1.0, mode="nearest")
                standardized3 = (1.0 - args.normality_smoothing_blend) * standardized3 + args.normality_smoothing_blend * smooth_neuron3
            standardized3 = (standardized3 - standardized3.mean()) / max(float(standardized3.std()), 1e-6)
            decision = float(video_model.decision_function(np.asarray(joint_video_features(base, neuron))[None])[0])
            normality_decision = float(normality_video_model.decision_function(np.asarray(normality_video_features(base, neuron, neuron2, blend_normality(neuron3_context, args.normality_smoothing_blend)))[None])[0])
            video_anomaly_gate = float(expit(min(decision, normality_decision)))
            standardized_base = (base - base.mean()) / max(float(base.std()), 1e-6)
            neuron_consensus = np.minimum(
                np.maximum(standardized, 0.0), np.maximum(standardized3, 0.0)
            )
            if not args.disable_agreement:
                residual_scale = (
                    (1.0 - duration_factor) * 0.3
                    + duration_factor * video_anomaly_gate * neuron_consensus_weight
                )
                corrected = expit(
                    logit(corrected)
                    + residual_scale * neuron_consensus
                )
                neuron_conflict = np.minimum(
                    np.maximum(standardized_base, 0.0),
                    np.minimum(np.maximum(-standardized, 0.0), np.maximum(-standardized3, 0.0)),
                )
                corrected = expit(logit(corrected) - neuron_conflict_weight * neuron_conflict)
            high_high = np.minimum(np.maximum(standardized_base, 0.0), np.maximum(standardized, 0.0))
            if not args.disable_agreement:
                corrected = expit(logit(corrected) + agreement_residual_weight * high_high)
                triple_high = np.minimum(
                    high_high,
                    np.minimum(
                        np.maximum(standardized2, 0.0),
                        np.maximum(standardized3, 0.0),
                    ),
                )
                corrected = expit(logit(corrected) + triple_agreement_weight * triple_high)
            consensus_weights = spectral_consensus_weights(
                standardized, standardized2, standardized3
            )
            neuron_gate = expit(
                consensus_weights[0] * standardized
                + consensus_weights[1] * standardized2
                + normality_gate_weight * consensus_weights[2] * standardized3
            )
            if not args.disable_event_gate:
                expanded = maximum_filter1d(corrected, args.event_width, mode="nearest")
                corrected = corrected + args.event_weight * neuron_gate * (expanded - corrected)
            normal_shift = (
                decision + normality_decision
                if decision < 0.0 and normality_decision < 0.0
                else 0.0
            )
            if not args.disable_video_suppression:
                corrected = expit(logit(corrected) + normal_suppression_weight * normal_shift)
            if not args.disable_temporal:
                persistent = median_filter(corrected, persistence_width, mode="nearest")
                if duration_factor > 0.0:
                    long_width = 2 * persistence_width - 1
                    long_persistent = median_filter(corrected, long_width, mode="nearest")
                    persistent = (
                        (1.0 - 0.5 * duration_factor) * persistent
                        + 0.5 * duration_factor * long_persistent
                    )
                corrected = (1.0 - args.persistence_weight) * corrected + args.persistence_weight * persistent
                if final_dilation_width > 1 and final_dilation_weight > 0.0:
                    dilated = maximum_filter1d(corrected, final_dilation_width, mode="nearest")
                    corrected = corrected + final_dilation_weight * (dilated - corrected)
                if effective_gaussian_sigma > 0:
                    corrected = gaussian_filter1d(corrected, effective_gaussian_sigma, mode="nearest")
                if not 0.0 <= args.advance_snippets < len(corrected):
                    raise ValueError("advance-snippets must be non-negative and shorter than every video")
                if args.advance_snippets:
                    positions = np.arange(len(corrected), dtype=np.float32)
                    corrected = np.interp(
                        positions + args.advance_snippets,
                        positions,
                        corrected,
                        right=float(corrected[-1]),
                    )
            baseline_curves.append(base)
            corrected_curves.append(corrected)
            curve_path = curve_dir / f"{row.key}.npz"
            np.savez_compressed(curve_path, baseline=base, corrected=corrected.astype(np.float32))
            rows.append({"key": str(row.key), "binary_label": int(row.binary_label), "snippets": len(base), "curve_path": str(curve_path), "corrected_mean": float(corrected.mean())})

    truth = np.load(args.gt_path).astype(np.int64).reshape(-1)
    baseline_frames = np.repeat(np.concatenate(baseline_curves), args.frames_per_snippet)
    corrected_frames = np.repeat(np.concatenate(corrected_curves), args.frames_per_snippet)
    if len(truth) != len(baseline_frames) or len(truth) != len(corrected_frames):
        raise RuntimeError(
            f"strict frame alignment failed: gt={len(truth)} baseline={len(baseline_frames)} corrected={len(corrected_frames)}"
        )
    elapsed_seconds = time.perf_counter() - evaluation_start
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
            "correction_weight": correction_weight,
            "neuron_weight": neuron_weight,
            "event_width": args.event_width,
            "event_weight": args.event_weight,
            "event_gate": "sigmoid(sum of video-standardized CLS-neuron evidence)",
            "event_gate_reliability": "principal eigenvector of positive detector agreement",
            "event_gate_experts": "MIL experts plus weighted baseline-independent normality expert",
            "normality_gate_weight": normality_gate_weight,
            "normality_smoothing_blend": args.normality_smoothing_blend,
            "agreement_residual_weight": agreement_residual_weight,
            "neuron_consensus_weight": neuron_consensus_weight,
            "neuron_conflict_weight": neuron_conflict_weight,
            "neuron_consensus_context": "duration interpolation from fixed 0.3 to gated unit residual",
            "triple_agreement_weight": triple_agreement_weight,
            "normal_suppression_weight": normal_suppression_weight,
            "video_prior": "training-only one-sided classifier with all three CLS-neuron views and pairwise consensus",
            "persistence_width": persistence_width,
            "persistence_width_rule": "0.35 * training gate-run q75, nearest odd, clipped to [7, 21]",
            "duration_factor": duration_factor,
            "duration_adaptation": "clip((training persistence width - 11) / 4, 0, 1)",
            "context_diverse_weight": context_diverse_weight,
            "context_normality_weight": context_normality_weight,
            "final_dilation_width": final_dilation_width,
            "final_dilation_weight": final_dilation_weight,
            "persistence_weight": args.persistence_weight,
            "persistence_scales": [persistence_width, 2 * persistence_width - 1],
            "gaussian_sigma": effective_gaussian_sigma,
            "advance_snippets": args.advance_snippets,
            "disabled_components": [
                name for name, disabled in {
                    "correction": args.disable_correction,
                    "agreement": args.disable_agreement,
                    "event_gate": args.disable_event_gate,
                    "video_suppression": args.disable_video_suppression,
                    "temporal": args.disable_temporal,
                }.items() if disabled
            ],
        },
        "frames": len(truth),
        "performance": {
            "elapsed_seconds": elapsed_seconds,
            "frames_per_second": float(len(truth) / max(elapsed_seconds, 1e-9)),
            "peak_cuda_memory_mb": (
                float(torch.cuda.max_memory_allocated(device) / (1024 ** 2))
                if device.type == "cuda"
                else 0.0
            ),
            "scope": "cached-score adapter evaluation including manifest and curve loading",
        },
    }
    (output / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    pd.DataFrame(rows).to_csv(output / "per_video.csv", index=False)
    print(json.dumps(metrics, indent=2), flush=True)


if __name__ == "__main__":
    main()
