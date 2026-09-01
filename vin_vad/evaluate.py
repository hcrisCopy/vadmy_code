from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import average_precision_score, roc_auc_score
from torch.utils.data import DataLoader
from tqdm import tqdm

from vin_vad.data import FinalLayerDataset, collate_final_layer
from vin_vad.model import EventAblationModel


def file_sha256(path: str) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


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


def count_positive_runs(probabilities: np.ndarray, threshold: float = 0.5) -> int:
    positive = np.asarray(probabilities) >= threshold
    if not len(positive):
        return 0
    return int(positive[0]) + int(np.logical_and(positive[1:], ~positive[:-1]).sum())


def safe_correlation(first: list[float], second: list[float]) -> float:
    if len(first) < 2 or np.std(first) == 0 or np.std(second) == 0:
        return 0.0
    return float(np.corrcoef(first, second)[0, 1])


def auc_decomposition(
    frame_scores: list[np.ndarray],
    frame_labels: list[np.ndarray],
) -> dict[str, float | int]:
    """Decompose pooled AUROC into exact same-video and cross-video terms.

    The pair-count identity follows Song and Lee (arXiv:2608.21854).  Macro
    Within gives every mixed-label video one vote; Within uses the pair count
    required to reconstruct pooled AUROC exactly.
    """
    if len(frame_scores) != len(frame_labels) or not frame_scores:
        raise ValueError("frame_scores and frame_labels need the same non-zero length")
    scores = [np.asarray(value, dtype=np.float64).reshape(-1) for value in frame_scores]
    labels = [np.asarray(value, dtype=np.int8).reshape(-1) for value in frame_labels]
    if any(len(score) != len(label) for score, label in zip(scores, labels)):
        raise ValueError("every video score curve must match its frame labels")
    pooled_score = np.concatenate(scores)
    pooled_label = np.concatenate(labels)
    positives = int(pooled_label.sum())
    negatives = int(len(pooled_label) - positives)
    if positives == 0 or negatives == 0:
        raise ValueError("pooled labels must contain positive and negative frames")

    within_weighted_sum = 0.0
    within_pairs = 0
    per_video_auc: list[float] = []
    for score, label in zip(scores, labels):
        positive = int(label.sum())
        negative = int(len(label) - positive)
        if positive == 0 or negative == 0:
            continue
        pairs = positive * negative
        value = float(roc_auc_score(label, score))
        within_weighted_sum += pairs * value
        within_pairs += pairs
        per_video_auc.append(value)
    total_pairs = positives * negatives
    if within_pairs == 0 or within_pairs >= total_pairs:
        raise ValueError("both same-video and cross-video pair populations are required")
    pooled_auc = float(roc_auc_score(pooled_label, pooled_score))
    within_auc = within_weighted_sum / within_pairs
    cross_auc = (pooled_auc * total_pairs - within_auc * within_pairs) / (
        total_pairs - within_pairs
    )
    reconstructed = (within_pairs * within_auc + (total_pairs - within_pairs) * cross_auc) / total_pairs
    return {
        "pooled_auc": pooled_auc,
        "cross_auc": float(cross_auc),
        "within_auc": float(within_auc),
        "macro_within_auc": float(np.mean(per_video_auc)),
        "mixed_label_videos": len(per_video_auc),
        "same_video_pair_share": float(within_pairs / total_pairs),
        "decomposition_abs_error": float(abs(reconstructed - pooled_auc)),
    }


def fixed_tpr_false_alarm(
    frame_scores: list[np.ndarray],
    frame_labels: list[np.ndarray],
    target_tpr: float,
) -> dict[str, float]:
    """Measure false alarms on all-normal videos at a fixed positive recall."""
    if not 0.0 < target_tpr <= 1.0:
        raise ValueError("target_tpr must be in (0, 1]")
    positive_scores = np.concatenate(
        [score[label.astype(bool)] for score, label in zip(frame_scores, frame_labels)]
    )
    normal_video_scores = np.concatenate(
        [score for score, label in zip(frame_scores, frame_labels) if not np.any(label)]
    )
    if not len(positive_scores) or not len(normal_video_scores):
        raise ValueError("fixed-TPR evaluation needs positives and all-normal videos")
    ordered = np.sort(positive_scores)
    index = min(len(ordered) - 1, int(np.floor((1.0 - target_tpr) * len(ordered))))
    threshold = float(ordered[index])
    return {
        "target_tpr": float(target_tpr),
        "achieved_tpr": float(np.mean(positive_scores >= threshold)),
        "threshold": threshold,
        "normal_video_frame_fpr": float(np.mean(normal_video_scores >= threshold)),
    }


def video_constant_scores(frame_scores: list[np.ndarray]) -> list[np.ndarray]:
    """Remove temporal variation while retaining one mean score per video."""
    return [np.full_like(score, float(np.mean(score))) for score in frame_scores]


def score_curve_metrics(
    frame_scores: list[np.ndarray],
    frame_labels: list[np.ndarray],
    target_tpr: float,
) -> dict[str, float | int | dict[str, float]]:
    """Return the B0 metrics used by every later CVA-VAD experiment."""
    decomposition = auc_decomposition(frame_scores, frame_labels)
    pooled_scores = np.concatenate(frame_scores)
    pooled_labels = np.concatenate(frame_labels)
    constant = video_constant_scores(frame_scores)
    normal_top_scores = [
        float(np.max(score))
        for score, label in zip(frame_scores, frame_labels)
        if not np.any(label)
    ]
    if not normal_top_scores:
        raise ValueError("normal top-score evaluation needs all-normal videos")
    return {
        **decomposition,
        "pooled_ap": float(average_precision_score(pooled_labels, pooled_scores)),
        "video_constant_auc": float(
            roc_auc_score(pooled_labels, np.concatenate(constant))
        ),
        "video_constant_ap": float(
            average_precision_score(pooled_labels, np.concatenate(constant))
        ),
        "normal_fpr": fixed_tpr_false_alarm(
            frame_scores, frame_labels, target_tpr
        ),
        "normal_video_top_score": {
            "mean": float(np.mean(normal_top_scores)),
            "median": float(np.median(normal_top_scores)),
            "p95": float(np.quantile(normal_top_scores, 0.95)),
            "max": float(np.max(normal_top_scores)),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate one P1 event ablation without post-processing.")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--gt-path", required=True)
    parser.add_argument("--variant", choices=["e0", "e1", "e2", "e3"], required=True)
    parser.add_argument("--dataset", choices=["ucf", "xd"], required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--width", type=int, required=True)
    parser.add_argument("--dropout", type=float, required=True)
    parser.add_argument("--num-workers", type=int, required=True)
    parser.add_argument("--device", required=True)
    args = parser.parse_args()
    output = Path(args.out_dir)
    if "vadmy_data" not in output.resolve().parts:
        raise ValueError("out-dir must be inside the sibling vadmy_data directory")
    output.mkdir(parents=True, exist_ok=True)
    signature = {
        "manifest": {"path": args.manifest, "sha256": file_sha256(args.manifest)},
        "checkpoint": {"path": args.checkpoint, "sha256": file_sha256(args.checkpoint)},
        "gt": {"path": args.gt_path, "sha256": file_sha256(args.gt_path)},
        "dataset": args.dataset,
        "variant": args.variant,
        "width": args.width,
        "dropout": args.dropout,
        "post_processing": "none",
    }
    signature_path = output / "signature.json"
    if signature_path.exists() and json.loads(signature_path.read_text(encoding="utf-8")) != signature:
        raise RuntimeError("evaluation inputs changed; remove this variant output or retrain with --clean")
    signature_path.write_text(json.dumps(signature, indent=2), encoding="utf-8")
    metrics_path = output / "metrics.json"
    if metrics_path.exists() and (output / "per_video.csv").exists():
        print(f"reusing completed evaluation: {metrics_path}", flush=True)
        print(metrics_path.read_text(encoding="utf-8"), flush=True)
        return
    curves = output / "curves"
    curves.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    dataset = FinalLayerDataset(args.manifest, training=False)
    loader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        collate_fn=collate_final_layer,
    )
    model = EventAblationModel(args.variant, width=args.width, dropout=args.dropout).to(device)
    model.load_state_dict(
        torch.load(args.checkpoint, map_location=device, weights_only=True)
    )
    model.eval()
    rows = []
    frame_curves = []
    normal_lengths: list[float] = []
    normal_video_probabilities: list[float] = []
    with torch.no_grad():
        for batch in tqdm(loader, total=len(loader), desc=f"evaluate {args.dataset}/{args.variant}", unit="video"):
            key = batch["keys"][0]
            curve_path = curves / f"{key}.npz"
            if curve_path.exists():
                with np.load(curve_path, allow_pickle=False) as cached:
                    snippet_probability = np.asarray(cached["snippet_prob"], dtype=np.float32)
                    video_probability = float(cached["video_prob"])
            else:
                result = model(batch["features"].to(device), batch["mask"].to(device))
                valid = int(batch["lengths"][0])
                snippet_probability = result["snippet_prob"][0, :valid].cpu().numpy().astype(np.float32)
                video_probability = float(result["video_prob"][0].cpu())
                np.savez_compressed(
                    curve_path,
                    snippet_prob=snippet_probability,
                    video_prob=np.asarray(video_probability, dtype=np.float32),
                )
            frame_indices = np.load(batch["frame_indices_paths"][0], allow_pickle=False)
            frame_count = int(batch["evaluation_frames"][0])
            frame_curve = expand_snippet_scores(snippet_probability, frame_indices, frame_count)
            frame_curves.append(frame_curve)
            label = int(batch["labels"][0])
            fragments = count_positive_runs(snippet_probability)
            rows.append(
                {
                    "key": key,
                    "binary_label": label,
                    "snippets": len(snippet_probability),
                    "video_prob": video_probability,
                    "positive_runs_at_0.5": fragments,
                    "curve_path": str(curve_path),
                }
            )
            if label == 0:
                normal_lengths.append(float(len(snippet_probability)))
                normal_video_probabilities.append(video_probability)
    gt = np.asarray(np.load(args.gt_path, allow_pickle=False), dtype=np.float32).reshape(-1)
    frame_probability = np.concatenate(frame_curves)
    if len(frame_probability) != len(gt):
        raise RuntimeError(f"prediction frames {len(frame_probability)} != GT frames {len(gt)}")
    anomaly_fragments = [row["positive_runs_at_0.5"] for row in rows if row["binary_label"] == 1]
    metrics = {
        "status": "pass",
        "dataset": args.dataset,
        "variant": args.variant,
        "frame_auc": float(roc_auc_score(gt, frame_probability)),
        "frame_ap": float(average_precision_score(gt, frame_probability)),
        "primary_metric": "frame_auc" if args.dataset == "ucf" else "frame_ap",
        "normal_video_score_length_pearson": safe_correlation(normal_lengths, normal_video_probabilities),
        "mean_anomaly_video_positive_runs_at_0.5": float(np.mean(anomaly_fragments)),
        "post_processing": "none",
        "test_videos": len(rows),
        "test_frames": len(gt),
    }
    pd.DataFrame(rows).to_csv(output / "per_video.csv", index=False)
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(json.dumps(metrics, indent=2), flush=True)


if __name__ == "__main__":
    main()
