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
    model.load_state_dict(torch.load(args.checkpoint, map_location=device))
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
