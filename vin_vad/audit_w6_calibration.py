from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from vin_vad.evaluate import expand_snippet_scores, score_curve_metrics


def group_summary(values: np.ndarray, mask: np.ndarray) -> dict[str, float | int]:
    selected = values[mask]
    return {
        "frames": int(mask.sum()),
        "mean": float(selected.mean()),
        "positive_share": float((selected > 0.0).mean()),
        "mean_positive": float(selected[selected > 0.0].mean())
        if np.any(selected > 0.0)
        else 0.0,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Audit raw witness calibration versus within-video residual correction"
    )
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--gt-path", required=True)
    parser.add_argument("--curves-dir", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--target-tpr", type=float, default=0.95)
    args = parser.parse_args()

    manifest = pd.read_csv(args.manifest)
    ground_truth = np.asarray(np.load(args.gt_path, allow_pickle=False), dtype=np.int8)
    names = ("host", "corrected", "evidence", "delta_anomaly")
    curves: dict[str, list[np.ndarray]] = {name: [] for name in names}
    labels: list[np.ndarray] = []
    normal_video_frames: list[np.ndarray] = []
    offset = 0
    for row in manifest.itertuples(index=False):
        with np.load(Path(args.curves_dir) / f"{row.key}.npz", allow_pickle=False) as item:
            frame_indices = np.asarray(item["frame_indices"], dtype=np.int64)
            snippet = {
                "host": np.asarray(item["host_score"], dtype=np.float32),
                "corrected": np.asarray(item["corrected_score"], dtype=np.float32),
                "evidence": np.asarray(item["evidence"], dtype=np.float32),
                "delta_anomaly": np.asarray(item["delta_anomaly"], dtype=np.float32),
            }
        frame_count = int(row.evaluation_frames)
        frame_labels = ground_truth[offset : offset + frame_count]
        offset += frame_count
        labels.append(frame_labels)
        normal_video_frames.append(
            np.full(frame_count, int(row.binary_label) == 0, dtype=bool)
        )
        for name in names:
            curves[name].append(
                expand_snippet_scores(snippet[name], frame_indices, frame_count)
            )
    if offset != len(ground_truth):
        raise RuntimeError(f"evaluated frames {offset} != GT frames {len(ground_truth)}")

    metrics = {
        name: score_curve_metrics(value, labels, args.target_tpr)
        for name, value in curves.items()
    }
    flat = {name: np.concatenate(value) for name, value in curves.items()}
    flat_labels = np.concatenate(labels).astype(bool)
    flat_normal_videos = np.concatenate(normal_video_frames)
    host_missed_positive = flat_labels & (flat["host"] < 0.5)
    delta = flat["delta_anomaly"]
    report = {
        "protocol": "read_only_reporting_no_fit",
        "metrics": metrics,
        "delta_anomaly_groups": {
            "normal_video_frames": group_summary(delta, flat_normal_videos),
            "positive_gt_frames": group_summary(delta, flat_labels),
            "host_below_0_5_positive_gt_frames": group_summary(
                delta, host_missed_positive
            ),
        },
        "raw_evidence_mean": {
            "normal_video_frames": float(flat["evidence"][flat_normal_videos].mean()),
            "positive_gt_frames": float(flat["evidence"][flat_labels].mean()),
            "host_below_0_5_positive_gt_frames": float(
                flat["evidence"][host_missed_positive].mean()
            ),
        },
    }
    output = Path(args.out)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
