#!/usr/bin/env python3
"""Evaluate NREE and the released author checkpoint with one protocol."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import average_precision_score, roc_auc_score
from tqdm import tqdm


PACKAGE_ROOT = Path(__file__).resolve().parent.parent
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))

from neuron_responsibility.baselines import build_baseline
from neuron_responsibility.common import clean_output
from neuron_responsibility.event_experts import (
    NeuronRoutedEventExperts,
    route_targets,
)
from neuron_responsibility.train_feature_modulation import (
    add_baseline_arguments,
    pad_chunks,
)


def frame_metrics(gt: np.ndarray, scores: np.ndarray) -> dict[str, float | int]:
    usable = min(len(gt), len(scores))
    if usable != len(gt) or usable != len(scores):
        print(
            f"metric length alignment: gt={len(gt)} prediction={len(scores)} usable={usable}",
            flush=True,
        )
    return {
        "auc": float(roc_auc_score(gt[:usable], scores[:usable])),
        "ap": float(average_precision_score(gt[:usable], scores[:usable])),
        "frames": int(usable),
    }


def evaluate(
    adapter,
    aligned_csv: str,
    device: torch.device,
    frames_per_snippet: int,
    use_experts: bool,
) -> tuple[np.ndarray, list[list[object]], dict[str, object]]:
    adapter.eval()
    frame = pd.read_csv(aligned_csv)
    if "key" not in frame.columns:
        frame["key"] = frame["clip_path"].map(lambda value: Path(str(value)).stem)
    snippets, rows = [], []
    route_total = None
    route_videos = 0
    route_correct = 0
    abnormal_correct = 0
    abnormal_videos = 0
    normal_correct = 0
    normal_videos = 0
    prediction_histogram: dict[str, int] = {}
    fast_total = 0.0
    fast_count = 0.0
    groups = list(frame.groupby("key", sort=False))
    with torch.no_grad():
        for key, group in tqdm(groups, desc="NREE evaluation", unit="video"):
            clip = np.concatenate([
                np.load(str(path)).astype(np.float32) for path in group["clip_path"]
            ])
            neurons = np.concatenate([
                np.load(str(path)).astype(np.float32) for path in group["neuron_path"]
            ])
            clip_chunks, lengths = pad_chunks(clip, adapter.visual_length)
            neuron_chunks, neuron_lengths = pad_chunks(neurons, adapter.visual_length)
            if not torch.equal(lengths, neuron_lengths):
                raise RuntimeError(f"{key}: CLIP and neuron lengths differ")
            lengths_device = lengths.to(device)
            if use_experts:
                output, records = adapter.forward_modulated(
                    clip_chunks.to(device), neuron_chunks.to(device), lengths_device
                )
                routes = torch.stack([
                    record["route"] for record in records
                    if isinstance(record["route"], torch.Tensor)
                ]).mean(dim=0)
                video_route = routes.mean(dim=0)
                route_sum = video_route.cpu().numpy()
                route_total = route_sum if route_total is None else route_total + route_sum
                route_videos += 1
                class_names = ["Normal"] + list(adapter.feature_modulator.class_names)
                target = route_targets(
                    [str(group.iloc[0]["label"])],
                    list(adapter.feature_modulator.class_names),
                    device,
                )[0]
                target_index = int(target.argmax().item())
                prediction_index = int(video_route.argmax().item())
                route_correct += int(target_index == prediction_index)
                prediction_name = class_names[prediction_index]
                prediction_histogram[prediction_name] = (
                    prediction_histogram.get(prediction_name, 0) + 1
                )
                if target_index == 0:
                    normal_videos += 1
                    normal_correct += int(prediction_index == 0)
                else:
                    abnormal_videos += 1
                    abnormal_correct += int(target_index == prediction_index)
                for record in records:
                    fast_gate = record["fast_gate"]
                    mask = record["mask"]
                    if isinstance(fast_gate, torch.Tensor) and isinstance(mask, torch.Tensor):
                        fast_total += float((fast_gate * mask).sum().cpu())
                        fast_count += float(mask.sum().cpu())
            else:
                output = adapter.forward_baseline(clip_chunks.to(device), lengths_device)
            video_scores = []
            for index, length in enumerate(lengths.tolist()):
                values = torch.sigmoid(output.binary_logits[index, :length]).cpu().numpy()
                snippets.append(values)
                video_scores.append(values)
            merged = np.concatenate(video_scores)
            rows.append([
                key,
                str(group.iloc[0]["label"]),
                len(merged),
                float(merged.mean()),
                float(merged.max()),
            ])
    snippet_scores = np.concatenate(snippets).astype(np.float32)
    frame_scores = np.repeat(snippet_scores, frames_per_snippet)
    diagnostics: dict[str, object] = {}
    if route_total is not None and route_videos:
        mean_route = route_total / route_videos
        for index, value in enumerate(mean_route):
            diagnostics[f"route_{index}_mean"] = float(value)
        diagnostics["fast_gate_mean"] = fast_total / max(1.0, fast_count)
        diagnostics["video_route_accuracy"] = route_correct / route_videos
        diagnostics["abnormal_class_accuracy"] = abnormal_correct / max(1, abnormal_videos)
        diagnostics["normal_recall"] = normal_correct / max(1, normal_videos)
        diagnostics["prediction_histogram"] = prediction_histogram
    return frame_scores, rows, diagnostics


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare NREE with released baseline.")
    add_baseline_arguments(parser)
    parser.add_argument("--dataset", choices=["ucf", "xd"], required=True)
    parser.add_argument("--test-list", required=True)
    parser.add_argument("--gt-path", required=True)
    parser.add_argument("--atlas", required=True)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--frames-per-snippet", type=int, default=16)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--clean", action="store_true")
    args = parser.parse_args()
    out_dir = clean_output(args.out_dir, args.clean)
    device = torch.device(
        args.device if torch.cuda.is_available() and args.device.startswith("cuda") else "cpu"
    )
    checkpoint = torch.load(args.model_path, map_location="cpu")
    config = checkpoint["expert_config"]

    author = build_baseline(args, str(device)).to(device)
    author_scores, author_rows, _ = evaluate(
        author, args.test_list, device, args.frames_per_snippet, use_experts=False
    )
    del author
    if device.type == "cuda":
        torch.cuda.empty_cache()

    adapted = build_baseline(args, str(device)).to(device)
    experts = NeuronRoutedEventExperts(
        args.atlas,
        feature_width=int(config["feature_width"]),
        rank=int(config["rank"]),
        slow_dilation=int(config["slow_dilation"]),
        route_top_fraction=float(config["route_top_fraction"]),
    ).to(device)
    adapted.attach_feature_modulator(experts)
    adapted.load_state_dict(checkpoint["model_state_dict"], strict=True)
    nree_scores, nree_rows, diagnostics = evaluate(
        adapted, args.test_list, device, args.frames_per_snippet, use_experts=True
    )

    gt = np.load(args.gt_path).astype(np.int64).reshape(-1)
    author_metrics = frame_metrics(gt, author_scores)
    nree_metrics = frame_metrics(gt, nree_scores)
    primary = "auc" if args.dataset == "ucf" else "ap"
    metrics = {
        "released_author": author_metrics,
        "nree": nree_metrics,
        "primary_metric": primary,
        "absolute_gain": float(nree_metrics[primary]) - float(author_metrics[primary]),
        "checkpoint": args.model_path,
        "validation_tag": checkpoint.get("validation_tag"),
        "route_diagnostics": diagnostics,
    }
    (out_dir / "metrics.json").write_text(
        json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    np.savez_compressed(
        out_dir / "frame_scores.npz", author=author_scores, nree=nree_scores
    )
    with (out_dir / "per_video.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow([
            "key", "label", "snippets", "author_mean", "author_max",
            "nree_mean", "nree_max",
        ])
        for author_row, nree_row in zip(author_rows, nree_rows):
            if author_row[:3] != nree_row[:3]:
                raise RuntimeError("author and NREE video order differs")
            writer.writerow(author_row + nree_row[3:])
    print(json.dumps(metrics, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
