#!/usr/bin/env python3
"""Evaluate one Shift residual on an otherwise frozen baseline."""

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
from neuron_responsibility.evaluate import class_probabilities, load_detection_map, pad_chunks
from shift_residual_head_tuning.train import add_baseline_arguments
from shift_single_frozen.method import FrozenSingleResidualInjector
from shift_single_frozen.provenance import sha256


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    add_baseline_arguments(parser)
    parser.add_argument("--dataset", choices=["ucf", "xd"], required=True)
    parser.add_argument("--test-list", required=True)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--gt-path", required=True)
    parser.add_argument("--gt-segment-path", default="")
    parser.add_argument("--gt-label-path", default="")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--frames-per-snippet", type=int, default=16)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--clean", action="store_true")
    args = parser.parse_args()

    checkpoint = torch.load(args.model_path, map_location="cpu")
    if checkpoint.get("method") != FrozenSingleResidualInjector.method_name:
        raise ValueError(f"unsupported checkpoint method: {checkpoint.get('method')}")
    config = checkpoint.get("run_config", {})
    if config.get("baseline") != args.baseline or config.get("dataset") != args.dataset:
        raise ValueError("checkpoint baseline/dataset differs from command")
    output = clean_output(args.out_dir, args.clean)
    model_sha256 = sha256(args.model_path)
    score_dir = output / "scores" / model_sha256[:16]
    score_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device if torch.cuda.is_available() and args.device.startswith("cuda") else "cpu")
    adapter = build_baseline(args, str(device)).to(device)
    injector = FrozenSingleResidualInjector.from_config(checkpoint["injector_config"]).to(device)
    injector.load_state_dict(checkpoint["injector_state_dict"], strict=True)
    adapter.attach_pre_temporal_conditioner(injector)
    adapter.requires_grad_(False)
    adapter.eval()

    frame = pd.read_csv(args.test_list)
    missing = {"clip_path", "neuron_path", "label"} - set(frame.columns)
    if missing:
        raise ValueError(f"{args.test_list}: missing columns {sorted(missing)}")
    if "key" not in frame.columns:
        frame["key"] = frame["clip_path"].map(lambda value: Path(str(value)).stem)
    temperature = args.temperature if args.temperature > 0 else (
        5.0 if args.baseline == "dsanet" and args.dataset == "ucf" else 1.0
    )
    binary_all, refined_all, class_predictions, rows = [], [], [], []
    with torch.no_grad():
        for key_value, group in tqdm(frame.groupby("key", sort=False), desc="evaluate single frozen", unit="video"):
            key = str(key_value)
            labels = set(group["label"].astype(str))
            if len(labels) != 1:
                raise ValueError(f"{key}: inconsistent labels")
            cache_path = score_dir / f"{key}.npz"
            if cache_path.exists() and not args.clean:
                cached = np.load(cache_path)
                binary, refined, class_prob = cached["binary"], cached["refined"], cached["class_prob"]
            else:
                clip = np.concatenate([np.load(str(path)).astype(np.float32) for path in group["clip_path"]])
                neurons = np.concatenate([np.load(str(path)).astype(np.float32) for path in group["neuron_path"]])
                clip_chunks, lengths = pad_chunks(clip, adapter.visual_length)
                neuron_chunks, neuron_lengths = pad_chunks(neurons, adapter.visual_length)
                if not torch.equal(lengths, neuron_lengths):
                    raise RuntimeError(f"{key}: modality lengths differ")
                result, _ = adapter.forward_conditioned(
                    clip_chunks.to(device), neuron_chunks.to(device), lengths.to(device),
                )
                class_batch = class_probabilities(
                    result.binary_logits, result.semantic_logits, args.baseline, temperature,
                )
                binary_parts, refined_parts, class_parts = [], [], []
                for index, length in enumerate(lengths.tolist()):
                    binary_parts.append(torch.sigmoid(result.binary_logits[index, :length]).cpu())
                    class_parts.append(class_batch[index, :length].cpu())
                    refined_parts.append((1.0 - class_batch[index, :length, 0]).cpu())
                binary = torch.cat(binary_parts).numpy().astype(np.float32)
                refined = torch.cat(refined_parts).numpy().astype(np.float32)
                class_prob = torch.cat(class_parts).numpy().astype(np.float32)
                np.savez_compressed(cache_path, binary=binary, refined=refined, class_prob=class_prob)
            binary_all.append(binary)
            refined_all.append(refined)
            class_predictions.append(np.repeat(class_prob, args.frames_per_snippet, axis=0))
            rows.append([key, next(iter(labels)), len(binary), float(binary.mean()), float(refined.mean())])

    ground_truth = np.load(args.gt_path).astype(np.int64).reshape(-1)
    metrics = {}
    for name, pieces in (("binary_score_head", binary_all), ("official_refined", refined_all)):
        scores = np.repeat(np.concatenate(pieces), args.frames_per_snippet)
        if len(scores) != len(ground_truth):
            raise RuntimeError(
                f"strict frame alignment failed for {name}: prediction={len(scores)} gt={len(ground_truth)}"
            )
        metrics[name] = {
            "auc": float(roc_auc_score(ground_truth, scores)),
            "ap": float(average_precision_score(ground_truth, scores)),
            "frames": len(ground_truth),
        }
    if args.gt_segment_path and args.gt_label_path:
        utility = Path(args.baseline_root) / "src" / "utils" / f"{args.dataset}_detectionMAP.py"
        if utility.exists():
            detection_map = load_detection_map(utility)
            maps, thresholds = detection_map(
                class_predictions,
                np.load(args.gt_segment_path, allow_pickle=True),
                np.load(args.gt_label_path, allow_pickle=True),
                excludeNormal=False,
            )
            metrics["detection_map"] = {
                f"iou_{float(iou):.1f}": float(value) for iou, value in zip(thresholds, maps)
            }
            metrics["detection_map_average"] = float(np.mean(maps[:5]))
    metrics["checkpoint_sha256"] = model_sha256
    (output / "metrics.json").write_text(json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8")
    with (output / "per_video.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["key", "label", "snippets", "binary_mean", "refined_mean"])
        writer.writerows(rows)
    print(json.dumps(metrics, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
