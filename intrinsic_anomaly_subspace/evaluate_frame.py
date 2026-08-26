#!/usr/bin/env python3
"""Evaluate the hidden-state linear readout with official frame-level metrics."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import average_precision_score, roc_auc_score
from tqdm import tqdm

from .common import base_key, clean_output, load_hidden, project, read_hidden_manifest, resample, save_json, write_csv


def official_snippet_length(paths: list[str]) -> int:
    total = 0
    for path in paths:
        feature = np.load(path, mmap_mode="r", allow_pickle=False)
        if feature.ndim != 2:
            raise ValueError(f"{path}: expected [T,D] official CLIP feature, got {feature.shape}")
        total += int(feature.shape[0])
    return total


def main() -> None:
    parser = argparse.ArgumentParser(description="Official UCF frame AUC / XD frame AP for the hidden-state readout.")
    parser.add_argument("--dataset", choices=["ucf", "xd"], required=True)
    parser.add_argument("--test-list", required=True)
    parser.add_argument("--hidden-manifest", required=True)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--gt-path", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--frames-per-snippet", type=int, default=16)
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--allow-length-crop", action="store_true")
    parser.add_argument("--clean", action="store_true")
    args = parser.parse_args()
    output = clean_output(args.out_dir, args.clean)
    metrics_path = output / "metrics.json"
    if metrics_path.exists() and not args.clean:
        print(metrics_path.read_text(encoding="utf-8"), flush=True)
        return
    state = torch.load(args.model_path, map_location="cpu")
    coordinates = [(int(layer), int(dim)) for layer, dim in state["coordinates"]]
    mean, std = np.asarray(state["mean"], dtype=np.float32), np.asarray(state["std"], dtype=np.float32)
    model = torch.nn.Linear(len(coordinates), 1)
    model.load_state_dict(state["model"])
    device = torch.device(args.device if torch.cuda.is_available() and args.device.startswith("cuda") else "cpu")
    model = model.to(device).eval()
    hidden_paths = read_hidden_manifest(args.hidden_manifest)
    source = pd.read_csv(args.test_list)
    missing = {"path", "label"} - set(source.columns)
    if missing:
        raise ValueError(f"{args.test_list}: missing columns {sorted(missing)}")
    source["key"] = source["path"].map(base_key)
    cache = output / "per_video"
    cache.mkdir(parents=True, exist_ok=True)
    all_scores, boundaries, skipped = [], [], []
    offset = 0
    grouped = source.groupby("key", sort=False)
    with torch.no_grad():
        for key, group in tqdm(grouped, desc=f"{args.dataset} frame evaluation", unit="video"):
            if key not in hidden_paths:
                raise FileNotFoundError(f"test hidden state missing for {key}; test videos cannot be skipped")
            score_path = cache / f"{key}.npy"
            target_length = official_snippet_length(group["path"].astype(str).tolist())
            if score_path.exists() and not args.clean:
                scores = np.load(score_path, allow_pickle=False).astype(np.float32)
                if len(scores) != target_length:
                    raise ValueError(f"{score_path}: cached length {len(scores)} != current official length {target_length}; use --clean")
            else:
                hidden = resample(load_hidden(hidden_paths[key]), target_length)
                features = (project(hidden, coordinates) - mean) / std
                parts = []
                for start in range(0, len(features), args.batch_size):
                    tensor = torch.from_numpy(features[start:start + args.batch_size]).to(device)
                    parts.append(torch.sigmoid(model(tensor).squeeze(1)).cpu().numpy())
                scores = np.concatenate(parts).astype(np.float32)
                np.save(score_path, scores)
            all_scores.append(scores)
            frame_length = len(scores) * args.frames_per_snippet
            boundaries.append([key, str(group.iloc[0]["label"]), offset, offset + frame_length, len(scores)])
            offset += frame_length
    prediction = np.repeat(np.concatenate(all_scores), args.frames_per_snippet).astype(np.float32)
    truth = np.load(args.gt_path, allow_pickle=False).astype(np.int64).reshape(-1)
    original_lengths = {"truth": int(len(truth)), "prediction": int(len(prediction))}
    if len(truth) != len(prediction):
        if not args.allow_length_crop:
            raise RuntimeError(
                f"strict frame alignment failed: GT={len(truth)}, prediction={len(prediction)}. "
                "Inspect official feature/test order; use --allow-length-crop only after confirming the cause."
            )
        usable = min(len(truth), len(prediction))
        truth, prediction = truth[:usable], prediction[:usable]
    np.save(output / "frame_scores.npy", prediction)
    np.save(output / "frame_truth.npy", truth)
    write_csv(output / "video_boundaries.csv", ["key", "label", "frame_start", "frame_end", "snippet_count"], boundaries)
    metrics = {
        "method": "linear_readout_on_clip_hidden_states",
        "dataset": args.dataset,
        "feature_mode": state["config"]["feature_mode"],
        "selected_width": len(coordinates),
        "frame_auc": float(roc_auc_score(truth, prediction)),
        "frame_ap": float(average_precision_score(truth, prediction)),
        "frames": int(len(truth)),
        "strict_alignment": not args.allow_length_crop,
        "original_lengths": original_lengths,
        "validation": state.get("validation"),
        "frame_ground_truth_used_for_training_or_selection": False,
    }
    save_json(metrics_path, metrics)
    print(json.dumps(metrics, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
