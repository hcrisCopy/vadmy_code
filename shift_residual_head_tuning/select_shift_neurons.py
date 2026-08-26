#!/usr/bin/env python3
"""Reproduce Shift-Global768 top/bottom neuron selection."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm


PACKAGE_ROOT = Path(__file__).resolve().parent.parent
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))

from neuron_responsibility.common import (
    clean_output, grouped_rows, is_normal_label, load_hidden, read_feature_csv,
    read_hidden_manifest, resample_feature, save_json, uniform_indices, write_csv,
)


def normal_statistics(paths: list[str], snippets_per_video: int) -> tuple[np.ndarray, np.ndarray, int]:
    count, mean, m2 = 0, None, None
    for path in tqdm(paths, desc="normal z-score statistics", unit="video"):
        hidden, _ = load_hidden(path)
        for sample in hidden[uniform_indices(len(hidden), snippets_per_video)].astype(np.float64):
            if mean is None:
                mean = np.zeros_like(sample)
                m2 = np.zeros_like(sample)
            count += 1
            difference = sample - mean
            mean += difference / count
            m2 += difference * (sample - mean)
    if count < 2 or mean is None or m2 is None:
        raise RuntimeError("at least two normal snippets are required")
    return mean.astype(np.float32), np.sqrt(np.maximum(m2 / (count - 1), 1e-12)).astype(np.float32), count


def paired_indices(scores: np.ndarray, top_p: float) -> tuple[np.ndarray, np.ndarray]:
    count = max(1, int(np.ceil(len(scores) * top_p)))
    count = min(count, max(1, len(scores) // 2))
    order = np.argsort(scores, kind="mergesort")
    return order[-count:][::-1], order[:count]


def main() -> None:
    parser = argparse.ArgumentParser(description="Select 64 neurons per CLIP layer from within-video score tails.")
    parser.add_argument("--dataset", choices=["ucf", "xd"], required=True)
    parser.add_argument("--source-train-csv", required=True)
    parser.add_argument("--hidden-manifest", required=True)
    parser.add_argument("--pseudo-csv", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--top-p", type=float, default=0.10)
    parser.add_argument("--topk-per-layer", type=int, default=64)
    parser.add_argument("--normal-stat-snippets-per-video", type=int, default=256)
    parser.add_argument("--sigma-min", type=float, default=1e-6)
    parser.add_argument("--clean", action="store_true")
    args = parser.parse_args()
    if not 0.0 < args.top_p <= 0.5:
        parser.error("--top-p must be in (0, 0.5]")
    if min(args.topk_per_layer, args.normal_stat_snippets_per_video, args.sigma_min) <= 0:
        parser.error("top-k, snippet count, and sigma must be positive")

    out_dir = clean_output(args.out_dir, args.clean)
    complete = out_dir / "selected_neurons.json"
    if complete.exists() and not args.clean:
        print(f"reuse completed selection: {complete}", flush=True)
        return
    groups = grouped_rows(read_feature_csv(args.source_train_csv))
    hidden_by_key, token_pool = read_hidden_manifest(args.hidden_manifest)
    pseudo_frame = pd.read_csv(args.pseudo_csv)
    required = {"key", "label", "score_path"}
    if required - set(pseudo_frame.columns):
        raise ValueError(f"{args.pseudo_csv}: missing {sorted(required - set(pseudo_frame.columns))}")
    pseudo = {str(row["key"]): (str(row["label"]), str(row["score_path"])) for _, row in pseudo_frame.iterrows()}

    labels = {}
    for key, group in groups.items():
        values = set(group["label"].astype(str))
        if len(values) != 1:
            raise ValueError(f"{key}: inconsistent labels {sorted(values)}")
        labels[key] = next(iter(values))
    normal_keys = [key for key, label in labels.items() if is_normal_label(args.dataset, label) and key in hidden_by_key]
    abnormal_keys = [key for key, label in labels.items() if not is_normal_label(args.dataset, label)]
    if not normal_keys:
        raise RuntimeError("no matched normal videos for z-score statistics")
    normal_mean, normal_std, normal_count = normal_statistics(
        [hidden_by_key[key] for key in normal_keys], args.normal_stat_snippets_per_video
    )
    if normal_mean.shape != (12, 768):
        raise ValueError(
            f"Shift-Global768 requires 12 CLIP ViT-B/16 layers x 768 hidden dimensions, got {normal_mean.shape}"
        )
    np.save(out_dir / "normal_mean.npy", normal_mean)
    np.save(out_dir / "normal_std.npy", normal_std)

    delta_dir = out_dir / "per_video_deltas"
    delta_dir.mkdir(exist_ok=True)
    deltas, pair_rows, skipped = [], [], []
    for key in tqdm(abnormal_keys, desc="video-internal top/bottom deltas", unit="video"):
        label = labels[key]
        if key not in hidden_by_key:
            skipped.append([key, label, "missing_hidden"])
            continue
        if key not in pseudo:
            skipped.append([key, label, "missing_pseudo_score"])
            continue
        pseudo_label, score_path = pseudo[key]
        if pseudo_label != label:
            raise ValueError(f"{key}: pseudo/source labels differ")
        cache_path = delta_dir / f"{key}.npz"
        if cache_path.exists() and not args.clean:
            cached = np.load(cache_path, allow_pickle=False)
            delta = cached["delta"].astype(np.float32)
            pair_count = int(cached["pair_count"].item())
            top_mean = float(cached["top_score_mean"].item())
            bottom_mean = float(cached["bottom_score_mean"].item())
            hidden_length = int(cached["hidden_length"].item())
            status = "reused"
        else:
            hidden, _ = load_hidden(hidden_by_key[key])
            raw_scores = np.load(score_path).astype(np.float32).reshape(-1)
            scores = resample_feature(raw_scores[:, None], len(hidden))[:, 0]
            top, bottom = paired_indices(scores, args.top_p)
            z_hidden = (hidden - normal_mean) / (normal_std + args.sigma_min)
            delta = (z_hidden[top].mean(axis=0) - z_hidden[bottom].mean(axis=0)).astype(np.float32)
            pair_count, top_mean, bottom_mean, hidden_length = len(top), float(scores[top].mean()), float(scores[bottom].mean()), len(hidden)
            np.savez_compressed(
                cache_path, delta=delta, pair_count=np.asarray(pair_count), hidden_length=np.asarray(hidden_length),
                top_score_mean=np.asarray(top_mean), bottom_score_mean=np.asarray(bottom_mean),
            )
            status = "computed"
        if delta.shape != normal_mean.shape or not np.isfinite(delta).all():
            raise RuntimeError(f"{key}: invalid cached delta {delta.shape}")
        deltas.append(delta)
        pair_rows.append([key, label, pair_count, hidden_length, top_mean, bottom_mean, status])
    if len(deltas) < 2:
        raise RuntimeError("at least two abnormal-video deltas are required")

    delta_array = np.stack(deltas)
    mean_delta = delta_array.mean(axis=0)
    std_delta = delta_array.std(axis=0, ddof=1)
    shift_score = np.abs(mean_delta) / (std_delta + args.sigma_min)
    if args.topk_per_layer > shift_score.shape[1]:
        raise ValueError("--topk-per-layer exceeds hidden width")
    selected = []
    selected_mask = np.zeros_like(shift_score, dtype=np.uint8)
    for layer in range(shift_score.shape[0]):
        dims = np.argsort(-shift_score[layer], kind="mergesort")[:args.topk_per_layer]
        selected_mask[layer, dims] = 1
        selected.append({
            "layer_index": layer,
            "dims": dims.astype(int).tolist(),
            "scores": shift_score[layer, dims].astype(float).tolist(),
            "mean_deltas": mean_delta[layer, dims].astype(float).tolist(),
        })
    np.save(out_dir / "mean_delta.npy", mean_delta.astype(np.float32))
    np.save(out_dir / "std_delta.npy", std_delta.astype(np.float32))
    np.save(out_dir / "shift_scores.npy", shift_score.astype(np.float32))
    np.save(out_dir / "selected_mask.npy", selected_mask)
    write_csv(out_dir / "video_pairs.csv", ["key", "label", "paired_count", "hidden_length", "top_score_mean", "bottom_score_mean", "status"], pair_rows)
    write_csv(out_dir / "skipped_videos.csv", ["key", "label", "reason"], skipped)
    save_json(complete, {
        "method": "intravideo_paired_shift_global768",
        "dataset": args.dataset,
        "positive_definition": "top 10% frozen-baseline-score snippets within each abnormal training video",
        "negative_definition": "bottom 10% frozen-baseline-score snippets from the same abnormal training video",
        "baseline_score_dependency": True,
        "frame_ground_truth_used": False,
        "token_pool": token_pool,
        "top_p": args.top_p,
        "topk_per_layer": args.topk_per_layer,
        "neuron_width": int(len(selected) * args.topk_per_layer),
        "sigma_min": args.sigma_min,
        "normal_stat_snippet_count": normal_count,
        "normal_mean_path": "normal_mean.npy",
        "normal_std_path": "normal_std.npy",
        "shift_scores_path": "shift_scores.npy",
        "selected_mask_path": "selected_mask.npy",
        "selected": selected,
    })
    print(f"wrote {complete}: {len(selected)} layers x {args.topk_per_layer} = {len(selected) * args.topk_per_layer} neurons", flush=True)


if __name__ == "__main__":
    main()
