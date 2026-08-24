#!/usr/bin/env python3
"""Select CLIP CLS hidden coordinates without any baseline pseudo scores."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
from tqdm import tqdm


PACKAGE_ROOT = Path(__file__).resolve().parent.parent
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))

from neuron_responsibility.common import (
    base_key,
    clean_output,
    grouped_rows,
    is_normal_label,
    load_hidden,
    read_feature_csv,
    read_hidden_manifest,
    save_json,
    uniform_indices,
    write_csv,
)


def update_welford(
    count: int,
    mean: np.ndarray | None,
    m2: np.ndarray | None,
    samples: np.ndarray,
) -> tuple[int, np.ndarray, np.ndarray]:
    for sample in samples.astype(np.float64, copy=False):
        if mean is None:
            mean = np.zeros_like(sample, dtype=np.float64)
            m2 = np.zeros_like(sample, dtype=np.float64)
        count += 1
        delta = sample - mean
        mean += delta / count
        m2 += delta * (sample - mean)
    assert mean is not None and m2 is not None
    return count, mean, m2


def collect_normal_stats(
    paths: list[str],
    snippets_per_video: int,
) -> tuple[np.ndarray, np.ndarray, int]:
    count = 0
    mean = None
    m2 = None
    for path in tqdm(paths, desc="normal neuron statistics", unit="video"):
        hidden, _ = load_hidden(path)
        indices = uniform_indices(hidden.shape[0], snippets_per_video)
        count, mean, m2 = update_welford(count, mean, m2, hidden[indices])
    if count < 2 or mean is None or m2 is None:
        raise RuntimeError("at least two normal hidden snippets are required")
    variance = m2 / (count - 1)
    return mean.astype(np.float32), np.sqrt(np.maximum(variance, 1e-12)).astype(np.float32), count


def bag_extremes(z_hidden: np.ndarray, top_p: float) -> np.ndarray:
    """Return positive and negative tail evidence with shape [2,L,D]."""
    count = max(1, int(np.ceil(z_hidden.shape[0] * top_p)))
    count = min(count, z_hidden.shape[0])
    positive = np.partition(z_hidden, z_hidden.shape[0] - count, axis=0)[-count:].mean(axis=0)
    negative = -np.partition(z_hidden, count - 1, axis=0)[:count].mean(axis=0)
    return np.stack([positive, negative], axis=0).astype(np.float32)


def cache_bag_statistics(
    key: str,
    hidden_path: str,
    cache_dir: Path,
    normal_mean: np.ndarray,
    normal_std: np.ndarray,
    top_p: float,
    sigma_min: float,
) -> np.ndarray:
    cache_path = cache_dir / f"{key}.npy"
    if cache_path.exists():
        cached = np.load(cache_path).astype(np.float32)
        if cached.shape == (2, *normal_mean.shape):
            return cached
    hidden, _ = load_hidden(hidden_path)
    if hidden.shape[1:] != normal_mean.shape:
        raise ValueError(f"{hidden_path}: hidden shape {hidden.shape[1:]} != {normal_mean.shape}")
    extremes = bag_extremes((hidden - normal_mean) / (normal_std + sigma_min), top_p)
    np.save(cache_path, extremes)
    return extremes


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Baseline-independent sparse CLIP-neuron selection from video-level labels."
    )
    parser.add_argument("--dataset", choices=["ucf", "xd"], required=True)
    parser.add_argument("--source-train-csv", required=True)
    parser.add_argument("--hidden-manifest", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--top-p", type=float, default=0.10)
    parser.add_argument("--topk-global", type=int, default=768)
    parser.add_argument("--normal-stat-snippets-per-video", type=int, default=256)
    parser.add_argument("--sigma-min", type=float, default=1e-6)
    parser.add_argument("--clean", action="store_true")
    args = parser.parse_args()

    if not 0.0 < args.top_p <= 0.5:
        parser.error("--top-p must be in (0, 0.5]")
    if args.topk_global <= 0 or args.normal_stat_snippets_per_video <= 0 or args.sigma_min <= 0:
        parser.error("top-k, sample count and sigma must be positive")

    out_dir = clean_output(args.out_dir, args.clean)
    cache_dir = out_dir / "bag_cache"
    cache_dir.mkdir(exist_ok=True)
    groups = grouped_rows(read_feature_csv(args.source_train_csv))
    hidden_by_key, token_pool = read_hidden_manifest(args.hidden_manifest)

    labels: dict[str, str] = {}
    missing_rows = []
    for key, group in groups.items():
        unique_labels = set(group["label"].astype(str))
        if len(unique_labels) != 1:
            raise ValueError(f"{key}: inconsistent labels {sorted(unique_labels)}")
        labels[key] = next(iter(unique_labels))
        if key not in hidden_by_key:
            missing_rows.append([key, labels[key], "missing_hidden"])

    normal_keys = [key for key, label in labels.items() if is_normal_label(args.dataset, label) and key in hidden_by_key]
    abnormal_keys = [key for key, label in labels.items() if not is_normal_label(args.dataset, label) and key in hidden_by_key]
    if not normal_keys or not abnormal_keys:
        raise RuntimeError("selection requires matched normal and abnormal training videos")
    print(
        f"matched normal={len(normal_keys)} abnormal={len(abnormal_keys)} missing={len(missing_rows)}",
        flush=True,
    )

    normal_mean, normal_std, normal_count = collect_normal_stats(
        [hidden_by_key[key] for key in normal_keys], args.normal_stat_snippets_per_video
    )
    np.save(out_dir / "normal_mean.npy", normal_mean)
    np.save(out_dir / "normal_std.npy", normal_std)

    normal_bags = []
    for key in tqdm(normal_keys, desc="normal bag evidence", unit="video"):
        normal_bags.append(cache_bag_statistics(
            key, hidden_by_key[key], cache_dir, normal_mean, normal_std, args.top_p, args.sigma_min
        ))
    abnormal_bags = []
    for key in tqdm(abnormal_keys, desc="abnormal bag evidence", unit="video"):
        abnormal_bags.append(cache_bag_statistics(
            key, hidden_by_key[key], cache_dir, normal_mean, normal_std, args.top_p, args.sigma_min
        ))

    normal_array = np.stack(normal_bags).astype(np.float64)
    abnormal_array = np.stack(abnormal_bags).astype(np.float64)
    normal_tail_mean = normal_array.mean(axis=0)
    abnormal_tail_mean = abnormal_array.mean(axis=0)
    delta = abnormal_tail_mean - normal_tail_mean
    pooled_variance = normal_array.var(axis=0, ddof=1) + abnormal_array.var(axis=0, ddof=1)
    stability = delta / (np.sqrt(np.maximum(pooled_variance, 0.0)) + args.sigma_min)

    direction_index = np.argmax(stability, axis=0)
    layer_indices, dimensions = np.indices(normal_mean.shape)
    chosen_score = np.take_along_axis(stability, direction_index[None, ...], axis=0)[0]
    total = chosen_score.size
    if args.topk_global > total:
        raise ValueError(f"--topk-global={args.topk_global} exceeds available neurons={total}")
    flat = np.argsort(-chosen_score.reshape(-1), kind="mergesort")[:args.topk_global]

    selected = []
    for layer in range(normal_mean.shape[0]):
        layer_flat = flat[layer_indices.reshape(-1)[flat] == layer]
        if not len(layer_flat):
            continue
        dims = dimensions.reshape(-1)[layer_flat]
        dirs = direction_index.reshape(-1)[layer_flat]
        selected.append({
            "layer_index": int(layer),
            "dims": dims.astype(int).tolist(),
            "directions": np.where(dirs == 0, 1, -1).astype(int).tolist(),
            "scores": chosen_score.reshape(-1)[layer_flat].astype(float).tolist(),
        })

    np.save(out_dir / "tail_delta.npy", delta.astype(np.float32))
    np.save(out_dir / "tail_stability.npy", stability.astype(np.float32))
    metadata = {
        "method": "baseline_independent_bag_tail_v1",
        "dataset": args.dataset,
        "token_pool": token_pool,
        "hidden_shape": list(normal_mean.shape),
        "neuron_width": int(args.topk_global),
        "top_p": float(args.top_p),
        "sigma_min": float(args.sigma_min),
        "normal_stat_snippet_count": int(normal_count),
        "normal_video_count": len(normal_keys),
        "abnormal_video_count": len(abnormal_keys),
        "normal_mean_path": "normal_mean.npy",
        "normal_std_path": "normal_std.npy",
        "selected": selected,
    }
    save_json(out_dir / "selected_neurons.json", metadata)
    write_csv(out_dir / "skipped_videos.csv", ["key", "label", "reason"], missing_rows)
    print(f"wrote {out_dir / 'selected_neurons.json'} with {args.topk_global} neurons", flush=True)


if __name__ == "__main__":
    main()
