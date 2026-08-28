from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

from universal_neuron_adapter.category_neurons import atomic_categories
from universal_neuron_adapter.data import load_hidden_array
from universal_neuron_adapter.normality import layer_normalize, load_normality_model


def main() -> None:
    parser = argparse.ArgumentParser(description="Fit a category-conditional Fisher CLS-neuron bank.")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--normality-model", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--active-per-layer", type=int, default=16)
    parser.add_argument("--maximum-length", type=int, default=256)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    output = Path(args.out_dir)
    output.mkdir(parents=True, exist_ok=True)
    model_path = output / "category_expert.npz"
    if args.resume and model_path.exists():
        print(f"reuse {model_path}", flush=True)
        return
    frame = pd.read_csv(args.manifest)
    categories = sorted(
        {category for row in frame.itertuples(index=False) for category in atomic_categories(row.label, int(row.binary_label))}
    )
    if not categories:
        raise ValueError("training manifest contains no abnormal categories")
    category_index = {name: index for index, name in enumerate(categories)}
    normality = load_normality_model(args.normality_model)
    sums = np.zeros((len(categories) + 1, 2, 12, 768), dtype=np.float64)
    squares = np.zeros_like(sums)
    counts = np.zeros(len(categories) + 1, dtype=np.int64)
    for row in tqdm(frame.itertuples(index=False), total=len(frame), desc="fit category Fisher neurons"):
        hidden = load_hidden_array(str(row.hidden_path))
        if len(hidden) > args.maximum_length:
            positions = np.linspace(0, len(hidden) - 1, args.maximum_length).round().astype(np.int64)
            hidden = hidden[positions]
        normalized = layer_normalize(hidden)
        deviation = (normalized - normality["normal_mean"]) / normality["normal_scale"]
        top_count = min(len(deviation), max(1, len(deviation) // 16 + 1))
        upper = np.partition(deviation, len(deviation) - top_count, axis=0)[-top_count:].mean(axis=0)
        lower = -np.partition(deviation, top_count - 1, axis=0)[:top_count].mean(axis=0)
        summary = np.stack([upper, lower])
        memberships = [0] if not int(row.binary_label) else [category_index[name] + 1 for name in atomic_categories(row.label, 1)]
        for membership in memberships:
            sums[membership] += summary
            squares[membership] += np.square(summary, dtype=np.float64)
            counts[membership] += 1
    if counts[0] == 0 or np.any(counts[1:] == 0):
        raise ValueError("every category and the normal class require training videos")
    means = sums / counts[:, None, None, None]
    variances = squares / counts[:, None, None, None] - np.square(means)
    effect = (means[1:] - means[0]) / np.sqrt(np.maximum(variances[1:] + variances[0], 1e-6))
    effect = np.maximum(effect, 0.0)
    directions = np.argmax(effect, axis=1)
    directional = np.take_along_axis(effect, directions[:, None], axis=1)[:, 0]
    active = min(args.active_per_layer, directional.shape[-1])
    indices = np.argsort(directional, axis=2)[:, :, -active:]
    selected_directions = np.take_along_axis(directions, indices, axis=2)
    weights = np.take_along_axis(directional, indices, axis=2)
    weights = weights / np.maximum(weights.mean(axis=2, keepdims=True), 1e-6)
    np.savez_compressed(
        model_path,
        normal_mean=normality["normal_mean"].astype(np.float32),
        normal_scale=normality["normal_scale"].astype(np.float32),
        categories=np.asarray(categories),
        indices=indices.astype(np.int64),
        directions=selected_directions.astype(np.int64),
        weights=weights.astype(np.float32),
    )
    report = {
        "definition": "category-conditional Fisher effects of CLIP ViT-B/16 CLS coordinates",
        "categories": categories,
        "active_per_layer": active,
        "training_counts": {"normal": int(counts[0]), **{name: int(counts[index + 1]) for index, name in enumerate(categories)}},
    }
    (output / "selected_neurons.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
