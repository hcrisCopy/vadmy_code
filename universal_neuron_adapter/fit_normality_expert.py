from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

from universal_neuron_adapter.data import load_hidden_array
from universal_neuron_adapter.normality import layer_normalize


def bounded_hidden(path: str, maximum_length: int) -> np.ndarray:
    hidden = load_hidden_array(path)
    if len(hidden) > maximum_length:
        indices = np.linspace(0, len(hidden) - 1, maximum_length).round().astype(np.int64)
        hidden = hidden[indices]
    return layer_normalize(hidden)


def main() -> None:
    parser = argparse.ArgumentParser(description="Fit a baseline-independent normality CLS-neuron expert.")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--active-per-layer", type=int, default=32)
    parser.add_argument("--maximum-length", type=int, default=256)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--clean", action="store_true")
    args = parser.parse_args()

    output = Path(args.out_dir)
    model_path = output / "normality_expert.npz"
    if args.clean and output.exists():
        shutil.rmtree(output)
    if args.resume and model_path.exists():
        print(f"reuse {model_path}", flush=True)
        return
    output.mkdir(parents=True, exist_ok=True)
    frame = pd.read_csv(args.manifest)
    normal = frame[frame["binary_label"] == 0]
    if normal.empty or len(normal) == len(frame):
        raise ValueError("manifest must contain both normal and abnormal videos")

    total = np.zeros((12, 768), dtype=np.float64)
    square = np.zeros_like(total)
    count = 0
    for row in tqdm(normal.itertuples(index=False), total=len(normal), desc="fit normal neuron moments"):
        hidden = bounded_hidden(str(row.hidden_path), args.maximum_length)
        total += hidden.sum(axis=0)
        square += np.square(hidden, dtype=np.float64).sum(axis=0)
        count += len(hidden)
    normal_mean = total / max(count, 1)
    normal_variance = square / max(count, 1) - np.square(normal_mean)
    normal_scale = np.sqrt(np.maximum(normal_variance, 1e-4))

    class_sum = np.zeros((2, 2, 12, 768), dtype=np.float64)
    class_square = np.zeros_like(class_sum)
    class_count = np.zeros(2, dtype=np.int64)
    for row in tqdm(frame.itertuples(index=False), total=len(frame), desc="rank abnormality-sensitive neurons"):
        hidden = bounded_hidden(str(row.hidden_path), args.maximum_length)
        deviation = (hidden - normal_mean) / normal_scale
        top_count = min(len(hidden), max(1, len(hidden) // 16 + 1))
        upper = np.partition(deviation, len(hidden) - top_count, axis=0)[-top_count:].mean(axis=0)
        lower = -np.partition(deviation, top_count - 1, axis=0)[:top_count].mean(axis=0)
        summary = np.stack([upper, lower])
        label = int(row.binary_label)
        class_sum[label] += summary
        class_square[label] += np.square(summary, dtype=np.float64)
        class_count[label] += 1
    means = class_sum / class_count[:, None, None, None]
    variances = class_square / class_count[:, None, None, None] - np.square(means)
    effect = (means[1] - means[0]) / np.sqrt(np.maximum(variances[0] + variances[1], 1e-6))
    effect = np.maximum(effect, 0.0)
    directions = np.argmax(effect, axis=0)
    directional_effect = np.take_along_axis(effect, directions[None], axis=0)[0]
    active = min(args.active_per_layer, directional_effect.shape[1])
    indices = np.argsort(directional_effect, axis=1)[:, -active:]
    selected_directions = np.take_along_axis(directions, indices, axis=1)
    weights = np.take_along_axis(directional_effect, indices, axis=1)
    weights = weights / np.maximum(weights.mean(axis=1, keepdims=True), 1e-6)
    np.savez_compressed(
        model_path,
        normal_mean=normal_mean.astype(np.float32),
        normal_scale=normal_scale.astype(np.float32),
        indices=indices.astype(np.int64),
        directions=selected_directions.astype(np.int64),
        weights=weights.astype(np.float32),
    )
    selected = [
        {
            "layer": layer + 1,
            "dimension": int(dimension),
            "direction": "above_normal" if int(direction) == 0 else "below_normal",
            "effect": float(weight),
        }
        for layer in range(12)
        for dimension, direction, weight in zip(indices[layer], selected_directions[layer], weights[layer])
    ]
    (output / "selected_neurons.json").write_text(
        json.dumps({"definition": "CLIP ViT-B/16 CLS hidden-state coordinate", "neurons": selected}, indent=2),
        encoding="utf-8",
    )
    print(json.dumps({"videos": len(frame), "normal_snippets": count, "selected_neurons": len(selected)}), flush=True)


if __name__ == "__main__":
    main()
