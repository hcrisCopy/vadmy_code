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


def categories(label: str) -> list[str]:
    text = str(label)
    tokens = [token for token in text.split("-") if token and token != "0"]
    if tokens and all(token == "G" or (token.startswith("B") and token[1:].isdigit()) for token in tokens):
        return sorted(set(tokens))
    return [text]


def bounded_hidden(path: str, maximum_length: int) -> np.ndarray:
    hidden = load_hidden_array(path)
    if len(hidden) > maximum_length:
        indices = np.linspace(0, len(hidden) - 1, maximum_length).round().astype(np.int64)
        hidden = hidden[indices]
    return layer_normalize(hidden)


def main() -> None:
    parser = argparse.ArgumentParser(description="Fit category-conditional normality CLS-neuron experts.")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--active-per-layer", type=int, default=16)
    parser.add_argument("--maximum-length", type=int, default=256)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--clean", action="store_true")
    args = parser.parse_args()
    output = Path(args.out_dir)
    model_path = output / "category_normality_expert.npz"
    if args.clean and output.exists():
        shutil.rmtree(output)
    if args.resume and model_path.exists():
        print(f"reuse {model_path}", flush=True)
        return
    output.mkdir(parents=True, exist_ok=True)
    frame = pd.read_csv(args.manifest)
    normal = frame[frame["binary_label"] == 0]
    category_names = sorted({name for row in frame[frame["binary_label"] == 1].itertuples(index=False) for name in categories(row.label)})
    if normal.empty or not category_names:
        raise ValueError("manifest must contain normal videos and anomaly categories")

    total = np.zeros((12, 768), dtype=np.float64)
    square = np.zeros_like(total)
    snippet_count = 0
    for row in tqdm(normal.itertuples(index=False), total=len(normal), desc="fit category normal moments"):
        hidden = bounded_hidden(str(row.hidden_path), args.maximum_length)
        total += hidden.sum(axis=0)
        square += np.square(hidden, dtype=np.float64).sum(axis=0)
        snippet_count += len(hidden)
    normal_mean = total / max(snippet_count, 1)
    normal_scale = np.sqrt(np.maximum(square / max(snippet_count, 1) - np.square(normal_mean), 1e-4))

    names = ["__normal__", *category_names]
    name_to_index = {name: index for index, name in enumerate(names)}
    sums = np.zeros((len(names), 12, 768), dtype=np.float64)
    squares = np.zeros_like(sums)
    counts = np.zeros(len(names), dtype=np.int64)
    for row in tqdm(frame.itertuples(index=False), total=len(frame), desc="rank category-sensitive neurons"):
        hidden = bounded_hidden(str(row.hidden_path), args.maximum_length)
        deviation = np.abs((hidden - normal_mean) / normal_scale)
        top_count = min(len(hidden), max(1, len(hidden) // 16 + 1))
        summary = np.partition(deviation, len(hidden) - top_count, axis=0)[-top_count:].mean(axis=0)
        row_categories = ["__normal__"] if int(row.binary_label) == 0 else categories(row.label)
        for name in row_categories:
            index = name_to_index[name]
            sums[index] += summary
            squares[index] += np.square(summary, dtype=np.float64)
            counts[index] += 1
    means = sums / counts[:, None, None]
    variances = squares / counts[:, None, None] - np.square(means)
    effects = []
    for index in range(1, len(names)):
        effect = (means[index] - means[0]) / np.sqrt(np.maximum(variances[index] + variances[0], 1e-6))
        effects.append(np.maximum(effect, 0.0))
    effect = np.stack(effects)
    active = min(args.active_per_layer, effect.shape[-1])
    indices = np.argsort(effect, axis=2)[:, :, -active:]
    weights = np.take_along_axis(effect, indices, axis=2)
    weights = weights / np.maximum(weights.mean(axis=2, keepdims=True), 1e-6)
    np.savez_compressed(model_path, normal_mean=normal_mean.astype(np.float32), normal_scale=normal_scale.astype(np.float32), indices=indices.astype(np.int64), weights=weights.astype(np.float32), categories=np.asarray(category_names))
    selected = [{"category": category_names[c], "layer": layer + 1, "dimension": int(dimension), "effect": float(weight)} for c in range(len(category_names)) for layer in range(12) for dimension, weight in zip(indices[c, layer], weights[c, layer])]
    (output / "selected_neurons.json").write_text(json.dumps({"definition": "CLIP ViT-B/16 CLS hidden-state coordinate", "categories": category_names, "neurons": selected}, indent=2), encoding="utf-8")
    print(json.dumps({"categories": len(category_names), "selected_entries": len(selected)}), flush=True)


if __name__ == "__main__":
    main()
