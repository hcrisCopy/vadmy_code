from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

from universal_neuron_adapter.data import load_hidden_array
from universal_neuron_adapter.fit_normality_expert import bounded_hidden


def greedy_mrmr(relevance: np.ndarray, features: np.ndarray, count: int, penalty: float) -> np.ndarray:
    """Select relevant coordinates while penalizing maximum pairwise redundancy."""
    standardized = (features - features.mean(axis=0)) / np.maximum(features.std(axis=0), 1e-6)
    correlation = np.abs(standardized.T @ standardized / max(len(features), 1))
    np.fill_diagonal(correlation, 0.0)
    normalized_relevance = relevance / max(float(relevance.max()), 1e-8)
    selected = [int(np.argmax(normalized_relevance))]
    available = np.ones(len(relevance), dtype=bool)
    available[selected[0]] = False
    while len(selected) < count:
        redundancy = correlation[:, selected].max(axis=1)
        objective = normalized_relevance - penalty * redundancy
        objective[~available] = -np.inf
        choice = int(np.argmax(objective))
        selected.append(choice)
        available[choice] = False
    return np.asarray(selected, dtype=np.int64)


def main() -> None:
    parser = argparse.ArgumentParser(description="Fit a deterministic mRMR CLS-neuron expert.")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--active-per-layer", type=int, default=32)
    parser.add_argument("--redundancy-penalty", type=float, default=0.25)
    parser.add_argument("--maximum-length", type=int, default=256)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--clean", action="store_true")
    args = parser.parse_args()
    output = Path(args.out_dir)
    model_path = output / "mrmr_expert.npz"
    if args.clean and output.exists():
        shutil.rmtree(output)
    if args.resume and model_path.exists():
        print(f"reuse {model_path}", flush=True)
        return
    output.mkdir(parents=True, exist_ok=True)
    frame = pd.read_csv(args.manifest)
    normal = frame[frame["binary_label"] == 0]
    if normal.empty or len(normal) == len(frame):
        raise ValueError("manifest must contain normal and abnormal training videos")

    total = np.zeros((12, 768), dtype=np.float64)
    square = np.zeros_like(total)
    normal_count = 0
    for row in tqdm(normal.itertuples(index=False), total=len(normal), desc="fit normal moments"):
        hidden = bounded_hidden(str(row.hidden_path), args.maximum_length)
        total += hidden.sum(axis=0)
        square += np.square(hidden, dtype=np.float64).sum(axis=0)
        normal_count += len(hidden)
    normal_mean = total / normal_count
    normal_scale = np.sqrt(np.maximum(square / normal_count - np.square(normal_mean), 1e-4))

    summaries, labels = [], []
    for row in tqdm(frame.itertuples(index=False), total=len(frame), desc="summarize directional neurons"):
        hidden = bounded_hidden(str(row.hidden_path), args.maximum_length)
        deviation = (hidden - normal_mean) / normal_scale
        top_count = min(len(hidden), max(1, len(hidden) // 16 + 1))
        upper = np.partition(deviation, len(hidden) - top_count, axis=0)[-top_count:].mean(axis=0)
        lower = -np.partition(deviation, top_count - 1, axis=0)[:top_count].mean(axis=0)
        summaries.append(np.stack([upper, lower]).astype(np.float32))
        labels.append(int(row.binary_label))
    summaries = np.stack(summaries)
    labels = np.asarray(labels)
    normal_summary = summaries[labels == 0]
    abnormal_summary = summaries[labels == 1]
    means = np.stack([normal_summary.mean(axis=0), abnormal_summary.mean(axis=0)])
    variances = np.stack([normal_summary.var(axis=0), abnormal_summary.var(axis=0)])
    effect = np.maximum((means[1] - means[0]) / np.sqrt(np.maximum(variances.sum(axis=0), 1e-6)), 0.0)
    directions = np.argmax(effect, axis=0)
    relevance = np.take_along_axis(effect, directions[None], axis=0)[0]

    indices = []
    for layer in tqdm(range(12), desc="select mRMR neurons"):
        directional_features = np.where(
            directions[layer][None] == 0,
            summaries[:, 0, layer],
            summaries[:, 1, layer],
        )
        indices.append(greedy_mrmr(
            relevance[layer], directional_features, args.active_per_layer, args.redundancy_penalty
        ))
    indices = np.stack(indices)
    selected_directions = np.take_along_axis(directions, indices, axis=1)
    weights = np.take_along_axis(relevance, indices, axis=1)
    weights /= np.maximum(weights.mean(axis=1, keepdims=True), 1e-6)
    np.savez_compressed(
        model_path,
        normal_mean=normal_mean.astype(np.float32),
        normal_scale=normal_scale.astype(np.float32),
        indices=indices,
        directions=selected_directions.astype(np.int64),
        weights=weights.astype(np.float32),
    )
    selected = [{"layer": layer + 1, "dimension": int(dimension), "direction": int(direction), "weight": float(weight)} for layer in range(12) for dimension, direction, weight in zip(indices[layer], selected_directions[layer], weights[layer])]
    (output / "selected_neurons.json").write_text(json.dumps({"definition": "mRMR CLS coordinates", "neurons": selected}, indent=2), encoding="utf-8")
    print(json.dumps({"selected_neurons": len(selected), "redundancy_penalty": args.redundancy_penalty}), flush=True)


if __name__ == "__main__":
    main()
