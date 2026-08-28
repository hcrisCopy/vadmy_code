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
from universal_neuron_adapter.temporal_information import local_temporal_contrast, temporal_surprise


def bounded_hidden(path: str, maximum_length: int) -> np.ndarray:
    hidden = load_hidden_array(path)
    if len(hidden) > maximum_length:
        positions = np.linspace(0, len(hidden) - 1, maximum_length).round().astype(np.int64)
        hidden = hidden[positions]
    return hidden


def greedy_logdet(kernel: np.ndarray, count: int) -> np.ndarray:
    """Deterministic pivoted-Cholesky MAP approximation for fixed-size DPP selection."""
    size = len(kernel)
    count = min(int(count), size)
    residual = np.diag(kernel).astype(np.float64).copy()
    factors = np.zeros((count, size), dtype=np.float64)
    selected: list[int] = []
    available = np.ones(size, dtype=bool)
    for step in range(count):
        scores = np.where(available, residual, -np.inf)
        pivot = int(np.argmax(scores))
        selected.append(pivot)
        available[pivot] = False
        denominator = np.sqrt(max(float(residual[pivot]), 1e-12))
        projection = factors[:step, pivot] @ factors[:step] if step else 0.0
        row = (kernel[pivot] - projection) / denominator
        factors[step] = row
        residual = np.maximum(residual - np.square(row), 0.0)
    return np.asarray(selected, dtype=np.int64)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Fit a deterministic log-determinant temporal CLS-neuron expert."
    )
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--active-per-layer", type=int, default=32)
    parser.add_argument("--candidate-per-layer", type=int, default=128)
    parser.add_argument("--scales", type=int, nargs="+", default=[1, 2, 4])
    parser.add_argument("--maximum-length", type=int, default=256)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--clean", action="store_true")
    args = parser.parse_args()

    if args.active_per_layer < 1 or args.candidate_per_layer < args.active_per_layer:
        raise ValueError("candidate-per-layer must be at least active-per-layer, both positive")
    if args.maximum_length < 1 or not args.scales or min(args.scales) < 1:
        raise ValueError("maximum-length and every temporal scale must be positive")

    output = Path(args.out_dir)
    model_path = output / "temporal_information_expert.npz"
    if args.clean and output.exists():
        shutil.rmtree(output)
    if args.resume and model_path.exists():
        print(f"reuse {model_path}", flush=True)
        return
    output.mkdir(parents=True, exist_ok=True)

    frame = pd.read_csv(args.manifest)
    normal = frame[frame["binary_label"] == 0]
    if normal.empty or len(normal) == len(frame):
        raise ValueError("manifest must contain both normal and abnormal training videos")

    scales = np.asarray(args.scales, dtype=np.int64)
    total = np.zeros((len(scales), 12, 768), dtype=np.float64)
    square = np.zeros_like(total)
    snippet_count = 0
    for row in tqdm(normal.itertuples(index=False), total=len(normal), desc="fit normal temporal moments"):
        normalized = layer_normalize(bounded_hidden(str(row.hidden_path), args.maximum_length))
        for position, scale in enumerate(scales.tolist()):
            contrast = local_temporal_contrast(normalized, int(scale))
            total[position] += contrast.sum(axis=0)
            square[position] += np.square(contrast, dtype=np.float64).sum(axis=0)
        snippet_count += len(normalized)
    normal_mean = total / max(snippet_count, 1)
    normal_variance = square / max(snippet_count, 1) - np.square(normal_mean)
    normal_scale = np.sqrt(np.maximum(normal_variance, 1e-4))
    provisional = {
        "scales": scales,
        "normal_mean": normal_mean.astype(np.float32),
        "normal_scale": normal_scale.astype(np.float32),
    }

    summaries, labels = [], []
    for row in tqdm(frame.itertuples(index=False), total=len(frame), desc="summarize temporal neurons"):
        surprise = temporal_surprise(
            bounded_hidden(str(row.hidden_path), args.maximum_length), provisional
        )
        top_count = min(len(surprise), max(1, len(surprise) // 16 + 1))
        summary = np.partition(surprise, len(surprise) - top_count, axis=0)[-top_count:].mean(axis=0)
        summaries.append(summary.astype(np.float32))
        labels.append(int(row.binary_label))
    responses = np.stack(summaries)
    targets = np.asarray(labels, dtype=np.int64)

    means = np.stack([responses[targets == label].mean(axis=0) for label in (0, 1)])
    variances = np.stack([responses[targets == label].var(axis=0) for label in (0, 1)])
    effect = (means[1] - means[0]) / np.sqrt(np.maximum(variances.sum(axis=0), 1e-6))
    effect = np.maximum(effect, 0.0)

    selected_indices, selected_weights = [], []
    selected_rows = []
    for layer in range(12):
        candidate_count = min(args.candidate_per_layer, effect.shape[1])
        candidates = np.argsort(effect[layer])[-candidate_count:]
        values = responses[:, layer, candidates].astype(np.float64)
        values = (values - values.mean(axis=0)) / np.maximum(values.std(axis=0), 1e-6)
        gram = values.T @ values / max(len(values), 1)
        diagonal = np.sqrt(np.maximum(np.diag(gram), 1e-8))
        gram = gram / np.maximum(diagonal[:, None] * diagonal[None], 1e-8)
        quality = 1.0 + effect[layer, candidates] / max(float(effect[layer, candidates].max()), 1e-6)
        kernel = np.outer(quality, quality) * gram
        kernel = 0.5 * (kernel + kernel.T) + np.eye(candidate_count) * 1e-5
        positions = greedy_logdet(kernel, args.active_per_layer)
        dimensions = candidates[positions]
        weights = effect[layer, dimensions]
        weights = weights / max(float(weights.mean()), 1e-6)
        selected_indices.append(dimensions)
        selected_weights.append(weights)
        for rank, (dimension, weight) in enumerate(zip(dimensions, weights), start=1):
            selected_rows.append(
                {
                    "layer": layer + 1,
                    "rank": rank,
                    "dimension": int(dimension),
                    "effect_weight": float(weight),
                }
            )

    indices = np.stack(selected_indices).astype(np.int64)
    weights = np.stack(selected_weights).astype(np.float32)
    np.savez_compressed(
        model_path,
        scales=scales,
        normal_mean=normal_mean.astype(np.float32),
        normal_scale=normal_scale.astype(np.float32),
        indices=indices,
        weights=weights,
    )
    (output / "selected_neurons.json").write_text(
        json.dumps(
            {
                "definition": "CLIP ViT-B/16 CLS hidden-state coordinate",
                "selection": "training-only relevance plus fixed-size log-determinant diversity",
                "active_per_layer": int(args.active_per_layer),
                "scales": scales.tolist(),
                "neurons": selected_rows,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "videos": len(frame),
                "normal_snippets": snippet_count,
                "selected_neurons": int(indices.size),
                "seed": "deterministic",
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()

