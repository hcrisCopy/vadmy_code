#!/usr/bin/env python3
"""Select score-free temporal-responsibility neurons from cached CLS states."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

PACKAGE_ROOT = Path(__file__).resolve().parent.parent
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))

from neuron_responsibility.common import clean_output, is_normal_label, load_hidden, save_json, uniform_indices


def representatives(csv_path: str) -> pd.DataFrame:
    frame = pd.read_csv(csv_path)
    missing = {"hidden_path", "label", "key"} - set(frame.columns)
    if missing:
        raise ValueError(f"{csv_path} is missing columns: {sorted(missing)}")
    return frame.groupby("key", sort=True).first().reset_index()


def sample_hidden(path: str, snippets: int) -> np.ndarray:
    hidden, _ = load_hidden(path)
    return hidden[uniform_indices(len(hidden), snippets)].astype(np.float32)


def neighbor_innovation(hidden: np.ndarray) -> np.ndarray:
    if len(hidden) == 1:
        return np.zeros_like(hidden)
    left = np.concatenate([hidden[:1], hidden[:-1]], axis=0)
    right = np.concatenate([hidden[1:], hidden[-1:]], axis=0)
    return np.abs(hidden - 0.5 * (left + right))


def moments(rows: pd.DataFrame, snippets: int, description: str) -> tuple[np.ndarray, np.ndarray, int]:
    total = square = None
    count = 0
    for _, row in tqdm(rows.iterrows(), total=len(rows), desc=description, unit="video"):
        value = neighbor_innovation(sample_hidden(str(row["hidden_path"]), snippets)).astype(np.float64)
        if total is None:
            total = np.zeros(value.shape[1:], dtype=np.float64)
            square = np.zeros_like(total)
        total += value.sum(axis=0); square += np.square(value).sum(axis=0); count += len(value)
    if total is None or square is None or count < 2:
        raise RuntimeError("insufficient normal videos for temporal statistics")
    center = total / count
    scale = np.sqrt(np.maximum(square / count - np.square(center), 1e-12))
    return center.astype(np.float32), scale.astype(np.float32), count


def top_fraction_mean(value: np.ndarray, fraction: float) -> np.ndarray:
    count = max(1, int(round(len(value) * fraction)))
    split = max(0, len(value) - count)
    return np.partition(value, split, axis=0)[split:].mean(axis=0)


def evidence_statistics(
    rows: pd.DataFrame,
    dataset: str,
    center: np.ndarray,
    scale: np.ndarray,
    innovation_center: np.ndarray,
    innovation_scale: np.ndarray,
    snippets: int,
    top_fraction: float,
) -> tuple[dict[str, np.ndarray], dict[str, int]]:
    shape = center.shape
    accumulators = {
        name: {"normal": np.zeros(shape, dtype=np.float64), "abnormal": np.zeros(shape, dtype=np.float64)}
        for name in ("semantic", "temporal", "agreement")
    }
    counts = {"normal": 0, "abnormal": 0}
    for _, row in tqdm(rows.iterrows(), total=len(rows), desc="score layers and neurons", unit="video"):
        hidden = sample_hidden(str(row["hidden_path"]), snippets)
        standardized = np.abs((hidden - center) / np.maximum(scale, 1e-6))
        innovation = neighbor_innovation(hidden)
        temporal = np.maximum((innovation - innovation_center) / np.maximum(innovation_scale, 1e-6), 0.0)
        agreement = np.sqrt(np.maximum(standardized * temporal, 0.0))
        group = "normal" if is_normal_label(dataset, str(row["label"])) else "abnormal"
        counts[group] += 1
        for name, value in (("semantic", standardized), ("temporal", temporal), ("agreement", agreement)):
            accumulators[name][group] += top_fraction_mean(value, top_fraction)
    result = {}
    for name, values in accumulators.items():
        normal = values["normal"] / max(1, counts["normal"])
        abnormal = values["abnormal"] / max(1, counts["abnormal"])
        result[name] = (abnormal - normal).astype(np.float32)
    return result, counts


def choose_layers(
    neuron_score: np.ndarray,
    neurons_per_layer: int,
    coverage: float,
    max_layers: int,
) -> tuple[list[int], np.ndarray]:
    layer_quality = np.zeros(neuron_score.shape[0], dtype=np.float64)
    for layer in range(neuron_score.shape[0]):
        positive = np.maximum(neuron_score[layer], 0.0)
        count = min(neurons_per_layer, positive.size)
        layer_quality[layer] = np.sort(positive)[-count:].sum()
    order = np.argsort(layer_quality)[::-1]
    target = float(layer_quality.sum()) * coverage
    selected, cumulative = [], 0.0
    for layer in order:
        if layer_quality[layer] <= 0 and selected:
            break
        selected.append(int(layer)); cumulative += float(layer_quality[layer])
        if cumulative >= target or len(selected) >= max_layers:
            break
    if not selected:
        selected = [int(order[0])]
    return selected, layer_quality.astype(np.float32)


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare TRACE score-free neuron evidence.")
    parser.add_argument("--dataset", choices=["ucf", "xd"], required=True)
    parser.add_argument("--train-csv", required=True)
    parser.add_argument("--normal-artifact", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--snippets-per-video", type=int, default=256)
    parser.add_argument("--top-fraction", type=float, default=0.10)
    parser.add_argument("--neurons-per-layer", type=int, default=64)
    parser.add_argument("--layer-coverage", type=float, default=0.80)
    parser.add_argument("--max-layers", type=int, default=3)
    parser.add_argument("--clean", action="store_true")
    args = parser.parse_args()
    if not 0 < args.top_fraction <= 1 or not 0 < args.layer_coverage <= 1:
        parser.error("fractions must be in (0,1]")
    output = clean_output(args.out_dir, args.clean)
    artifact_path, report_path = output / "trace_artifact.npz", output / "prepare_report.json"
    if artifact_path.exists() and report_path.exists() and not args.clean:
        print(report_path.read_text(encoding="utf-8"), flush=True); return
    rows = representatives(args.train_csv)
    source = np.load(args.normal_artifact)
    center = source["center"].astype(np.float32)
    scale = source["scale"].astype(np.float32)
    normal_rows = rows[rows["label"].map(lambda value: is_normal_label(args.dataset, str(value)))]
    innovation_center, innovation_scale, innovation_samples = moments(
        normal_rows, args.snippets_per_video, "normal temporal statistics"
    )
    effects, counts = evidence_statistics(
        rows, args.dataset, center, scale, innovation_center, innovation_scale,
        args.snippets_per_video, args.top_fraction,
    )
    positive_semantic = np.maximum(effects["semantic"], 0.0)
    positive_temporal = np.maximum(effects["temporal"], 0.0)
    positive_agreement = np.maximum(effects["agreement"], 0.0)
    neuron_score = positive_agreement + 0.5 * np.sqrt(positive_semantic * positive_temporal + 1e-12)
    selected_layers, layer_quality = choose_layers(
        neuron_score, args.neurons_per_layer, args.layer_coverage, args.max_layers
    )
    selected_layer_indices, selected_dimensions = [], []
    for layer in selected_layers:
        order = np.argsort(neuron_score[layer])[::-1][:args.neurons_per_layer]
        selected_layer_indices.extend([layer] * len(order)); selected_dimensions.extend(map(int, order))
    layer_array = np.asarray(selected_layer_indices, dtype=np.int64)
    dimension_array = np.asarray(selected_dimensions, dtype=np.int64)
    np.savez_compressed(
        artifact_path,
        selected_layers=layer_array, selected_dimensions=dimension_array,
        center=center[layer_array, dimension_array], scale=scale[layer_array, dimension_array],
        innovation_center=innovation_center[layer_array, dimension_array],
        innovation_scale=innovation_scale[layer_array, dimension_array],
        neuron_score=neuron_score[layer_array, dimension_array],
        layer_quality=layer_quality,
    )
    report = {
        "method": "temporally_responsible_anomaly_concept_evidence_v1",
        "selection_source": "video labels + normal CLS statistics + normal temporal innovation; no baseline scores",
        "dataset": args.dataset, "videos": counts,
        "layers": int(center.shape[0]), "hidden_width": int(center.shape[1]),
        "selected_layers_zero_based": selected_layers,
        "selected_layers_one_based": [value + 1 for value in selected_layers],
        "selected_neurons": int(len(layer_array)),
        "neurons_per_layer": int(args.neurons_per_layer),
        "layer_quality": layer_quality.tolist(),
        "innovation_normal_samples": int(innovation_samples),
        "artifact": str(artifact_path), "reused_train_csv": args.train_csv,
        "reused_normal_artifact": args.normal_artifact,
    }
    save_json(report_path, report)
    print(json.dumps(report, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()

