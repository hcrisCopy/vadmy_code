from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm import tqdm

from innovation_evidence.common import (
    DATASETS, DATASET_NAMES, expert_curves, frame_metric, merge_expert_manifests,
    prepare_output, require_relative, sigmoid_gate,
)
from universal_neuron_adapter.evaluate import spectral_consensus_weights


def evaluate_dataset(dataset: str, source: Path, normality: Path, context: Path, truth_path: Path, frames_per_snippet: int):
    frame = merge_expert_manifests(
        source / dataset / "expert" / "test" / "expert_scores.csv",
        context / dataset / "top32_multiscale_seed234" / "test" / "student_scores.csv",
        normality / dataset / "top32_signed_v1" / "test" / "expert3_scores.csv",
    )
    uniform_parts, spectral_parts, weight_rows = [], [], []
    for row in tqdm(frame.itertuples(index=False), total=len(frame), desc=f"{dataset}: spectral consensus"):
        curves = expert_curves(row)
        weights = spectral_consensus_weights(*curves)
        uniform_parts.append(sigmoid_gate(curves, np.ones(3, dtype=np.float32)))
        spectral_parts.append(sigmoid_gate(curves, weights))
        weight_rows.append({"key": row.key, "primary": weights[0], "context": weights[1], "normality": weights[2]})
    truth = np.load(truth_path).reshape(-1)
    uniform = np.repeat(np.concatenate(uniform_parts), frames_per_snippet)
    spectral = np.repeat(np.concatenate(spectral_parts), frames_per_snippet)
    if len(truth) != len(uniform):
        raise RuntimeError(f"{dataset}: strict frame alignment failed: gt={len(truth)}, predictions={len(uniform)}")
    metrics = pd.DataFrame([
        {"dataset": dataset, "method": "Uniform", "metric": frame_metric(dataset, truth, uniform)},
        {"dataset": dataset, "method": "Spectral reliability", "metric": frame_metric(dataset, truth, spectral)},
    ])
    return metrics, pd.DataFrame(weight_rows)


def render(dataset: str, metrics: pd.DataFrame, weights: pd.DataFrame, output: Path) -> None:
    metric_name = "Frame AUC (%)" if dataset == "ucf" else "Frame AP (%)"
    figure, axis = plt.subplots(figsize=(4.2, 3.0), constrained_layout=True)
    values = 100 * metrics.metric.to_numpy()
    positions = np.arange(len(values))
    axis.plot(positions, values, color="#777777", linewidth=1.2)
    axis.scatter(positions, values, color=("#A7A7A7", "#0072B2"), s=55, zorder=3)
    for position, value in zip(positions, values):
        axis.text(position, value, f" {value:.2f}", va="center", fontsize=8)
    axis.set_xticks(positions, metrics.method)
    padding = max(0.2, float(np.ptp(values)) * 0.5)
    axis.set_ylim(float(values.min() - padding), float(values.max() + padding))
    axis.set_ylabel(metric_name)
    axis.set_title(f"{DATASET_NAMES[dataset]}: consensus-gate reliability")
    axis.tick_params(axis="x", rotation=12)
    axis.grid(axis="y", alpha=0.25)
    figure.savefig(output / f"{dataset}_spectral_ablation.png", dpi=300)
    figure.savefig(output / f"{dataset}_spectral_ablation.pdf")
    plt.close(figure)

    figure, axis = plt.subplots(figsize=(4.8, 3.0), constrained_layout=True)
    axis.boxplot(
        [weights[name] for name in ("primary", "context", "normality")],
        labels=("Sparse", "Context", "Normality"),
        showfliers=False,
    )
    axis.axhline(1.0, color="#777777", linestyle="--", linewidth=1)
    axis.set_ylabel("Spectral weight (mean normalized to 1)")
    axis.set_title(f"{DATASET_NAMES[dataset]}: per-video expert reliability")
    axis.grid(axis="y", alpha=0.25)
    figure.savefig(output / f"{dataset}_spectral_weights.png", dpi=300)
    figure.savefig(output / f"{dataset}_spectral_weights.pdf")
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare uniform and principal-eigenvector consensus gates on cached scores.")
    parser.add_argument("--source-root", required=True)
    parser.add_argument("--normality-root", required=True)
    parser.add_argument("--context-root", required=True)
    parser.add_argument("--annotation-root", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--frames-per-snippet", type=int, default=16)
    parser.add_argument("--clean", action="store_true")
    args = parser.parse_args()
    roots = {name: Path(value) for name, value in vars(args).items() if name.endswith("root") or name == "out_dir"}
    for name, path in roots.items():
        require_relative(path, f"--{name.replace('_', '-')}")
    output = roots["out_dir"]
    if not prepare_output(output, args.clean, "innovation2_metrics.csv"):
        return
    metric_tables = []
    for dataset in DATASETS:
        metrics, weights = evaluate_dataset(
            dataset, roots["source_root"], roots["normality_root"], roots["context_root"],
            roots["annotation_root"] / dataset / "gt.npy", args.frames_per_snippet,
        )
        render(dataset, metrics, weights, output)
        weights.to_csv(output / f"{dataset}_spectral_weights.csv", index=False)
        metric_tables.append(metrics)
    pd.concat(metric_tables, ignore_index=True).to_csv(output / "innovation2_metrics.csv", index=False)


if __name__ == "__main__":
    main()
