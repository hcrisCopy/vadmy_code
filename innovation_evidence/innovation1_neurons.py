from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm import tqdm

from innovation_evidence.common import DATASETS, DATASET_NAMES, prepare_output, require_relative


CONTROL_LABELS = {
    "global_random": "Global random",
    "same_layer_random": "Layer-matched random",
    "hard_nonselected": "Hard non-selected",
    "selected": "Detected neurons",
}


def plot_probe_control(table: pd.DataFrame, dataset: str, output: Path) -> None:
    summary = table.groupby("control", as_index=False).agg(
        mean=("test_auc", "mean"), std=("test_auc", "std")
    ).fillna(0.0)
    order = list(CONTROL_LABELS)
    summary["order"] = summary["control"].map({name: i for i, name in enumerate(order)})
    summary = summary.sort_values("order")
    colors = ["#A7A7A7" if name != "selected" else "#0072B2" for name in summary.control]
    figure, axis = plt.subplots(figsize=(5.2, 3.2), constrained_layout=True)
    positions = np.arange(len(summary))
    axis.bar(positions, 100 * summary["mean"], yerr=100 * summary["std"], color=colors, capsize=3)
    axis.set_xticks(positions, [CONTROL_LABELS[name] for name in summary.control], rotation=18, ha="right")
    axis.set_ylabel("Video-level test AUC (%)")
    axis.set_title(f"{DATASET_NAMES[dataset]}: fixed-budget neuron probe")
    axis.grid(axis="y", alpha=0.25)
    figure.savefig(output / f"{dataset}_primary_fixed_budget.png", dpi=300)
    figure.savefig(output / f"{dataset}_primary_fixed_budget.pdf")
    plt.close(figure)
    summary.drop(columns="order").to_csv(output / f"{dataset}_primary_fixed_budget.csv", index=False)


def plot_directional_heatmap(model_path: Path, dataset: str, output: Path) -> None:
    with np.load(model_path, allow_pickle=False) as model:
        directions = np.asarray(model["directions"], dtype=np.int64)
        weights = np.asarray(model["weights"], dtype=np.float64)
    signed = weights * np.where(directions == 0, 1.0, -1.0)
    signed /= max(float(np.abs(signed).max()), 1e-12)
    order = np.argsort(-np.abs(signed), axis=1)
    ranked = np.take_along_axis(signed, order, axis=1)
    figure, axis = plt.subplots(figsize=(6.0, 3.4), constrained_layout=True)
    image = axis.imshow(ranked, cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto", interpolation="nearest")
    axis.set_yticks(np.arange(12), np.arange(1, 13))
    axis.set_xticks([0, 7, 15, 23, 31], [1, 8, 16, 24, 32])
    axis.set_xlabel("Selected-neuron rank within layer")
    axis.set_ylabel("CLIP visual layer")
    axis.set_title(f"{DATASET_NAMES[dataset]}: directional normality neurons")
    colorbar = figure.colorbar(image, ax=axis)
    colorbar.set_label("Signed importance")
    figure.savefig(output / f"{dataset}_directional_neurons.png", dpi=300)
    figure.savefig(output / f"{dataset}_directional_neurons.pdf")
    plt.close(figure)
    pd.DataFrame(ranked, index=np.arange(1, 13)).rename_axis("layer").to_csv(
        output / f"{dataset}_directional_neurons.csv"
    )


def plot_context_scales(student_path: Path, dataset: str, output: Path) -> None:
    with np.load(student_path, allow_pickle=False) as student:
        coefficient = np.asarray(student["coef"], dtype=np.float64).reshape(-1)
        scale = np.asarray(student["scale"], dtype=np.float64).reshape(-1)
    effective = np.abs(coefficient / np.maximum(scale, 1e-12)).reshape(3, -1).sum(axis=1)
    effective /= max(float(effective.sum()), 1e-12)
    labels = ("Current", "Short context\n(σ=1.5)", "Long context\n(σ=4.0)")
    figure, axis = plt.subplots(figsize=(4.6, 3.1), constrained_layout=True)
    axis.bar(labels, 100 * effective, color=("#6C757D", "#56B4E9", "#0072B2"))
    axis.set_ylabel("Absolute coefficient mass (%)")
    axis.set_title(f"{DATASET_NAMES[dataset]}: context-scale usage")
    axis.grid(axis="y", alpha=0.25)
    figure.savefig(output / f"{dataset}_context_scales.png", dpi=300)
    figure.savefig(output / f"{dataset}_context_scales.pdf")
    plt.close(figure)
    pd.DataFrame({"scale": labels, "coefficient_mass": effective}).to_csv(
        output / f"{dataset}_context_scales.csv", index=False
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Render compact evidence for discriminative, directional, and contextual neuron modeling.")
    parser.add_argument("--probe-root", required=True)
    parser.add_argument("--normality-root", required=True)
    parser.add_argument("--context-root", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--clean", action="store_true")
    args = parser.parse_args()
    paths = {name: Path(value) for name, value in vars(args).items() if name.endswith("root") or name == "out_dir"}
    for name, path in paths.items():
        require_relative(path, f"--{name.replace('_', '-')}")
    output = paths["out_dir"]
    if not prepare_output(output, args.clean, "innovation1_complete.txt"):
        return
    for dataset in tqdm(DATASETS, desc="innovation 1 figures"):
        controls = pd.read_csv(paths["probe_root"] / dataset / "fixed_budget_controls.csv")
        plot_probe_control(controls, dataset, output)
        plot_directional_heatmap(
            paths["normality_root"] / dataset / "top32_signed_v1" / "normality_expert.npz",
            dataset,
            output,
        )
        plot_context_scales(
            paths["context_root"] / dataset / "top32_multiscale_seed234" / "context_student.npz",
            dataset,
            output,
        )
    (output / "innovation1_complete.txt").write_text("complete\n", encoding="utf-8")


if __name__ == "__main__":
    main()
