#!/usr/bin/env python3
"""Create compact diagnostics that test the method's important assumptions."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from semantic_knn_splicing.common import clean_output


def plot_layer_evidence(atlas_path: str, output: Path) -> list[str]:
    atlas = json.loads(Path(atlas_path).read_text(encoding="utf-8"))
    ranking = sorted(atlas["layer_ranking"], key=lambda value: value["layer_zero_based"])
    chosen = {int(value["layer_zero_based"]) for value in atlas["blocks"]}
    layers = np.asarray([int(value["layer_zero_based"]) + 1 for value in ranking])
    quality = np.asarray([float(value["quality"]) for value in ranking])
    colors = ["#d62728" if int(value) - 1 in chosen else "#9ecae1" for value in layers]
    fig, axis = plt.subplots(figsize=(9, 4.5))
    axis.bar(layers, quality, color=colors)
    axis.set(xlabel="CLIP layer (1-based)", ylabel="responsibility quality", title="Layer responsibility ranking")
    axis.set_xticks(layers)
    axis.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    path_a = output / "layer_responsibility.png"
    fig.savefig(path_a, dpi=180)
    plt.close(fig)

    class_names = atlas["class_names"]
    blocks = sorted(atlas["blocks"], key=lambda value: value["layer_zero_based"])
    selected_layers = np.asarray([int(value["layer_zero_based"]) + 1 for value in blocks])
    # Each block stores non-negative responsibility weights for every class and
    # selected neuron.  Summing within a layer measures how much responsibility
    # evidence that layer contributes; row normalization makes classes comparable.
    matrix = np.stack(
        [np.asarray(value["weights"], dtype=np.float32).sum(axis=1) for value in blocks], axis=1
    )
    matrix /= np.maximum(matrix.sum(axis=1, keepdims=True), 1e-8)
    fig, axis = plt.subplots(figsize=(6.5, max(4.5, 0.42 * len(class_names))))
    image = axis.imshow(matrix, aspect="auto", cmap="YlGnBu", vmin=0.0, vmax=1.0)
    axis.set_xticks(np.arange(len(selected_layers)), selected_layers)
    axis.set_yticks(np.arange(len(class_names)), class_names)
    axis.set(xlabel="selected CLIP layer (1-based)", title="Per-class responsibility across selected layers")
    for row in range(matrix.shape[0]):
        for column in range(matrix.shape[1]):
            color = "white" if matrix[row, column] >= 0.55 else "#222222"
            axis.text(column, row, f"{matrix[row, column]:.2f}", ha="center", va="center", color=color)
    fig.colorbar(image, ax=axis, label="fraction of class responsibility")
    fig.tight_layout()
    path_b = output / "class_layer_responsibility_heatmap.png"
    fig.savefig(path_b, dpi=180)
    plt.close(fig)
    return [str(path_a), str(path_b)]


def plot_temporal_localization(
    pseudo_csv: str, output: Path, examples: int, layer_numbers: list[int]
) -> str | None:
    frame = pd.read_csv(pseudo_csv)
    if frame.empty:
        return None
    selected = frame.sort_values("score", ascending=False).drop_duplicates("key").head(examples)
    fig, axes = plt.subplots(len(selected), 1, figsize=(12, 2.4 * len(selected)), squeeze=False)
    for axis, (_, row) in zip(axes[:, 0], selected.iterrows()):
        cache = np.load(str(row["score_cache"]))
        score = cache["snippet_score"]
        layer = cache["layer_score"]
        axis.plot(score, color="#111111", linewidth=1.8, label="weighted text margin")
        for index in range(layer.shape[1]):
            number = layer_numbers[index] if index < len(layer_numbers) else index + 1
            axis.plot(layer[:, index], linewidth=1.0, alpha=0.65, label=f"CLIP layer {number}")
        matches = frame[frame["key"].astype(str) == str(row["key"])]
        for _, span in matches.iterrows():
            axis.axvspan(int(span["start"]), int(span["end"]), color="#d62728", alpha=0.18)
        axis.axhline(0.0, color="#777777", linewidth=0.7, linestyle="--")
        axis.set_title(f"{row['key']} | {row['label']}", loc="left", fontsize=10)
        axis.set_ylabel("margin")
        axis.grid(alpha=0.18)
    axes[-1, 0].set_xlabel("snippet index")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper right", frameon=False)
    fig.tight_layout(rect=(0, 0, 0.92, 1))
    path = output / "temporal_text_margin.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return str(path)


def plot_synthetic_labels(synthetic_csv: str, output: Path, examples: int) -> str | None:
    frame = pd.read_csv(synthetic_csv).head(examples)
    if frame.empty:
        return None
    fig, axes = plt.subplots(len(frame), 1, figsize=(12, 1.25 * len(frame)), squeeze=False)
    for axis, (_, row) in zip(axes[:, 0], frame.iterrows()):
        stored = np.load(str(row["feature_path"]))
        label = stored["frame_label"].astype(np.float32)
        axis.imshow(label[None, :], aspect="auto", cmap="RdYlBu_r", vmin=0, vmax=1)
        changes = np.where(np.diff(label) != 0)[0] + 1
        for boundary in changes:
            axis.axvline(boundary - 0.5, color="white", linewidth=1.2)
        axis.set_yticks([])
        axis.set_ylabel(str(row["sample_id"]), rotation=0, ha="right", va="center", fontsize=8)
    axes[-1, 0].set_xlabel("synthetic snippet index (blue=normal, red=abnormal candidate)")
    fig.suptitle("LaGoVAD-style KNN temporal splice boundaries", y=1.01)
    fig.tight_layout()
    path = output / "synthetic_boundaries.png"
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return str(path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Visualize layer, localization, and synthesis evidence.")
    parser.add_argument("--layer-atlas", required=True)
    parser.add_argument("--pseudo-csv", required=True)
    parser.add_argument("--synthetic-list", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--examples", type=int, default=8)
    parser.add_argument("--clean", action="store_true")
    args = parser.parse_args()
    if args.examples <= 0:
        parser.error("--examples must be positive")
    output = clean_output(args.out_dir, args.clean)
    paths = plot_layer_evidence(args.layer_atlas, output)
    atlas = json.loads(Path(args.layer_atlas).read_text(encoding="utf-8"))
    layer_numbers = [
        int(value["layer_zero_based"]) + 1
        for value in sorted(atlas["blocks"], key=lambda value: value["layer_zero_based"])
    ]
    for value in (
        plot_temporal_localization(args.pseudo_csv, output, args.examples, layer_numbers),
        plot_synthetic_labels(args.synthetic_list, output, args.examples),
    ):
        if value is not None:
            paths.append(value)
    report = {"figures": paths, "purpose": "method diagnostics, not decorative plots"}
    (output / "visualization_report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(report, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
