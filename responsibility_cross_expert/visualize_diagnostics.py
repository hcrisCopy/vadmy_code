#!/usr/bin/env python3
"""Create only diagnostics that test the new method's central assumptions."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from .common import clean_output, is_normal
from .prompts import abnormal_class_names, label_targets
from .semantic_model import selected_layer_spec


def main() -> None:
    parser = argparse.ArgumentParser(description="Visualize cross-expert agreement and layer evidence.")
    parser.add_argument("--dataset", choices=["ucf", "xd"], required=True)
    parser.add_argument("--consensus-csv", required=True)
    parser.add_argument("--layer-atlas", required=True)
    parser.add_argument("--semantic-checkpoint", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--examples", type=int, default=8)
    parser.add_argument("--clean", action="store_true")
    args = parser.parse_args()
    if args.examples <= 0:
        parser.error("--examples must be positive")
    output_dir = clean_output(args.out_dir, args.clean)
    frame = pd.read_csv(args.consensus_csv)
    layers, initial_weights = selected_layer_spec(args.layer_atlas)
    semantic_checkpoint = torch.load(args.semantic_checkpoint, map_location="cpu")
    learned_weights = semantic_checkpoint.get("model_metadata", {}).get("layer_weights")
    if learned_weights is None or len(learned_weights) != len(layers):
        raise ValueError("semantic checkpoint has no layer weights matching the layer atlas")
    class_names = abnormal_class_names(args.dataset)
    quadrants = np.zeros((2, 2), dtype=np.int64)
    class_layer_sum = np.zeros((len(class_names), len(layers)), dtype=np.float64)
    class_layer_count = np.zeros(len(class_names), dtype=np.int64)
    examples = []
    for _, row in frame.iterrows():
        score = np.load(str(row["score_path"]))
        consensus = np.load(str(row["consensus_path"]))
        length = int(row["length"])
        semantic = score["semantic_score"][:length]
        baseline = score["baseline_score"][:length]
        semantic_high = semantic >= 0.5
        baseline_high = baseline >= 0.5
        quadrants[0, 0] += int((~semantic_high & ~baseline_high).sum())
        quadrants[0, 1] += int((~semantic_high & baseline_high).sum())
        quadrants[1, 0] += int((semantic_high & ~baseline_high).sum())
        quadrants[1, 1] += int((semantic_high & baseline_high).sum())
        disagreement = float((semantic_high != baseline_high).mean()) if length else 0.0
        agreement_positive = int((consensus["target"][:length] > 0).sum())
        examples.append((disagreement, agreement_positive, row))
        positive = consensus["target"][:length] > 0
        if not positive.any() or is_normal(args.dataset, str(row["label"])):
            continue
        layer_margin = score["layer_margin"][:length]
        for class_index in label_targets(args.dataset, str(row["label"])):
            class_layer_sum[class_index] += layer_margin[positive, :, class_index].mean(axis=0)
            class_layer_count[class_index] += 1

    quadrant_fraction = quadrants / max(1, quadrants.sum())
    figure, axis = plt.subplots(figsize=(5.8, 4.8))
    image = axis.imshow(quadrant_fraction, cmap="Blues", vmin=0, vmax=max(0.01, quadrant_fraction.max()))
    axis.set_xticks([0, 1], ["baseline low", "baseline high"])
    axis.set_yticks([0, 1], ["semantic low", "semantic high"])
    axis.set_title("Raw expert agreement over valid snippets")
    for row in range(2):
        for column in range(2):
            axis.text(column, row, f"{quadrant_fraction[row, column]:.1%}\n(n={quadrants[row, column]})", ha="center", va="center")
    figure.colorbar(image, ax=axis, label="fraction of snippets")
    figure.tight_layout()
    agreement_path = output_dir / "expert_agreement_matrix.png"
    figure.savefig(agreement_path, dpi=180)
    plt.close(figure)

    matrix = class_layer_sum / np.maximum(class_layer_count[:, None], 1)
    figure, axis = plt.subplots(figsize=(max(5.5, len(layers) * 1.1), max(4.5, len(class_names) * 0.42)))
    limit = max(1e-4, float(np.abs(matrix).max()))
    image = axis.imshow(matrix, aspect="auto", cmap="coolwarm", vmin=-limit, vmax=limit)
    axis.set_xticks(np.arange(len(layers)), [value + 1 for value in layers])
    axis.set_yticks(np.arange(len(class_names)), class_names)
    axis.set_xlabel("selected CLIP layer (1-based)")
    axis.set_title("Class-layer margin on consensus-positive snippets")
    for row in range(matrix.shape[0]):
        for column in range(matrix.shape[1]):
            axis.text(column, row, f"{matrix[row, column]:.3f}", ha="center", va="center", fontsize=8)
    figure.colorbar(image, ax=axis, label="mean anomaly-minus-normal margin")
    figure.tight_layout()
    layer_path = output_dir / "class_layer_evidence_heatmap.png"
    figure.savefig(layer_path, dpi=180)
    plt.close(figure)

    positions = np.arange(len(layers))
    figure, axis = plt.subplots(figsize=(max(6.0, len(layers) * 1.1), 4.6))
    width = 0.36
    axis.bar(positions - width / 2, initial_weights, width, label="responsibility initialization")
    axis.bar(positions + width / 2, learned_weights, width, label="learned semantic gate")
    axis.set_xticks(positions, [value + 1 for value in layers])
    axis.set_xlabel("selected CLIP layer (1-based)")
    axis.set_ylabel("normalized weight")
    axis.set_title("Responsibility evidence before and after target-domain adaptation")
    axis.grid(axis="y", alpha=0.25)
    axis.legend(frameon=False)
    figure.tight_layout()
    gate_path = output_dir / "layer_gate_weights.png"
    figure.savefig(gate_path, dpi=180)
    plt.close(figure)

    ranked = sorted(examples, key=lambda value: (value[0], value[1]), reverse=True)
    selected = ranked[: args.examples]
    temporal_path = output_dir / "temporal_consensus_examples.png"
    if selected:
        figure, axes = plt.subplots(len(selected), 1, figsize=(12, 2.35 * len(selected)), squeeze=False)
        for axis, (disagreement, _, row) in zip(axes[:, 0], selected):
            score = np.load(str(row["score_path"]))
            consensus = np.load(str(row["consensus_path"]))
            length = int(row["length"])
            x = np.arange(length)
            axis.plot(x, score["semantic_score"][:length], label="semantic expert", color="#d95f02")
            axis.plot(x, score["baseline_score"][:length], label="baseline temporal expert", color="#1b9e77")
            axis.plot(x, consensus["semantic_refined"][:length], color="#e6ab02", alpha=0.65, linestyle="--", label="semantic refined")
            axis.plot(x, consensus["baseline_refined"][:length], color="#377eb8", alpha=0.65, linestyle="--", label="baseline refined")
            positive = consensus["target"][:length] > 0
            for start, end in _runs(positive):
                axis.axvspan(start, end, color="#984ea3", alpha=0.18)
            axis.set_ylim(-0.05, 1.05)
            axis.set_title(f"{row['key']} | {row['label']} | raw disagreement={disagreement:.1%}", loc="left", fontsize=9)
            axis.grid(alpha=0.18)
        axes[-1, 0].set_xlabel("snippet index; purple = retained consensus positive")
        handles, labels = axes[0, 0].get_legend_handles_labels()
        figure.legend(handles, labels, loc="upper right", frameon=False)
        figure.tight_layout(rect=(0, 0, 0.88, 1))
        figure.savefig(temporal_path, dpi=180)
        plt.close(figure)
    report = {
        "figures": [str(agreement_path), str(layer_path), str(gate_path), str(temporal_path)],
        "purposes": {
            "expert_agreement_matrix.png": "measure whether two experts supply independent but compatible evidence",
            "class_layer_evidence_heatmap.png": "show which complete CLIP layers support each anomaly class",
            "layer_gate_weights.png": "show whether target-domain training preserves or changes the responsibility-based layer prior",
            "temporal_consensus_examples.png": "inspect retained spans and expert disagreement instead of decorative examples",
        },
    }
    (output_dir / "visualization_report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(report, indent=2, ensure_ascii=False), flush=True)


def _runs(mask: np.ndarray) -> list[tuple[int, int]]:
    indices = np.where(mask)[0]
    if not len(indices):
        return []
    output, start, previous = [], int(indices[0]), int(indices[0])
    for raw in indices[1:]:
        current = int(raw)
        if current != previous + 1:
            output.append((start, previous + 1))
            start = current
        previous = current
    output.append((start, previous + 1))
    return output


if __name__ == "__main__":
    main()
