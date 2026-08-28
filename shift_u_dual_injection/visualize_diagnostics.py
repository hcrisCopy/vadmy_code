#!/usr/bin/env python3
"""Plot whether early and late U branches are used and improve frame metrics."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser(description="Visualize frozen-baseline U-dual training evidence.")
    parser.add_argument("--selection-dir", required=True)
    parser.add_argument("--training-dir", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--clean", action="store_true")
    args = parser.parse_args()
    output = Path(args.out_dir)
    if args.clean and output.exists():
        for path in output.glob("*.png"):
            path.unlink()
    output.mkdir(parents=True, exist_ok=True)
    history_path = Path(args.training_dir) / "history.jsonl"
    if not history_path.exists():
        raise FileNotFoundError(f"missing training history: {history_path}")
    records = [json.loads(line) for line in history_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    epochs = [record["epoch"] for record in records]

    selection = Path(args.selection_dir)
    scores = np.load(selection / "shift_scores.npy")
    mask = np.load(selection / "selected_mask.npy").astype(bool)
    pairs = np.genfromtxt(selection / "video_pairs.csv", delimiter=",", names=True, dtype=None, encoding="utf-8")
    figure, axes = plt.subplots(1, 2, figsize=(13, 4.5), constrained_layout=True)
    clipped = np.clip(scores, 0, np.quantile(scores, 0.99))
    image = axes[0].imshow(clipped, aspect="auto", cmap="magma", interpolation="nearest")
    selected_y, selected_x = np.where(mask)
    axes[0].scatter(selected_x, selected_y, s=2, c="cyan", alpha=0.7, label="selected Top-64")
    axes[0].set(title="Reused ShiftScore selection", xlabel="hidden dimension", ylabel="CLIP layer")
    axes[0].legend(loc="upper right", fontsize=8)
    figure.colorbar(image, ax=axes[0], label="ShiftScore (clipped at q99)")
    score_gap = np.asarray(pairs["top_score_mean"] - pairs["bottom_score_mean"], dtype=np.float32)
    axes[1].hist(score_gap, bins=30, color="#3976af", edgecolor="white")
    axes[1].axvline(float(np.median(score_gap)), color="#c33", linestyle="--", label=f"median={np.median(score_gap):.3f}")
    axes[1].set(title="Unchanged top/bottom construction", xlabel="top mean - bottom mean", ylabel="abnormal videos")
    axes[1].legend()
    figure.savefig(output / "reused_selection_evidence.png", dpi=180)
    plt.close(figure)

    figure, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
    axes[0, 0].plot(epochs, [record["loss"] for record in records], marker="o")
    axes[0, 0].set(title="Frozen-baseline training objective", xlabel="epoch", ylabel="author loss")
    axes[0, 1].plot(epochs, [record["early_gate"] for record in records], marker="o", label="early gate")
    axes[0, 1].plot(epochs, [record["late_gate"] for record in records], marker="s", label="late gate")
    axes[0, 1].set(title="Independent U-branch gates", xlabel="epoch", ylabel="sigmoid gate")
    axes[0, 1].legend()
    axes[1, 0].plot(epochs, [record["early_applied_delta_rms"] for record in records], marker="o", label="early RMS")
    axes[1, 0].plot(epochs, [record["late_applied_delta_rms"] for record in records], marker="s", label="late RMS")
    axes[1, 0].set(title="Actually applied residual energy", xlabel="epoch", ylabel="RMS")
    axes[1, 0].legend()
    auc = [record.get("metrics", {}).get("frame_auc", np.nan) for record in records]
    ap = [record.get("metrics", {}).get("frame_ap", np.nan) for record in records]
    axes[1, 1].plot(epochs, auc, marker="o", label="frame AUC")
    axes[1, 1].plot(epochs, ap, marker="s", label="frame AP")
    axes[1, 1].set(title="Author-style selection metrics", xlabel="epoch", ylabel="metric")
    axes[1, 1].legend()
    figure.savefig(output / "u_branch_diagnostics.png", dpi=180)
    plt.close(figure)

    ratios = np.asarray([record["late_to_early_rms_ratio"] for record in records], dtype=np.float32)
    figure, axis = plt.subplots(figsize=(7, 4), constrained_layout=True)
    axis.plot(epochs, ratios, marker="o", color="#7a3db8")
    axis.axhline(1.0, color="black", linestyle="--", linewidth=1)
    axis.set(
        title="Does the late skip preserve more neuron evidence?",
        xlabel="epoch", ylabel="late residual RMS / early residual RMS",
    )
    figure.savefig(output / "late_early_energy_ratio.png", dpi=180)
    plt.close(figure)
    print(f"wrote U-branch diagnostics to {output}", flush=True)


if __name__ == "__main__":
    main()
