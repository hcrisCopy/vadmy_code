#!/usr/bin/env python3
"""Plot evidence that selection is baseline-specific and the single residual is used."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
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
    selection = Path(args.selection_dir)
    provenance = json.loads((selection / "selection_provenance.json").read_text(encoding="utf-8"))
    scores = np.load(selection / "shift_scores.npy")
    mask = np.load(selection / "selected_mask.npy").astype(bool)
    pairs = np.genfromtxt(
        selection / "video_pairs.csv", delimiter=",", names=True, dtype=None, encoding="utf-8",
    )
    baseline = provenance["baseline"]
    dataset = provenance["dataset"]

    figure, axes = plt.subplots(1, 2, figsize=(13, 4.5), constrained_layout=True)
    clipped = np.clip(scores, 0, np.quantile(scores, 0.99))
    image = axes[0].imshow(clipped, aspect="auto", cmap="magma", interpolation="nearest")
    selected_y, selected_x = np.where(mask)
    axes[0].scatter(selected_x, selected_y, s=2, c="cyan", alpha=0.7, label="selected Top-64")
    axes[0].set(
        title=f"{baseline}/{dataset}: baseline-specific ShiftScore",
        xlabel="hidden dimension", ylabel="CLIP layer",
    )
    axes[0].legend(loc="upper right", fontsize=8)
    figure.colorbar(image, ax=axes[0], label="ShiftScore (clipped at q99)")
    gap = np.asarray(pairs["top_score_mean"] - pairs["bottom_score_mean"], dtype=np.float32)
    axes[1].hist(gap, bins=30, color="#3976af", edgecolor="white")
    axes[1].axvline(float(np.median(gap)), color="#c33", linestyle="--", label=f"median={np.median(gap):.3f}")
    axes[1].set(
        title=f"Frozen {baseline} top/bottom separation",
        xlabel="top mean - bottom mean", ylabel="abnormal videos",
    )
    axes[1].legend()
    figure.savefig(output / "baseline_specific_selection.png", dpi=180)
    plt.close(figure)

    history = Path(args.training_dir) / "history.jsonl"
    records = [json.loads(line) for line in history.read_text(encoding="utf-8").splitlines() if line.strip()]
    epochs = [record["epoch"] for record in records]
    figure, axes = plt.subplots(1, 3, figsize=(14, 4), constrained_layout=True)
    axes[0].plot(epochs, [record["loss"] for record in records], marker="o")
    axes[0].set(title="Frozen-baseline objective", xlabel="epoch", ylabel="author loss")
    axes[1].plot(epochs, [record["gate"] for record in records], marker="o", label="gate")
    axes[1].plot(epochs, [record["applied_delta_rms"] for record in records], marker="s", label="residual RMS")
    axes[1].set(title="Single residual usage", xlabel="epoch")
    axes[1].legend()
    auc = [record.get("metrics", {}).get("frame_auc", np.nan) for record in records]
    ap = [record.get("metrics", {}).get("frame_ap", np.nan) for record in records]
    axes[2].plot(epochs, auc, marker="o", label="frame AUC")
    axes[2].plot(epochs, ap, marker="s", label="frame AP")
    axes[2].set(title="Author-style selection metric", xlabel="epoch")
    axes[2].legend()
    figure.savefig(output / "single_residual_training.png", dpi=180)
    plt.close(figure)
    print(f"wrote diagnostics to {output}", flush=True)


if __name__ == "__main__":
    main()
