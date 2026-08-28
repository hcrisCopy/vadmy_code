#!/usr/bin/env python3
"""Create evidence-focused Shift selection and training diagnostics."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser(description="Visualize selected layers, score tails, and residual training.")
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
    scores = np.load(selection / "shift_scores.npy")
    mask = np.load(selection / "selected_mask.npy").astype(bool)
    pairs = np.genfromtxt(selection / "video_pairs.csv", delimiter=",", names=True, dtype=None, encoding="utf-8")

    figure, axes = plt.subplots(1, 2, figsize=(13, 4.5), constrained_layout=True)
    clipped = np.clip(scores, 0, np.quantile(scores, 0.99))
    image = axes[0].imshow(clipped, aspect="auto", cmap="magma", interpolation="nearest")
    selected_y, selected_x = np.where(mask)
    axes[0].scatter(selected_x, selected_y, s=2, c="cyan", alpha=0.7, label="selected Top-64")
    axes[0].set(title="ShiftScore and selected neurons", xlabel="hidden dimension", ylabel="CLIP layer")
    axes[0].legend(loc="upper right", fontsize=8)
    figure.colorbar(image, ax=axes[0], label="ShiftScore (clipped at q99)")
    gap = np.asarray(pairs["top_score_mean"] - pairs["bottom_score_mean"], dtype=np.float32)
    axes[1].hist(gap, bins=30, color="#3976af", edgecolor="white")
    axes[1].axvline(float(np.median(gap)), color="#c33", linestyle="--", label=f"median={np.median(gap):.3f}")
    axes[1].set(title="Baseline score-tail separation", xlabel="top mean - bottom mean", ylabel="abnormal videos")
    axes[1].legend()
    figure.savefig(output / "selection_evidence.png", dpi=180)
    plt.close(figure)

    history_path = Path(args.training_dir) / "history.jsonl"
    if not history_path.exists():
        print(f"selection figure written; training history not found: {history_path}", flush=True)
        return
    records = [json.loads(line) for line in history_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    epochs = [record["epoch"] for record in records]
    figure, axes = plt.subplots(1, 3, figsize=(14, 4), constrained_layout=True)
    axes[0].plot(epochs, [record["loss"] for record in records], marker="o")
    axes[0].set(title="Training objective", xlabel="epoch", ylabel="baseline loss")
    axes[1].plot(epochs, [record["gate"] for record in records], marker="o", label="gate")
    axes[1].plot(epochs, [record["applied_delta_rms"] for record in records], marker="s", label="residual RMS")
    axes[1].set(title="Residual usage", xlabel="epoch")
    axes[1].legend()
    auc = [record.get("metrics", {}).get("frame_auc", np.nan) for record in records]
    ap = [record.get("metrics", {}).get("frame_ap", np.nan) for record in records]
    axes[2].plot(epochs, auc, marker="o", label="frame AUC")
    axes[2].plot(epochs, ap, marker="s", label="frame AP")
    axes[2].set(title="Author-style model selection metrics", xlabel="epoch", ylabel="metric")
    axes[2].legend()
    figure.savefig(output / "training_diagnostics.png", dpi=180)
    plt.close(figure)
    print(f"wrote diagnostic figures to {output}", flush=True)


if __name__ == "__main__":
    main()
