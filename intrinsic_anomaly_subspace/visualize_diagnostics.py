#!/usr/bin/env python3
"""Create evidence-oriented diagnostics for layer, neuron, control, and timeline behavior."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .common import clean_output, save_json


def layer_figure(metrics: pd.DataFrame, output: Path) -> None:
    figure, axes = plt.subplots(2, 1, figsize=(9, 6), sharex=True)
    for axis, column, threshold in zip(axes, ("d_cos", "d_shift"), ("tau_cos", "tau_shift")):
        axis.plot(metrics["layer_index"], metrics[column], marker="o", linewidth=2, label=column)
        axis.axhline(float(metrics[threshold].iloc[0]), color="#d95f02", linestyle="--", label=f"{threshold}")
        critical = metrics[metrics["critical"].astype(str).str.lower().isin(["true", "1"])]
        axis.scatter(critical["layer_index"], critical[column], color="#1b9e77", s=80, zorder=3, label="critical")
        axis.set_ylabel(column)
        axis.grid(alpha=0.25)
        axis.legend(frameon=False)
    axes[-1].set_xlabel("CLIP transformer layer index")
    figure.suptitle("V-FIND critical-layer intersection")
    figure.tight_layout()
    figure.savefig(output / "critical_layers.png", dpi=180)
    plt.close(figure)


def neuron_figure(neurons: pd.DataFrame, output: Path) -> None:
    pivot = neurons.pivot(index="layer_index", columns="dimension", values="effect_size").sort_index()
    figure, axis = plt.subplots(figsize=(12, max(2.5, 0.65 * len(pivot))))
    image = axis.imshow(pivot.to_numpy(), aspect="auto", cmap="magma", interpolation="nearest")
    axis.set_yticks(np.arange(len(pivot)))
    axis.set_yticklabels(pivot.index.tolist())
    axis.set_xlabel("hidden dimension")
    axis.set_ylabel("critical layer")
    axis.set_title("Neuron response effect size (V-FIND Eq. 14)")
    figure.colorbar(image, ax=axis, label="effect size")
    figure.tight_layout()
    figure.savefig(output / "neuron_effect_heatmap.png", dpi=180)
    plt.close(figure)


def control_figure(reports: list[dict], output: Path, dataset: str) -> None:
    labels = [report["feature_mode"] for report in reports]
    auc = [100 * report["frame_auc"] for report in reports]
    ap = [100 * report["frame_ap"] for report in reports]
    x = np.arange(len(labels))
    figure, axis = plt.subplots(figsize=(9, 4.8))
    width = 0.36
    axis.bar(x - width / 2, auc, width, label="frame AUC")
    axis.bar(x + width / 2, ap, width, label="frame AP")
    axis.set_xticks(x)
    axis.set_xticklabels(labels, rotation=15, ha="right")
    axis.set_ylabel("metric (%)")
    axis.set_title(f"Equal-budget neuron controls on {dataset.upper()}")
    axis.grid(axis="y", alpha=0.25)
    axis.legend(frameon=False)
    figure.tight_layout()
    figure.savefig(output / "fixed_budget_controls.png", dpi=180)
    plt.close(figure)


def timeline_figure(evaluation_dir: Path, output: Path, examples: int) -> None:
    scores = np.load(evaluation_dir / "frame_scores.npy", allow_pickle=False)
    truth = np.load(evaluation_dir / "frame_truth.npy", allow_pickle=False)
    boundaries = pd.read_csv(evaluation_dir / "video_boundaries.csv")
    ranked = []
    for _, row in boundaries.iterrows():
        start, end = int(row["frame_start"]), min(int(row["frame_end"]), len(scores))
        if end <= start or end > len(truth):
            continue
        local_truth, local_score = truth[start:end], scores[start:end]
        if len(np.unique(local_truth)) < 2:
            continue
        separation = float(local_score[local_truth == 1].mean() - local_score[local_truth == 0].mean())
        ranked.append((abs(separation), str(row["key"]), start, end))
    ranked.sort(reverse=True)
    chosen = ranked[:examples]
    if not chosen:
        return
    figure, axes = plt.subplots(len(chosen), 1, figsize=(12, 2.6 * len(chosen)), squeeze=False)
    for axis, (_, key, start, end) in zip(axes[:, 0], chosen):
        local_score, local_truth = scores[start:end], truth[start:end]
        axis.plot(local_score, color="#377eb8", linewidth=1.3, label="prediction")
        axis.fill_between(np.arange(len(local_truth)), 0, local_truth, color="#e41a1c", alpha=0.22, label="frame GT")
        axis.set_ylim(-0.03, 1.03)
        axis.set_title(key)
        axis.legend(frameon=False, ncol=2)
    axes[-1, 0].set_xlabel("frame index within video")
    figure.suptitle("Representative temporal localization (ranked by GT separation for diagnosis only)")
    figure.tight_layout()
    figure.savefig(output / "temporal_localization_examples.png", dpi=180)
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(description="Visualize intrinsic anomaly subspace diagnostics.")
    parser.add_argument("--dataset", choices=["ucf", "xd"], required=True)
    parser.add_argument("--discovery-dir", required=True)
    parser.add_argument("--selected-eval-dir", required=True)
    parser.add_argument("--same-layer-random-eval-dir", required=True)
    parser.add_argument("--global-random-eval-dir", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--timeline-examples", type=int, default=4)
    parser.add_argument("--clean", action="store_true")
    args = parser.parse_args()
    output = clean_output(args.out_dir, args.clean)
    discovery = Path(args.discovery_dir)
    layer_figure(pd.read_csv(discovery / "layer_metrics.csv"), output)
    neuron_figure(pd.read_csv(discovery / "neuron_metrics.csv"), output)
    eval_dirs = [Path(args.selected_eval_dir), Path(args.same_layer_random_eval_dir), Path(args.global_random_eval_dir)]
    reports = [json.loads((directory / "metrics.json").read_text(encoding="utf-8")) for directory in eval_dirs]
    widths = {report["selected_width"] for report in reports}
    if len(widths) != 1:
        raise ValueError(f"fixed-budget controls have unequal widths: {sorted(widths)}")
    control_figure(reports, output, args.dataset)
    timeline_figure(Path(args.selected_eval_dir), output, args.timeline_examples)
    selected = reports[0]
    control_best = max(reports[1:], key=lambda item: item["frame_auc"] if args.dataset == "ucf" else item["frame_ap"])
    primary = "frame_auc" if args.dataset == "ucf" else "frame_ap"
    summary = {
        "dataset": args.dataset,
        "primary_metric": primary,
        "selected": selected,
        "controls": reports[1:],
        "selected_minus_best_random_points": 100 * (selected[primary] - control_best[primary]),
        "interpretation": (
            "Positive evidence requires the selected subspace to beat both equal-width random controls; "
            "otherwise the result supports layer capacity, not functionally special neurons."
        ),
    }
    save_json(output / "diagnostic_summary.json", summary)
    print(json.dumps(summary, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
