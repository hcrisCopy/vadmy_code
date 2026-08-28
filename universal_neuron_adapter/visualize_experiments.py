from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import average_precision_score, roc_auc_score


def score(dataset: str, truth: np.ndarray, values: np.ndarray) -> float:
    if len(np.unique(truth)) < 2:
        return float("nan")
    return float(roc_auc_score(truth, values) if dataset == "ucf" else average_precision_score(truth, values))


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate ablation and qualitative figures focused on the method claim.")
    parser.add_argument("--experiment-root", required=True)
    parser.add_argument("--ucf-gt", required=True)
    parser.add_argument("--xd-gt", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--frames-per-snippet", type=int, default=16)
    args = parser.parse_args()
    root, output = Path(args.experiment_root), Path(args.out_dir)
    output.mkdir(parents=True, exist_ok=True)
    variants = ["baseline", "correction", "agreement", "event_gate", "video_suppression", "full"]
    rows = []
    for variant in variants:
        for dataset in ("ucf", "xd"):
            for baseline in ("lagovad", "desc", "dsanet"):
                path = root / "ablations" / variant / dataset / baseline / "metrics.json"
                if not path.exists():
                    continue
                metrics = json.loads(path.read_text(encoding="utf-8"))
                name = "auc" if dataset == "ucf" else "ap"
                rows.append({"variant": variant, "setting": f"{dataset.upper()}-{baseline}", "gain": 100 * (metrics["corrected"][name] - metrics["baseline"][name])})
    if rows:
        table = pd.DataFrame(rows).pivot(index="variant", columns="setting", values="gain").reindex(variants)
        table.to_csv(output / "ablation_gains.csv")
        sns.set_theme(style="white", context="paper")
        figure, axis = plt.subplots(figsize=(9.0, 3.8))
        maximum = max(0.25, float(np.nanmax(np.abs(table.to_numpy()))))
        sns.heatmap(table, annot=True, fmt=".2f", cmap="RdBu_r", center=0, vmin=-maximum, vmax=maximum, linewidths=0.5, cbar_kws={"label": "gain (points)"}, ax=axis)
        axis.set_xlabel(""); axis.set_ylabel(""); axis.set_title("Cumulative contribution of each method component")
        figure.tight_layout(); figure.savefig(output / "ablation_heatmap.png", dpi=300, bbox_inches="tight"); figure.savefig(output / "ablation_heatmap.pdf", bbox_inches="tight"); plt.close(figure)

    figure, axes = plt.subplots(2, 3, figsize=(12, 5.8), sharey=False)
    plotted = 0
    for row_index, (dataset, gt_path) in enumerate((("ucf", args.ucf_gt), ("xd", args.xd_gt))):
        truth = np.load(gt_path).reshape(-1)
        for column, baseline in enumerate(("lagovad", "desc", "dsanet")):
            evaluation = root / "ablations" / "full" / dataset / baseline
            manifest_path = evaluation / "per_video.csv"
            if not manifest_path.exists():
                continue
            manifest = pd.read_csv(manifest_path)
            offset, candidates = 0, []
            for video in manifest.itertuples(index=False):
                curves = np.load(str(video.curve_path)); length = int(video.snippets) * args.frames_per_snippet
                target = truth[offset:offset + length]; offset += length
                base = np.repeat(curves["baseline"], args.frames_per_snippet); method = np.repeat(curves["corrected"], args.frames_per_snippet)
                gain = score(dataset, target, method) - score(dataset, target, base)
                if np.isfinite(gain): candidates.append((gain, str(video.key), target, base, method))
            if not candidates:
                continue
            median_gain = float(np.median([item[0] for item in candidates]))
            gain, key, target, base, method = min(
                candidates, key=lambda item: abs(item[0] - median_gain)
            )
            axis = axes[row_index, column]; time = np.arange(len(target)) / args.frames_per_snippet
            axis.fill_between(time, 0, target, color="#D1495B", alpha=0.18, label="ground truth")
            axis.plot(time, base, color="#6C757D", linewidth=1.0, label="baseline")
            axis.plot(time, method, color="#176B87", linewidth=1.2, label="adapter")
            axis.set_title(
                f"{dataset.upper()} / {baseline}, median-gain representative\n"
                f"{key}, gain {100 * gain:.2f}"
            )
            axis.set_xlabel("snippet index"); axis.set_ylim(-0.03, 1.03); plotted += 1
    if plotted:
        axes[0, 0].legend(loc="upper right", fontsize=7)
        figure.suptitle("Neuron evidence expands verified events while retaining baseline structure", y=1.01)
        figure.tight_layout(); figure.savefig(output / "qualitative_curves.png", dpi=300, bbox_inches="tight"); figure.savefig(output / "qualitative_curves.pdf", bbox_inches="tight")
    plt.close(figure)


if __name__ == "__main__":
    main()
