from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from innovation_evidence.common import BASELINES, DATASETS, DATASET_NAMES, prepare_output, require_relative


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare uniform and spectral reliability inside the complete adapter.")
    parser.add_argument("--evaluation-root", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--clean", action="store_true")
    args = parser.parse_args()
    root, output = Path(args.evaluation_root), Path(args.out_dir)
    require_relative(root, "--evaluation-root")
    if not prepare_output(output, args.clean, "innovation2_metrics.csv"):
        return
    rows = []
    for dataset in DATASETS:
        metric_name = "auc" if dataset == "ucf" else "ap"
        for baseline in BASELINES:
            for method in ("uniform", "spectral"):
                payload = json.loads(
                    (root / method / dataset / baseline / "metrics.json").read_text(encoding="utf-8")
                )
                rows.append({
                    "dataset": dataset,
                    "baseline": baseline,
                    "method": method,
                    "metric": payload["corrected"][metric_name],
                })
    table = pd.DataFrame(rows)
    table.to_csv(output / "innovation2_metrics.csv", index=False)
    for dataset in DATASETS:
        selected = table[table.dataset == dataset]
        pivot = selected.pivot(index="baseline", columns="method", values="metric").reindex(BASELINES)
        gains = 100 * (pivot.spectral - pivot.uniform)
        figure, axis = plt.subplots(figsize=(5.3, 3.1), constrained_layout=True)
        bars = axis.bar(
            ("LaGoVAD", "DeSC", "DSANet"),
            gains,
            color=np.where(gains >= 0, "#0072B2", "#D55E00"),
        )
        axis.axhline(0.0, color="#444444", linewidth=0.8)
        axis.bar_label(bars, labels=[f"{value:+.3f}" for value in gains], padding=2, fontsize=8)
        axis.set_ylabel("Spectral minus uniform (points)")
        axis.set_title(f"{DATASET_NAMES[dataset]}: spectral reliability in full adapter")
        axis.grid(axis="y", alpha=0.25)
        figure.savefig(output / f"{dataset}_spectral_ablation.png", dpi=300)
        figure.savefig(output / f"{dataset}_spectral_ablation.pdf")
        plt.close(figure)


if __name__ == "__main__":
    main()
