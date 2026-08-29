from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize post-hoc parameter robustness and adapter cost.")
    parser.add_argument("--root", required=True)
    args = parser.parse_args()
    root = Path(args.root)
    rows = []
    for variant_dir in sorted((root / "evaluations").iterdir()):
        for dataset in ("ucf", "xd"):
            metric = "auc" if dataset == "ucf" else "ap"
            for baseline in ("lagovad", "desc", "dsanet", "vadclip"):
                path = variant_dir / dataset / baseline / "metrics.json"
                payload = json.loads(path.read_text(encoding="utf-8"))
                rows.append({
                    "variant": variant_dir.name,
                    "dataset": dataset,
                    "baseline": baseline,
                    "metric": metric,
                    "baseline_value": 100 * float(payload["baseline"][metric]),
                    "corrected_value": 100 * float(payload["corrected"][metric]),
                    "gain": 100 * (float(payload["corrected"][metric]) - float(payload["baseline"][metric])),
                    **payload["performance"],
                })
    frame = pd.DataFrame(rows)
    frame.to_csv(root / "robustness_efficiency.csv", index=False)

    order = ["width_33", "width_41", "width_49", "advance_0", "advance_2"]
    labels = ["width 33", "width 41 / advance 1", "width 49", "advance 0", "advance 2"]
    settings = [
        ("ucf", "lagovad"), ("ucf", "desc"), ("ucf", "dsanet"), ("ucf", "vadclip"),
        ("xd", "lagovad"), ("xd", "desc"), ("xd", "dsanet"), ("xd", "vadclip"),
    ]
    values = np.asarray([
        [float(frame[(frame.variant == variant) & (frame.dataset == dataset) &
                     (frame.baseline == baseline)].gain.iloc[0]) for variant in order]
        for dataset, baseline in settings
    ])
    fig, ax = plt.subplots(figsize=(8.4, 4.4), constrained_layout=True)
    image = ax.imshow(
        values, cmap="RdYlGn", aspect="auto",
        vmin=min(0.0, float(values.min())), vmax=float(values.max()),
    )
    ax.set_xticks(range(len(labels)), labels, rotation=18, ha="right")
    ax.set_yticks(range(len(settings)), [f"{d.upper()}–{b}" for d, b in settings])
    ax.set_title("Gain under temporal-parameter perturbations")
    for row in range(values.shape[0]):
        for column in range(values.shape[1]):
            ax.text(column, row, f"{values[row, column]:+.2f}", ha="center", va="center", fontsize=8)
    fig.colorbar(image, ax=ax, label="AUC/AP gain (percentage points)", shrink=0.85)
    fig.savefig(root / "robustness_heatmap.png", dpi=240)
    fig.savefig(root / "robustness_heatmap.pdf")
    plt.close(fig)

    central = frame[frame.variant == "width_41"]
    summary = {
        "robustness": {
            "minimum_gain_points": float(frame.gain.min()),
            "maximum_gain_points": float(frame.gain.max()),
            "all_tested_variants_positive": bool((frame.gain > 0).all()),
            "note": "Post-hoc robustness only; no formal setting was selected from these test results.",
        },
        "efficiency_central_setting": {
            "mean_elapsed_seconds": float(central.elapsed_seconds.mean()),
            "mean_frames_per_second": float(central.frames_per_second.mean()),
            "max_peak_cuda_memory_mb": float(central.peak_cuda_memory_mb.max()),
            "scope": "Cached-score adapter evaluation; baseline and CLIP extraction are excluded.",
        },
    }
    (root / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(frame.to_string(index=False), flush=True)
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
