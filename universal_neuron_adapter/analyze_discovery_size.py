from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def selected_set(path: Path) -> set[tuple[int, int]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    entries = payload if isinstance(payload, list) else payload["neurons"]
    return {(int(item["layer"]), int(item["dimension"])) for item in entries}


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize neuron-discovery data-size sensitivity.")
    parser.add_argument("--root", required=True)
    parser.add_argument("--full-selected", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--dataset", choices=["ucf", "xd"], required=True)
    args = parser.parse_args()
    root, output = Path(args.root), Path(args.out_dir)
    output.mkdir(parents=True, exist_ok=True)
    reference = selected_set(Path(args.full_selected))
    rows = []
    for fraction in (25, 50, 100):
        directory = root / f"fraction_{fraction:03d}"
        metrics = json.loads((directory / "test_metrics.json").read_text(encoding="utf-8"))
        chosen = selected_set(directory / "selected_neurons.json")
        union = reference | chosen
        rows.append({
            "dataset": args.dataset,
            "train_fraction": fraction / 100,
            "video_auc": metrics["video_auc"],
            "video_ap": metrics["video_ap"],
            "selection_jaccard_with_full": len(reference & chosen) / max(1, len(union)),
        })
    frame = pd.DataFrame(rows)
    frame.to_csv(output / "discovery_size.csv", index=False)

    plt.rcParams.update({"font.size": 8, "font.family": "DejaVu Sans"})
    figure, left = plt.subplots(figsize=(5.4, 3.4), constrained_layout=True)
    right = left.twinx()
    left.plot(100 * frame.train_fraction, 100 * frame.video_auc, marker="o", linewidth=2, color="#0072B2", label="Video AUC")
    left.plot(100 * frame.train_fraction, 100 * frame.video_ap, marker="s", linewidth=2, color="#D55E00", label="Video AP")
    right.plot(100 * frame.train_fraction, 100 * frame.selection_jaccard_with_full, marker="^", linestyle="--", linewidth=1.8, color="#009E73", label="Neuron Jaccard")
    left.set_xlabel("Discovery-training videos retained (%)")
    left.set_ylabel("Video metric (%)")
    right.set_ylabel("Selection overlap with full data (%)")
    left.set_title(f"Neuron discovery data efficiency - {args.dataset.upper()}")
    left.spines["top"].set_visible(False)
    right.spines["top"].set_visible(False)
    handles_left, labels_left = left.get_legend_handles_labels()
    handles_right, labels_right = right.get_legend_handles_labels()
    left.legend(handles_left + handles_right, labels_left + labels_right, frameon=False, fontsize=7, loc="lower right")
    figure.savefig(output / "discovery_size.png", dpi=300)
    figure.savefig(output / "discovery_size.pdf")
    plt.close(figure)
    print(frame.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
