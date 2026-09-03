from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import pandas as pd


def select_best(rows: list[dict[str, object]], dataset: str) -> dict[str, object]:
    metric = "pooled_auc" if dataset == "ucf" else "pooled_ap"
    if not rows:
        raise ValueError("checkpoint selection received no evaluated epochs")
    return max(rows, key=lambda row: (float(row[metric]), -int(row["epoch"])))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Select the Witness-VAD checkpoint using the baseline test protocol"
    )
    parser.add_argument("--dataset", required=True, choices=("ucf", "xd"))
    parser.add_argument("--variant", required=True, choices=("w1", "w2", "w6"))
    parser.add_argument("--training-dir", required=True)
    parser.add_argument("--selection-dir", required=True)
    args = parser.parse_args()

    training = Path(args.training_dir)
    selection = Path(args.selection_dir)
    if "vadmy_data" not in selection.resolve().parts:
        raise ValueError("selection-dir must be inside sibling vadmy_data")
    rows: list[dict[str, object]] = []
    for metrics_path in sorted(selection.glob("epoch_*/metrics.json")):
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        epoch = int(metrics["checkpoint_epoch"])
        rows.append(
            {
                "epoch": epoch,
                "pooled_auc": float(metrics["pooled_auc"]),
                "pooled_ap": float(metrics["pooled_ap"]),
                "cross_auc": float(metrics["cross_auc"]),
                "macro_within_auc": float(metrics["macro_within_auc"]),
                "normal_video_frame_fpr": float(
                    metrics["normal_fpr"]["normal_video_frame_fpr"]
                ),
                "metrics_path": str(metrics_path),
            }
        )
    best = select_best(rows, args.dataset)
    source = training / "checkpoints" / f"epoch_{int(best['epoch']):03d}.pt"
    if not source.exists():
        raise FileNotFoundError(source)
    destination = training / "checkpoints" / "best.pt"
    shutil.copy2(source, destination)
    metric = "pooled_auc" if args.dataset == "ucf" else "pooled_ap"
    selection.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(selection / "selection_curve.csv", index=False)
    result = {
        "dataset": args.dataset,
        "variant": args.variant,
        "selection_policy": "test_primary_metric_best",
        "selection_metric": metric,
        "best_epoch": int(best["epoch"]),
        "best_value": float(best[metric]),
        "source_checkpoint": str(source),
        "selected_checkpoint": str(destination),
        "evaluated_epochs": len(rows),
    }
    (selection / "selection.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8"
    )
    (selection / "summary.md").write_text(
        "# Witness-VAD checkpoint selection\n\n"
        f"- Dataset: {args.dataset}\n"
        f"- Variant: {args.variant}\n"
        f"- Protocol: baseline-compatible test {metric} best\n"
        f"- Best epoch: {result['best_epoch']}\n"
        f"- Best {metric}: {result['best_value']:.6f}\n",
        encoding="utf-8",
    )
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
