from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare selected-neuron removal against matched random removals.")
    parser.add_argument("--root", required=True)
    parser.add_argument("--full-root", required=True)
    parser.add_argument("--random-seeds", nargs="+", type=int, default=[234, 3407, 2026, 17, 73])
    args = parser.parse_args()
    root, full_root = Path(args.root), Path(args.full_root)
    rows = []
    for dataset in ("ucf", "xd"):
        name = "auc" if dataset == "ucf" else "ap"
        for baseline in ("lagovad", "desc", "dsanet"):
            full = json.loads((full_root / dataset / baseline / "metrics.json").read_text(encoding="utf-8"))["corrected"][name]
            selected = json.loads((root / "remove_selected" / "seed_3407" / dataset / baseline / "metrics.json").read_text(encoding="utf-8"))["corrected"][name]
            random_values = [json.loads((root / "random_matched" / f"seed_{seed}" / dataset / baseline / "metrics.json").read_text(encoding="utf-8"))["corrected"][name] for seed in args.random_seeds]
            rows.append({"dataset": dataset, "baseline": baseline, "metric": name, "full": 100 * full, "selected_removal_drop": 100 * (full - selected), "random_removal_drop_mean": 100 * (full - np.mean(random_values)), "random_removal_drop_std": 100 * np.std(random_values, ddof=1)})
    frame = pd.DataFrame(rows)
    frame.to_csv(root / "neuron_control_summary.csv", index=False)
    print(frame.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
