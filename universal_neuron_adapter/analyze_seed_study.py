from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser(description="Aggregate three-seed metrics and neuron-selection stability.")
    parser.add_argument("--root", required=True)
    parser.add_argument("--seeds", nargs="+", type=int, default=[234, 3407, 2026])
    args = parser.parse_args()
    root = Path(args.root)
    rows = []
    for dataset in ("ucf", "xd"):
        metric = "auc" if dataset == "ucf" else "ap"
        for baseline in ("lagovad", "desc", "dsanet"):
            values = []
            for seed in args.seeds:
                path = root / f"seed_{seed}" / dataset / baseline / "evaluation" / "metrics.json"
                payload = json.loads(path.read_text(encoding="utf-8"))
                values.append(100 * float(payload["corrected"][metric]))
            rows.append({"dataset": dataset, "baseline": baseline, "metric": metric, "mean": float(np.mean(values)), "std": float(np.std(values, ddof=1)), **{f"seed_{seed}": value for seed, value in zip(args.seeds, values)}})
    pd.DataFrame(rows).to_csv(root / "seed_metrics.csv", index=False)
    stability = {}
    for dataset in ("ucf", "xd"):
        sets = []
        for seed in args.seeds:
            payload = json.loads((root / f"seed_{seed}" / dataset / "primary" / "selected_neurons.json").read_text(encoding="utf-8"))
            sets.append({(int(row["layer"]), int(row["dimension"])) for row in payload["neurons"]})
        pairwise = [len(sets[left] & sets[right]) / len(sets[left] | sets[right]) for left in range(len(sets)) for right in range(left + 1, len(sets))]
        stability[dataset] = {"pairwise_jaccard": pairwise, "mean_jaccard": float(np.mean(pairwise))}
    (root / "neuron_stability.json").write_text(json.dumps(stability, indent=2), encoding="utf-8")
    print(pd.DataFrame(rows).to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
