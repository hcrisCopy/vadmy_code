from __future__ import annotations

import argparse
import json
import math
from pathlib import Path


PAPER_BASELINES = {
    ("lagovad", "ucf"): ("auc", 81.12),
    ("lagovad", "xd"): ("ap", 74.25),
    ("desc", "ucf"): ("auc", 89.37),
    ("desc", "xd"): ("ap", 87.18),
    ("dsanet", "ucf"): ("auc", 89.44),
    ("dsanet", "xd"): ("ap", 86.95),
}


def main() -> None:
    parser = argparse.ArgumentParser(description="Return the minimum paper-baseline gain across all six experiments.")
    parser.add_argument("--results-root", required=True)
    args = parser.parse_args()
    root = Path(args.results_root)
    current = root / "current_run.txt"
    if current.exists():
        root = root / "runs" / current.read_text(encoding="utf-8").strip()
    results, gains = {}, []
    for (baseline, dataset), (metric, paper_value) in PAPER_BASELINES.items():
        path = root / dataset / baseline / "evaluation" / "metrics.json"
        if not path.exists():
            raise FileNotFoundError(path)
        payload = json.loads(path.read_text(encoding="utf-8"))
        value = 100.0 * float(payload["corrected"][metric])
        gain = value - paper_value
        if not math.isfinite(gain):
            raise ValueError(f"non-finite gain for {baseline}/{dataset}")
        gains.append(gain)
        results[f"{baseline}_{dataset}"] = {
            "metric": metric, "paper": paper_value, "corrected": value, "gain_pp": gain,
        }
    summary = {"minimum_gain_pp": min(gains), "experiments": results}
    (root / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False), flush=True)
    print(f"{min(gains):.8f}", flush=True)


if __name__ == "__main__":
    main()
