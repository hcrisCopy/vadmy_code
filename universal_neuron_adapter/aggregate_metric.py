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
    ("vadclip", "ucf"): ("auc", 88.02),
    ("vadclip", "xd"): ("ap", 84.51),
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-root", required=True)
    args = parser.parse_args()
    root = Path(args.results_root)
    current = root / "current_run.txt"
    if current.exists():
        root = root / "runs" / current.read_text(encoding="utf-8").strip()
    gains, experiments = [], {}
    for (baseline, dataset), (metric, paper) in PAPER_BASELINES.items():
        path = root / dataset / baseline / "evaluation" / "metrics.json"
        payload = json.loads(path.read_text(encoding="utf-8"))
        value = 100.0 * float(payload["corrected"][metric])
        gain = value - paper
        if not math.isfinite(gain):
            raise ValueError(f"non-finite gain for {baseline}/{dataset}")
        gains.append(gain)
        experiments[f"{baseline}_{dataset}"] = {"metric": metric, "paper": paper, "corrected": value, "gain_pp": gain}
    summary = {"minimum_gain_pp": min(gains), "experiments": experiments}
    (root / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False), flush=True)
    print(f"{min(gains):.8f}", flush=True)


if __name__ == "__main__":
    main()

