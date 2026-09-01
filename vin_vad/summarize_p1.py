from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize P1 E0--E3 results.")
    parser.add_argument("--root", required=True)
    args = parser.parse_args()
    root = Path(args.root)
    rows = []
    for dataset in ("ucf", "xd"):
        for variant in ("e0", "e1", "e2", "e3"):
            path = root / dataset / variant / "evaluation" / "metrics.json"
            if not path.exists():
                raise FileNotFoundError(path)
            metrics = json.loads(path.read_text(encoding="utf-8"))
            rows.append(metrics)
    frame = pd.DataFrame(rows)
    frame.to_csv(root / "summary.csv", index=False)
    payload = {"status": "pass", "results": rows}
    (root / "summary.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(frame[["dataset", "variant", "frame_auc", "frame_ap", "normal_video_score_length_pearson", "mean_anomaly_video_positive_runs_at_0.5"]].to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
