#!/usr/bin/env python3
"""Evaluate a semantic-KNN checkpoint with the baseline's binary score."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from semantic_knn_splicing.baselines import build_baseline
from semantic_knn_splicing.common import clean_output
from semantic_knn_splicing.train_baseline import official_frame_metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Official frame AUC/AP evaluation.")
    parser.add_argument("--baseline", choices=["dsanet", "desc", "lagovad"], required=True)
    parser.add_argument("--baseline-root", required=True)
    parser.add_argument("--baseline-weight", default="")
    parser.add_argument("--sensitivity-weight", default="")
    parser.add_argument("--consistency-weight", default="")
    parser.add_argument("--dataset", choices=["ucf", "xd"], required=True)
    parser.add_argument("--test-list", required=True)
    parser.add_argument("--gt-path", required=True)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--frames-per-snippet", type=int, default=16)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--clean", action="store_true")
    args = parser.parse_args()
    output = clean_output(args.out_dir, args.clean)
    metrics_path = output / "metrics.json"
    if metrics_path.exists() and not args.clean:
        print(metrics_path.read_text(encoding="utf-8"), flush=True)
        return
    device = torch.device(args.device if torch.cuda.is_available() and args.device.startswith("cuda") else "cpu")
    adapter = build_baseline(args, str(device)).to(device)
    checkpoint = torch.load(args.model_path, map_location="cpu")
    adapter.load_state_dict(checkpoint["model_state_dict"], strict=True)
    metrics = official_frame_metrics(
        adapter, args.test_list, args.gt_path, args.frames_per_snippet, device,
        score_cache=output / "score_cache",
    )
    report = {
        "baseline": args.baseline, "dataset": args.dataset,
        "selection_rule": checkpoint.get("selection_rule"), "checkpoint_metrics": checkpoint.get("metrics"),
        "official_test": metrics,
    }
    metrics_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
