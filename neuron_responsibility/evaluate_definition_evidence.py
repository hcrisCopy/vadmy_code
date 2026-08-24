#!/usr/bin/env python3
"""Evaluate a definition-evidence checkpoint through the original baseline path."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import average_precision_score, roc_auc_score
from tqdm import tqdm

PACKAGE_ROOT = Path(__file__).resolve().parent.parent
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))

from neuron_responsibility.baselines import build_baseline
from neuron_responsibility.common import clean_output
from neuron_responsibility.evaluate import pad_chunks


def collect(adapter, csv_path: str, device: torch.device, description: str) -> tuple[np.ndarray, list[list[object]]]:
    frame = pd.read_csv(csv_path); all_scores, rows = [], []
    adapter.eval()
    with torch.no_grad():
        for key, group in tqdm(list(frame.groupby("key", sort=False)), desc=description, unit="video"):
            clip = np.concatenate([np.load(str(path)).astype(np.float32) for path in group["clip_path"]])
            chunks, lengths = pad_chunks(clip, adapter.visual_length)
            output = adapter.forward_baseline(chunks.to(device), lengths.to(device))
            scores = torch.cat([
                torch.sigmoid(output.binary_logits[index, :length]).cpu()
                for index, length in enumerate(lengths.tolist())
            ]).numpy()
            all_scores.append(scores)
            rows.append([str(key), str(group.iloc[0]["label"]), len(scores), float(scores.mean()), float(scores.max())])
    return np.concatenate(all_scores), rows


def metrics(truth: np.ndarray, snippets: np.ndarray, repeat: int) -> dict[str, float]:
    prediction = np.repeat(snippets, repeat); usable = min(len(truth), len(prediction))
    return {"auc": float(roc_auc_score(truth[:usable], prediction[:usable])),
            "ap": float(average_precision_score(truth[:usable], prediction[:usable])), "frames": int(usable)}


def main() -> None:
    parser = argparse.ArgumentParser(description="Original-path evaluation after definition-evidence training.")
    parser.add_argument("--baseline", choices=["dsanet", "desc", "lagovad"], required=True)
    parser.add_argument("--baseline-root", required=True); parser.add_argument("--baseline-weight", default="")
    parser.add_argument("--sensitivity-weight", default=""); parser.add_argument("--consistency-weight", default="")
    parser.add_argument("--dataset", choices=["ucf", "xd"], required=True)
    parser.add_argument("--test-list", required=True); parser.add_argument("--gt-path", required=True)
    parser.add_argument("--model-path", required=True); parser.add_argument("--out-dir", required=True)
    parser.add_argument("--frames-per-snippet", type=int, default=16); parser.add_argument("--device", default="cuda")
    parser.add_argument("--clean", action="store_true")
    args = parser.parse_args()
    output = clean_output(args.out_dir, args.clean); metrics_path = output / "metrics.json"
    if metrics_path.exists() and not args.clean:
        print(metrics_path.read_text(encoding="utf-8")); return
    device = torch.device(args.device if torch.cuda.is_available() and args.device.startswith("cuda") else "cpu")
    adapter = build_baseline(args, str(device)).to(device)
    released, _ = collect(adapter, args.test_list, device, "released baseline")
    checkpoint = torch.load(args.model_path, map_location="cpu")
    adapter.load_state_dict(checkpoint["model_state_dict"], strict=True)
    trained, rows = collect(adapter, args.test_list, device, "definition-evidence model")
    truth = np.load(args.gt_path).astype(np.int64).reshape(-1)
    result = {
        "released_author": metrics(truth, released, args.frames_per_snippet),
        "definition_evidence": metrics(truth, trained, args.frames_per_snippet),
        "inference": "original baseline path; no circuit feature is loaded",
        "checkpoint": args.model_path,
    }
    metrics_path.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    with (output / "per_video.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle); writer.writerow(["key", "label", "snippets", "mean", "max"]); writer.writerows(rows)
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
