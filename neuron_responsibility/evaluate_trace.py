#!/usr/bin/env python3
"""Official frame evaluation and evidence ablations for TRACE."""

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
from neuron_responsibility.trace import TraceNeuronEvidence, TraceThresholds
from neuron_responsibility.train_cacc import group_features, pad_chunks


def frame_metrics(truth: np.ndarray, snippets: list[np.ndarray], repeat: int) -> dict[str, float]:
    prediction = np.repeat(np.concatenate(snippets), repeat)
    usable = min(len(truth), len(prediction))
    return {
        "frame_auc": float(roc_auc_score(truth[:usable], prediction[:usable])),
        "frame_ap": float(average_precision_score(truth[:usable], prediction[:usable])),
        "frames": int(usable),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate released and TRACE-adapted baselines.")
    parser.add_argument("--baseline", choices=["dsanet", "desc", "lagovad"], required=True)
    parser.add_argument("--baseline-root", required=True); parser.add_argument("--baseline-weight", default="")
    parser.add_argument("--sensitivity-weight", default=""); parser.add_argument("--consistency-weight", default="")
    parser.add_argument("--dataset", choices=["ucf", "xd"], required=True)
    parser.add_argument("--test-list", required=True); parser.add_argument("--gt-path", required=True)
    parser.add_argument("--model-path", required=True); parser.add_argument("--out-dir", required=True)
    parser.add_argument("--frames-per-snippet", type=int, default=16)
    parser.add_argument("--device", default="cuda"); parser.add_argument("--clean", action="store_true")
    args = parser.parse_args()
    checkpoint = torch.load(args.model_path, map_location="cpu")
    if checkpoint.get("method") != TraceNeuronEvidence.method_name:
        raise ValueError(f"unsupported checkpoint method: {checkpoint.get('method')}")
    config = checkpoint["run_config"]
    if config["baseline"] != args.baseline or config["dataset"] != args.dataset:
        raise ValueError("checkpoint baseline or dataset differs from command")
    if not checkpoint.get("thresholds"):
        raise ValueError("TRACE checkpoint has no calibrated evidence thresholds")
    output = clean_output(args.out_dir, args.clean)
    score_dir = output / "scores"; score_dir.mkdir(exist_ok=True)
    device = torch.device(args.device if torch.cuda.is_available() and args.device.startswith("cuda") else "cpu")
    author = build_baseline(args, str(device)).to(device).eval()
    adapted = build_baseline(args, str(device)).to(device)
    adapted.load_state_dict(checkpoint["baseline_state_dict"], strict=True); adapted.eval()
    evidence = TraceNeuronEvidence.from_config(checkpoint["evidence_config"]).to(device)
    evidence.load_state_dict(checkpoint["evidence_state_dict"], strict=True); evidence.eval()
    thresholds = TraceThresholds.from_dict(checkpoint["thresholds"])
    frame = pd.read_csv(args.test_list)
    truth = np.load(args.gt_path).astype(np.int64).reshape(-1)
    names = ["released_author", "trace_adapted", "evidence_semantic", "evidence_temporal", "evidence_joint"]
    all_scores: dict[str, list[np.ndarray]] = {name: [] for name in names}
    rows = []
    with torch.no_grad():
        for key_value, group in tqdm(list(frame.groupby("key", sort=False)), desc="evaluate TRACE", unit="video"):
            key = str(key_value); cache_path = score_dir / f"{key}.npz"
            if cache_path.exists() and not args.clean:
                cached = np.load(cache_path); values = {name: cached[name] for name in names}
            else:
                clip, hidden = group_features(group)
                clip_chunks, lengths = pad_chunks(clip, author.visual_length)
                hidden_chunks, hidden_lengths = pad_chunks(hidden, author.visual_length)
                if not torch.equal(lengths, hidden_lengths):
                    raise RuntimeError(f"{key}: modality lengths differ")
                lengths_device = lengths.to(device)
                author_output = author.forward_baseline(clip_chunks.to(device), lengths_device)
                adapted_output = adapted.forward_baseline(clip_chunks.to(device), lengths_device)
                record = evidence(hidden_chunks.to(device), lengths_device)
                semantic = torch.sigmoid(record["semantic_logits"])
                temporal = record["temporal_score"]
                semantic_scaled = torch.sigmoid(
                    (semantic - thresholds.semantic_high)
                    / max(0.02, thresholds.semantic_high - thresholds.semantic_low)
                )
                temporal_scaled = torch.sigmoid(
                    (temporal - thresholds.temporal_high)
                    / max(0.02, thresholds.temporal_high - thresholds.temporal_low)
                )
                joint = 0.5 * semantic_scaled + 0.5 * temporal_scaled
                parts = {name: [] for name in names}
                for index, length in enumerate(lengths.tolist()):
                    parts["released_author"].append(torch.sigmoid(author_output.binary_logits[index, :length]).cpu())
                    parts["trace_adapted"].append(torch.sigmoid(adapted_output.binary_logits[index, :length]).cpu())
                    parts["evidence_semantic"].append(semantic[index, :length].cpu())
                    parts["evidence_temporal"].append(temporal[index, :length].cpu())
                    parts["evidence_joint"].append(joint[index, :length].cpu())
                values = {name: torch.cat(value).numpy().astype(np.float32) for name, value in parts.items()}
                np.savez_compressed(cache_path, **values)
            for name, value in values.items():
                all_scores[name].append(value)
            rows.append([key, str(group.iloc[0]["label"]), len(values["released_author"])]
                        + [float(values[name].mean()) for name in names])
    result = {name: frame_metrics(truth, values, args.frames_per_snippet) for name, values in all_scores.items()}
    primary = "frame_auc" if args.dataset == "ucf" else "frame_ap"
    result["primary_metric"] = primary
    result["absolute_gain"] = result["trace_adapted"][primary] - result["released_author"][primary]
    result["checkpoint_validation"] = checkpoint.get("metrics", {})
    result["thresholds"] = thresholds.as_dict()
    (output / "metrics.json").write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    with (output / "per_video.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["key", "label", "snippets"] + [f"{name}_mean" for name in names]); writer.writerows(rows)
    print(json.dumps(result, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
