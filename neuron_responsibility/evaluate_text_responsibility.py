#!/usr/bin/env python3
"""Falsification gate for a text-grounded neuron prior and its complementarity."""

from __future__ import annotations

import argparse
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
from neuron_responsibility.common import clean_output, load_json
from neuron_responsibility.evaluate import pad_chunks


def frame_metrics(truth: np.ndarray, snippets: np.ndarray, repeat: int) -> dict[str, float]:
    prediction = np.repeat(snippets, repeat)
    usable = min(len(truth), len(prediction))
    return {
        "auc": float(roc_auc_score(truth[:usable], prediction[:usable])),
        "ap": float(average_precision_score(truth[:usable], prediction[:usable])),
        "frames": int(usable),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate the neuron prior before baseline training.")
    parser.add_argument("--baseline", choices=["dsanet", "desc", "lagovad"], required=True)
    parser.add_argument("--baseline-root", required=True)
    parser.add_argument("--baseline-weight", default="")
    parser.add_argument("--sensitivity-weight", default="")
    parser.add_argument("--consistency-weight", default="")
    parser.add_argument("--joint-model", default="")
    parser.add_argument("--dataset", choices=["ucf", "xd"], required=True)
    parser.add_argument("--test-list", required=True)
    parser.add_argument("--neuron-json", required=True)
    parser.add_argument("--gt-path", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--frames-per-snippet", type=int, default=16)
    parser.add_argument("--projection-cosine-min", type=float, default=0.90)
    parser.add_argument("--prior-auc-min", type=float, default=0.60)
    parser.add_argument("--random-auc-margin", type=float, default=0.01)
    parser.add_argument("--stability-min", type=float, default=0.75)
    parser.add_argument("--correlation-max", type=float, default=0.95)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--clean", action="store_true")
    args = parser.parse_args()
    output = clean_output(args.out_dir, args.clean)
    metrics_path = output / "gate_metrics.json"
    if metrics_path.exists() and not args.clean:
        print(metrics_path.read_text(encoding="utf-8"), flush=True)
        return
    device = torch.device(args.device if torch.cuda.is_available() and args.device.startswith("cuda") else "cpu")
    adapter = build_baseline(args, str(device)).to(device).eval()
    if args.joint_model:
        checkpoint = torch.load(args.joint_model, map_location="cpu")
        adapter.load_state_dict(checkpoint["model_state_dict"], strict=True)
    frame = pd.read_csv(args.test_list)
    required = {"clip_path", "neuron_path", "label", "key"}
    if required - set(frame.columns):
        raise ValueError(f"{args.test_list}: missing {sorted(required - set(frame.columns))}")
    baseline_all, selected_all, random_all = [], [], []
    groups = list(frame.groupby("key", sort=False))
    with torch.no_grad():
        for key, group in tqdm(groups, desc="responsibility gate", unit="video"):
            clip = np.concatenate([np.load(str(path)).astype(np.float32) for path in group["clip_path"]])
            prior = np.concatenate([np.load(str(path)).astype(np.float32) for path in group["neuron_path"]])
            if prior.ndim != 2 or prior.shape[1] < 2 or len(prior) != len(clip):
                raise ValueError(f"{key}: invalid prior {prior.shape} for CLIP {clip.shape}")
            chunks, lengths = pad_chunks(clip, adapter.visual_length)
            result = adapter.forward_baseline(chunks.to(device), lengths.to(device))
            valid = [torch.sigmoid(result.binary_logits[i, :length]).cpu() for i, length in enumerate(lengths.tolist())]
            baseline_all.append(torch.cat(valid).numpy())
            selected_all.append(prior[:, 0])
            random_all.append(prior[:, 1])
    truth = np.load(args.gt_path).astype(np.int64).reshape(-1)
    baseline = np.concatenate(baseline_all)
    selected = np.concatenate(selected_all)
    random_control = np.concatenate(random_all)
    # Rank normalization makes this diagnostic independent of score calibration.
    baseline_rank = np.argsort(np.argsort(baseline)).astype(np.float32) / max(1, len(baseline) - 1)
    selected_rank = np.argsort(np.argsort(selected)).astype(np.float32) / max(1, len(selected) - 1)
    fused = 0.5 * baseline_rank + 0.5 * selected_rank
    artifact = load_json(args.neuron_json)
    stability = np.asarray(artifact["fold_stability"])[artifact["selected_indices"]]
    projection = float(artifact["projection_validation"]["mean_cosine"])
    correlation = float(np.corrcoef(baseline, selected)[0, 1])
    metrics = {
        "baseline": frame_metrics(truth, baseline, args.frames_per_snippet),
        "selected_prior": frame_metrics(truth, selected, args.frames_per_snippet),
        "random_control": frame_metrics(truth, random_control, args.frames_per_snippet),
        "rank_fusion_diagnostic": frame_metrics(truth, fused, args.frames_per_snippet),
        "projection_mean_cosine": projection,
        "selected_stability_mean": float(stability.mean()),
        "selected_concept_entropy_mean": artifact["selected_concept_entropy_mean"],
        "baseline_prior_pearson": correlation,
    }
    checks = {
        "projection": projection >= args.projection_cosine_min,
        "localization": metrics["selected_prior"]["auc"] >= args.prior_auc_min,
        "beats_random": metrics["selected_prior"]["auc"] >= metrics["random_control"]["auc"] + args.random_auc_margin,
        "stable": metrics["selected_stability_mean"] >= args.stability_min,
        "complementary": abs(correlation) <= args.correlation_max,
    }
    metrics["checks"] = checks
    metrics["gate_passed"] = bool(all(checks.values()))
    metrics["training_policy"] = "train last temporal block only if gate_passed is true"
    metrics_path.write_text(json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(metrics, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
