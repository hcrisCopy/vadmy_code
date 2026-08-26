from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.ndimage import maximum_filter1d
from sklearn.metrics import average_precision_score, roc_auc_score
from tqdm import tqdm

from universal_neuron_adapter.data import resample_curve


def load_curves(manifest: str) -> tuple[list[str], list[np.ndarray]]:
    frame = pd.read_csv(manifest)
    keys, curves = [], []
    for row in frame.itertuples(index=False):
        keys.append(str(row.key))
        curves.append(np.load(str(row.baseline_score_path)).astype(np.float32))
    return keys, curves


def empirical_cdf(reference: np.ndarray, values: np.ndarray) -> np.ndarray:
    return np.searchsorted(reference, values, side="right").astype(np.float32) / float(len(reference) + 1)


def metrics(truth: np.ndarray, scores: np.ndarray) -> dict[str, float]:
    return {
        "auc": float(roc_auc_score(truth, scores)),
        "ap": float(average_precision_score(truth, scores)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate neuron-gated dual-anchor consensus.")
    parser.add_argument("--desc-train", required=True)
    parser.add_argument("--dsanet-train", required=True)
    parser.add_argument("--desc-test", required=True)
    parser.add_argument("--dsanet-test", required=True)
    parser.add_argument("--lagovad-test", required=True)
    parser.add_argument("--expert-test", required=True)
    parser.add_argument("--gt-path", required=True)
    parser.add_argument("--dataset", choices=["ucf", "xd"], required=True)
    parser.add_argument("--out-root", required=True)
    parser.add_argument("--rank-weight", type=float, default=0.5)
    parser.add_argument("--event-width", type=int, default=25)
    parser.add_argument("--event-weight", type=float, default=0.5)
    parser.add_argument("--neuron-weight", type=float, default=0.15)
    parser.add_argument("--frames-per-snippet", type=int, default=16)
    args = parser.parse_args()

    desc_train = np.sort(np.concatenate(load_curves(args.desc_train)[1]))
    dsanet_train = np.sort(np.concatenate(load_curves(args.dsanet_train)[1]))
    desc_keys, desc_curves = load_curves(args.desc_test)
    dsanet_keys, dsanet_curves = load_curves(args.dsanet_test)
    lagovad_keys, lagovad_curves = load_curves(args.lagovad_test)
    if desc_keys != dsanet_keys or desc_keys != lagovad_keys:
        raise RuntimeError("strict test video ordering differs between anchor baselines")
    expert_frame = pd.read_csv(args.expert_test).set_index("key")

    corrected_curves, rows = [], []
    for key, desc, dsanet in tqdm(
        zip(desc_keys, desc_curves, dsanet_curves), total=len(desc_keys), desc=f"dual-anchor consensus/{args.dataset}"
    ):
        if len(desc) != len(dsanet):
            raise RuntimeError(f"anchor snippet lengths differ for {key}: {len(desc)} vs {len(dsanet)}")
        probability_consensus = 0.5 * (desc + dsanet)
        rank_consensus = 0.5 * (empirical_cdf(desc_train, desc) + empirical_cdf(dsanet_train, dsanet))
        calibrated = (1.0 - args.rank_weight) * probability_consensus + args.rank_weight * rank_consensus
        expanded = maximum_filter1d(calibrated, args.event_width, mode="nearest")
        event_score = (1.0 - args.event_weight) * calibrated + args.event_weight * expanded
        expert_path = str(expert_frame.loc[key, "expert_score_path"])
        neuron = resample_curve(np.load(expert_path), len(event_score))
        neuron = (neuron - neuron.mean()) / max(float(neuron.std()), 1e-6)
        event_logit = np.log(np.clip(event_score, 1e-5, 1.0 - 1e-5) / np.clip(1.0 - event_score, 1e-5, 1.0))
        corrected = 1.0 / (1.0 + np.exp(-(event_logit + args.neuron_weight * neuron)))
        corrected_curves.append(corrected.astype(np.float32))
        rows.append({"key": key, "snippets": len(corrected), "corrected_mean": float(corrected.mean())})

    truth = np.load(args.gt_path).astype(np.int64).reshape(-1)
    corrected_frames = np.repeat(np.concatenate(corrected_curves), args.frames_per_snippet)
    if len(truth) != len(corrected_frames):
        raise RuntimeError(f"strict frame alignment failed: gt={len(truth)} corrected={len(corrected_frames)}")
    corrected_metrics = metrics(truth, corrected_frames)
    baseline_curves = {"lagovad": lagovad_curves, "desc": desc_curves, "dsanet": dsanet_curves}
    configuration = {
        "anchors": ["desc", "dsanet"],
        "cdf_source": "training scores only",
        "rank_weight": args.rank_weight,
        "event_width": args.event_width,
        "event_weight": args.event_weight,
        "neuron_weight": args.neuron_weight,
    }
    for baseline, curves in baseline_curves.items():
        baseline_frames = np.repeat(np.concatenate(curves), args.frames_per_snippet)
        if len(truth) != len(baseline_frames):
            raise RuntimeError(f"strict baseline alignment failed for {baseline}")
        output = Path(args.out_root) / baseline / "evaluation"
        output.mkdir(parents=True, exist_ok=True)
        payload = {
            "baseline": metrics(truth, baseline_frames),
            "corrected": corrected_metrics,
            "configuration": configuration,
            "frames": len(truth),
        }
        (output / "metrics.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
        pd.DataFrame(rows).to_csv(output / "per_video.csv", index=False)
        print(json.dumps({"baseline": baseline, "dataset": args.dataset, **payload}), flush=True)


if __name__ == "__main__":
    main()
