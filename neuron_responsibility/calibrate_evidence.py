#!/usr/bin/env python3
"""Calibrate score-free neuron evidence from fully normal training videos."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm


PACKAGE_ROOT = Path(__file__).resolve().parent.parent
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))

from neuron_responsibility.common import clean_output, is_normal_label, load_json, uniform_indices


def file_hash(path: str) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def flattened_neurons(config: dict) -> list[dict[str, float | int]]:
    flattened = []
    feature_index = 0
    for group in config["selected"]:
        layer = int(group["layer_index"])
        directions = group.get("directions", [1] * len(group["dims"]))
        scores = group.get("scores", [0.0] * len(group["dims"]))
        for dimension, direction, score in zip(group["dims"], directions, scores):
            flattened.append({
                "feature_index": feature_index,
                "layer_index": layer,
                "dimension": int(dimension),
                "direction": int(direction),
                "selection_score": float(score),
            })
            feature_index += 1
    return flattened


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Estimate per-neuron normal thresholds without baseline predictions."
    )
    parser.add_argument("--dataset", choices=["ucf", "xd"], required=True)
    parser.add_argument("--train-list", required=True)
    parser.add_argument("--neuron-json", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--active-neurons", type=int, default=128)
    parser.add_argument("--normal-quantile", type=float, default=0.99)
    parser.add_argument("--snippets-per-video", type=int, default=64)
    parser.add_argument("--clean", action="store_true")
    args = parser.parse_args()
    if args.active_neurons <= 0 or args.snippets_per_video <= 0:
        parser.error("--active-neurons and --snippets-per-video must be positive")
    if not 0.5 < args.normal_quantile < 1.0:
        parser.error("--normal-quantile must be in (0.5, 1.0)")

    out_dir = clean_output(args.out_dir, args.clean)
    signature = {
        "dataset": args.dataset,
        "train_list_sha256": file_hash(args.train_list),
        "neuron_json_sha256": file_hash(args.neuron_json),
        "active_neurons": args.active_neurons,
        "normal_quantile": args.normal_quantile,
        "snippets_per_video": args.snippets_per_video,
    }
    signature_path = out_dir / "calibration_signature.json"
    config_path = out_dir / "evidence_config.json"
    array_path = out_dir / "evidence_config.npz"
    if signature_path.exists() and config_path.exists() and array_path.exists() and not args.clean:
        previous = json.loads(signature_path.read_text(encoding="utf-8"))
        if previous == signature:
            print(f"reusing {array_path}", flush=True)
            return
        raise RuntimeError("calibration inputs changed; rerun with --clean or use another --out-dir")

    frame = pd.read_csv(args.train_list)
    missing = {"neuron_path", "label", "key"} - set(frame.columns)
    if missing:
        raise ValueError(f"{args.train_list} is missing columns: {sorted(missing)}")
    normal = frame[frame["label"].map(lambda value: is_normal_label(args.dataset, str(value)))]
    if normal.empty:
        raise RuntimeError("normal training videos are required for evidence calibration")

    # Each UCF video has ten crop rows that share the same hidden states.  One
    # row per video prevents duplicated crops from biasing normal quantiles.
    representatives = normal.groupby("key", sort=True).first().reset_index()
    samples = []
    width = None
    for _, row in tqdm(
        representatives.iterrows(),
        total=len(representatives),
        desc="normal evidence calibration",
        unit="video",
    ):
        neuron = np.load(str(row["neuron_path"]), mmap_mode="r")
        if neuron.ndim != 2 or not len(neuron):
            raise ValueError(f"{row['neuron_path']}: expected non-empty [T,K], got {neuron.shape}")
        width = int(neuron.shape[1]) if width is None else width
        if neuron.shape[1] != width:
            raise ValueError("aligned neuron files have inconsistent widths")
        samples.append(np.asarray(neuron[uniform_indices(len(neuron), args.snippets_per_video)], dtype=np.float32))
    normal_samples = np.concatenate(samples, axis=0)
    thresholds_all = np.quantile(normal_samples, args.normal_quantile, axis=0).astype(np.float32)

    selection = load_json(args.neuron_json)
    neurons = flattened_neurons(selection)
    if width != len(neurons) or int(selection["neuron_width"]) != width:
        raise ValueError("neuron selection metadata does not match aligned feature width")
    active_count = min(args.active_neurons, width)
    ranked = sorted(neurons, key=lambda item: item["selection_score"], reverse=True)
    active = ranked[:active_count]
    active_indices = np.asarray([item["feature_index"] for item in active], dtype=np.int64)
    thresholds = thresholds_all[active_indices]
    exceedance = (normal_samples[:, active_indices] > thresholds[None, :]).mean(axis=0)

    np.savez_compressed(
        array_path,
        active_indices=active_indices,
        thresholds=thresholds,
        selection_scores=np.asarray([item["selection_score"] for item in active], dtype=np.float32),
        normal_exceedance=exceedance.astype(np.float32),
    )
    metadata = {
        "method": "normal_quantile_signed_neuron_evidence_v1",
        "dataset": args.dataset,
        "neuron_width": width,
        "active_neurons": active_count,
        "normal_quantile": args.normal_quantile,
        "normal_video_count": len(representatives),
        "normal_sample_count": int(len(normal_samples)),
        "mean_normal_exceedance": float(exceedance.mean()),
        "array_path": array_path.name,
        "active": active,
    }
    config_path.write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")
    signature_path.write_text(json.dumps(signature, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({key: value for key, value in metadata.items() if key != "active"}, indent=2), flush=True)


if __name__ == "__main__":
    main()
