#!/usr/bin/env python3
"""Build lightweight CACC indices, normal statistics and CLIP text anchors."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

PACKAGE_ROOT = Path(__file__).resolve().parent.parent
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))

from neuron_responsibility.common import (
    base_key, clean_output, is_normal_label, load_hidden, read_feature_csv,
    read_hidden_manifest, save_json, uniform_indices, write_csv,
)
from neuron_responsibility.text_responsibility import (
    NORMAL_CONCEPTS, abnormal_concepts, encode_concepts, load_reference_clip,
)


def build_index(name: str, source_csv: str, manifest: str, output: Path, skip_missing: bool) -> dict:
    frame = read_feature_csv(source_csv)
    mapping, _ = read_hidden_manifest(manifest)
    rows, skipped = [], []
    for _, row in tqdm(frame.iterrows(), total=len(frame), desc=f"index {name}", unit="crop"):
        clip_path = str(row["path"])
        key = base_key(clip_path)
        if key not in mapping:
            if not skip_missing:
                raise FileNotFoundError(f"missing hidden state for {key}")
            skipped.append([key, str(row["label"]), clip_path, "missing_hidden"])
            continue
        rows.append([clip_path, mapping[key], str(row["label"]), key])
    csv_path = output / f"{name}.csv"
    write_csv(csv_path, ["clip_path", "hidden_path", "label", "key"], rows)
    write_csv(output / f"{name}_skipped.csv", ["key", "label", "clip_path", "reason"], skipped)
    return {"csv": str(csv_path), "rows": len(rows), "skipped": len(skipped)}


def normal_statistics(
    train_csv: str,
    dataset: str,
    samples_per_video: int,
) -> tuple[np.ndarray, np.ndarray, int, int]:
    frame = pd.read_csv(train_csv)
    normal = frame[frame["label"].map(lambda value: is_normal_label(dataset, str(value)))]
    representatives = normal.groupby("key", sort=True).first().reset_index()
    total = square = None
    count = 0
    for _, row in tqdm(representatives.iterrows(), total=len(representatives), desc="normal CLS statistics", unit="video"):
        hidden, _ = load_hidden(str(row["hidden_path"]))
        sample = hidden[uniform_indices(len(hidden), samples_per_video)].astype(np.float64)
        if total is None:
            total = np.zeros(sample.shape[1:], dtype=np.float64)
            square = np.zeros_like(total)
        total += sample.sum(axis=0)
        square += np.square(sample).sum(axis=0)
        count += len(sample)
    if total is None or square is None or count < 2:
        raise RuntimeError("no normal hidden states were available")
    center = total / count
    variance = np.maximum(square / count - np.square(center), 1e-12)
    return center.astype(np.float32), np.sqrt(variance).astype(np.float32), len(representatives), count


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare score-free cross-layer anomaly concept circuits.")
    parser.add_argument("--dataset", choices=["ucf", "xd"], required=True)
    parser.add_argument("--train-list", required=True); parser.add_argument("--test-list", required=True)
    parser.add_argument("--train-hidden-manifest", required=True); parser.add_argument("--test-hidden-manifest", required=True)
    parser.add_argument("--clip-root", required=True); parser.add_argument("--clip-weight", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--normal-stat-snippets-per-video", type=int, default=256)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--skip-missing-hidden", action="store_true")
    parser.add_argument("--clean", action="store_true")
    args = parser.parse_args()
    output = clean_output(args.out_dir, args.clean)
    artifact_path = output / "cacc_artifact.npz"
    report_path = output / "prepare_report.json"
    if artifact_path.exists() and report_path.exists() and (output / "train.csv").exists() and (output / "test.csv").exists() and not args.clean:
        print(report_path.read_text(encoding="utf-8"), flush=True)
        return
    train_report = build_index("train", args.train_list, args.train_hidden_manifest, output, args.skip_missing_hidden)
    test_report = build_index("test", args.test_list, args.test_hidden_manifest, output, args.skip_missing_hidden)
    center, scale, normal_videos, normal_samples = normal_statistics(
        train_report["csv"], args.dataset, args.normal_stat_snippets_per_video
    )
    device = torch.device(args.device if torch.cuda.is_available() and args.device.startswith("cuda") else "cpu")
    model, tokenize = load_reference_clip(args.clip_root, args.clip_weight, device)
    normal_anchors = encode_concepts(model, tokenize, NORMAL_CONCEPTS, device).cpu().numpy()
    abnormal_names = abnormal_concepts(args.dataset)
    abnormal_anchors = encode_concepts(model, tokenize, abnormal_names, device).cpu().numpy()
    np.savez_compressed(
        artifact_path, center=center, scale=scale,
        normal_anchors=normal_anchors.astype(np.float32),
        abnormal_anchors=abnormal_anchors.astype(np.float32),
    )
    report = {
        "method": "cross_layer_anomaly_concept_circuit_v1",
        "selection_source": "normal videos + frozen CLIP text anchors; no baseline anomaly scores",
        "dataset": args.dataset, "layers": int(center.shape[0]), "hidden_width": int(center.shape[1]),
        "normal_concepts": list(NORMAL_CONCEPTS), "abnormal_concepts": abnormal_names,
        "normal_videos": normal_videos, "normal_samples": normal_samples,
        "artifact": str(artifact_path), "train": train_report, "test": test_report,
    }
    save_json(report_path, report)
    print(json.dumps(report, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
