#!/usr/bin/env python3
"""Official evaluation for pre-temporal neuron boundary conditioning."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from tqdm import tqdm


PACKAGE_ROOT = Path(__file__).resolve().parent.parent
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))

from neuron_responsibility.baselines import build_baseline
from neuron_responsibility.boundary_localization import NeuronBoundaryConditioner
from neuron_responsibility.common import clean_output
from neuron_responsibility.evaluate import (
    class_probabilities,
    file_signature,
    load_detection_map,
    pad_chunks,
    safe_frame_metrics,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Official-style evaluation for neuron boundary conditioning."
    )
    parser.add_argument("--baseline", choices=["dsanet", "desc", "lagovad"], required=True)
    parser.add_argument("--baseline-root", required=True)
    parser.add_argument("--baseline-weight", default="")
    parser.add_argument("--sensitivity-weight", default="")
    parser.add_argument("--consistency-weight", default="")
    parser.add_argument("--joint-model", required=True)
    parser.add_argument("--dataset", choices=["ucf", "xd"], required=True)
    parser.add_argument("--test-list", required=True)
    parser.add_argument("--gt-path", required=True)
    parser.add_argument("--gt-segment-path", default="")
    parser.add_argument("--gt-label-path", default="")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--frames-per-snippet", type=int, default=16)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--clean", action="store_true")
    args = parser.parse_args()

    checkpoint = torch.load(args.joint_model, map_location="cpu")
    if checkpoint.get("method") != NeuronBoundaryConditioner.method_name:
        raise ValueError(f"unsupported joint checkpoint method: {checkpoint.get('method')}")
    run_config = checkpoint.get("run_config", {})
    if run_config.get("baseline") != args.baseline or run_config.get("dataset") != args.dataset:
        raise ValueError("checkpoint baseline/dataset differs from command")

    out_dir = clean_output(args.out_dir, args.clean)
    signature = {
        "method": NeuronBoundaryConditioner.method_name,
        "baseline": args.baseline,
        "dataset": args.dataset,
        "baseline_weight": file_signature(args.baseline_weight),
        "sensitivity_weight": file_signature(args.sensitivity_weight),
        "consistency_weight": file_signature(args.consistency_weight),
        "joint_model": file_signature(args.joint_model),
        "test_list": file_signature(args.test_list),
        "frames_per_snippet": args.frames_per_snippet,
        "temperature": args.temperature,
    }
    signature_path = out_dir / "run_signature.json"
    if signature_path.exists() and not args.clean:
        previous = json.loads(signature_path.read_text(encoding="utf-8"))
        if previous != signature:
            raise RuntimeError("evaluation inputs changed; use --clean or another --out-dir")
    signature_path.write_text(json.dumps(signature, indent=2), encoding="utf-8")

    device = torch.device(
        args.device if torch.cuda.is_available() and args.device.startswith("cuda") else "cpu"
    )
    adapter = build_baseline(args, str(device)).to(device)
    conditioner = NeuronBoundaryConditioner.from_config(checkpoint["conditioner_config"]).to(device)
    adapter.attach_pre_temporal_conditioner(conditioner)
    adapter.load_state_dict(checkpoint["model_state_dict"], strict=True)
    adapter.eval()

    frame = pd.read_csv(args.test_list)
    missing = {"clip_path", "neuron_path", "label"} - set(frame.columns)
    if missing:
        raise ValueError(f"{args.test_list} is missing columns: {sorted(missing)}")
    if "key" not in frame.columns:
        frame["key"] = frame["clip_path"].map(lambda value: Path(str(value)).stem)
    temperature = args.temperature
    if temperature <= 0:
        temperature = 5.0 if args.dataset == "ucf" and args.baseline == "dsanet" else 1.0

    score_dir = out_dir / "scores"
    score_dir.mkdir(exist_ok=True)
    binary_all, semantic_all, neuron_all = [], [], []
    detection_predictions, summary_rows = [], []
    with torch.no_grad():
        for key_value, group in tqdm(
            list(frame.groupby("key", sort=False)), desc="evaluate videos", unit="video"
        ):
            key = str(key_value)
            labels = set(group["label"].astype(str))
            if len(labels) != 1:
                raise ValueError(f"{key}: inconsistent labels {sorted(labels)}")
            label = next(iter(labels))
            cache_path = score_dir / f"{key}.npz"
            if cache_path.exists() and not args.clean:
                cached = np.load(cache_path)
                binary = cached["binary"]
                semantic = cached["semantic"]
                neuron = cached["neuron"]
                class_probability = cached["class_prob"]
            else:
                clip = np.concatenate([
                    np.load(str(path)).astype(np.float32) for path in group["clip_path"]
                ])
                neurons = np.concatenate([
                    np.load(str(path)).astype(np.float32) for path in group["neuron_path"]
                ])
                clip_chunks, lengths = pad_chunks(clip, adapter.visual_length)
                neuron_chunks, neuron_lengths = pad_chunks(neurons, adapter.visual_length)
                if not torch.equal(lengths, neuron_lengths):
                    raise RuntimeError(f"{key}: modality chunk lengths differ")
                output, records = adapter.forward_conditioned(
                    clip_chunks.to(device), neuron_chunks.to(device), lengths.to(device)
                )
                neuron_batch = torch.stack([
                    torch.sigmoid(record["localizer_logits"])
                    for record in records
                    if isinstance(record["localizer_logits"], torch.Tensor)
                ]).mean(dim=0)
                semantic_batch = 1.0 - F.softmax(
                    output.semantic_logits / temperature, dim=-1
                )[..., 0]
                class_batch = class_probabilities(
                    output.binary_logits, output.semantic_logits, args.baseline, temperature
                )
                valid_binary, valid_semantic, valid_neuron, valid_class = [], [], [], []
                for index, length in enumerate(lengths.tolist()):
                    valid_binary.append(torch.sigmoid(output.binary_logits[index, :length]).cpu())
                    valid_semantic.append(semantic_batch[index, :length].cpu())
                    valid_neuron.append(neuron_batch[index, :length].cpu())
                    valid_class.append(class_batch[index, :length].cpu())
                binary = torch.cat(valid_binary).numpy().astype(np.float32)
                semantic = torch.cat(valid_semantic).numpy().astype(np.float32)
                neuron = torch.cat(valid_neuron).numpy().astype(np.float32)
                class_probability = torch.cat(valid_class).numpy().astype(np.float32)
                np.savez_compressed(
                    cache_path,
                    binary=binary,
                    semantic=semantic,
                    neuron=neuron,
                    class_prob=class_probability,
                )
            binary_all.append(binary)
            semantic_all.append(semantic)
            neuron_all.append(neuron)
            detection_predictions.append(
                np.repeat(class_probability, args.frames_per_snippet, axis=0)
            )
            summary_rows.append([
                key, label, len(binary), float(binary.mean()),
                float(semantic.mean()), float(neuron.mean()),
            ])

    truth = np.load(args.gt_path).astype(np.int64).reshape(-1)
    metrics = {}
    for name, values in {
        "binary": binary_all,
        "semantic": semantic_all,
        "independent_neuron": neuron_all,
    }.items():
        prediction = np.repeat(np.concatenate(values), args.frames_per_snippet)
        metrics[name] = safe_frame_metrics(truth, prediction)

    if args.gt_segment_path and args.gt_label_path:
        utility = Path(args.baseline_root) / "src" / "utils" / f"{args.dataset}_detectionMAP.py"
        if utility.is_file():
            detection_map = load_detection_map(utility)
            maps, thresholds = detection_map(
                detection_predictions,
                np.load(args.gt_segment_path, allow_pickle=True),
                np.load(args.gt_label_path, allow_pickle=True),
                excludeNormal=False,
            )
            metrics["detection_map"] = {
                f"iou_{float(iou):.1f}": float(value)
                for iou, value in zip(thresholds, maps)
            }
            metrics["detection_map_average"] = float(np.mean(maps[:5]))

    (out_dir / "metrics.json").write_text(
        json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    with (out_dir / "per_video.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow([
            "key", "label", "snippets", "binary_mean", "semantic_mean", "neuron_mean"
        ])
        writer.writerows(summary_rows)
    print(json.dumps(metrics, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()

