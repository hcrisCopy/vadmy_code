#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.metrics import average_precision_score, roc_auc_score
from tqdm import tqdm


PACKAGE_ROOT = Path(__file__).resolve().parent.parent
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))

from neuron_responsibility.baselines import build_baseline
from neuron_responsibility.common import clean_output
from neuron_responsibility.model import NeuronResponsibilityProbe, ResponsibilityCorrectionHead


def load_probe(path: str, device: torch.device) -> NeuronResponsibilityProbe:
    checkpoint = torch.load(path, map_location="cpu")
    config = checkpoint["config"]
    probe = NeuronResponsibilityProbe(
        config["neuron_width"], config["hidden_width"],
        active_neurons=config.get("active_neurons", config["neuron_width"]),
    )
    probe.load_state_dict(checkpoint["model_state_dict"])
    return probe.to(device).eval()


def pad_chunks(array: np.ndarray, chunk_length: int) -> tuple[torch.Tensor, torch.Tensor]:
    chunks, lengths = [], []
    for start in range(0, max(1, len(array)), chunk_length):
        part = array[start:start + chunk_length]
        if not len(part):
            continue
        lengths.append(len(part))
        if len(part) < chunk_length:
            part = np.pad(part, ((0, chunk_length - len(part)), (0, 0)), mode="constant")
        chunks.append(part)
    return torch.from_numpy(np.stack(chunks).astype(np.float32)), torch.tensor(lengths, dtype=torch.long)


def class_probabilities(binary_logits: torch.Tensor, semantic_logits: torch.Tensor, baseline: str, temperature: float) -> torch.Tensor:
    semantic = F.softmax(semantic_logits / temperature, dim=-1)
    if baseline == "dsanet":
        abnormal_total = torch.sigmoid(binary_logits / temperature).unsqueeze(-1)
        abnormal_distribution = semantic[..., 1:] / semantic[..., 1:].sum(dim=-1, keepdim=True).clamp_min(1e-12)
        return torch.cat([1.0 - abnormal_total, abnormal_total * abnormal_distribution], dim=-1)
    return semantic


def load_detection_map(path: Path):
    spec = importlib.util.spec_from_file_location("responsibility_detection_map", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load detection mAP utility: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.getDetectionMAP


def safe_frame_metrics(gt: np.ndarray, scores: np.ndarray) -> dict[str, float]:
    usable = min(len(gt), len(scores))
    if usable == 0:
        raise ValueError("empty ground truth or scores")
    return {
        "auc": float(roc_auc_score(gt[:usable], scores[:usable])),
        "ap": float(average_precision_score(gt[:usable], scores[:usable])),
        "frames": int(usable),
    }


def file_signature(path: str) -> dict[str, object] | None:
    if not path:
        return None
    item = Path(path)
    stat = item.stat()
    return {"path": path, "size": int(stat.st_size), "mtime_ns": int(stat.st_mtime_ns)}


def main() -> None:
    parser = argparse.ArgumentParser(description="Official-style evaluation and complementarity diagnostics.")
    parser.add_argument("--baseline", choices=["dsanet", "desc", "lagovad"], required=True)
    parser.add_argument("--baseline-root", required=True)
    parser.add_argument("--baseline-weight", default="")
    parser.add_argument("--sensitivity-weight", default="")
    parser.add_argument("--consistency-weight", default="")
    parser.add_argument("--joint-model", default="")
    parser.add_argument("--probe-model", required=True)
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

    out_dir = clean_output(args.out_dir, args.clean)
    run_signature = {
        "baseline": args.baseline,
        "dataset": args.dataset,
        "baseline_weight": file_signature(args.baseline_weight),
        "sensitivity_weight": file_signature(args.sensitivity_weight),
        "consistency_weight": file_signature(args.consistency_weight),
        "joint_model": file_signature(args.joint_model),
        "probe_model": file_signature(args.probe_model),
        "test_list": file_signature(args.test_list),
        "frames_per_snippet": args.frames_per_snippet,
        "temperature": args.temperature,
    }
    signature_path = out_dir / "run_signature.json"
    if signature_path.exists() and not args.clean:
        with signature_path.open("r", encoding="utf-8") as handle:
            previous_signature = json.load(handle)
        if previous_signature != run_signature:
            raise RuntimeError("evaluation inputs changed; rerun with --clean or choose another --out-dir")
    with signature_path.open("w", encoding="utf-8") as handle:
        json.dump(run_signature, handle, indent=2, ensure_ascii=False)
    score_dir = out_dir / "scores"
    score_dir.mkdir(exist_ok=True)
    device = torch.device(args.device if torch.cuda.is_available() and args.device.startswith("cuda") else "cpu")
    adapter = build_baseline(args, str(device)).to(device).eval()
    correction = ResponsibilityCorrectionHead().to(device).eval()
    if args.joint_model:
        checkpoint = torch.load(args.joint_model, map_location="cpu")
        adapter.load_state_dict(checkpoint["model_state_dict"], strict=True)
        correction.load_state_dict(checkpoint["correction_state_dict"], strict=True)
    probe = load_probe(args.probe_model, device)
    frame = pd.read_csv(args.test_list)
    missing = {"clip_path", "neuron_path", "label"} - set(frame.columns)
    if missing:
        raise ValueError(f"{args.test_list} is missing columns: {sorted(missing)}")

    temperature = args.temperature
    if temperature <= 0:
        temperature = 5.0 if args.dataset == "ucf" and args.baseline == "dsanet" else 1.0
    binary_all, semantic_all, neuron_all, fused_all = [], [], [], []
    detection_predictions = []
    summary_rows = []
    if "key" not in frame.columns:
        frame["key"] = frame["clip_path"].map(lambda value: Path(str(value)).stem)
    groups = list(frame.groupby("key", sort=False))
    with torch.no_grad():
        for key_value, group in tqdm(groups, total=len(groups), desc="evaluate videos", unit="video"):
            key = str(key_value)
            labels = set(group["label"].astype(str))
            if len(labels) != 1:
                raise ValueError(f"{key}: inconsistent labels {sorted(labels)}")
            label = next(iter(labels))
            cache_path = score_dir / f"{key}.npz"
            if cache_path.exists() and not args.clean:
                cached = np.load(cache_path)
                binary = cached["binary"]
                semantic_score = cached["semantic"]
                neuron = cached["neuron"]
                fused = cached["fused"]
                class_prob = cached["class_prob"]
            else:
                clip = np.concatenate([
                    np.load(str(path)).astype(np.float32) for path in group["clip_path"]
                ], axis=0)
                neurons = np.concatenate([
                    np.load(str(path)).astype(np.float32) for path in group["neuron_path"]
                ], axis=0)
                if clip.shape[0] != neurons.shape[0] or clip.shape[1] != 512:
                    raise ValueError(f"{key}: invalid aligned shapes clip={clip.shape}, neurons={neurons.shape}")
                clip_chunks, lengths = pad_chunks(clip, adapter.visual_length)
                neuron_chunks, neuron_lengths = pad_chunks(neurons, adapter.visual_length)
                if not torch.equal(lengths, neuron_lengths):
                    raise RuntimeError(f"{key}: modality chunk lengths differ")
                output = adapter.forward_baseline(clip_chunks.to(device), lengths.to(device))
                neuron_prob = torch.sigmoid(probe(neuron_chunks.to(device), lengths.to(device)))
                corrected_logits = correction(output.binary_logits, neuron_prob, lengths.to(device))
                valid_binary, valid_semantic, valid_neuron, valid_class = [], [], [], [],
                class_prob_batch = class_probabilities(
                    corrected_logits, output.semantic_logits, args.baseline, temperature
                )
                semantic_batch = 1.0 - F.softmax(output.semantic_logits / temperature, dim=-1)[..., 0]
                for index, length in enumerate(lengths.tolist()):
                    valid_binary.append(torch.sigmoid(corrected_logits[index, :length]).cpu())
                    valid_semantic.append(semantic_batch[index, :length].cpu())
                    valid_neuron.append(neuron_prob[index, :length].cpu())
                    valid_class.append(class_prob_batch[index, :length].cpu())
                binary = torch.cat(valid_binary).numpy().astype(np.float32)
                semantic_score = torch.cat(valid_semantic).numpy().astype(np.float32)
                neuron = torch.cat(valid_neuron).numpy().astype(np.float32)
                class_prob = torch.cat(valid_class).numpy().astype(np.float32)
                # All three released baselines report frame anomaly AUC/AP from
                # their binary branch.  Semantic scores remain a secondary
                # diagnostic and provide DSANet's class distribution for mAP.
                baseline_final = binary
                fused = (0.5 * baseline_final + 0.5 * neuron).astype(np.float32)
                np.savez_compressed(
                    cache_path, binary=binary, semantic=semantic_score,
                    neuron=neuron, fused=fused, class_prob=class_prob,
                )
            binary_all.append(binary)
            semantic_all.append(semantic_score)
            neuron_all.append(neuron)
            fused_all.append(fused)
            detection_predictions.append(np.repeat(class_prob, args.frames_per_snippet, axis=0))
            summary_rows.append([
                key, label, len(binary), float(binary.mean()),
                float(semantic_score.mean()), float(neuron.mean()), float(fused.mean()),
            ])

    gt = np.load(args.gt_path).astype(np.int64).reshape(-1)
    metrics = {}
    for name, values in {
        "binary": binary_all,
        "semantic": semantic_all,
        "neuron": neuron_all,
        "fused_diagnostic": fused_all,
    }.items():
        snippet = np.concatenate(values)
        frame_scores = np.repeat(snippet, args.frames_per_snippet)
        metrics[name] = safe_frame_metrics(gt, frame_scores)

    baseline_for_diagnostic = binary_all
    baseline_frames = np.repeat(np.concatenate(baseline_for_diagnostic), args.frames_per_snippet)
    neuron_frames = np.repeat(np.concatenate(neuron_all), args.frames_per_snippet)
    usable = min(len(gt), len(baseline_frames), len(neuron_frames))
    truth = gt[:usable]
    base_high = baseline_frames[:usable] >= 0.5
    neuron_high = neuron_frames[:usable] >= 0.5
    partitions = {
        "agree_high": base_high & neuron_high,
        "agree_low": ~base_high & ~neuron_high,
        "neuron_only": ~base_high & neuron_high,
        "baseline_only": base_high & ~neuron_high,
    }
    metrics["disagreement_diagnostic"] = {}
    for name, selection in partitions.items():
        count = int(selection.sum())
        metrics["disagreement_diagnostic"][name] = {
            "frames": count,
            "true_anomaly_fraction": float(truth[selection].mean()) if count else None,
        }

    if args.gt_segment_path and args.gt_label_path:
        utility_path = Path(args.baseline_root) / "src" / "utils" / f"{args.dataset}_detectionMAP.py"
        if utility_path.is_file():
            detection_map = load_detection_map(utility_path)
            maps, thresholds = detection_map(
                detection_predictions,
                np.load(args.gt_segment_path, allow_pickle=True),
                np.load(args.gt_label_path, allow_pickle=True),
                excludeNormal=False,
            )
            metrics["detection_map"] = {
                f"iou_{float(iou):.1f}": float(value) for iou, value in zip(thresholds, maps)
            }
            metrics["detection_map_average"] = float(np.mean(maps[:5]))
        else:
            metrics["detection_map_skipped"] = (
                f"baseline-specific utility not found: {utility_path.as_posix()}"
            )

    with (out_dir / "metrics.json").open("w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2, ensure_ascii=False)
    with (out_dir / "per_video.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["key", "label", "snippets", "binary_mean", "semantic_mean", "neuron_mean", "fused_mean"])
        writer.writerows(summary_rows)
    print(json.dumps(metrics, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
