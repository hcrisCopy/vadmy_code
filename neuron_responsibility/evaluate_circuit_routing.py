#!/usr/bin/env python3
"""Official frame evaluation for concept-conditioned neuron circuit routing."""

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
from neuron_responsibility.circuit_routing import ConceptCircuitRouter, load_circuit_router
from neuron_responsibility.common import clean_output, is_normal_label
from neuron_responsibility.evaluate import pad_chunks


def frame_metrics(truth: np.ndarray, snippets: np.ndarray, repeat: int) -> dict[str, float]:
    prediction = np.repeat(snippets, repeat)
    usable = min(len(truth), len(prediction))
    return {
        "auc": float(roc_auc_score(truth[:usable], prediction[:usable])),
        "ap": float(average_precision_score(truth[:usable], prediction[:usable])),
        "frames": int(usable),
    }


def evaluate_models(
    adapter,
    router: ConceptCircuitRouter,
    csv_path: str,
    gt_path: str,
    dataset: str,
    repeat: int,
    device: torch.device,
    include_author: bool = False,
    description: str = "CNCR validation",
) -> tuple[dict, list[list[object]]]:
    frame = pd.read_csv(csv_path)
    required = {"clip_path", "neuron_path", "label", "key"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"{csv_path}: missing columns {sorted(missing)}")
    enhanced_all, author_all, rows = [], [], []
    normal_gates, abnormal_gates = [], []
    adapter.eval(); router.eval()
    with torch.no_grad():
        for key, group in tqdm(
            list(frame.groupby("key", sort=False)), desc=description, unit="video", leave=False
        ):
            clip = np.concatenate([np.load(str(path)).astype(np.float32) for path in group["clip_path"]])
            compact = np.concatenate([np.load(str(path)).astype(np.float32) for path in group["neuron_path"]])
            clip_chunks, lengths = pad_chunks(clip, adapter.visual_length)
            compact_chunks, compact_lengths = pad_chunks(compact, adapter.visual_length)
            if not torch.equal(lengths, compact_lengths):
                raise RuntimeError(f"{key}: CLIP and compact circuit lengths differ")
            clip_device = clip_chunks.to(device); compact_device = compact_chunks.to(device)
            views = router(clip_device, compact_device)
            routed = adapter.forward_baseline(views.enhanced, lengths.to(device))
            routed_scores = torch.cat([
                torch.sigmoid(routed.binary_logits[index, :length]).cpu()
                for index, length in enumerate(lengths.tolist())
            ]).numpy()
            enhanced_all.append(routed_scores)
            if include_author:
                author = adapter.forward_baseline(clip_device, lengths.to(device))
                author_scores = torch.cat([
                    torch.sigmoid(author.binary_logits[index, :length]).cpu()
                    for index, length in enumerate(lengths.tolist())
                ]).numpy()
                author_all.append(author_scores)
            gate = torch.cat([
                views.anomaly_gate[index, :length].cpu()
                for index, length in enumerate(lengths.tolist())
            ]).numpy()
            label = str(group.iloc[0]["label"])
            (normal_gates if is_normal_label(dataset, label) else abnormal_gates).append(gate)
            rows.append([
                str(key), label, len(routed_scores), float(routed_scores.mean()),
                float(gate.mean()), float(views.target_text_effect.mean().cpu()),
            ])
    truth = np.load(gt_path).astype(np.int64).reshape(-1)
    enhanced = np.concatenate(enhanced_all)
    metrics = {
        "cncr": frame_metrics(truth, enhanced, repeat),
        "routing": {
            "normal_gate_mean": float(np.concatenate(normal_gates).mean()),
            "abnormal_gate_mean": float(np.concatenate(abnormal_gates).mean()),
        },
    }
    if include_author:
        metrics["author_same_adapter"] = frame_metrics(truth, np.concatenate(author_all), repeat)
    return metrics, rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate CNCR with the official frame metric.")
    parser.add_argument("--baseline", choices=["dsanet", "desc", "lagovad"], required=True)
    parser.add_argument("--baseline-root", required=True); parser.add_argument("--baseline-weight", default="")
    parser.add_argument("--sensitivity-weight", default=""); parser.add_argument("--consistency-weight", default="")
    parser.add_argument("--dataset", choices=["ucf", "xd"], required=True)
    parser.add_argument("--test-list", required=True); parser.add_argument("--atlas", required=True)
    parser.add_argument("--model-path", default=""); parser.add_argument("--gt-path", required=True)
    parser.add_argument("--out-dir", required=True); parser.add_argument("--frames-per-snippet", type=int, default=16)
    parser.add_argument("--gate-temperature", type=float, default=0.05)
    parser.add_argument("--max-gain", type=float, default=0.50); parser.add_argument("--initial-gain", type=float, default=0.10)
    parser.add_argument("--device", default="cuda"); parser.add_argument("--clean", action="store_true")
    args = parser.parse_args()
    output = clean_output(args.out_dir, args.clean)
    metrics_path = output / "metrics.json"
    if metrics_path.exists() and not args.clean:
        print(metrics_path.read_text(encoding="utf-8"), flush=True)
        return
    device = torch.device(args.device if torch.cuda.is_available() and args.device.startswith("cuda") else "cpu")
    adapter = build_baseline(args, str(device)).to(device)
    router, atlas = load_circuit_router(
        args.atlas, args.gate_temperature, args.max_gain, args.initial_gain
    )
    router = router.to(device)
    released_metrics, _ = evaluate_models(
        adapter, router, args.test_list, args.gt_path, args.dataset,
        args.frames_per_snippet, device, include_author=True,
        description="evaluate released baseline",
    )
    if args.model_path:
        checkpoint = torch.load(args.model_path, map_location="cpu")
        adapter.load_state_dict(checkpoint["model_state_dict"], strict=True)
        router.load_state_dict(checkpoint["router_state_dict"], strict=True)
    metrics, rows = evaluate_models(
        adapter, router, args.test_list, args.gt_path, args.dataset,
        args.frames_per_snippet, device, include_author=True, description="evaluate CNCR videos",
    )
    metrics["released_author"] = released_metrics["author_same_adapter"]
    metrics["initial_router"] = released_metrics["cncr"]
    metrics["atlas_gate_passed"] = bool(atlas.get("gate_passed", False))
    metrics["checkpoint"] = args.model_path or "released baseline plus initial router"
    metrics_path.write_text(json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8")
    with (output / "per_video.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["key", "label", "snippets", "cncr_mean", "gate_mean", "text_effect_mean"])
        writer.writerows(rows)
    print(json.dumps(metrics, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
