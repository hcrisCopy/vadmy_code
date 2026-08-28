#!/usr/bin/env python3
"""Export independent semantic- and baseline-expert curves for every train crop."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from .baselines import build_baseline
from .common import clean_output, write_csv
from .data import WholeLayerDataset
from .semantic_model import build_semantic_expert


def main() -> None:
    parser = argparse.ArgumentParser(description="Export two independent expert score curves.")
    parser.add_argument("--baseline", choices=["dsanet", "desc", "lagovad"], required=True)
    parser.add_argument("--baseline-root", required=True)
    parser.add_argument("--baseline-weight", default="")
    parser.add_argument("--sensitivity-weight", default="")
    parser.add_argument("--consistency-weight", default="")
    parser.add_argument("--dataset", choices=["ucf", "xd"], required=True)
    parser.add_argument("--train-csv", required=True)
    parser.add_argument("--layer-atlas", required=True)
    parser.add_argument("--clip-weight", required=True)
    parser.add_argument("--normal-prototype", required=True)
    parser.add_argument("--semantic-checkpoint", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--sequence-length", type=int, default=256)
    parser.add_argument("--bottleneck", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--clean", action="store_true")
    args = parser.parse_args()
    output_dir = clean_output(args.out_dir, args.clean)
    cache_dir = output_dir / "score_cache"
    cache_dir.mkdir(exist_ok=True)
    list_path = output_dir / "expert_scores.csv"
    device = torch.device(
        args.device if torch.cuda.is_available() and args.device.startswith("cuda") else "cpu"
    )
    semantic = build_semantic_expert(
        args.layer_atlas,
        args.clip_weight,
        args.dataset,
        args.normal_prototype,
        args.bottleneck,
        device,
    )
    checkpoint = torch.load(args.semantic_checkpoint, map_location="cpu")
    semantic.load_state_dict(checkpoint["model_state_dict"], strict=True)
    semantic.eval()
    baseline = build_baseline(args, str(device)).to(device)
    baseline.set_train_scope("frozen")
    baseline.eval()
    dataset = WholeLayerDataset(
        args.train_csv, args.dataset, args.sequence_length, kind="all", fold="all"
    )
    loader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )
    rows, correlations = [], []
    with torch.no_grad():
        for index, batch in enumerate(tqdm(loader, desc="export expert scores", unit="crop")):
            cache_path = cache_dir / f"{index:07d}.npz"
            if cache_path.exists() and not args.clean:
                stored = np.load(cache_path)
                semantic_score = stored["semantic_score"]
                baseline_score = stored["baseline_score"]
                length = int(stored["length"])
            else:
                hidden = batch["hidden"].to(device, non_blocking=True)
                clip = batch["clip"].to(device, non_blocking=True)
                lengths = batch["length"].to(device, non_blocking=True)
                semantic_output = semantic(hidden)
                baseline_output = baseline.forward_baseline(clip, lengths)
                length = int(lengths[0].item())
                semantic_score = torch.sigmoid(semantic_output["anomaly_logit"][0]).cpu().numpy()
                baseline_score = torch.sigmoid(baseline_output.binary_logits[0]).cpu().numpy()
                np.savez_compressed(
                    cache_path,
                    semantic_score=semantic_score.astype(np.float32),
                    baseline_score=baseline_score.astype(np.float32),
                    class_margin=semantic_output["class_margin"][0].cpu().numpy().astype(np.float32),
                    layer_margin=semantic_output["layer_margin"][0].cpu().numpy().astype(np.float32),
                    length=np.int64(length),
                )
            if length > 1 and np.std(semantic_score[:length]) > 0 and np.std(baseline_score[:length]) > 0:
                correlations.append(float(np.corrcoef(semantic_score[:length], baseline_score[:length])[0, 1]))
            rows.append([
                index,
                batch["key"][0],
                batch["clip_path"][0],
                batch["hidden_path"][0],
                batch["label_text"][0],
                length,
                str(cache_path),
            ])
    write_csv(
        list_path,
        ["row_id", "key", "clip_path", "hidden_path", "label", "length", "score_path"],
        rows,
    )
    report = {
        "method": "independent_responsibility_semantic_and_baseline_temporal_experts_v1",
        "baseline": args.baseline,
        "dataset": args.dataset,
        "crops": len(rows),
        "mean_within_crop_score_correlation": float(np.mean(correlations)) if correlations else None,
        "semantic_checkpoint": args.semantic_checkpoint,
        "scores": str(list_path),
        "baseline_used_for_layer_selection": False,
    }
    (output_dir / "score_report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(report, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
