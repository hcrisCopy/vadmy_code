#!/usr/bin/env python3
"""Train an independent neuron localizer with synthetic temporal boundaries."""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import average_precision_score, roc_auc_score
from torch.utils.data import DataLoader
from tqdm import tqdm


PACKAGE_ROOT = Path(__file__).resolve().parent.parent
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))

from neuron_responsibility.boundary_localization import (
    IndependentNeuronLocalizer,
    boundary_supervision_loss,
    load_evidence_localizer,
    synthesize_boundary_batch,
)
from neuron_responsibility.common import clean_output
from neuron_responsibility.data import AlignedFeatureDataset
from neuron_responsibility.evaluate import pad_chunks
from neuron_responsibility.model import probe_mil_loss


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def paired_loaders(
    normal_set: AlignedFeatureDataset,
    abnormal_set: AlignedFeatureDataset,
    batch_size: int,
    num_workers: int,
    pin_memory: bool,
    seed: int,
) -> tuple[DataLoader, DataLoader]:
    common = {
        "batch_size": batch_size,
        "shuffle": True,
        "drop_last": True,
        "num_workers": num_workers,
        "pin_memory": pin_memory,
    }
    normal = DataLoader(
        normal_set,
        generator=torch.Generator().manual_seed(seed),
        **common,
    )
    abnormal = DataLoader(
        abnormal_set,
        generator=torch.Generator().manual_seed(seed + 100_003),
        **common,
    )
    return normal, abnormal


def official_localizer_metrics(
    model: IndependentNeuronLocalizer,
    aligned_csv: str,
    gt_path: str,
    visual_length: int,
    frames_per_snippet: int,
    device: torch.device,
) -> dict[str, float]:
    model.eval()
    frame = pd.read_csv(aligned_csv)
    if "key" not in frame.columns:
        frame["key"] = frame["neuron_path"].map(lambda value: Path(str(value)).stem)
    scores = []
    with torch.no_grad():
        for _, group in tqdm(
            list(frame.groupby("key", sort=False)),
            desc="localizer frame validation",
            unit="video",
            leave=False,
        ):
            neurons = np.concatenate([
                np.load(str(path)).astype(np.float32) for path in group["neuron_path"]
            ])
            chunks, lengths = pad_chunks(neurons, visual_length)
            logits = model(chunks.to(device), lengths.to(device))
            for index, length in enumerate(lengths.tolist()):
                scores.append(torch.sigmoid(logits[index, :length]).cpu())
    snippet = torch.cat(scores).numpy()
    prediction = np.repeat(snippet, frames_per_snippet)
    truth = np.load(gt_path).astype(np.int64).reshape(-1)
    usable = min(len(prediction), len(truth))
    prediction, truth = prediction[:usable], truth[:usable]
    metrics = {
        "frame_auc": float(roc_auc_score(truth, prediction)),
        "frame_ap": float(average_precision_score(truth, prediction)),
    }
    for fraction in (0.01, 0.05, 0.10):
        threshold = np.quantile(prediction, 1.0 - fraction)
        selected = prediction >= threshold
        metrics[f"top_{int(fraction * 100)}pct_precision"] = float(truth[selected].mean())
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Independent neuron localization with dynamically synthesized boundaries."
    )
    parser.add_argument("--dataset", choices=["ucf", "xd"], required=True)
    parser.add_argument("--train-list", required=True)
    parser.add_argument("--val-list", required=True)
    parser.add_argument("--gt-path", required=True)
    parser.add_argument("--evidence-config", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--visual-length", type=int, default=256)
    parser.add_argument("--hidden-width", type=int, default=64)
    parser.add_argument("--active-neurons", type=int, default=64)
    parser.add_argument("--evidence-cap", type=float, default=6.0)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--max-epoch", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=7e-5)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--real-mil-weight", type=float, default=0.2)
    parser.add_argument("--synthetic-bce-weight", type=float, default=1.0)
    parser.add_argument("--synthetic-dice-weight", type=float, default=1.0)
    parser.add_argument("--boundary-weight", type=float, default=0.1)
    parser.add_argument("--sparsity-weight", type=float, default=1e-3)
    parser.add_argument("--min-segment", type=int, default=4)
    parser.add_argument("--max-segment", type=int, default=32)
    parser.add_argument("--frames-per-snippet", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=234)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--clean", action="store_true")
    args = parser.parse_args()
    if args.clean and args.resume:
        parser.error("--clean and --resume cannot be used together")
    if args.min_segment <= 0 or args.max_segment < args.min_segment:
        parser.error("require 0 < min-segment <= max-segment")
    if min(
        args.lr, args.weight_decay, args.real_mil_weight, args.synthetic_bce_weight,
        args.synthetic_dice_weight, args.boundary_weight, args.sparsity_weight,
    ) < 0:
        parser.error("learning rates and loss weights must be non-negative")

    out_dir = clean_output(args.out_dir, args.clean)
    checkpoint_path = out_dir / "checkpoint_last.pth"
    best_path = out_dir / "localizer_best.pth"
    if checkpoint_path.exists() and not args.resume:
        raise RuntimeError("checkpoint exists; use --resume, --clean, or another --out-dir")
    seed_everything(args.seed)
    device = torch.device(
        args.device if torch.cuda.is_available() and args.device.startswith("cuda") else "cpu"
    )
    normal_set = AlignedFeatureDataset(
        args.train_list, args.dataset, args.visual_length, split="normal"
    )
    abnormal_set = AlignedFeatureDataset(
        args.train_list, args.dataset, args.visual_length, split="abnormal"
    )
    if not len(normal_set) or not len(abnormal_set):
        raise RuntimeError("localizer training requires normal and abnormal videos")
    model = load_evidence_localizer(
        args.evidence_config,
        args.hidden_width,
        args.active_neurons,
        args.evidence_cap,
        args.dropout,
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.max_epoch)
    run_config = {
        "method": IndependentNeuronLocalizer.method_name,
        "dataset": args.dataset,
        "evidence_config": args.evidence_config,
        "visual_length": args.visual_length,
        "model": model.config(),
        "batch_size_per_class": args.batch_size,
        "min_segment": args.min_segment,
        "max_segment": args.max_segment,
    }
    (out_dir / "run_config.json").write_text(
        json.dumps(run_config, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    start_epoch, best = 0, -float("inf")
    if args.resume:
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"resume checkpoint not found: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        if checkpoint.get("run_config") != run_config:
            raise RuntimeError("resume configuration differs from checkpoint")
        model.load_state_dict(checkpoint["model_state_dict"], strict=True)
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        start_epoch = int(checkpoint["epoch"]) + 1
        best = float(checkpoint["best_metric"])

    history_path = out_dir / "history.jsonl"
    for epoch in range(start_epoch, args.max_epoch):
        normal_loader, abnormal_loader = paired_loaders(
            normal_set,
            abnormal_set,
            args.batch_size,
            args.num_workers,
            device.type == "cuda",
            args.seed + epoch,
        )
        steps = min(len(normal_loader), len(abnormal_loader))
        model.train()
        running = {name: 0.0 for name in ("total", "real_mil", "bce", "dice", "boundary", "sparse")}
        pairs = zip(normal_loader, abnormal_loader)
        progress = tqdm(pairs, total=steps, desc=f"boundary localizer {epoch + 1}/{args.max_epoch}")
        for step, (normal, abnormal) in enumerate(progress, 1):
            normal_neurons = normal["neurons"].to(device, non_blocking=True)
            abnormal_neurons = abnormal["neurons"].to(device, non_blocking=True)
            normal_clip = normal["clip"].to(device, non_blocking=True)
            abnormal_clip = abnormal["clip"].to(device, non_blocking=True)
            normal_lengths = normal["length"].to(device, non_blocking=True)
            abnormal_lengths = abnormal["length"].to(device, non_blocking=True)
            synthetic = synthesize_boundary_batch(
                model,
                normal_clip,
                normal_neurons,
                normal_lengths,
                abnormal_clip,
                abnormal_neurons,
                abnormal_lengths,
                args.min_segment,
                args.max_segment,
            )
            real_neurons = torch.cat([normal_neurons, abnormal_neurons], dim=0)
            real_lengths = torch.cat([normal_lengths, abnormal_lengths], dim=0)
            real_labels = torch.cat([
                torch.zeros(len(normal_neurons), device=device),
                torch.ones(len(abnormal_neurons), device=device),
            ])
            real_logits = model(real_neurons, real_lengths)
            real_mil = probe_mil_loss(real_logits, real_labels, real_lengths)
            synthetic_logits = model(synthetic["neurons"], synthetic["lengths"])
            boundary = boundary_supervision_loss(
                synthetic_logits,
                synthetic["targets"],
                synthetic["lengths"],
                synthetic["confidence"],
            )
            sparse = model.sparsity_loss()
            loss = (
                args.real_mil_weight * real_mil
                + args.synthetic_bce_weight * boundary["bce"]
                + args.synthetic_dice_weight * boundary["dice"]
                + args.boundary_weight * boundary["boundary"]
                + args.sparsity_weight * sparse
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            values = {"total": loss, "real_mil": real_mil, **boundary, "sparse": sparse}
            for name, value in values.items():
                running[name] += float(value.detach())
            progress.set_postfix(
                loss=f"{running['total'] / step:.4f}",
                bce=f"{running['bce'] / step:.4f}",
            )
        scheduler.step()
        metrics = official_localizer_metrics(
            model,
            args.val_list,
            args.gt_path,
            args.visual_length,
            args.frames_per_snippet,
            device,
        )
        selection = metrics["frame_auc" if args.dataset == "ucf" else "frame_ap"]
        best = max(best, selection)
        payload = {
            "method": IndependentNeuronLocalizer.method_name,
            "epoch": epoch,
            "best_metric": best,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "model_config": model.config(),
            "run_config": run_config,
            "metrics": metrics,
            "selection_rule": "UCF frame AUC" if args.dataset == "ucf" else "XD frame AP",
        }
        torch.save(payload, checkpoint_path)
        if selection >= best:
            torch.save(payload, best_path)
        record = {
            "epoch": epoch + 1,
            **{f"{key}_loss": value / max(1, steps) for key, value in running.items()},
            **metrics,
        }
        with history_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        print(json.dumps(record, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()

