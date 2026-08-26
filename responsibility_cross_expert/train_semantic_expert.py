#!/usr/bin/env python3
"""Train the selected-layer semantic expert with video-level labels only."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import average_precision_score, roc_auc_score
from torch.utils.data import DataLoader
from tqdm import tqdm

from .common import clean_output, seed_everything
from .data import WholeLayerDataset
from .losses import binary_topk_mil
from .prompts import abnormal_class_names, label_targets
from .semantic_model import build_semantic_expert


def merge_batches(normal: dict, abnormal: dict) -> dict:
    merged = {}
    for key in ("hidden", "length", "binary_label"):
        merged[key] = torch.cat([normal[key], abnormal[key]], dim=0)
    merged["label_text"] = list(normal["label_text"]) + list(abnormal["label_text"])
    return merged


def category_targets(dataset: str, labels: list[str], device: torch.device) -> torch.Tensor:
    targets = torch.zeros(len(labels), len(abnormal_class_names(dataset)), device=device)
    for row, label in enumerate(labels):
        for class_index in label_targets(dataset, label):
            targets[row, class_index] = 1.0
    return targets


def category_mil_loss(
    class_logits: torch.Tensor,
    targets: torch.Tensor,
    lengths: torch.Tensor,
) -> torch.Tensor:
    bags = []
    for row, length in zip(class_logits, lengths):
        valid = max(1, int(length.item()))
        count = min(valid, valid // 16 + 1)
        bags.append(row[:valid].topk(count, dim=0).values.mean(dim=0))
    return F.binary_cross_entropy_with_logits(torch.stack(bags), targets)


def bag_probabilities(logits: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
    values = []
    for row, length in zip(logits, lengths):
        valid = max(1, int(length.item()))
        count = min(valid, valid // 16 + 1)
        values.append(torch.sigmoid(row[:valid]).topk(count).values.mean())
    return torch.stack(values)


def validate(model, loader: DataLoader, device: torch.device) -> dict[str, float]:
    model.eval()
    labels, scores, losses = [], [], []
    with torch.no_grad():
        for batch in tqdm(loader, desc="semantic validation", unit="batch", leave=False):
            hidden = batch["hidden"].to(device, non_blocking=True)
            lengths = batch["length"].to(device, non_blocking=True)
            target = batch["binary_label"].to(device, non_blocking=True)
            output = model(hidden)
            loss = binary_topk_mil(output["anomaly_logit"], target, lengths)
            losses.append(float(loss))
            labels.extend(target.cpu().tolist())
            scores.extend(bag_probabilities(output["anomaly_logit"], lengths).cpu().tolist())
    if len(set(labels)) < 2:
        raise RuntimeError("validation split must contain both normal and abnormal videos")
    return {
        "video_auc": float(roc_auc_score(labels, scores)),
        "video_ap": float(average_precision_score(labels, scores)),
        "binary_loss": float(np.mean(losses)),
        "videos": len(labels),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Train responsibility-selected semantic expert.")
    parser.add_argument("--dataset", choices=["ucf", "xd"], required=True)
    parser.add_argument("--train-csv", required=True)
    parser.add_argument("--layer-atlas", required=True)
    parser.add_argument("--clip-weight", required=True)
    parser.add_argument("--normal-prototype", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--sequence-length", type=int, default=256)
    parser.add_argument("--bottleneck", type=int, default=64)
    parser.add_argument("--max-epoch", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--category-weight", type=float, default=1.0)
    parser.add_argument("--validation-fraction", type=float, default=0.1)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=234)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--clean", action="store_true")
    args = parser.parse_args()
    if args.clean and args.resume:
        parser.error("--clean and --resume are mutually exclusive")
    output_dir = clean_output(args.out_dir, args.clean)
    checkpoint_path = output_dir / "checkpoint_last.pth"
    best_path = output_dir / "semantic_expert_best.pth"
    resume = args.resume or checkpoint_path.exists()
    if args.resume and not checkpoint_path.exists():
        raise FileNotFoundError(f"resume checkpoint does not exist: {checkpoint_path}")
    seed_everything(args.seed)
    device = torch.device(
        args.device if torch.cuda.is_available() and args.device.startswith("cuda") else "cpu"
    )
    model = build_semantic_expert(
        args.layer_atlas,
        args.clip_weight,
        args.dataset,
        args.normal_prototype,
        args.bottleneck,
        device,
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.max_epoch)
    dataset_options = dict(
        csv_path=args.train_csv,
        dataset=args.dataset,
        sequence_length=args.sequence_length,
        seed=args.seed,
        validation_fraction=args.validation_fraction,
        include_clip=False,
    )
    loader_options = dict(
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )
    normal_loader = DataLoader(
        WholeLayerDataset(**dataset_options, kind="normal", fold="train"),
        shuffle=True,
        drop_last=True,
        **loader_options,
    )
    abnormal_loader = DataLoader(
        WholeLayerDataset(**dataset_options, kind="abnormal", fold="train"),
        shuffle=True,
        drop_last=True,
        **loader_options,
    )
    validation_loader = DataLoader(
        WholeLayerDataset(**dataset_options, kind="all", fold="validation"),
        shuffle=False,
        drop_last=False,
        **loader_options,
    )
    steps = min(len(normal_loader), len(abnormal_loader))
    if steps == 0 or len(validation_loader) == 0:
        raise RuntimeError("semantic train/validation loaders must be non-empty")
    run_config = {
        "method": model.method_name,
        "dataset": args.dataset,
        "sequence_length": args.sequence_length,
        "bottleneck": args.bottleneck,
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "category_weight": args.category_weight,
        "validation_fraction": args.validation_fraction,
        "selection_rule": "held-out train-video AUC; no frame GT",
    }
    start_epoch, best = 0, -float("inf")
    if resume:
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        if checkpoint["run_config"] != run_config:
            raise RuntimeError("resume configuration differs from checkpoint")
        model.load_state_dict(checkpoint["model_state_dict"], strict=True)
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        start_epoch = int(checkpoint["epoch"]) + 1
        best = float(checkpoint["best_metric"])
    history = output_dir / "history.jsonl"
    for epoch in range(start_epoch, args.max_epoch):
        model.train()
        running = {"total": 0.0, "binary": 0.0, "category": 0.0}
        paired = zip(normal_loader, abnormal_loader)
        progress = tqdm(paired, total=steps, desc=f"semantic {epoch + 1}/{args.max_epoch}", unit="batch")
        for step, (normal, abnormal) in enumerate(progress, 1):
            batch = merge_batches(normal, abnormal)
            hidden = batch["hidden"].to(device, non_blocking=True)
            lengths = batch["length"].to(device, non_blocking=True)
            binary_target = batch["binary_label"].to(device, non_blocking=True)
            class_target = category_targets(args.dataset, batch["label_text"], device)
            result = model(hidden)
            binary = binary_topk_mil(result["anomaly_logit"], binary_target, lengths)
            category = category_mil_loss(result["class_margin"], class_target, lengths)
            loss = binary + args.category_weight * category
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            for name, value in (("total", loss), ("binary", binary), ("category", category)):
                running[name] += float(value.detach())
            progress.set_postfix(
                loss=f"{running['total'] / step:.4f}",
                layer="/".join(f"{v:.2f}" for v in model.layer_weights.detach().cpu().tolist()),
            )
        metrics = validate(model, validation_loader, device)
        value = metrics["video_auc"]
        if value > best:
            best = value
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "run_config": run_config,
                    "model_metadata": model.metadata(),
                    "metrics": metrics,
                    "epoch": epoch,
                    "best_metric": best,
                },
                best_path,
            )
        scheduler.step()
        record = {
            "epoch": epoch + 1,
            **{f"{key}_loss": value / steps for key, value in running.items()},
            "metrics": metrics,
            "best_metric": best,
            "layer_weights": model.layer_weights.detach().cpu().tolist(),
        }
        with history.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "run_config": run_config,
                "model_metadata": model.metadata(),
                "metrics": metrics,
                "epoch": epoch,
                "best_metric": best,
            },
            checkpoint_path,
        )
        print(json.dumps(record, ensure_ascii=False), flush=True)
    (output_dir / "training_report.json").write_text(
        json.dumps({**run_config, "best_metric": best, **model.metadata()}, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
