#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import random
import shutil
import sys
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import average_precision_score, roc_auc_score
from torch.utils.data import DataLoader
from tqdm import tqdm


PACKAGE_ROOT = Path(__file__).resolve().parent.parent
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))

from neuron_responsibility.data import AlignedFeatureDataset
from neuron_responsibility.model import NeuronResponsibilityProbe, probe_mil_loss, topk_mil_probability


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def evaluate_probe(model, loader, device) -> dict[str, float]:
    model.eval()
    targets, scores = [], []
    with torch.no_grad():
        for batch in tqdm(loader, desc="probe validation", leave=False):
            neurons = batch["neurons"].to(device)
            lengths = batch["length"].to(device)
            logits = model(neurons, lengths)
            score = topk_mil_probability(torch.sigmoid(logits), lengths)
            scores.extend(score.cpu().tolist())
            targets.extend(batch["binary_label"].tolist())
    return {
        "video_auc": float(roc_auc_score(targets, scores)),
        "video_ap": float(average_precision_score(targets, scores)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Stage A: train the baseline-independent neuron probe.")
    parser.add_argument("--dataset", choices=["ucf", "xd"], required=True)
    parser.add_argument("--train-list", required=True)
    parser.add_argument("--val-list", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--visual-length", type=int, default=256)
    parser.add_argument("--hidden-width", type=int, default=128)
    parser.add_argument("--max-epoch", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=7e-5)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--sparsity-weight", type=float, default=1e-3)
    parser.add_argument("--normal-instance-weight", type=float, default=0.25)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=234)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--clean", action="store_true")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    if args.clean and out_dir.exists():
        shutil.rmtree(out_dir.resolve())
    out_dir.mkdir(parents=True, exist_ok=True)
    seed_everything(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() and args.device.startswith("cuda") else "cpu")

    train_set = AlignedFeatureDataset(args.train_list, args.dataset, args.visual_length)
    val_set = AlignedFeatureDataset(args.val_list, args.dataset, args.visual_length)
    if not len(train_set) or not len(val_set):
        raise RuntimeError("train and validation lists must be non-empty")
    neuron_width = int(train_set[0]["neurons"].shape[-1])
    train_loader = DataLoader(
        train_set, batch_size=args.batch_size, shuffle=True, drop_last=True,
        num_workers=args.num_workers, pin_memory=device.type == "cuda",
    )
    val_loader = DataLoader(
        val_set, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=device.type == "cuda",
    )
    model = NeuronResponsibilityProbe(neuron_width, args.hidden_width).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.max_epoch)
    checkpoint_path = out_dir / "checkpoint_last.pth"
    best_path = out_dir / "probe_best.pth"
    start_epoch, best = 0, -float("inf")
    if args.resume and checkpoint_path.exists():
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        start_epoch = int(checkpoint["epoch"]) + 1
        best = float(checkpoint["best_metric"])
        print(f"resumed at epoch {start_epoch + 1}; best={best:.6f}", flush=True)

    history_path = out_dir / "history.jsonl"
    for epoch in range(start_epoch, args.max_epoch):
        model.train()
        running = 0.0
        progress = tqdm(train_loader, desc=f"probe epoch {epoch + 1}/{args.max_epoch}")
        for step, batch in enumerate(progress, 1):
            neurons = batch["neurons"].to(device, non_blocking=True)
            lengths = batch["length"].to(device, non_blocking=True)
            labels = batch["binary_label"].to(device, non_blocking=True)
            logits = model(neurons, lengths)
            loss = probe_mil_loss(logits, labels, lengths, args.normal_instance_weight)
            loss = loss + args.sparsity_weight * model.sparsity_loss()
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            running += float(loss.detach())
            progress.set_postfix(loss=f"{running / step:.4f}", gate=f"{model.feature_gates().mean().item():.3f}")
        scheduler.step()
        metrics = evaluate_probe(model, val_loader, device)
        selection_metric = metrics["video_auc" if args.dataset == "ucf" else "video_ap"]
        record = {"epoch": epoch + 1, "train_loss": running / max(1, len(train_loader)), **metrics}
        with history_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        checkpoint = {
            "epoch": epoch,
            "best_metric": max(best, selection_metric),
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "config": {"neuron_width": neuron_width, "hidden_width": args.hidden_width},
            "metrics": metrics,
        }
        torch.save(checkpoint, checkpoint_path)
        if selection_metric > best:
            best = selection_metric
            torch.save(checkpoint, best_path)
        print(f"epoch {epoch + 1}: {metrics} | best={best:.6f}", flush=True)


if __name__ == "__main__":
    main()
