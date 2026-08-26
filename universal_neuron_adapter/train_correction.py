from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import average_precision_score, roc_auc_score
from torch.utils.data import DataLoader
from tqdm import tqdm

from universal_neuron_adapter.data import ScorePairDataset, collate_scores
from universal_neuron_adapter.model import ScoreCorrectionHead, topk_bag, weak_supervision_loss


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def validate(model: ScoreCorrectionHead, loader: DataLoader, device: torch.device) -> dict[str, float]:
    model.eval()
    targets, corrected, baseline = [], [], []
    with torch.no_grad():
        for batch in tqdm(loader, desc="correction validation", leave=False):
            base = batch["baseline"].to(device)
            expert = batch["expert"].to(device)
            lengths = batch["lengths"].to(device)
            logits = model(base, expert, lengths)
            corrected.extend(topk_bag(torch.sigmoid(logits), lengths).cpu().tolist())
            baseline.extend(topk_bag(base, lengths).cpu().tolist())
            targets.extend(batch["labels"].tolist())
    return {
        "corrected_video_auc": float(roc_auc_score(targets, corrected)),
        "corrected_video_ap": float(average_precision_score(targets, corrected)),
        "baseline_video_auc": float(roc_auc_score(targets, baseline)),
        "baseline_video_ap": float(average_precision_score(targets, baseline)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Train one universal score-space correction head.")
    parser.add_argument("--baseline-manifest", required=True)
    parser.add_argument("--expert-manifest", required=True)
    parser.add_argument("--train-keys", required=True)
    parser.add_argument("--val-keys", required=True)
    parser.add_argument("--baseline", choices=["dsanet", "desc", "lagovad"], required=True)
    parser.add_argument("--dataset", choices=["ucf", "xd"], required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--width", type=int, default=32)
    parser.add_argument("--max-epoch", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--maximum-length", type=int, default=256)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=234)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    output = Path(args.out_dir)
    output.mkdir(parents=True, exist_ok=True)
    seed_everything(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    train_loader = DataLoader(
        ScorePairDataset(args.baseline_manifest, args.expert_manifest, args.train_keys, args.maximum_length),
        batch_size=args.batch_size, shuffle=True, drop_last=True, num_workers=args.num_workers,
        pin_memory=device.type == "cuda", collate_fn=collate_scores,
    )
    val_loader = DataLoader(
        ScorePairDataset(args.baseline_manifest, args.expert_manifest, args.val_keys, args.maximum_length),
        batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers,
        pin_memory=device.type == "cuda", collate_fn=collate_scores,
    )
    model = ScoreCorrectionHead(args.width).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.max_epoch)
    checkpoint_path = output / "checkpoint_last.pth"
    best_path = output / "model_best.pth"
    start_epoch, best = 0, -float("inf")
    if args.resume and checkpoint_path.exists():
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        start_epoch = int(checkpoint["epoch"]) + 1
        best = float(checkpoint["best_metric"])

    for epoch in range(start_epoch, args.max_epoch):
        model.train()
        totals = {key: 0.0 for key in ("loss", "bag", "normal", "ranking", "anchor", "smooth")}
        progress = tqdm(train_loader, desc=f"score correction {args.baseline}/{args.dataset} {epoch + 1}/{args.max_epoch}")
        for step, batch in enumerate(progress, 1):
            baseline = batch["baseline"].to(device, non_blocking=True)
            expert = batch["expert"].to(device, non_blocking=True)
            lengths = batch["lengths"].to(device, non_blocking=True)
            labels = batch["labels"].to(device, non_blocking=True)
            logits = model(baseline, expert, lengths)
            loss, parts = weak_supervision_loss(logits, baseline, labels, lengths)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            totals["loss"] += float(loss.detach())
            for key, value in parts.items():
                totals[key] += value
            progress.set_postfix(loss=f"{totals['loss'] / step:.4f}")
        scheduler.step()
        metrics = validate(model, val_loader, device)
        selection = 0.5 * (metrics["corrected_video_auc"] + metrics["corrected_video_ap"])
        payload = {
            "epoch": epoch,
            "best_metric": max(best, selection),
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "config": {"width": args.width},
            "baseline": args.baseline,
            "dataset": args.dataset,
            "metrics": metrics,
        }
        torch.save(payload, checkpoint_path)
        if selection > best:
            best = selection
            torch.save(payload, best_path)
        record = {
            "epoch": epoch + 1,
            **{key: value / max(1, len(train_loader)) for key, value in totals.items()},
            **metrics,
        }
        with (output / "history.jsonl").open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record) + "\n")
        print(json.dumps(record), flush=True)


if __name__ == "__main__":
    main()

