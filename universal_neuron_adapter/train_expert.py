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

from universal_neuron_adapter.data import HiddenVideoDataset, collate_hidden
from universal_neuron_adapter.model import SparseNeuronExpert, expert_mil_loss, topk_bag


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def evaluate(model: SparseNeuronExpert, loader: DataLoader, device: torch.device) -> dict[str, float]:
    model.eval()
    targets, scores = [], []
    with torch.no_grad():
        for batch in tqdm(loader, desc="expert validation", leave=False):
            logits = model(batch["hidden"].to(device), batch["lengths"].to(device))
            bag = topk_bag(torch.sigmoid(logits), batch["lengths"].to(device))
            scores.extend(bag.cpu().tolist())
            targets.extend(batch["labels"].tolist())
    return {
        "video_auc": float(roc_auc_score(targets, scores)),
        "video_ap": float(average_precision_score(targets, scores)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Train one baseline-independent sparse CLS-neuron expert.")
    parser.add_argument("--train-manifest", required=True)
    parser.add_argument("--val-manifest", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--active-per-layer", type=int, default=32)
    parser.add_argument("--temporal-width", type=int, default=64)
    parser.add_argument("--max-epoch", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--sparsity-weight", type=float, default=1e-3)
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
        HiddenVideoDataset(args.train_manifest, args.maximum_length),
        batch_size=args.batch_size, shuffle=True, drop_last=True,
        num_workers=args.num_workers, pin_memory=device.type == "cuda", collate_fn=collate_hidden,
    )
    val_loader = DataLoader(
        HiddenVideoDataset(args.val_manifest, args.maximum_length),
        batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=device.type == "cuda", collate_fn=collate_hidden,
    )
    model = SparseNeuronExpert(args.active_per_layer, args.temporal_width).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.max_epoch)
    checkpoint_path = output / "checkpoint_last.pth"
    best_path = output / "expert_best.pth"
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
        running = 0.0
        progress = tqdm(train_loader, desc=f"neuron expert {epoch + 1}/{args.max_epoch}")
        for step, batch in enumerate(progress, 1):
            hidden = batch["hidden"].to(device, non_blocking=True)
            lengths = batch["lengths"].to(device, non_blocking=True)
            labels = batch["labels"].to(device, non_blocking=True)
            logits = model(hidden, lengths)
            loss = expert_mil_loss(logits, labels, lengths) + args.sparsity_weight * model.sparsity_loss()
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            running += float(loss.detach())
            progress.set_postfix(loss=f"{running / step:.4f}")
        scheduler.step()
        metrics = evaluate(model, val_loader, device)
        selection_metric = 0.5 * (metrics["video_auc"] + metrics["video_ap"])
        payload = {
            "epoch": epoch,
            "best_metric": max(best, selection_metric),
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "config": {
                "active_per_layer": args.active_per_layer,
                "temporal_width": args.temporal_width,
            },
            "metrics": metrics,
        }
        torch.save(payload, checkpoint_path)
        if selection_metric > best:
            best = selection_metric
            torch.save(payload, best_path)
            (output / "selected_neurons.json").write_text(
                json.dumps(model.selection(), indent=2, ensure_ascii=False), encoding="utf-8"
            )
        record = {"epoch": epoch + 1, "loss": running / max(1, len(train_loader)), **metrics}
        with (output / "history.jsonl").open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record) + "\n")
        print(json.dumps(record), flush=True)


if __name__ == "__main__":
    main()

