from __future__ import annotations

import argparse
import json
import random
import shutil
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


def evaluate(
    model: SparseNeuronExpert, loader: DataLoader, device: torch.device
) -> float:
    model.eval()
    targets: list[int] = []
    scores: list[float] = []
    with torch.no_grad():
        for batch in tqdm(loader, desc="primary expert validation", leave=False):
            lengths = batch["lengths"].to(device)
            logits = model(batch["hidden"].to(device), lengths)
            scores.extend(topk_bag(torch.sigmoid(logits), lengths).cpu().tolist())
            targets.extend(batch["labels"].tolist())
    return 0.5 * (
        roc_auc_score(targets, scores) + average_precision_score(targets, scores)
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the primary sparse CLS-neuron detector.")
    for name in ("train-manifest", "val-manifest", "out-dir"):
        parser.add_argument(f"--{name}", required=True)
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
    parser.add_argument("--clean", action="store_true")
    args = parser.parse_args()

    output = Path(args.out_dir)
    if args.clean and output.exists():
        shutil.rmtree(output)
    output.mkdir(parents=True, exist_ok=True)
    seed_everything(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    train_loader = DataLoader(
        HiddenVideoDataset(args.train_manifest, args.maximum_length),
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=args.num_workers,
        collate_fn=collate_hidden,
    )
    validation_loader = DataLoader(
        HiddenVideoDataset(args.val_manifest, args.maximum_length),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_hidden,
    )
    model = SparseNeuronExpert(args.active_per_layer, args.temporal_width).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.max_epoch
    )
    last_path = output / "checkpoint_last.pth"
    best_path = output / "expert_best.pth"
    start_epoch, best_metric = 0, -float("inf")
    if args.resume and last_path.exists():
        checkpoint = torch.load(last_path, map_location="cpu", weights_only=False)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        start_epoch = int(checkpoint["epoch"]) + 1
        best_metric = float(checkpoint["best_metric"])

    for epoch in range(start_epoch, args.max_epoch):
        model.train()
        running_loss = 0.0
        for batch in tqdm(
            train_loader, desc=f"primary expert {epoch + 1}/{args.max_epoch}"
        ):
            lengths = batch["lengths"].to(device)
            logits = model(batch["hidden"].to(device), lengths)
            loss = expert_mil_loss(logits, batch["labels"].to(device), lengths)
            loss = loss + args.sparsity_weight * model.sparsity_loss()
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            running_loss += float(loss.detach())
        scheduler.step()
        validation_metric = evaluate(model, validation_loader, device)
        average_loss = running_loss / max(1, len(train_loader))
        payload = {
            "epoch": epoch,
            "best_metric": max(best_metric, validation_metric),
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "config": {
                "active_per_layer": args.active_per_layer,
                "temporal_width": args.temporal_width,
            },
        }
        torch.save(payload, last_path)
        if validation_metric > best_metric:
            best_metric = validation_metric
            torch.save(payload, best_path)
            (output / "selected_neurons.json").write_text(
                json.dumps(model.selection(), indent=2), encoding="utf-8"
            )
        record = {
            "epoch": epoch + 1,
            "loss": average_loss,
            "validation_metric": validation_metric,
        }
        with (output / "history.jsonl").open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record) + "\n")
        print(json.dumps(record), flush=True)


if __name__ == "__main__":
    main()
