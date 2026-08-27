from __future__ import annotations

import argparse
import json
import random
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from universal_neuron_adapter.data import resample_curve
from universal_neuron_adapter.model import ScoreCorrectionHead


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class GuidedScoreDataset(Dataset):
    def __init__(self, baseline_manifest: str, expert_manifest: str, keys_path: str, maximum_length: int) -> None:
        baseline = pd.read_csv(baseline_manifest)
        expert = pd.read_csv(expert_manifest)[["key", "expert_score_path"]]
        keys = set(pd.read_csv(keys_path)["key"].astype(str))
        self.frame = baseline.merge(expert, on="key", validate="one_to_one")
        self.frame = self.frame[self.frame["key"].astype(str).isin(keys)].reset_index(drop=True)
        if self.frame.empty:
            raise ValueError(f"no matching rows for split {keys_path}")
        self.maximum_length = int(maximum_length)

    def __len__(self) -> int:
        return len(self.frame)

    def __getitem__(self, index: int) -> dict[str, object]:
        row = self.frame.iloc[index]
        baseline = np.load(str(row.baseline_score_path)).astype(np.float32)
        expert = resample_curve(np.load(str(row.expert_score_path)), len(baseline))
        length = min(len(baseline), self.maximum_length)
        if len(baseline) != length:
            baseline = resample_curve(baseline, length)
            expert = resample_curve(expert, length)
        return {
            "baseline": baseline,
            "expert": expert,
            "label": float(row.binary_label),
            "key": str(row.key),
        }


def collate(batch: list[dict[str, object]]) -> dict[str, object]:
    lengths = torch.tensor([len(item["baseline"]) for item in batch], dtype=torch.long)
    steps = int(lengths.max())
    baseline = torch.zeros(len(batch), steps)
    expert = torch.zeros(len(batch), steps)
    for index, item in enumerate(batch):
        length = int(lengths[index])
        baseline[index, :length] = torch.from_numpy(item["baseline"])
        expert[index, :length] = torch.from_numpy(item["expert"])
    return {
        "baseline": baseline,
        "expert": expert,
        "lengths": lengths,
        "labels": torch.tensor([item["label"] for item in batch], dtype=torch.float32),
        "keys": [item["key"] for item in batch],
    }


def guided_loss(
    logits: torch.Tensor,
    baseline: torch.Tensor,
    expert: torch.Tensor,
    labels: torch.Tensor,
    lengths: torch.Tensor,
) -> tuple[torch.Tensor, dict[str, float]]:
    positions = torch.arange(logits.shape[1], device=logits.device).unsqueeze(0)
    valid = positions < lengths.unsqueeze(1)
    target = torch.zeros_like(logits)
    weight = torch.zeros_like(logits)
    bags = []
    for index, length_tensor in enumerate(lengths):
        length = int(length_tensor.item())
        probability = torch.sigmoid(logits[index, :length])
        top_count = max(1, length // 16 + 1)
        bags.append(probability.topk(min(top_count, length)).values.mean())
        if labels[index] < 0.5:
            weight[index, :length] = 1.0
            continue
        base = baseline[index, :length]
        neuron = expert[index, :length]
        positive_cut = torch.quantile(base, 0.90)
        negative_cut = torch.quantile(base, 0.50)
        positive = base >= positive_cut
        negative = base <= negative_cut
        standardized = (neuron - neuron.mean()) / neuron.std(unbiased=False).clamp_min(1e-6)
        neuron_gate = torch.sigmoid(2.0 * standardized)
        target[index, :length][positive] = 1.0
        weight[index, :length][negative] = 0.5
        weight[index, :length][positive] = 1.0 + neuron_gate[positive]
    dense = (F.binary_cross_entropy_with_logits(logits, target, reduction="none") * weight).sum() / weight.sum().clamp_min(1.0)
    bag = F.binary_cross_entropy(torch.stack(bags), labels)
    baseline_logits = torch.logit(baseline.clamp(1e-5, 1.0 - 1e-5))
    anchor = (((logits - baseline_logits).square() * valid).sum() / valid.sum().clamp_min(1))
    pair = valid[:, 1:] & valid[:, :-1]
    probability = torch.sigmoid(logits)
    smooth = (((probability[:, 1:] - probability[:, :-1]).square() * pair).sum() / pair.sum().clamp_min(1))
    loss = dense + 0.5 * bag + 0.02 * anchor + 0.02 * smooth
    return loss, {
        "dense": float(dense.detach()),
        "bag": float(bag.detach()),
        "anchor": float(anchor.detach()),
        "smooth": float(smooth.detach()),
    }


def validate(model: ScoreCorrectionHead, loader: DataLoader, device: torch.device) -> float:
    model.eval()
    total = 0.0
    with torch.no_grad():
        for batch in tqdm(loader, desc="guided validation", leave=False):
            baseline = batch["baseline"].to(device)
            expert = batch["expert"].to(device)
            loss, _ = guided_loss(
                model(baseline, expert), baseline, expert,
                batch["labels"].to(device), batch["lengths"].to(device),
            )
            total += float(loss)
    return total / max(1, len(loader))


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a current-baseline-guided CLS-neuron correction head.")
    parser.add_argument("--baseline-manifest", required=True)
    parser.add_argument("--expert-manifest", required=True)
    parser.add_argument("--train-keys", required=True)
    parser.add_argument("--val-keys", required=True)
    parser.add_argument("--baseline", choices=["lagovad", "desc", "dsanet"], required=True)
    parser.add_argument("--dataset", choices=["ucf", "xd"], required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--width", type=int, default=32)
    parser.add_argument("--max-epoch", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--maximum-length", type=int, default=256)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=3407)
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
    train_data = GuidedScoreDataset(args.baseline_manifest, args.expert_manifest, args.train_keys, args.maximum_length)
    val_data = GuidedScoreDataset(args.baseline_manifest, args.expert_manifest, args.val_keys, args.maximum_length)
    generator = torch.Generator().manual_seed(args.seed)
    train_loader = DataLoader(
        train_data, batch_size=args.batch_size, shuffle=True, drop_last=False,
        num_workers=args.num_workers, pin_memory=device.type == "cuda", collate_fn=collate, generator=generator,
    )
    val_loader = DataLoader(
        val_data, batch_size=args.batch_size, shuffle=False, drop_last=False,
        num_workers=args.num_workers, pin_memory=device.type == "cuda", collate_fn=collate,
    )
    model = ScoreCorrectionHead(args.width).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.max_epoch)
    checkpoint_path = output / "checkpoint_last.pth"
    best_path = output / "model_best.pth"
    start_epoch, best = 0, float("inf")
    if args.resume and checkpoint_path.exists():
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        start_epoch = int(checkpoint["epoch"]) + 1
        best = float(checkpoint["best_metric"])

    for epoch in range(start_epoch, args.max_epoch):
        model.train()
        totals = {key: 0.0 for key in ("loss", "dense", "bag", "anchor", "smooth")}
        progress = tqdm(train_loader, desc=f"guided correction {args.baseline}/{args.dataset} {epoch + 1}/{args.max_epoch}")
        for step, batch in enumerate(progress, 1):
            baseline = batch["baseline"].to(device, non_blocking=True)
            expert = batch["expert"].to(device, non_blocking=True)
            loss, parts = guided_loss(
                model(baseline, expert), baseline, expert,
                batch["labels"].to(device, non_blocking=True), batch["lengths"].to(device, non_blocking=True),
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            totals["loss"] += float(loss.detach())
            for key, value in parts.items():
                totals[key] += value
            progress.set_postfix(loss=f"{totals['loss'] / step:.4f}")
        scheduler.step()
        validation_loss = validate(model, val_loader, device)
        payload = {
            "epoch": epoch,
            "best_metric": min(best, validation_loss),
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "config": {"width": args.width},
            "baseline": args.baseline,
            "dataset": args.dataset,
            "validation_loss": validation_loss,
        }
        torch.save(payload, checkpoint_path)
        if validation_loss < best:
            best = validation_loss
            torch.save(payload, best_path)
        record = {
            "epoch": epoch + 1,
            **{key: value / max(1, len(train_loader)) for key, value in totals.items()},
            "validation_loss": validation_loss,
        }
        with (output / "history.jsonl").open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record) + "\n")
        print(json.dumps(record), flush=True)


if __name__ == "__main__":
    main()
