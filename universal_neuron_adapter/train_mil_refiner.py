from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from universal_neuron_adapter.data import resample_curve, resample_matrix
from universal_neuron_adapter.model import NeuronMILRefiner


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class RefinerDataset(Dataset):
    def __init__(self, selected: str, baseline: str, expert: str, keys: str, maximum_length: int) -> None:
        frame = pd.read_csv(selected)
        frame = frame.merge(pd.read_csv(baseline)[["key", "baseline_score_path"]], on="key", validate="one_to_one")
        frame = frame.merge(pd.read_csv(expert)[["key", "expert_score_path"]], on="key", validate="one_to_one")
        keep = set(pd.read_csv(keys)["key"].astype(str))
        self.frame = frame[frame["key"].astype(str).isin(keep)].reset_index(drop=True)
        if self.frame.empty:
            raise ValueError(f"empty refiner split: {keys}")
        self.maximum_length = int(maximum_length)

    def __len__(self) -> int:
        return len(self.frame)

    def __getitem__(self, index: int) -> dict[str, object]:
        row = self.frame.iloc[index]
        neurons = np.load(str(row.selected_path)).astype(np.float32)
        length = min(len(neurons), self.maximum_length)
        neurons = resample_matrix(neurons, length)
        baseline = resample_curve(np.load(str(row.baseline_score_path)), length)
        expert = resample_curve(np.load(str(row.expert_score_path)), length)
        return {"neurons": neurons, "baseline": baseline, "expert": expert, "label": float(row.binary_label)}


def collate(batch: list[dict[str, object]]) -> dict[str, torch.Tensor]:
    lengths = torch.tensor([len(row["baseline"]) for row in batch], dtype=torch.long)
    steps, width = int(lengths.max()), int(batch[0]["neurons"].shape[1])
    neurons = torch.zeros(len(batch), steps, width)
    baseline = torch.full((len(batch), steps), 0.5)
    expert = torch.full((len(batch), steps), 0.5)
    for index, row in enumerate(batch):
        length = int(lengths[index])
        neurons[index, :length] = torch.from_numpy(row["neurons"])
        baseline[index, :length] = torch.from_numpy(row["baseline"])
        expert[index, :length] = torch.from_numpy(row["expert"])
    return {
        "neurons": neurons, "baseline": baseline, "expert": expert,
        "labels": torch.tensor([row["label"] for row in batch], dtype=torch.float32), "lengths": lengths,
    }


def mil_loss(logits: torch.Tensor, batch: dict[str, torch.Tensor]) -> torch.Tensor:
    positions = torch.arange(logits.shape[1], device=logits.device).unsqueeze(0)
    mask = positions < batch["lengths"].unsqueeze(1)
    probability = torch.sigmoid(logits)
    bags = []
    for row, length_tensor in zip(probability, batch["lengths"]):
        length = int(length_tensor.item())
        count = max(1, length // 16 + 1)
        bags.append(row[:length].topk(count).values.mean())
    bags = torch.stack(bags)
    bag_loss = F.binary_cross_entropy(bags, batch["labels"])
    normal_mask = mask * (batch["labels"] < 0.5).unsqueeze(1)
    normal_loss = (
        F.binary_cross_entropy_with_logits(logits, torch.zeros_like(logits), reduction="none") * normal_mask
    ).sum() / normal_mask.sum().clamp_min(1)
    abnormal_bags, normal_bags = bags[batch["labels"] > 0.5], bags[batch["labels"] < 0.5]
    ranking = (
        F.softplus(0.5 - abnormal_bags[:, None] + normal_bags[None, :]).mean()
        if abnormal_bags.numel() and normal_bags.numel() else bag_loss * 0.0
    )
    baseline_logit = torch.logit(batch["baseline"].clamp(1e-5, 1.0 - 1e-5))
    anchor = (((logits - baseline_logit).square()) * mask).sum() / mask.sum().clamp_min(1)
    pair_mask = mask[:, 1:] & mask[:, :-1]
    smooth = (((probability[:, 1:] - probability[:, :-1]).square()) * pair_mask).sum() / pair_mask.sum().clamp_min(1)
    abnormal_mask = mask * (batch["labels"] > 0.5).unsqueeze(1)
    sparse = (probability * abnormal_mask).sum() / abnormal_mask.sum().clamp_min(1)
    return bag_loss + 0.5 * normal_loss + 0.5 * ranking + 0.01 * anchor + 0.02 * smooth + 0.005 * sparse


def normalization(dataset: RefinerDataset) -> tuple[torch.Tensor, torch.Tensor]:
    width = int(dataset[0]["neurons"].shape[1])
    total, square, count = np.zeros(width, np.float64), np.zeros(width, np.float64), 0
    for index in tqdm(range(len(dataset)), desc="refiner normalization"):
        values = dataset[index]["neurons"].astype(np.float64)
        total += values.sum(axis=0)
        square += np.square(values).sum(axis=0)
        count += len(values)
    mean = total / count
    std = np.sqrt(np.maximum(square / count - np.square(mean), 1e-5))
    return torch.from_numpy(mean.astype(np.float32)), torch.from_numpy(std.astype(np.float32))


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a single-baseline CLS-neuron MIL residual refiner.")
    parser.add_argument("--selected-manifest", required=True)
    parser.add_argument("--baseline-manifest", required=True)
    parser.add_argument("--expert-manifest", required=True)
    parser.add_argument("--train-keys", required=True)
    parser.add_argument("--val-keys", required=True)
    parser.add_argument("--baseline", required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--width", type=int, default=64)
    parser.add_argument("--max-epoch", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--maximum-length", type=int, default=256)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    output = Path(args.out_dir)
    output.mkdir(parents=True, exist_ok=True)
    seed_everything(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    train = RefinerDataset(args.selected_manifest, args.baseline_manifest, args.expert_manifest, args.train_keys, args.maximum_length)
    val = RefinerDataset(args.selected_manifest, args.baseline_manifest, args.expert_manifest, args.val_keys, args.maximum_length)
    mean, std = normalization(train)
    generator = torch.Generator().manual_seed(args.seed)
    loader = DataLoader(train, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, collate_fn=collate, generator=generator)
    val_loader = DataLoader(val, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, collate_fn=collate)
    model = NeuronMILRefiner(int(mean.numel()), args.width).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.max_epoch)
    checkpoint_path, best_path = output / "checkpoint_last.pth", output / "model_best.pth"
    start_epoch, best = 0, float("inf")
    if args.resume and checkpoint_path.exists():
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        start_epoch, best = int(checkpoint["epoch"]) + 1, float(checkpoint["best_metric"])
    mean_device, std_device = mean.to(device), std.to(device)
    for epoch in range(start_epoch, args.max_epoch):
        model.train()
        total = 0.0
        progress = tqdm(loader, desc=f"MIL refiner {args.baseline}/{args.dataset} {epoch + 1}/{args.max_epoch}")
        for step, batch in enumerate(progress, 1):
            batch = {key: value.to(device, non_blocking=True) for key, value in batch.items()}
            logits = model((batch["neurons"] - mean_device) / std_device, batch["baseline"], batch["expert"])
            loss = mil_loss(logits, batch)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            total += float(loss.detach())
            progress.set_postfix(loss=f"{total / step:.4f}")
        scheduler.step()
        model.eval()
        validation = 0.0
        with torch.no_grad():
            for batch in val_loader:
                batch = {key: value.to(device, non_blocking=True) for key, value in batch.items()}
                validation += float(mil_loss(model((batch["neurons"] - mean_device) / std_device, batch["baseline"], batch["expert"]), batch))
        validation /= max(1, len(val_loader))
        payload = {
            "epoch": epoch, "best_metric": min(best, validation), "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(), "scheduler_state_dict": scheduler.state_dict(),
            "config": {"input_dim": int(mean.numel()), "width": args.width},
            "normalization_mean": mean, "normalization_std": std,
            "baseline": args.baseline, "dataset": args.dataset,
        }
        torch.save(payload, checkpoint_path)
        if validation < best:
            best = validation
            torch.save(payload, best_path)
        with (output / "history.jsonl").open("a", encoding="utf-8") as handle:
            handle.write(json.dumps({"epoch": epoch + 1, "loss": total / len(loader), "validation_loss": validation}) + "\n")
        print(json.dumps({"epoch": epoch + 1, "loss": total / len(loader), "validation_loss": validation}), flush=True)


if __name__ == "__main__":
    main()
