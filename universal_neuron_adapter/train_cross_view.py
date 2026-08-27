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

from universal_neuron_adapter.data import normalize_selected_layers, resample_curve, resample_matrix
from universal_neuron_adapter.model import TemporalNeuronHead


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class CrossViewDataset(Dataset):
    def __init__(self, selected: str, teacher: str, expert: str, keys: str, maximum_length: int) -> None:
        frame = pd.read_csv(selected)
        frame = frame.merge(pd.read_csv(teacher)[["key", "baseline_score_path"]], on="key", validate="one_to_one")
        frame = frame.merge(pd.read_csv(expert)[["key", "expert_score_path"]], on="key", validate="one_to_one")
        keep = set(pd.read_csv(keys)["key"].astype(str))
        self.frame = frame[frame["key"].astype(str).isin(keep)].reset_index(drop=True)
        if self.frame.empty:
            raise ValueError(f"empty cross-view split: {keys}")
        self.maximum_length = int(maximum_length)

    def __len__(self) -> int:
        return len(self.frame)

    def __getitem__(self, index: int) -> dict[str, object]:
        row = self.frame.iloc[index]
        neurons = np.load(str(row.selected_path)).astype(np.float32)
        length = min(len(neurons), self.maximum_length)
        neurons = normalize_selected_layers(resample_matrix(neurons, length))
        return {
            "neurons": neurons,
            "teacher": resample_curve(np.load(str(row.baseline_score_path)), length),
            "expert": resample_curve(np.load(str(row.expert_score_path)), length),
            "label": float(row.binary_label),
        }


def collate(batch: list[dict[str, object]]) -> dict[str, torch.Tensor]:
    lengths = torch.tensor([len(row["teacher"]) for row in batch], dtype=torch.long)
    steps, width = int(lengths.max()), int(batch[0]["neurons"].shape[1])
    neurons = torch.zeros(len(batch), steps, width)
    teacher, expert = torch.zeros(len(batch), steps), torch.zeros(len(batch), steps)
    for index, row in enumerate(batch):
        length = int(lengths[index])
        neurons[index, :length] = torch.from_numpy(row["neurons"])
        teacher[index, :length] = torch.from_numpy(row["teacher"])
        expert[index, :length] = torch.from_numpy(row["expert"])
    return {"neurons": neurons, "teacher": teacher, "expert": expert, "labels": torch.tensor([row["label"] for row in batch]), "lengths": lengths}


def normalized_rank(values: torch.Tensor) -> torch.Tensor:
    order = torch.argsort(values, stable=True)
    ranks = torch.empty_like(values)
    ranks[order] = (torch.arange(len(values), device=values.device, dtype=values.dtype) + 1.0) / (len(values) + 1.0)
    return ranks


def cross_view_loss(logits: torch.Tensor, batch: dict[str, torch.Tensor]) -> torch.Tensor:
    target, weight, bags = torch.zeros_like(logits), torch.zeros_like(logits), []
    for index, length_tensor in enumerate(batch["lengths"]):
        length = int(length_tensor.item())
        probability = torch.sigmoid(logits[index, :length])
        bags.append(probability.topk(max(1, length // 16 + 1)).values.mean())
        if batch["labels"][index] < 0.5:
            weight[index, :length] = 1.0
            continue
        teacher_rank = normalized_rank(batch["teacher"][index, :length])
        expert_rank = normalized_rank(batch["expert"][index, :length])
        positive = (teacher_rank >= 0.8) & (expert_rank >= 0.8)
        negative = (teacher_rank <= 0.5) & (expert_rank <= 0.5)
        if not positive.any():
            positive[(teacher_rank + expert_rank).topk(max(1, length // 20)).indices] = True
        valid_target, valid_weight = target[index, :length], weight[index, :length]
        valid_target[positive] = 1.0
        valid_weight[positive] = 2.0
        valid_weight[negative] = 0.5
    dense = (F.binary_cross_entropy_with_logits(logits, target, reduction="none") * weight).sum() / weight.sum().clamp_min(1.0)
    return dense + 0.5 * F.binary_cross_entropy(torch.stack(bags), batch["labels"])


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a layer-normalized baseline-neuron agreement head.")
    parser.add_argument("--selected-manifest", required=True)
    parser.add_argument("--teacher-manifest", required=True)
    parser.add_argument("--expert-manifest", required=True)
    parser.add_argument("--train-keys", required=True)
    parser.add_argument("--val-keys", required=True)
    parser.add_argument("--baseline", required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--width", type=int, default=64)
    parser.add_argument("--max-epoch", type=int, default=8)
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
    train = CrossViewDataset(args.selected_manifest, args.teacher_manifest, args.expert_manifest, args.train_keys, args.maximum_length)
    val = CrossViewDataset(args.selected_manifest, args.teacher_manifest, args.expert_manifest, args.val_keys, args.maximum_length)
    generator = torch.Generator().manual_seed(args.seed)
    loader = DataLoader(train, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, collate_fn=collate, generator=generator)
    val_loader = DataLoader(val, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, collate_fn=collate)
    model = TemporalNeuronHead(int(train[0]["neurons"].shape[1]), args.width).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    checkpoint_path, best_path = output / "checkpoint_last.pth", output / "model_best.pth"
    start_epoch, best = 0, float("inf")
    if args.resume and checkpoint_path.exists():
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        start_epoch, best = int(checkpoint["epoch"]) + 1, float(checkpoint["best_metric"])
    for epoch in range(start_epoch, args.max_epoch):
        model.train()
        total = 0.0
        progress = tqdm(loader, desc=f"normalized cross-view {args.baseline}/{args.dataset} {epoch + 1}/{args.max_epoch}")
        for step, batch in enumerate(progress, 1):
            batch = {key: value.to(device, non_blocking=True) for key, value in batch.items()}
            loss = cross_view_loss(model(batch["neurons"]), batch)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            total += float(loss.detach())
            progress.set_postfix(loss=f"{total / step:.4f}")
        model.eval()
        validation = 0.0
        with torch.no_grad():
            for batch in val_loader:
                batch = {key: value.to(device, non_blocking=True) for key, value in batch.items()}
                validation += float(cross_view_loss(model(batch["neurons"]), batch))
        validation /= max(1, len(val_loader))
        payload = {"epoch": epoch, "best_metric": min(best, validation), "model_state_dict": model.state_dict(), "optimizer_state_dict": optimizer.state_dict(), "config": {"input_dim": int(train[0]["neurons"].shape[1]), "width": args.width}, "baseline": args.baseline, "dataset": args.dataset}
        torch.save(payload, checkpoint_path)
        if validation < best:
            best = validation
            torch.save(payload, best_path)
        with (output / "history.jsonl").open("a", encoding="utf-8") as handle:
            handle.write(json.dumps({"epoch": epoch + 1, "loss": total / len(loader), "validation_loss": validation}) + "\n")
        print(json.dumps({"epoch": epoch + 1, "loss": total / len(loader), "validation_loss": validation}), flush=True)


if __name__ == "__main__":
    main()
