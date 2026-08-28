from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import average_precision_score, roc_auc_score
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from universal_neuron_adapter.data import HiddenVideoDataset, resample_curve
from universal_neuron_adapter.model import SparseNeuronExpert, expert_mil_loss, topk_bag


class ResidualVideoDataset(Dataset):
    def __init__(self, manifest: str, primary_manifest: str, maximum_length: int) -> None:
        self.hidden = HiddenVideoDataset(manifest, maximum_length)
        scores = pd.read_csv(primary_manifest)[["key", "expert_score_path"]]
        self.score_paths = dict(zip(scores["key"].astype(str), scores["expert_score_path"].astype(str)))
        missing = set(self.hidden.frame["key"].astype(str)) - set(self.score_paths)
        if missing:
            raise ValueError(f"primary score manifest misses {len(missing)} dataset keys")

    def __len__(self) -> int:
        return len(self.hidden)

    def __getitem__(self, index: int) -> dict[str, object]:
        item = self.hidden[index]
        primary = resample_curve(np.load(self.score_paths[str(item["key"])]), len(item["hidden"]))
        return {**item, "primary": torch.from_numpy(primary)}


def collate_residual(items: list[dict[str, object]]) -> dict[str, object]:
    lengths = torch.tensor([len(item["hidden"]) for item in items], dtype=torch.long)
    hidden = torch.zeros(len(items), int(lengths.max()), 12, 768)
    primary = torch.full((len(items), int(lengths.max())), 0.5)
    for index, item in enumerate(items):
        length = len(item["hidden"])
        hidden[index, :length] = item["hidden"]
        primary[index, :length] = item["primary"]
    return {
        "hidden": hidden,
        "primary": primary,
        "lengths": lengths,
        "labels": torch.tensor([item["label"] for item in items], dtype=torch.float32),
    }


def primary_indices(checkpoint_path: str) -> list[list[int]]:
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    primary = SparseNeuronExpert(**checkpoint["config"])
    primary.load_state_dict(checkpoint["model_state_dict"])
    return torch.sigmoid(primary.gate_logits.detach()).topk(primary.active_per_layer, dim=-1).indices.tolist()


def union_probability(primary: torch.Tensor, complement: torch.Tensor) -> torch.Tensor:
    return 1.0 - (1.0 - primary.clamp(0.0, 1.0)) * (1.0 - complement.clamp(0.0, 1.0))


def validation_metric(model: SparseNeuronExpert, loader: DataLoader, device: torch.device) -> float:
    model.eval()
    targets, scores = [], []
    with torch.no_grad():
        for batch in tqdm(loader, desc="residual expert validation", leave=False):
            lengths = batch["lengths"].to(device)
            complement = torch.sigmoid(model(batch["hidden"].to(device), lengths))
            union = union_probability(batch["primary"].to(device), complement)
            scores.extend(topk_bag(union, lengths).cpu().tolist())
            targets.extend(batch["labels"].tolist())
    return 0.5 * (roc_auc_score(targets, scores) + average_precision_score(targets, scores))


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a coordinate-disjoint residual CLS-neuron expert.")
    for name in ("train-manifest", "val-manifest", "primary-manifest", "primary-model", "out-dir"):
        parser.add_argument(f"--{name}", required=True)
    parser.add_argument("--active-per-layer", type=int, default=32)
    parser.add_argument("--temporal-width", type=int, default=64)
    parser.add_argument("--max-epoch", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--sparsity-weight", type=float, default=1e-3)
    parser.add_argument("--standalone-weight", type=float, default=0.25)
    parser.add_argument("--maximum-length", type=int, default=256)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=234)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    output = Path(args.out_dir)
    output.mkdir(parents=True, exist_ok=True)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    config = {
        "active_per_layer": args.active_per_layer,
        "temporal_width": args.temporal_width,
        "forbidden_indices": primary_indices(args.primary_model),
    }
    model = SparseNeuronExpert(**config).to(device)
    train = DataLoader(ResidualVideoDataset(args.train_manifest, args.primary_manifest, args.maximum_length), batch_size=args.batch_size, shuffle=True, drop_last=True, num_workers=args.num_workers, collate_fn=collate_residual)
    validation = DataLoader(ResidualVideoDataset(args.val_manifest, args.primary_manifest, args.maximum_length), batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, collate_fn=collate_residual)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.max_epoch)
    last_path, best_path = output / "checkpoint_last.pth", output / "expert_best.pth"
    start_epoch, best_metric = 0, -1.0
    if args.resume and last_path.exists():
        checkpoint = torch.load(last_path, map_location="cpu", weights_only=False)
        if checkpoint.get("objective") != "residual_probability_union_v1":
            raise ValueError("checkpoint objective does not match residual expert v1")
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        start_epoch, best_metric = int(checkpoint["epoch"]) + 1, float(checkpoint["best_metric"])
    for epoch in range(start_epoch, args.max_epoch):
        model.train()
        total_loss = 0.0
        for batch in tqdm(train, desc=f"residual expert {epoch + 1}/{args.max_epoch}"):
            lengths, labels = batch["lengths"].to(device), batch["labels"].to(device)
            logits = model(batch["hidden"].to(device), lengths)
            complement = torch.sigmoid(logits)
            union = union_probability(batch["primary"].to(device), complement)
            union_logits = torch.logit(union.clamp(1e-5, 1.0 - 1e-5))
            loss = expert_mil_loss(union_logits, labels, lengths)
            loss = loss + args.standalone_weight * expert_mil_loss(logits, labels, lengths)
            loss = loss + args.sparsity_weight * model.sparsity_loss()
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            total_loss += float(loss)
        scheduler.step()
        metric = validation_metric(model, validation, device)
        payload = {
            "epoch": epoch,
            "best_metric": max(best_metric, metric),
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "config": config,
            "seed": args.seed,
            "objective": "residual_probability_union_v1",
            "standalone_weight": args.standalone_weight,
        }
        torch.save(payload, last_path)
        if metric > best_metric:
            best_metric = metric
            torch.save(payload, best_path)
            (output / "selected_neurons.json").write_text(json.dumps(model.selection(), indent=2), encoding="utf-8")
        with (output / "history.jsonl").open("a", encoding="utf-8") as history:
            history.write(json.dumps({"epoch": epoch + 1, "loss": total_loss / len(train), "validation_metric": metric}) + "\n")
        print(json.dumps({"epoch": epoch + 1, "loss": total_loss / len(train), "validation_metric": metric}), flush=True)


if __name__ == "__main__":
    main()
