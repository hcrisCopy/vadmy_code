from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from universal_neuron_adapter.data import ConsensusHiddenDataset, collate_consensus
from universal_neuron_adapter.model import ConsensusNeuronExpert, valid_mask


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def loss_parts(logits: torch.Tensor, target: torch.Tensor, labels: torch.Tensor, lengths: torch.Tensor) -> tuple[torch.Tensor, dict]:
    mask = valid_mask(lengths, logits.shape[1], logits.dtype)
    dense = (F.binary_cross_entropy_with_logits(logits, target, reduction="none") * mask).sum() / mask.sum()
    probability = torch.sigmoid(logits)
    predicted_bag, target_bag = [], []
    for pred, truth, length in zip(probability, target, lengths):
        count = int(length.item())
        top = max(1, count // 16 + 1)
        predicted_bag.append(pred[:count].topk(top).values.mean())
        target_bag.append(truth[:count].topk(top).values.mean())
    predicted_bag = torch.stack(predicted_bag)
    target_bag = torch.stack(target_bag)
    bag_target = torch.maximum(target_bag, 0.9 * labels)
    bag = F.binary_cross_entropy(predicted_bag, bag_target)
    pair = mask[:, 1:] * mask[:, :-1]
    smooth = (((probability[:, 1:] - probability[:, :-1]).square() * pair).sum() / pair.sum().clamp_min(1.0))
    loss = dense + 0.5 * bag + 0.02 * smooth
    return loss, {"dense": float(dense.detach()), "bag": float(bag.detach()), "smooth": float(smooth.detach())}


def validate(model: ConsensusNeuronExpert, loader: DataLoader, device: torch.device) -> float:
    model.eval()
    total, batches = 0.0, 0
    with torch.no_grad():
        for batch in tqdm(loader, desc="consensus validation", leave=False):
            logits = model(batch["hidden"].to(device), batch["lengths"].to(device))
            loss, _ = loss_parts(logits, batch["target"].to(device), batch["labels"].to(device), batch["lengths"].to(device))
            total += float(loss)
            batches += 1
    return total / max(1, batches)


def main() -> None:
    parser = argparse.ArgumentParser(description="Distill train-only strong-baseline consensus into sparse CLS neurons.")
    parser.add_argument("--train-keys", required=True)
    parser.add_argument("--val-keys", required=True)
    parser.add_argument("--desc-manifest", required=True)
    parser.add_argument("--dsanet-manifest", required=True)
    parser.add_argument("--dataset", choices=["ucf", "xd"], required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--max-epoch", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--maximum-length", type=int, default=256)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=234)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    seed_everything(args.seed)
    output = Path(args.out_dir)
    output.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    common = (args.desc_manifest, args.dsanet_manifest, args.maximum_length)
    train = DataLoader(ConsensusHiddenDataset(args.train_keys, *common), batch_size=args.batch_size, shuffle=True,
                       num_workers=args.num_workers, pin_memory=device.type == "cuda", collate_fn=collate_consensus)
    val = DataLoader(ConsensusHiddenDataset(args.val_keys, *common), batch_size=args.batch_size, shuffle=False,
                     num_workers=args.num_workers, pin_memory=device.type == "cuda", collate_fn=collate_consensus)
    model = ConsensusNeuronExpert().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.max_epoch)
    best = float("inf")
    for epoch in range(args.max_epoch):
        model.train()
        totals = {"loss": 0.0, "dense": 0.0, "bag": 0.0, "smooth": 0.0}
        progress = tqdm(train, desc=f"consensus expert {args.dataset} {epoch + 1}/{args.max_epoch}")
        for step, batch in enumerate(progress, 1):
            lengths = batch["lengths"].to(device)
            logits = model(batch["hidden"].to(device, non_blocking=True), lengths)
            loss, parts = loss_parts(logits, batch["target"].to(device), batch["labels"].to(device), lengths)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            totals["loss"] += float(loss.detach())
            for key, value in parts.items():
                totals[key] += value
            progress.set_postfix(loss=f"{totals['loss'] / step:.4f}")
        scheduler.step()
        validation = validate(model, val, device)
        record = {"epoch": epoch + 1, "validation_loss": validation,
                  **{key: value / max(1, len(train)) for key, value in totals.items()}}
        with (output / "history.jsonl").open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record) + "\n")
        print(json.dumps(record), flush=True)
        payload = {"model_state_dict": model.state_dict(), "dataset": args.dataset,
                   "config": {"active_per_layer": 32, "temporal_width": 64}, "record": record}
        torch.save(payload, output / "checkpoint_last.pth")
        if validation < best:
            best = validation
            torch.save(payload, output / "model_best.pth")
            (output / "selected_neurons.json").write_text(json.dumps(model.selection(), indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()

