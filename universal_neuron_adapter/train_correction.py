from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as functional
from sklearn.metrics import average_precision_score, roc_auc_score
from torch.utils.data import DataLoader
from tqdm import tqdm

from universal_neuron_adapter.data import ScorePairDataset, collate_scores
from universal_neuron_adapter.model import ScoreCorrectionHead, topk_bag, valid_mask


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def loss_terms(logits, baseline, labels, lengths, anchor_weight):
    probability = torch.sigmoid(logits)
    mask = valid_mask(lengths, logits.shape[1], logits.dtype)
    bag = topk_bag(probability, lengths)
    bag_loss = functional.binary_cross_entropy(bag, labels)
    normal_mask = (1.0 - labels).unsqueeze(1) * mask
    normal_loss = -(torch.log1p(-probability.clamp(max=1 - 1e-6)) * normal_mask).sum() / normal_mask.sum().clamp_min(1)
    abnormal, normal = bag[labels.bool()], bag[~labels.bool()]
    ranking = functional.softplus(0.5 - abnormal[:, None] + normal[None, :]).mean() if abnormal.numel() and normal.numel() else bag_loss * 0
    anchor = (((probability - baseline) ** 2) * mask).sum() / mask.sum().clamp_min(1)
    pair = mask[:, 1:] * mask[:, :-1]
    smooth = (((probability[:, 1:] - probability[:, :-1]) ** 2) * pair).sum() / pair.sum().clamp_min(1)
    return bag_loss + 0.5 * normal_loss + 0.5 * ranking + anchor_weight * anchor + 0.02 * smooth


def validate(model, loader, device):
    targets, corrected, baseline = [], [], []
    model.eval()
    with torch.no_grad():
        for batch in loader:
            base, expert, lengths = batch["baseline"].to(device), batch["expert"].to(device), batch["lengths"].to(device)
            corrected.extend(topk_bag(torch.sigmoid(model(base, expert)), lengths).cpu().tolist())
            baseline.extend(topk_bag(base, lengths).cpu().tolist())
            targets.extend(batch["labels"].tolist())
    return {"corrected_video_auc": roc_auc_score(targets, corrected), "corrected_video_ap": average_precision_score(targets, corrected), "baseline_video_auc": roc_auc_score(targets, baseline), "baseline_video_ap": average_precision_score(targets, baseline)}


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a conservative single-baseline score correction head.")
    for name in ("baseline-manifest", "expert-manifest", "train-keys", "val-keys", "out-dir"):
        parser.add_argument(f"--{name}", required=True)
    parser.add_argument("--baseline", choices=["dsanet", "desc", "lagovad"], required=True)
    parser.add_argument("--dataset", choices=["ucf", "xd"], required=True)
    parser.add_argument("--width", type=int, default=32)
    parser.add_argument("--max-epoch", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--anchor-weight", type=float, default=2.0)
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
    def loader(keys, shuffle):
        return DataLoader(ScorePairDataset(args.baseline_manifest, args.expert_manifest, keys, args.maximum_length), batch_size=args.batch_size, shuffle=shuffle, drop_last=shuffle, num_workers=args.num_workers, collate_fn=collate_scores)
    train, val = loader(args.train_keys, True), loader(args.val_keys, False)
    model = ScoreCorrectionHead(args.width).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.max_epoch)
    last, best_path = output / "checkpoint_last.pth", output / "model_best.pth"
    start, best = 0, -float("inf")
    if args.resume and last.exists():
        checkpoint = torch.load(last, map_location="cpu", weights_only=False)
        if float(checkpoint["config"].get("anchor_weight", 2.0)) != args.anchor_weight:
            raise ValueError("checkpoint anchor weight mismatch")
        model.load_state_dict(checkpoint["model_state_dict"]); optimizer.load_state_dict(checkpoint["optimizer_state_dict"]); scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        start, best = int(checkpoint["epoch"]) + 1, float(checkpoint["best_metric"])
    for epoch in range(start, args.max_epoch):
        model.train(); running = 0.0
        for batch in tqdm(train, desc=f"correction {args.baseline}/{args.dataset} {epoch + 1}/{args.max_epoch}"):
            base, expert, lengths, labels = batch["baseline"].to(device), batch["expert"].to(device), batch["lengths"].to(device), batch["labels"].to(device)
            loss = loss_terms(model(base, expert), base, labels, lengths, args.anchor_weight)
            optimizer.zero_grad(set_to_none=True); loss.backward(); optimizer.step(); running += float(loss.detach())
        scheduler.step(); metrics = validate(model, val, device); selection = 0.5 * (metrics["corrected_video_auc"] + metrics["corrected_video_ap"])
        payload = {"epoch": epoch, "best_metric": max(best, selection), "model_state_dict": model.state_dict(), "optimizer_state_dict": optimizer.state_dict(), "scheduler_state_dict": scheduler.state_dict(), "config": {"width": args.width, "anchor_weight": args.anchor_weight}, "baseline": args.baseline, "dataset": args.dataset, "metrics": metrics}
        torch.save(payload, last)
        if selection > best: best = selection; torch.save(payload, best_path)
        with (output / "history.jsonl").open("a", encoding="utf-8") as handle: handle.write(json.dumps({"epoch": epoch + 1, "loss": running / max(1, len(train)), **metrics}) + "\n")


if __name__ == "__main__":
    main()
