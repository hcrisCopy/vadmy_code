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


def probability_anchor_loss(
    logits: torch.Tensor,
    baseline: torch.Tensor,
    labels: torch.Tensor,
    lengths: torch.Tensor,
) -> torch.Tensor:
    probability = torch.sigmoid(logits)
    mask = valid_mask(lengths, logits.shape[1], logits.dtype)
    bag = topk_bag(probability, lengths)
    bag_loss = functional.binary_cross_entropy(bag, labels)
    normal_mask = (1.0 - labels).unsqueeze(1) * mask
    normal_loss = -(
        torch.log1p(-probability.clamp(max=1.0 - 1e-6)) * normal_mask
    ).sum() / normal_mask.sum().clamp_min(1.0)
    abnormal, normal = bag[labels.bool()], bag[~labels.bool()]
    ranking = (
        functional.softplus(0.5 - abnormal[:, None] + normal[None, :]).mean()
        if abnormal.numel() and normal.numel()
        else bag_loss * 0.0
    )
    anchor = ((probability - baseline).square() * mask).sum()
    anchor = anchor / mask.sum().clamp_min(1.0)
    pair = mask[:, 1:] * mask[:, :-1]
    smooth = ((probability[:, 1:] - probability[:, :-1]).square() * pair).sum()
    smooth = smooth / pair.sum().clamp_min(1.0)
    return bag_loss + 0.5 * normal_loss + 0.5 * ranking + 2.0 * anchor + 0.02 * smooth


def conservative_logit_loss(
    logits: torch.Tensor,
    baseline: torch.Tensor,
    labels: torch.Tensor,
    lengths: torch.Tensor,
) -> torch.Tensor:
    """Identity-initialized objective from the frozen formal protocol."""
    probability = torch.sigmoid(logits)
    mask = valid_mask(lengths, logits.shape[1], logits.dtype)
    bag = topk_bag(probability, lengths)
    bag_loss = functional.binary_cross_entropy(bag, labels)
    normal_mask = (1.0 - labels).unsqueeze(1) * mask
    normal_loss = -(
        torch.log1p(-probability.clamp(max=1.0 - 1e-6)) * normal_mask
    ).sum() / normal_mask.sum().clamp_min(1.0)
    abnormal = bag[labels.bool()]
    normal_hard = [
        row[: int(length.item())].max()
        for row, length, label in zip(probability, lengths, labels)
        if label < 0.5
    ]
    ranking = (
        functional.softplus(
            0.5 - abnormal[:, None] + torch.stack(normal_hard)[None, :]
        ).mean()
        if abnormal.numel() and normal_hard
        else bag_loss * 0.0
    )
    baseline_logit = torch.logit(baseline.clamp(1e-5, 1.0 - 1e-5))
    anchor = (((logits - baseline_logit) * mask).square()).sum()
    anchor = anchor / mask.sum().clamp_min(1.0)
    pair = mask[:, 1:] * mask[:, :-1]
    smooth = ((probability[:, 1:] - probability[:, :-1]).square() * pair).sum()
    smooth = smooth / pair.sum().clamp_min(1.0)
    return bag_loss + 0.5 * normal_loss + 0.5 * ranking + 0.02 * anchor + 0.02 * smooth


def validate(
    model: ScoreCorrectionHead,
    loader: DataLoader,
    device: torch.device,
) -> dict[str, float]:
    targets: list[int] = []
    corrected: list[float] = []
    baseline: list[float] = []
    model.eval()
    with torch.no_grad():
        for batch in tqdm(loader, desc="correction validation", leave=False):
            base = batch["baseline"].to(device)
            expert = batch["expert"].to(device)
            lengths = batch["lengths"].to(device)
            corrected.extend(
                topk_bag(torch.sigmoid(model(base, expert, lengths)), lengths)
                .cpu()
                .tolist()
            )
            baseline.extend(topk_bag(base, lengths).cpu().tolist())
            targets.extend(batch["labels"].tolist())
    return {
        "corrected_video_auc": float(roc_auc_score(targets, corrected)),
        "corrected_video_ap": float(average_precision_score(targets, corrected)),
        "baseline_video_auc": float(roc_auc_score(targets, baseline)),
        "baseline_video_ap": float(average_precision_score(targets, baseline)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train a conservative single-baseline score correction head."
    )
    for name in (
        "baseline-manifest",
        "expert-manifest",
        "train-keys",
        "val-keys",
        "out-dir",
    ):
        parser.add_argument(f"--{name}", required=True)
    parser.add_argument(
        "--baseline",
        choices=["dsanet", "desc", "lagovad", "vadclip"],
        required=True,
    )
    parser.add_argument("--dataset", choices=["ucf", "xd"], required=True)
    parser.add_argument(
        "--loss-protocol",
        choices=["probability-anchor-v2", "conservative-logit-v1"],
        default="probability-anchor-v2",
    )
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

    def make_loader(keys: str, shuffle: bool) -> DataLoader:
        return DataLoader(
            ScorePairDataset(
                args.baseline_manifest,
                args.expert_manifest,
                keys,
                args.maximum_length,
            ),
            batch_size=args.batch_size,
            shuffle=shuffle,
            drop_last=shuffle,
            num_workers=args.num_workers,
            collate_fn=collate_scores,
        )

    train_loader = make_loader(args.train_keys, True)
    validation_loader = make_loader(args.val_keys, False)
    identity_init = args.loss_protocol == "conservative-logit-v1"
    loss_function = conservative_logit_loss if identity_init else probability_anchor_loss
    model = ScoreCorrectionHead(args.width, identity_init=identity_init).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.max_epoch
    )
    last_path = output / "checkpoint_last.pth"
    best_path = output / "model_best.pth"
    start_epoch, best_metric = 0, -float("inf")
    if args.resume and last_path.exists():
        checkpoint = torch.load(last_path, map_location="cpu", weights_only=False)
        checkpoint_protocol = checkpoint.get("config", {}).get(
            "loss_protocol", "probability-anchor-v2"
        )
        if checkpoint_protocol != args.loss_protocol:
            raise ValueError(
                f"checkpoint protocol {checkpoint_protocol!r} does not match "
                f"requested {args.loss_protocol!r}"
            )
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        start_epoch = int(checkpoint["epoch"]) + 1
        best_metric = float(checkpoint["best_metric"])

    for epoch in range(start_epoch, args.max_epoch):
        model.train()
        running_loss = 0.0
        progress = tqdm(
            train_loader,
            desc=f"correction {args.baseline}/{args.dataset} {epoch + 1}/{args.max_epoch}",
        )
        for batch in progress:
            base = batch["baseline"].to(device)
            expert = batch["expert"].to(device)
            lengths = batch["lengths"].to(device)
            labels = batch["labels"].to(device)
            loss = loss_function(model(base, expert, lengths), base, labels, lengths)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            running_loss += float(loss.detach())
        scheduler.step()
        metrics = validate(model, validation_loader, device)
        selection = 0.5 * (
            metrics["corrected_video_auc"] + metrics["corrected_video_ap"]
        )
        payload = {
            "epoch": epoch,
            "best_metric": max(best_metric, selection),
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "config": {
                "width": args.width,
                "loss_protocol": args.loss_protocol,
                "identity_init": identity_init,
            },
            "baseline": args.baseline,
            "dataset": args.dataset,
            "metrics": metrics,
        }
        torch.save(payload, last_path)
        if selection > best_metric:
            best_metric = selection
            torch.save(payload, best_path)
        record = {
            "epoch": epoch + 1,
            "loss": running_loss / max(1, len(train_loader)),
            **metrics,
        }
        with (output / "history.jsonl").open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record) + "\n")
        print(json.dumps(record), flush=True)


if __name__ == "__main__":
    main()
