#!/usr/bin/env python3
"""Train the baseline-score-free whole-layer semantic localizer."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from semantic_knn_splicing.common import clean_output, is_normal, seed_everything
from semantic_knn_splicing.data import LocalizerDataset
from semantic_knn_splicing.model import WholeLayerSemanticLocalizer, load_frozen_clip, localizer_loss


def paired_batches(normal_loader: DataLoader, abnormal_loader: DataLoader):
    for normal, abnormal in zip(normal_loader, abnormal_loader):
        merged = {}
        for key in ("clip", "hidden", "length", "binary_label", "target_mask"):
            merged[key] = torch.cat([abnormal[key], normal[key]], dim=0)
        for key in ("label", "key", "clip_path", "hidden_path"):
            merged[key] = list(abnormal[key]) + list(normal[key])
        yield merged


def split_csv(source: str, dataset: str, output: Path, val_fraction: float, seed: int) -> tuple[Path, Path]:
    frame = pd.read_csv(source)
    videos = sorted(frame["key"].astype(str).unique())
    rng = np.random.default_rng(seed)
    rng.shuffle(videos)
    val_count = max(1, int(round(len(videos) * val_fraction)))
    val_keys = set(videos[:val_count])
    train = frame[~frame["key"].astype(str).isin(val_keys)]
    val = frame[frame["key"].astype(str).isin(val_keys)]
    # A deterministic split can rarely miss one side; move one key rather than
    # silently validating on a single class.
    for expect_normal in (True, False):
        present = val["label"].map(lambda value: is_normal(dataset, str(value)) == expect_normal).any()
        if not present:
            candidates = train[train["label"].map(lambda value: is_normal(dataset, str(value)) == expect_normal)]
            if candidates.empty:
                raise RuntimeError("training data has no normal/abnormal pair")
            key = str(candidates.iloc[0]["key"])
            moving = train[train["key"].astype(str) == key]
            train = train[train["key"].astype(str) != key]
            val = pd.concat([val, moving], ignore_index=True)
    train_path, val_path = output / "localizer_train.csv", output / "localizer_val.csv"
    train.to_csv(train_path, index=False)
    val.to_csv(val_path, index=False)
    return train_path, val_path


def normal_centroid(csv_path: Path, sequence_length: int, device: torch.device) -> torch.Tensor:
    frame = pd.read_csv(csv_path)
    values = []
    for path in tqdm(frame["clip_path"], desc="normal CLIP centroid", unit="crop"):
        clip = np.load(str(path), mmap_mode="r")
        if len(clip) > sequence_length:
            indices = np.linspace(0, len(clip) - 1, sequence_length, dtype=np.int64)
            clip = clip[indices]
        values.append(torch.from_numpy(np.asarray(clip, dtype=np.float32)).mean(dim=0))
    if not values:
        raise RuntimeError("cannot estimate normal centroid from an empty set")
    return F.normalize(torch.stack(values).mean(dim=0), dim=0).to(device)


def evaluate(model, loader: DataLoader, device: torch.device, args) -> dict[str, float]:
    model.eval()
    totals = {key: 0.0 for key in ("loss", "positive", "normal", "bottom", "smooth", "sparse")}
    steps = 0
    for batch in tqdm(loader, desc="localizer validation", unit="batch", leave=False):
        with torch.no_grad():
            record = model(batch["clip"].to(device), batch["hidden"].to(device))
            loss, pieces = localizer_loss(
                record, batch["target_mask"].to(device), batch["binary_label"].to(device),
                batch["length"].to(device), args.topk_ratio, args.smooth_weight, args.sparse_weight,
            )
        totals["loss"] += float(loss)
        for key, value in pieces.items():
            totals[key] += float(value)
        steps += 1
    return {key: value / max(1, steps) for key, value in totals.items()}


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the whole-layer AnomalyCLIP-style localizer.")
    parser.add_argument("--dataset", choices=["ucf", "xd"], required=True)
    parser.add_argument("--train-csv", required=True)
    parser.add_argument("--clip-weight", required=True)
    parser.add_argument("--layer-atlas", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--max-epoch", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--sequence-length", type=int, default=256)
    parser.add_argument("--context-length", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--topk-ratio", type=int, default=16)
    parser.add_argument("--smooth-weight", type=float, default=8e-4)
    parser.add_argument("--sparse-weight", type=float, default=8e-3)
    parser.add_argument("--val-fraction", type=float, default=0.10)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=234)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--clean", action="store_true")
    args = parser.parse_args()
    if args.clean and args.resume:
        parser.error("--clean and --resume are mutually exclusive")
    if not 0 < args.val_fraction < 0.5 or min(args.max_epoch, args.batch_size, args.topk_ratio) <= 0:
        parser.error("invalid split, epoch, batch, or top-k setting")
    output = clean_output(args.out_dir, args.clean)
    checkpoint_path, best_path = output / "checkpoint_last.pth", output / "localizer_best.pth"
    resume = args.resume or checkpoint_path.exists()
    if args.resume and not checkpoint_path.exists():
        raise FileNotFoundError(f"resume checkpoint does not exist: {checkpoint_path}")
    seed_everything(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() and args.device.startswith("cuda") else "cpu")
    train_csv, val_csv = split_csv(args.train_csv, args.dataset, output, args.val_fraction, args.seed)
    atlas = json.loads(Path(args.layer_atlas).read_text(encoding="utf-8"))
    layers = [int(block["layer_zero_based"]) for block in atlas["blocks"]]
    clip_model, tokenize = load_frozen_clip(args.clip_weight, device)
    model = WholeLayerSemanticLocalizer(layers, clip_model, tokenize, args.dataset, args.context_length).to(device)
    normal_frame = pd.read_csv(train_csv)
    normal_frame = normal_frame[normal_frame["label"].map(lambda value: is_normal(args.dataset, str(value)))]
    centroid_csv = output / "normal_rows.csv"
    normal_frame.to_csv(centroid_csv, index=False)
    model.set_normal_centroid(normal_centroid(centroid_csv, args.sequence_length, device))

    normal_set = LocalizerDataset(str(train_csv), args.dataset, args.sequence_length, "normal")
    abnormal_set = LocalizerDataset(str(train_csv), args.dataset, args.sequence_length, "abnormal")
    half_batch = max(1, args.batch_size // 2)
    common = dict(batch_size=half_batch, shuffle=True, drop_last=True, num_workers=args.num_workers, pin_memory=device.type == "cuda")
    normal_loader = DataLoader(normal_set, **common)
    abnormal_loader = DataLoader(abnormal_set, **common)
    val_loader = DataLoader(LocalizerDataset(str(val_csv), args.dataset, args.sequence_length), batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    if any(parameter.requires_grad for parameter in model.clip_model.parameters()):
        raise RuntimeError("CLIP parameters must remain frozen")
    optimizer = torch.optim.AdamW(parameters, lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.max_epoch)
    run_config = {
        "method": model.method_name, "dataset": args.dataset, "layers": layers,
        "sequence_length": args.sequence_length, "context_length": args.context_length,
        "lr": args.lr, "weight_decay": args.weight_decay, "topk_ratio": args.topk_ratio,
        "selection_rule": "lowest train-video held-out weak-label loss; no test labels",
    }
    (output / "parameter_report.json").write_text(json.dumps({
        **run_config,
        "trainable_parameters": sum(value.numel() for value in parameters),
        "clip_trainable_parameters": 0,
    }, indent=2, ensure_ascii=False), encoding="utf-8")
    start_epoch, best = 0, float("inf")
    if resume:
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        if checkpoint["run_config"] != run_config:
            raise RuntimeError("resume configuration differs from checkpoint")
        missing, unexpected = model.load_state_dict(checkpoint["model_state_dict"], strict=False)
        non_clip_missing = [key for key in missing if not key.startswith("clip_model.")]
        if non_clip_missing or unexpected:
            raise RuntimeError(f"invalid resume state: missing={non_clip_missing}, unexpected={unexpected}")
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        start_epoch, best = int(checkpoint["epoch"]) + 1, float(checkpoint["best_loss"])
    history = output / "history.jsonl"

    def method_state() -> dict[str, torch.Tensor]:
        # The released DSANet checkpoint is the single source of CLIP weights;
        # do not duplicate the frozen backbone in every localizer checkpoint.
        return {
            key: value for key, value in model.state_dict().items()
            if not key.startswith("clip_model.")
        }

    if start_epoch >= args.max_epoch:
        print(f"localizer already completed {args.max_epoch} epochs", flush=True)
        return
    for epoch in range(start_epoch, args.max_epoch):
        model.train()
        model.clip_model.eval()
        running, steps = 0.0, 0
        progress = tqdm(paired_batches(normal_loader, abnormal_loader), total=min(len(normal_loader), len(abnormal_loader)), desc=f"localizer {epoch + 1}/{args.max_epoch}", unit="batch")
        for batch in progress:
            record = model(batch["clip"].to(device, non_blocking=True), batch["hidden"].to(device, non_blocking=True))
            loss, _ = localizer_loss(record, batch["target_mask"].to(device), batch["binary_label"].to(device), batch["length"].to(device), args.topk_ratio, args.smooth_weight, args.sparse_weight)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            running += float(loss.detach()); steps += 1
            progress.set_postfix(loss=f"{running / steps:.4f}", layer_weight=f"{float(model.residual_weight.detach()):.3f}")
        scheduler.step()
        metrics = evaluate(model, val_loader, device, args)
        payload = {
            "epoch": epoch, "best_loss": min(best, metrics["loss"]), "run_config": run_config,
            "model_state_dict": method_state(), "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(), "metrics": metrics,
        }
        if metrics["loss"] < best:
            best = metrics["loss"]
            payload["best_loss"] = best
            torch.save(payload, best_path)
        torch.save(payload, checkpoint_path)
        record = {"epoch": epoch + 1, "train_loss": running / max(1, steps), "validation": metrics, "best_loss": best}
        with history.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        print(json.dumps(record, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
