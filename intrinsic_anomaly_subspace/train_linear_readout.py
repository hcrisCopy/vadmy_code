#!/usr/bin/env python3
"""Train a V-FIND-style linear readout on raw selected CLIP activations."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import average_precision_score, roc_auc_score
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from .common import (
    clean_output,
    load_pair,
    project,
    read_pair_manifest,
    save_json,
    seed_everything,
    selected_coordinates,
)


def build_fold(pair_manifest: str, fold: str, coordinates: list[tuple[int, int]]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    frame = read_pair_manifest(pair_manifest, fold)
    features, labels, weights = [], [], []
    for path in tqdm(frame["pair_path"], desc=f"load {fold} snippets", unit="video"):
        positive, negative = load_pair(str(path))
        features.extend([project(positive, coordinates), project(negative, coordinates)])
        labels.extend([np.ones(len(positive), dtype=np.float32), np.zeros(len(negative), dtype=np.float32)])
        # Every video and each of its two tails receive equal total loss weight.
        weights.extend([
            np.full(len(positive), 0.5 / len(positive), dtype=np.float32),
            np.full(len(negative), 0.5 / len(negative), dtype=np.float32),
        ])
    x = np.concatenate(features).astype(np.float32)
    y = np.concatenate(labels).astype(np.float32)
    w = np.concatenate(weights).astype(np.float32)
    w *= len(w) / w.sum()
    return x, y, w


@torch.no_grad()
def validate(model: torch.nn.Module, x: torch.Tensor, y: np.ndarray, batch_size: int) -> dict[str, float]:
    model.eval()
    logits = []
    for start in range(0, len(x), batch_size):
        logits.append(model(x[start:start + batch_size]).squeeze(1).cpu())
    scores = torch.sigmoid(torch.cat(logits)).numpy()
    return {
        "snippet_auc": float(roc_auc_score(y, scores)),
        "snippet_ap": float(average_precision_score(y, scores)),
        "snippets": int(len(y)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a linear anomaly classifier on a discovered CLIP subspace.")
    parser.add_argument("--pair-manifest", required=True)
    parser.add_argument("--subspace-json", required=True)
    parser.add_argument("--feature-mode", choices=["selected", "same_layer_random", "global_random"], required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--max-epoch", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--lr", type=float, default=1e-2)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--selection-metric", choices=["ap", "auc"], default="ap")
    parser.add_argument("--seed", type=int, default=234)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--clean", action="store_true")
    args = parser.parse_args()
    if args.clean and args.resume:
        parser.error("--clean and --resume are mutually exclusive")
    if args.max_epoch <= 0 or args.batch_size <= 0 or args.lr <= 0:
        parser.error("epochs, batch size, and learning rate must be positive")
    output = clean_output(args.out_dir, args.clean)
    checkpoint_path, best_path = output / "checkpoint_last.pth", output / "model_best.pth"
    if args.resume and not checkpoint_path.exists():
        raise FileNotFoundError(f"missing resume checkpoint: {checkpoint_path}")
    metadata = json.loads(Path(args.subspace_json).read_text(encoding="utf-8"))
    coordinates = selected_coordinates(metadata, args.feature_mode, args.seed)
    if not coordinates:
        raise RuntimeError("feature coordinate set is empty")
    seed_everything(args.seed)
    train_x, train_y, train_w = build_fold(args.pair_manifest, "train", coordinates)
    validation_x, validation_y, _ = build_fold(args.pair_manifest, "validation", coordinates)
    mean = train_x.mean(axis=0)
    std = train_x.std(axis=0)
    std = np.maximum(std, 1e-6)
    train_x = (train_x - mean) / std
    validation_x = (validation_x - mean) / std
    device = torch.device(args.device if torch.cuda.is_available() and args.device.startswith("cuda") else "cpu")
    dataset = TensorDataset(
        torch.from_numpy(train_x), torch.from_numpy(train_y), torch.from_numpy(train_w)
    )
    generator = torch.Generator().manual_seed(args.seed)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, generator=generator, pin_memory=device.type == "cuda")
    validation_tensor = torch.from_numpy(validation_x).to(device)
    model = torch.nn.Linear(len(coordinates), 1).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    config = {
        "method": "linear_readout_on_raw_hidden_coordinates",
        "pair_manifest": args.pair_manifest,
        "subspace_json": args.subspace_json,
        "feature_mode": args.feature_mode,
        "selected_width": len(coordinates),
        "max_epoch": args.max_epoch,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "selection_metric": args.selection_metric,
        "seed": args.seed,
        "model_selection_uses_frame_gt": False,
    }
    start, best = 0, -float("inf")
    if checkpoint_path.exists() and (args.resume or not args.clean):
        state = torch.load(checkpoint_path, map_location="cpu")
        if state["config"] != config:
            raise RuntimeError("existing checkpoint configuration differs; use --clean")
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        start, best = int(state["epoch"]) + 1, float(state["best"])
    history_path = output / "history.jsonl"
    for epoch in range(start, args.max_epoch):
        model.train()
        total = 0.0
        progress = tqdm(loader, desc=f"linear {args.feature_mode} {epoch + 1}/{args.max_epoch}", unit="batch", leave=False)
        for step, (features, target, sample_weight) in enumerate(progress, 1):
            features, target, sample_weight = features.to(device), target.to(device), sample_weight.to(device)
            loss_vector = F.binary_cross_entropy_with_logits(model(features).squeeze(1), target, reduction="none")
            loss = (loss_vector * sample_weight).mean()
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            total += float(loss.detach())
            progress.set_postfix(loss=f"{total / step:.4f}")
        metrics = validate(model, validation_tensor, validation_y, args.batch_size)
        score = metrics[f"snippet_{args.selection_metric}"]
        state = {"model": model.state_dict(), "optimizer": optimizer.state_dict(), "epoch": epoch, "best": max(best, score), "config": config}
        if score > best:
            best = score
            torch.save({
                "model": model.state_dict(),
                "config": config,
                "coordinates": coordinates,
                "mean": mean.astype(np.float32),
                "std": std.astype(np.float32),
                "validation": metrics,
                "epoch": epoch,
            }, best_path)
        torch.save(state, checkpoint_path)
        record = {"epoch": epoch + 1, "train_loss": total / max(1, len(loader)), **metrics, "best": best}
        with history_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        print(json.dumps(record, ensure_ascii=False), flush=True)
    save_json(output / "training_report.json", {**config, "best_validation_metric": best, "model_path": str(best_path)})


if __name__ == "__main__":
    main()
