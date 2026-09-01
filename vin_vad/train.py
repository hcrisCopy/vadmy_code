from __future__ import annotations

import argparse
import hashlib
import json
import random
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

from vin_vad.data import FinalLayerDataset, collate_final_layer
from vin_vad.losses import bag_loss
from vin_vad.model import EventAblationModel


def file_sha256(path: str) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def setup_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def merge_batches(first: dict[str, object], second: dict[str, object]) -> dict[str, torch.Tensor]:
    maximum = max(first["features"].shape[1], second["features"].shape[1])

    def pad(batch: dict[str, object]) -> tuple[torch.Tensor, torch.Tensor]:
        difference = maximum - batch["features"].shape[1]
        return F.pad(batch["features"], (0, 0, 0, difference)), F.pad(batch["mask"], (0, difference))

    first_features, first_mask = pad(first)
    second_features, second_mask = pad(second)
    return {
        "features": torch.cat([first_features, second_features], dim=0),
        "mask": torch.cat([first_mask, second_mask], dim=0),
        "labels": torch.cat([first["labels"], second["labels"]], dim=0),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Train one fixed-budget E0--E3 ablation.")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--variant", choices=["e0", "e1", "e2", "e3"], required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--epochs", type=int, required=True)
    parser.add_argument("--batch-size", type=int, required=True, help="Per-class batch size, matching DSANet.")
    parser.add_argument("--lr", type=float, required=True)
    parser.add_argument("--weight-decay", type=float, required=True)
    parser.add_argument("--maximum-length", type=int, required=True)
    parser.add_argument("--width", type=int, required=True)
    parser.add_argument("--dropout", type=float, required=True)
    parser.add_argument("--num-workers", type=int, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--device", required=True)
    parser.add_argument("--clean", action="store_true")
    args = parser.parse_args()
    output = Path(args.out_dir)
    if "vadmy_data" not in output.resolve().parts:
        raise ValueError("out-dir must be inside the sibling vadmy_data directory")
    if args.clean and output.exists():
        shutil.rmtree(output)
    output.mkdir(parents=True, exist_ok=True)
    config = vars(args).copy()
    config.pop("clean")
    config["manifest_sha256"] = file_sha256(args.manifest)
    config["checkpoint_rule"] = "final epoch; test set is never used for selection"
    config_path = output / "config.json"
    if config_path.exists() and json.loads(config_path.read_text(encoding="utf-8")) != config:
        raise RuntimeError("training configuration changed; use a new output directory or --clean")
    config_path.write_text(json.dumps(config, indent=2), encoding="utf-8")
    summary_path = output / "train_summary.json"
    if summary_path.exists() and (output / "model_final.pt").exists():
        print(f"reusing completed training: {summary_path}", flush=True)
        print(summary_path.read_text(encoding="utf-8"), flush=True)
        return

    setup_seed(args.seed)
    device = torch.device(args.device)
    dataset = FinalLayerDataset(args.manifest, training=True, maximum_length=args.maximum_length)
    labels = dataset.frame["binary_label"].astype(int).to_numpy()
    normal_indices = np.flatnonzero(labels == 0).tolist()
    anomaly_indices = np.flatnonzero(labels == 1).tolist()
    loader_kwargs = {
        "batch_size": args.batch_size,
        "shuffle": True,
        "drop_last": True,
        "num_workers": args.num_workers,
        "pin_memory": device.type == "cuda",
        "persistent_workers": args.num_workers > 0,
        "collate_fn": collate_final_layer,
    }
    normal_loader = DataLoader(Subset(dataset, normal_indices), **loader_kwargs)
    anomaly_loader = DataLoader(Subset(dataset, anomaly_indices), **loader_kwargs)
    steps_per_epoch = min(len(normal_loader), len(anomaly_loader))
    if steps_per_epoch < 1:
        raise RuntimeError("batch size leaves no balanced training step")

    model = EventAblationModel(args.variant, width=args.width, dropout=args.dropout).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    checkpoint_path = output / "checkpoint_latest.pt"
    history_path = output / "history.json"
    start_epoch = 0
    history: list[dict[str, float]] = []
    if checkpoint_path.exists():
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        scheduler.load_state_dict(checkpoint["scheduler"])
        start_epoch = int(checkpoint["epoch"])
        history = list(checkpoint["history"])
        if "python_rng" in checkpoint:
            random.setstate(checkpoint["python_rng"])
            np.random.set_state(checkpoint["numpy_rng"])
            torch.set_rng_state(checkpoint["torch_rng"])
            if device.type == "cuda":
                torch.cuda.set_rng_state_all(checkpoint["cuda_rng"])
        print(f"resume {args.variant} from epoch {start_epoch}/{args.epochs}", flush=True)

    for epoch in range(start_epoch, args.epochs):
        model.train()
        losses = []
        correct = 0
        examples = 0
        pairs = zip(normal_loader, anomaly_loader)
        progress = tqdm(pairs, total=steps_per_epoch, desc=f"{args.variant} epoch {epoch + 1}/{args.epochs}")
        for normal_batch, anomaly_batch in progress:
            batch = merge_batches(normal_batch, anomaly_batch)
            features = batch["features"].to(device, non_blocking=True)
            mask = batch["mask"].to(device, non_blocking=True)
            labels_tensor = batch["labels"].to(device, non_blocking=True)
            emissions = model.tcn(features, mask)
            loss, video_probability = bag_loss(args.variant, emissions, mask, labels_tensor, model.chain)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
            correct += int(((video_probability.detach() >= 0.5) == (labels_tensor >= 0.5)).sum().item())
            examples += len(labels_tensor)
            progress.set_postfix(loss=f"{np.mean(losses):.4f}", acc=f"{correct / examples:.3f}")
        scheduler.step()
        epoch_row = {
            "epoch": epoch + 1,
            "train_loss": float(np.mean(losses)),
            "train_video_accuracy": correct / examples,
            "lr": float(optimizer.param_groups[0]["lr"]),
        }
        history.append(epoch_row)
        history_path.write_text(json.dumps(history, indent=2), encoding="utf-8")
        torch.save(
            {
                "epoch": epoch + 1,
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "history": history,
                "python_rng": random.getstate(),
                "numpy_rng": np.random.get_state(),
                "torch_rng": torch.get_rng_state(),
                "cuda_rng": torch.cuda.get_rng_state_all() if device.type == "cuda" else [],
            },
            checkpoint_path,
        )
        print(json.dumps(epoch_row), flush=True)

    torch.save(model.state_dict(), output / "model_final.pt")
    chain_diagnostics = {}
    if model.chain is not None:
        length = torch.tensor([args.maximum_length], device=device)
        chain_diagnostics = {
            "onset_at_256": float(model.chain.onset_probability(length)[0].detach().cpu()),
            "persistence": float(model.chain.persistence_probability(length)[0].detach().cpu()),
        }
    summary = {
        "status": "pass",
        "variant": args.variant,
        "checkpoint_rule": config["checkpoint_rule"],
        "epochs": args.epochs,
        "steps_per_epoch": steps_per_epoch,
        "parameters": sum(parameter.numel() for parameter in model.parameters()),
        "final_train": history[-1],
        "chain": chain_diagnostics,
    }
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
