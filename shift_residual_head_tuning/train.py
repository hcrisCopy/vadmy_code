#!/usr/bin/env python3
"""Train Shift-Global768 residual injection and only the baseline score head."""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Iterator

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import average_precision_score, roc_auc_score
from torch.utils.data import DataLoader
from tqdm import tqdm


PACKAGE_ROOT = Path(__file__).resolve().parent.parent
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))

from neuron_responsibility.baselines import build_baseline
from neuron_responsibility.common import clean_output, load_json
from neuron_responsibility.data import AlignedFeatureDataset
from neuron_responsibility.evaluate import pad_chunks
from shift_residual_head_tuning.method import (
    ShiftResidualInjector, configure_score_head_only, score_head_train_mode,
)


def add_baseline_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--baseline", choices=["dsanet", "desc", "lagovad"], required=True)
    parser.add_argument("--baseline-root", required=True)
    parser.add_argument("--baseline-weight", default="")
    parser.add_argument("--sensitivity-weight", default="")
    parser.add_argument("--consistency-weight", default="")


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def merge_batches(normal: dict, abnormal: dict) -> dict:
    merged = {key: torch.cat([normal[key], abnormal[key]]) for key in ("clip", "neurons", "length", "binary_label")}
    for key in ("label_text", "key", "sample_id"):
        merged[key] = list(normal[key]) + list(abnormal[key])
    return merged


def epoch_batches(
    train_set: AlignedFeatureDataset,
    normal_set: AlignedFeatureDataset | None,
    abnormal_set: AlignedFeatureDataset | None,
    batch_size: int,
    num_workers: int,
    seed: int,
    paired: bool,
) -> tuple[Iterator[dict], int]:
    common = {"batch_size": batch_size, "shuffle": True, "drop_last": True, "num_workers": num_workers, "pin_memory": True}
    if not paired:
        loader = DataLoader(train_set, generator=torch.Generator().manual_seed(seed), **common)
        return iter(loader), len(loader)
    if normal_set is None or abnormal_set is None:
        raise RuntimeError("paired UCF training datasets are missing")
    normal_loader = DataLoader(normal_set, generator=torch.Generator().manual_seed(seed), **common)
    abnormal_loader = DataLoader(abnormal_set, generator=torch.Generator().manual_seed(seed + 100003), **common)

    def iterator() -> Iterator[dict]:
        for normal, abnormal in zip(normal_loader, abnormal_loader):
            yield merge_batches(normal, abnormal)

    return iterator(), min(len(normal_loader), len(abnormal_loader))


def frame_metrics(adapter, val_list: str, gt_path: str, frames_per_snippet: int, device: torch.device) -> dict[str, float]:
    adapter.eval()
    frame = pd.read_csv(val_list)
    if "key" not in frame.columns:
        frame["key"] = frame["clip_path"].map(lambda value: Path(str(value)).stem)
    predictions = []
    with torch.no_grad():
        for key, group in tqdm(frame.groupby("key", sort=False), desc="official frame validation", unit="video", leave=False):
            clip = np.concatenate([np.load(str(path)).astype(np.float32) for path in group["clip_path"]])
            neuron = np.concatenate([np.load(str(path)).astype(np.float32) for path in group["neuron_path"]])
            clip_chunks, lengths = pad_chunks(clip, adapter.visual_length)
            neuron_chunks, neuron_lengths = pad_chunks(neuron, adapter.visual_length)
            if not torch.equal(lengths, neuron_lengths):
                raise RuntimeError(f"{key}: CLIP/neuron validation lengths differ")
            output, _ = adapter.forward_conditioned(clip_chunks.to(device), neuron_chunks.to(device), lengths.to(device))
            for index, length in enumerate(lengths.tolist()):
                predictions.append(torch.sigmoid(output.binary_logits[index, :length]).cpu())
    snippet_scores = torch.cat(predictions).numpy()
    frame_scores = np.repeat(snippet_scores, frames_per_snippet)
    gt = np.load(gt_path).astype(np.int64).reshape(-1)
    if len(frame_scores) != len(gt):
        raise RuntimeError(f"strict frame alignment failed: prediction={len(frame_scores)} gt={len(gt)}")
    return {"frame_auc": float(roc_auc_score(gt, frame_scores)), "frame_ap": float(average_precision_score(gt, frame_scores))}


def main() -> None:
    parser = argparse.ArgumentParser(description="Shift-Global768 residual injection with score-head-only tuning.")
    add_baseline_arguments(parser)
    parser.add_argument("--dataset", choices=["ucf", "xd"], required=True)
    parser.add_argument("--train-list", required=True)
    parser.add_argument("--val-list", required=True)
    parser.add_argument("--gt-path", required=True)
    parser.add_argument("--neuron-json", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--max-epoch", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=7e-5)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--residual-hidden-width", type=int, default=1024)
    parser.add_argument("--residual-depth", type=int, default=3)
    parser.add_argument("--frames-per-snippet", type=int, default=16)
    parser.add_argument("--dsanet-ucf-eval-samples", type=int, default=1280)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=234)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--clean", action="store_true")
    args = parser.parse_args()
    if args.clean and args.resume:
        parser.error("--clean and --resume cannot be combined")
    if min(args.max_epoch, args.batch_size, args.lr, args.residual_hidden_width, args.residual_depth) <= 0:
        parser.error("epochs, batch size, learning rate, residual width, and depth must be positive")

    out_dir = clean_output(args.out_dir, args.clean)
    checkpoint_path, best_path = out_dir / "checkpoint_last.pth", out_dir / "model_best.pth"
    if checkpoint_path.exists() and not args.resume:
        raise RuntimeError("checkpoint exists; use --resume, --clean, or another --out-dir")
    seed_everything(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() and args.device.startswith("cuda") else "cpu")
    adapter = build_baseline(args, str(device)).to(device)
    head_parameters, head_names = configure_score_head_only(adapter, args.baseline)
    selection = load_json(args.neuron_json)
    injector = ShiftResidualInjector(
        neuron_width=int(selection["neuron_width"]), hidden_width=args.residual_hidden_width,
        depth=args.residual_depth,
    ).to(device)
    adapter.attach_pre_temporal_conditioner(injector)
    trainable_baseline_names = [
        name for name, parameter in adapter.named_parameters()
        if parameter.requires_grad and not name.startswith("pre_temporal_conditioner.")
    ]
    if set(trainable_baseline_names) != set(head_names):
        raise RuntimeError(
            f"score-head-only audit failed: actual={trainable_baseline_names}, expected={head_names}"
        )
    if any("clip" in name.lower() for name in trainable_baseline_names):
        raise RuntimeError("CLIP parameter unexpectedly became trainable")
    optimizer = torch.optim.AdamW(
        [
            {"params": injector.parameters(), "lr": args.lr, "name": "shift_residual"},
            {"params": head_parameters, "lr": args.lr, "name": "binary_score_head"},
        ],
        weight_decay=args.weight_decay,
    )

    train_set = AlignedFeatureDataset(args.train_list, args.dataset, adapter.visual_length)
    paired = args.dataset == "ucf" and args.baseline in {"dsanet", "desc"}
    normal_set = AlignedFeatureDataset(args.train_list, args.dataset, adapter.visual_length, "normal") if paired else None
    abnormal_set = AlignedFeatureDataset(args.train_list, args.dataset, adapter.visual_length, "abnormal") if paired else None
    if paired:
        assert normal_set is not None and abnormal_set is not None
        steps_per_epoch = min(len(normal_set) // args.batch_size, len(abnormal_set) // args.batch_size)
    else:
        steps_per_epoch = len(train_set) // args.batch_size
    if steps_per_epoch <= 0:
        raise RuntimeError("training set is smaller than one drop-last batch")
    if args.baseline == "dsanet":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.max_epoch)
        scheduler_rule = "CosineAnnealingLR per epoch"
        scheduler_per_step = False
    elif args.baseline == "desc":
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[4, 8], gamma=0.1)
        scheduler_rule = "MultiStepLR milestones=[4,8], gamma=0.1 per epoch"
        scheduler_per_step = False
    else:
        from transformers import get_scheduler
        scheduler = get_scheduler(
            "cosine", optimizer=optimizer, num_warmup_steps=20,
            num_training_steps=args.max_epoch * steps_per_epoch,
        )
        scheduler_rule = "Transformers cosine per step, warmup_steps=20"
        scheduler_per_step = True
    run_config = {
        "method": ShiftResidualInjector.method_name,
        "baseline": args.baseline,
        "dataset": args.dataset,
        "neuron_json": args.neuron_json,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "max_epoch": args.max_epoch,
        "paired_training": paired,
        "scheduler": scheduler_rule,
        "injector_config": injector.config(),
        "trainable_baseline_tensors": trainable_baseline_names,
    }
    report = {
        **run_config,
        "residual_parameters": sum(parameter.numel() for parameter in injector.parameters()),
        "score_head_parameters": sum(parameter.numel() for parameter in head_parameters),
        "clip_trainable_parameters": 0,
        "temporal_trainable_parameters": 0,
        "text_trainable_parameters": 0,
    }
    (out_dir / "parameter_report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False), flush=True)

    start_epoch, best, processed_samples = 0, -float("inf"), 0
    if args.resume:
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        if checkpoint.get("run_config") != run_config:
            raise RuntimeError("resume configuration differs from checkpoint")
        adapter.load_state_dict(checkpoint["model_state_dict"], strict=True)
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        start_epoch = int(checkpoint["epoch"]) + 1
        best = float(checkpoint["best_metric"])
        processed_samples = int(checkpoint["processed_samples"])

    history_path = out_dir / "history.jsonl"

    def payload(epoch: int, metrics: dict, tag: str) -> dict:
        return {
            "method": ShiftResidualInjector.method_name,
            "epoch": epoch,
            "best_metric": best,
            "processed_samples": processed_samples,
            "model_state_dict": adapter.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "run_config": run_config,
            "injector_config": injector.config(),
            "metrics": metrics,
            "selection_rule": "UCF frame AUC" if args.dataset == "ucf" else "XD frame AP",
            "validation_tag": tag,
        }

    def validate(epoch: int, tag: str) -> dict[str, float]:
        nonlocal best
        metrics = frame_metrics(adapter, args.val_list, args.gt_path, args.frames_per_snippet, device)
        chosen = metrics["frame_auc" if args.dataset == "ucf" else "frame_ap"]
        if chosen > best:
            best = chosen
            torch.save(payload(epoch, metrics, tag), best_path)
        print(f"validation {tag}: {metrics} | best={best:.6f}", flush=True)
        return metrics

    next_eval = ((processed_samples // args.dsanet_ucf_eval_samples) + 1) * args.dsanet_ucf_eval_samples
    for epoch in range(start_epoch, args.max_epoch):
        score_head_train_mode(adapter, injector, args.baseline)
        batches, steps = epoch_batches(
            train_set, normal_set, abnormal_set, args.batch_size, args.num_workers,
            args.seed + epoch, paired,
        )
        total_loss, total_delta = 0.0, 0.0
        progress = tqdm(batches, total=steps, desc=f"residual+score-head {epoch + 1}/{args.max_epoch}")
        metrics: dict = {}
        for step, batch in enumerate(progress, 1):
            clip = batch["clip"].to(device, non_blocking=True)
            neurons = batch["neurons"].to(device, non_blocking=True)
            lengths = batch["length"].to(device, non_blocking=True)
            labels = batch["binary_label"].to(device, non_blocking=True)
            output, records = adapter.forward_conditioned(clip, neurons, lengths)
            loss = adapter.original_loss(output, labels, list(batch["label_text"]), lengths)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            if scheduler_per_step:
                scheduler.step()
            processed_samples += int(labels.numel())
            delta_norm = torch.stack([record["applied_delta"].square().mean().sqrt() for record in records]).mean()
            total_loss += float(loss.detach())
            total_delta += float(delta_norm.detach())
            progress.set_postfix(loss=f"{total_loss / step:.4f}", gate=f"{float(injector.gate().detach()):.4f}", delta=f"{total_delta / step:.4f}")
            if args.baseline == "dsanet" and args.dataset == "ucf" and processed_samples >= next_eval:
                metrics = validate(epoch, f"sample_{processed_samples}")
                score_head_train_mode(adapter, injector, args.baseline)
                next_eval += args.dsanet_ucf_eval_samples
        if not (args.baseline == "dsanet" and args.dataset == "ucf"):
            metrics = validate(epoch, f"epoch_{epoch + 1}")
        elif not best_path.exists():
            metrics = validate(epoch, f"epoch_{epoch + 1}_bootstrap")
        if not scheduler_per_step:
            scheduler.step()
        record = {
            "epoch": epoch + 1,
            "loss": total_loss / max(1, steps),
            "gate": float(injector.gate().detach()),
            "applied_delta_rms": total_delta / max(1, steps),
            "learning_rate": float(optimizer.param_groups[0]["lr"]),
            "metrics": metrics,
        }
        with history_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        torch.save(payload(epoch, metrics, "epoch_recovery"), checkpoint_path)
        print(json.dumps(record, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
