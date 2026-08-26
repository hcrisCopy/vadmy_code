#!/usr/bin/env python3
"""Adapt only the released baseline's final binary head with sparse consensus."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from .baselines import build_baseline
from .common import clean_output, seed_everything
from .data import HeadTrainingDataset
from .evaluation import official_frame_metrics


def merge_batches(normal: dict, abnormal: dict) -> dict:
    merged = {}
    for key in ("clip", "target", "length", "binary_label"):
        merged[key] = torch.cat([normal[key], abnormal[key]], dim=0)
    merged["label_text"] = list(normal["label_text"]) + list(abnormal["label_text"])
    return merged


def consensus_loss(
    logits_list: list[torch.Tensor],
    target: torch.Tensor,
    lengths: torch.Tensor,
) -> torch.Tensor:
    positions = torch.arange(target.shape[1], device=target.device).unsqueeze(0)
    mask = (positions < lengths.unsqueeze(1)) & (target >= 0)
    if not mask.any():
        return sum(logits.sum() * 0.0 for logits in logits_list)
    losses = []
    for logits in logits_list:
        losses.append(F.binary_cross_entropy_with_logits(logits[mask], target[mask]))
    return torch.stack(losses).mean()


def keep_frozen_backbones_eval(adapter, baseline: str) -> None:
    if baseline == "dsanet":
        adapter.base.clipmodel.eval()
    elif baseline == "desc":
        adapter.sensitivity.clipmodel.eval()
        adapter.consistency.clipmodel.eval()
    else:
        adapter.base.clip_text_model.model.eval()


def parameter_groups(adapter, args) -> tuple[list[dict], list[str]]:
    adapter.set_train_scope("binary_head")
    names = [name for name, value in adapter.named_parameters() if value.requires_grad]
    if not names:
        raise RuntimeError("binary-head parameter group is empty")
    for name in names:
        if "classifier" not in name and "bin_head" not in name:
            raise RuntimeError(f"non-head parameter unexpectedly trainable: {name}")
    if args.baseline == "desc":
        groups = [
            {
                "params": list(adapter.sensitivity.classifier.parameters()),
                "lr": args.sensitivity_lr,
                "name": "sensitivity_binary_head",
            },
            {
                "params": list(adapter.consistency.classifier.parameters()),
                "lr": args.consistency_lr,
                "name": "consistency_binary_head",
            },
        ]
    else:
        groups = [{
            "params": [value for value in adapter.parameters() if value.requires_grad],
            "lr": args.lr,
            "name": "binary_head",
        }]
    return groups, names


def build_scheduler(optimizer, baseline: str, dataset: str, max_epoch: int, steps: int):
    if baseline == "desc":
        milestones = [4, 8] if dataset == "ucf" else [6, 8]
        return torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=milestones, gamma=0.1)
    if baseline == "lagovad":
        warmup_steps = 20
        total_steps = max(1, max_epoch * steps)

        def multiplier(step: int) -> float:
            if step < warmup_steps:
                return float(step + 1) / float(max(1, warmup_steps))
            progress = (step - warmup_steps) / float(max(1, total_steps - warmup_steps))
            return 0.5 * (1.0 + math.cos(min(progress, 1.0) * math.pi))

        return torch.optim.lr_scheduler.LambdaLR(optimizer, multiplier)
    return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max_epoch)


def main() -> None:
    parser = argparse.ArgumentParser(description="Head-only cross-expert baseline adaptation.")
    parser.add_argument("--baseline", choices=["dsanet", "desc", "lagovad"], required=True)
    parser.add_argument("--baseline-root", required=True)
    parser.add_argument("--baseline-weight", default="")
    parser.add_argument("--sensitivity-weight", default="")
    parser.add_argument("--consistency-weight", default="")
    parser.add_argument("--dataset", choices=["ucf", "xd"], required=True)
    parser.add_argument("--consensus-csv", required=True)
    parser.add_argument("--test-list", required=True)
    parser.add_argument("--gt-path", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--max-epoch", type=int, required=True)
    parser.add_argument("--batch-size", type=int, required=True)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--sensitivity-lr", type=float, default=1e-3)
    parser.add_argument("--consistency-lr", type=float, default=5e-5)
    parser.add_argument("--weight-decay", type=float, required=True)
    parser.add_argument("--author-loss-weight", type=float, default=1.0)
    parser.add_argument("--consensus-loss-weight", type=float, default=1.0)
    parser.add_argument("--frames-per-snippet", type=int, default=16)
    parser.add_argument("--dsanet-ucf-eval-samples", type=int, default=1280)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=234)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--clean", action="store_true")
    args = parser.parse_args()
    if args.clean and args.resume:
        parser.error("--clean and --resume are mutually exclusive")
    output_dir = clean_output(args.out_dir, args.clean)
    checkpoint_path = output_dir / "checkpoint_last.pth"
    best_path = output_dir / "model_best.pth"
    resume = args.resume or checkpoint_path.exists()
    if args.resume and not checkpoint_path.exists():
        raise FileNotFoundError(f"resume checkpoint does not exist: {checkpoint_path}")
    seed_everything(args.seed)
    device = torch.device(
        args.device if torch.cuda.is_available() and args.device.startswith("cuda") else "cpu"
    )
    adapter = build_baseline(args, str(device)).to(device)
    groups, trainable_names = parameter_groups(adapter, args)
    optimizer = torch.optim.AdamW(groups, weight_decay=args.weight_decay)
    dataset_options = dict(
        consensus_csv=args.consensus_csv,
        dataset=args.dataset,
        sequence_length=adapter.visual_length,
        baseline=args.baseline,
    )
    loader_options = dict(
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )
    normal_loader = DataLoader(HeadTrainingDataset(**dataset_options, kind="normal"), **loader_options)
    abnormal_loader = DataLoader(HeadTrainingDataset(**dataset_options, kind="abnormal"), **loader_options)
    steps = min(len(normal_loader), len(abnormal_loader))
    if steps == 0:
        raise RuntimeError("normal and abnormal head-training loaders must be non-empty")
    scheduler = build_scheduler(
        optimizer, args.baseline, args.dataset, args.max_epoch, steps
    )
    run_config = {
        "method": "responsibility_cross_expert_binary_head_only_v1",
        "baseline": args.baseline,
        "dataset": args.dataset,
        "max_epoch": args.max_epoch,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "sensitivity_lr": args.sensitivity_lr,
        "consistency_lr": args.consistency_lr,
        "weight_decay": args.weight_decay,
        "author_loss_weight": args.author_loss_weight,
        "consensus_loss_weight": args.consensus_loss_weight,
        "train_scope": "final binary head only",
    }
    (output_dir / "parameter_report.json").write_text(
        json.dumps({
            **run_config,
            "trainable_tensors": trainable_names,
            "trainable_parameters": sum(
                parameter.numel() for group in groups for parameter in group["params"]
            ),
            "clip_trainable_parameters": 0,
        }, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    start_epoch, best, processed = 0, -float("inf"), 0
    if resume:
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        if checkpoint["run_config"] != run_config:
            raise RuntimeError("resume configuration differs from checkpoint")
        adapter.load_state_dict(checkpoint["model_state_dict"], strict=True)
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        start_epoch = int(checkpoint["epoch"]) + 1
        best = float(checkpoint["best_metric"])
        processed = int(checkpoint["processed_samples"])
    history_path = output_dir / "history.jsonl"

    def payload(epoch: int, metrics: dict, tag: str) -> dict:
        return {
            "epoch": epoch,
            "best_metric": best,
            "processed_samples": processed,
            "run_config": run_config,
            "model_state_dict": adapter.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "metrics": metrics,
            "validation_tag": tag,
            "selection_rule": "UCF frame AUC" if args.dataset == "ucf" else "XD frame AP",
        }

    def validate(epoch: int, tag: str) -> dict:
        nonlocal best
        metrics = official_frame_metrics(
            adapter, args.test_list, args.gt_path, args.frames_per_snippet, device
        )
        value = metrics["frame_auc" if args.dataset == "ucf" else "frame_ap"]
        if value > best:
            best = value
            torch.save(payload(epoch, metrics, tag), best_path)
        print(f"validation {tag}: {metrics} | best={best:.6f}", flush=True)
        return metrics

    if not resume:
        adapter.set_train_scope("frozen")
        validate(-1, "author_initialization")
    next_eval = ((processed // args.dsanet_ucf_eval_samples) + 1) * args.dsanet_ucf_eval_samples
    for epoch in range(start_epoch, args.max_epoch):
        adapter.set_train_scope("binary_head")
        adapter.train()
        keep_frozen_backbones_eval(adapter, args.baseline)
        running = {"total": 0.0, "author": 0.0, "consensus": 0.0}
        progress = tqdm(
            zip(normal_loader, abnormal_loader),
            total=steps,
            desc=f"{args.baseline} head {epoch + 1}/{args.max_epoch}",
            unit="batch",
        )
        for step, (normal, abnormal) in enumerate(progress, 1):
            batch = merge_batches(normal, abnormal)
            clip = batch["clip"].to(device, non_blocking=True)
            lengths = batch["length"].to(device, non_blocking=True)
            labels = batch["binary_label"].to(device, non_blocking=True)
            target = batch["target"].to(device, non_blocking=True)
            output = adapter.forward_baseline(clip, lengths)
            author = adapter.original_loss(output, labels, batch["label_text"], lengths)
            dense = consensus_loss(adapter.binary_training_logits(output), target, lengths)
            total = args.author_loss_weight * author + args.consensus_loss_weight * dense
            optimizer.zero_grad(set_to_none=True)
            total.backward()
            optimizer.step()
            if args.baseline == "lagovad":
                scheduler.step()
            processed += int(labels.numel())
            for key, value in (("total", total), ("author", author), ("consensus", dense)):
                running[key] += float(value.detach())
            progress.set_postfix(
                loss=f"{running['total'] / step:.4f}",
                dense=f"{running['consensus'] / step:.4f}",
                scope="binary_head",
            )
            if args.baseline == "dsanet" and args.dataset == "ucf" and processed >= next_eval:
                validate(epoch, f"sample_{processed}")
                adapter.train()
                keep_frozen_backbones_eval(adapter, args.baseline)
                next_eval += args.dsanet_ucf_eval_samples
        if args.baseline != "lagovad":
            scheduler.step()
        metrics = {"selection_deferred_to_fixed_step": True}
        if not (args.baseline == "dsanet" and args.dataset == "ucf"):
            metrics = validate(epoch, f"epoch_{epoch + 1}")
        record = {
            "epoch": epoch + 1,
            **{f"{key}_loss": value / steps for key, value in running.items()},
            "metrics": metrics,
            "best_metric": best,
        }
        with history_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        # DSANet's released trainer begins every new epoch from its historical
        # best checkpoint.  Preserve that unusual but important behavior.
        if args.baseline == "dsanet" and best_path.exists():
            best_checkpoint = torch.load(best_path, map_location="cpu")
            adapter.load_state_dict(best_checkpoint["model_state_dict"], strict=True)
        torch.save(payload(epoch, metrics, "epoch_recovery"), checkpoint_path)
        print(json.dumps(record, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
