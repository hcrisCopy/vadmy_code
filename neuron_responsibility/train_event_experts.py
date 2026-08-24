#!/usr/bin/env python3
"""Train neuron-routed event experts with the released VAD objective."""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm


PACKAGE_ROOT = Path(__file__).resolve().parent.parent
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))

from neuron_responsibility.baselines import build_baseline
from neuron_responsibility.common import clean_output
from neuron_responsibility.data import AlignedFeatureDataset
from neuron_responsibility.event_experts import (
    NeuronRoutedEventExperts,
    event_expert_losses,
)
from neuron_responsibility.train_feature_modulation import (
    add_baseline_arguments,
    epoch_batches,
    is_head_parameter,
    official_frame_metrics,
    relative_parameter_anchor,
)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def maximum_scope(baseline: str) -> str:
    return "all_non_clip" if baseline in {"dsanet", "lagovad"} else "temporal_heads"


def scope_for_epoch(args: argparse.Namespace, epoch: int) -> tuple[str, str]:
    if epoch < args.head_start_epoch:
        return "experts", "frozen"
    if epoch < args.temporal_start_epoch:
        return "experts_heads", "heads"
    if epoch < args.full_start_epoch or args.baseline == "desc":
        return "experts_temporal_heads", "temporal_heads"
    return "experts_all_non_clip", maximum_scope(args.baseline)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Neuron-routed low-rank event experts for released VAD baselines."
    )
    add_baseline_arguments(parser)
    parser.add_argument("--dataset", choices=["ucf", "xd"], required=True)
    parser.add_argument("--train-list", required=True)
    parser.add_argument("--val-list", required=True)
    parser.add_argument("--gt-path", required=True)
    parser.add_argument("--atlas", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--max-epoch", type=int, default=10)
    parser.add_argument("--head-start-epoch", type=int, default=1)
    parser.add_argument("--temporal-start-epoch", type=int, default=2)
    parser.add_argument("--full-start-epoch", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--expert-lr", type=float, default=7e-5)
    parser.add_argument("--baseline-lr", type=float, default=7e-6)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--route-weight", type=float, default=0.10)
    parser.add_argument("--event-weight", type=float, default=0.05)
    parser.add_argument("--normal-weight", type=float, default=0.10)
    parser.add_argument("--smooth-weight", type=float, default=0.01)
    parser.add_argument("--anchor-weight", type=float, default=0.01)
    parser.add_argument("--expert-rank", type=int, default=32)
    parser.add_argument("--slow-dilation", type=int, default=4)
    parser.add_argument("--route-top-fraction", type=float, default=0.10)
    parser.add_argument("--frames-per-snippet", type=int, default=16)
    parser.add_argument("--dsanet-ucf-eval-samples", type=int, default=1280)
    parser.add_argument("--checkpoint-steps", type=int, default=10)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=234)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--clean", action="store_true")
    args = parser.parse_args()
    if args.clean and args.resume:
        parser.error("--clean and --resume cannot be used together")
    if not (
        0 <= args.head_start_epoch <= args.temporal_start_epoch
        <= args.full_start_epoch < args.max_epoch
    ):
        parser.error(
            "require 0 <= head-start <= temporal-start <= full-start < max-epoch"
        )
    non_negative = (
        args.expert_lr, args.baseline_lr, args.weight_decay, args.route_weight,
        args.event_weight, args.normal_weight, args.smooth_weight, args.anchor_weight,
    )
    if min(non_negative) < 0:
        parser.error("learning rates and loss weights must be non-negative")
    if args.checkpoint_steps <= 0 or args.dsanet_ucf_eval_samples <= 0:
        parser.error("checkpoint and evaluation intervals must be positive")

    out_dir = clean_output(args.out_dir, args.clean)
    recovery_path = out_dir / "checkpoint_last.pth"
    best_path = out_dir / "model_best.pth"
    if recovery_path.exists() and not args.resume:
        raise RuntimeError("checkpoint exists; use --resume, --clean, or a new --out-dir")
    seed_everything(args.seed)
    device = torch.device(
        args.device if torch.cuda.is_available() and args.device.startswith("cuda") else "cpu"
    )

    adapter = build_baseline(args, str(device)).to(device)
    adapter.set_train_scope(maximum_scope(args.baseline))
    baseline_parameters = {
        name: parameter
        for name, parameter in adapter.named_parameters()
        if parameter.requires_grad
    }
    if not baseline_parameters:
        raise RuntimeError("maximum baseline train scope has no parameters")
    frozen_clip = [
        name for name, parameter in adapter.named_parameters()
        if "clipmodel" in name.lower() and parameter.requires_grad
    ]
    if frozen_clip:
        raise RuntimeError(f"CLIP backbone must remain frozen: {frozen_clip[:3]}")
    initial_parameters = {
        name: parameter.detach().clone() for name, parameter in baseline_parameters.items()
    }
    adapter.set_train_scope("frozen")

    experts = NeuronRoutedEventExperts(
        args.atlas,
        feature_width=512,
        rank=args.expert_rank,
        slow_dilation=args.slow_dilation,
        route_top_fraction=args.route_top_fraction,
    ).to(device)
    adapter.attach_feature_modulator(experts)
    head_parameters = [
        parameter for name, parameter in baseline_parameters.items()
        if is_head_parameter(name)
    ]
    other_parameters = [
        parameter for name, parameter in baseline_parameters.items()
        if not is_head_parameter(name)
    ]
    if not head_parameters or not other_parameters:
        raise RuntimeError("baseline head/non-head parameter partition is empty")
    optimizer = torch.optim.AdamW([
        {"params": experts.parameters(), "lr": args.expert_lr, "name": "event_experts"},
        {"params": head_parameters, "lr": args.baseline_lr, "name": "baseline_heads"},
        {"params": other_parameters, "lr": args.baseline_lr, "name": "baseline_non_heads"},
    ], weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.max_epoch
    )

    train_set = AlignedFeatureDataset(args.train_list, args.dataset, adapter.visual_length)
    paired = args.baseline == "dsanet" and args.dataset == "ucf"
    normal_set = AlignedFeatureDataset(
        args.train_list, args.dataset, adapter.visual_length, split="normal"
    ) if paired else None
    abnormal_set = AlignedFeatureDataset(
        args.train_list, args.dataset, adapter.visual_length, split="abnormal"
    ) if paired else None

    run_config = {
        "method": experts.method_name,
        "baseline": args.baseline,
        "dataset": args.dataset,
        "atlas": args.atlas,
        "train_list": args.train_list,
        "val_list": args.val_list,
        "max_epoch": args.max_epoch,
        "head_start_epoch": args.head_start_epoch,
        "temporal_start_epoch": args.temporal_start_epoch,
        "full_start_epoch": args.full_start_epoch,
        "batch_size": args.batch_size,
        "expert_lr": args.expert_lr,
        "baseline_lr": args.baseline_lr,
        "loss_weights": {
            "route": args.route_weight,
            "event": args.event_weight,
            "normal": args.normal_weight,
            "smooth": args.smooth_weight,
            "anchor": args.anchor_weight,
        },
        "experts": experts.config(),
        "paired_normal_abnormal_batches": paired,
    }
    report = {
        **run_config,
        "expert_parameters": sum(value.numel() for value in experts.parameters()),
        "baseline_trainable_parameters": sum(
            value.numel() for value in baseline_parameters.values()
        ),
        "clip_backbone_trainable_parameters": 0,
        "selection": "author initial checkpoint and training checkpoints compete by frame AUC/AP",
        "score_dependency": "neurons route experts; current logits are used only for detached event growth",
    }
    (out_dir / "parameter_report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(report, indent=2, ensure_ascii=False), flush=True)

    start_epoch = 0
    resume_batch = 0
    processed_samples = 0
    best_metric = -float("inf")
    next_evaluation = args.dsanet_ucf_eval_samples
    history_path = out_dir / "history.jsonl"

    def model_payload(epoch: int, metrics: dict, tag: str) -> dict:
        return {
            "method": experts.method_name,
            "epoch": epoch,
            "best_metric": best_metric,
            "processed_samples": processed_samples,
            "model_state_dict": adapter.state_dict(),
            "expert_config": experts.config(),
            "run_config": run_config,
            "metrics": metrics,
            "validation_tag": tag,
            "selection_rule": "UCF frame AUC" if args.dataset == "ucf" else "XD frame AP",
        }

    def save_recovery(next_epoch: int, next_batch: int) -> None:
        payload = model_payload(next_epoch, {}, "recovery")
        payload.update({
            "next_epoch": next_epoch,
            "next_batch": next_batch,
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
        })
        torch.save(payload, recovery_path)

    def validate_and_select(epoch: int, tag: str) -> dict[str, float]:
        nonlocal best_metric
        metrics = official_frame_metrics(
            adapter, args.val_list, args.gt_path, args.frames_per_snippet, device
        )
        selection = metrics["frame_auc" if args.dataset == "ucf" else "frame_ap"]
        improved = selection > best_metric
        if improved:
            best_metric = selection
            torch.save(model_payload(epoch, metrics, tag), best_path)
        print(
            f"validation {tag}: {metrics} | best={best_metric:.6f} | improved={improved}",
            flush=True,
        )
        return metrics

    if args.resume:
        if not recovery_path.exists():
            raise FileNotFoundError(f"resume checkpoint not found: {recovery_path}")
        checkpoint = torch.load(recovery_path, map_location="cpu")
        if checkpoint.get("run_config") != run_config:
            raise RuntimeError("resume configuration differs from checkpoint")
        adapter.load_state_dict(checkpoint["model_state_dict"], strict=True)
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        start_epoch = int(checkpoint["next_epoch"])
        resume_batch = int(checkpoint["next_batch"])
        processed_samples = int(checkpoint["processed_samples"])
        best_metric = float(checkpoint["best_metric"])
        next_evaluation = (
            processed_samples // args.dsanet_ucf_eval_samples + 1
        ) * args.dsanet_ucf_eval_samples
    else:
        initial_metrics = validate_and_select(-1, "released_author_initialisation")
        with history_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps({
                "epoch": 0,
                "stage": "released_author_initialisation",
                "metrics": initial_metrics,
            }, ensure_ascii=False) + "\n")
        save_recovery(0, 0)

    for epoch in range(start_epoch, args.max_epoch):
        stage, scope = scope_for_epoch(args, epoch)
        adapter.set_train_scope(scope)
        experts.requires_grad_(True)
        adapter.train()
        batches, step_count = epoch_batches(
            train_set, normal_set, abnormal_set, args.batch_size, args.num_workers,
            device.type == "cuda", args.seed + epoch, paired,
        )
        running = {
            key: 0.0 for key in
            ("total", "baseline", "route", "event", "normal", "smooth", "anchor")
        }
        completed = resume_batch if epoch == start_epoch else 0
        progress = tqdm(
            batches,
            total=step_count,
            desc=f"NREE {epoch + 1}/{args.max_epoch}",
        )
        for step, batch in enumerate(progress, 1):
            if step <= completed:
                continue
            clip = batch["clip"].to(device, non_blocking=True)
            neurons = batch["neurons"].to(device, non_blocking=True)
            lengths = batch["length"].to(device, non_blocking=True)
            labels = batch["binary_label"].to(device, non_blocking=True)
            label_texts = list(batch["label_text"])
            output, records = adapter.forward_modulated(clip, neurons, lengths)
            baseline_loss = adapter.original_loss(output, labels, label_texts, lengths)
            auxiliary = event_expert_losses(
                records, output.binary_logits, labels, label_texts,
                experts.class_names, lengths,
            )
            anchor = relative_parameter_anchor(baseline_parameters, initial_parameters)
            loss = (
                baseline_loss
                + args.route_weight * auxiliary["route"]
                + args.event_weight * auxiliary["event"]
                + args.normal_weight * auxiliary["normal"]
                + args.smooth_weight * auxiliary["smooth"]
                + args.anchor_weight * anchor
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            processed_samples += int(labels.numel())
            values = {"total": loss, "baseline": baseline_loss, **auxiliary, "anchor": anchor}
            for key, value in values.items():
                running[key] += float(value.detach())
            progress.set_postfix(
                stage=stage,
                loss=f"{running['total'] / max(1, step - completed):.4f}",
                best=f"{best_metric:.4f}",
            )
            if step % args.checkpoint_steps == 0:
                save_recovery(epoch, step)
            while paired and processed_samples >= next_evaluation:
                validate_and_select(epoch, f"sample_{processed_samples}")
                adapter.train()
                next_evaluation += args.dsanet_ucf_eval_samples
                save_recovery(epoch, step)

        scheduler.step()
        if not paired:
            metrics = validate_and_select(epoch, f"epoch_{epoch + 1}")
        else:
            metrics = {"selection_deferred_to_fixed_sample_interval": True}
        denominator = max(1, step_count - completed)
        record = {
            "epoch": epoch + 1,
            "stage": stage,
            **{f"{key}_loss": value / denominator for key, value in running.items()},
            "best_metric": best_metric,
            "metrics": metrics,
        }
        with history_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        save_recovery(epoch + 1, 0)
        resume_batch = 0
        print(json.dumps(record, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
