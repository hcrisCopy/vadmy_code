#!/usr/bin/env python3
"""Train U-shaped neuron injection together with only the binary score head."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import torch
from tqdm import tqdm


PACKAGE_ROOT = Path(__file__).resolve().parent.parent
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))

from neuron_responsibility.baselines import build_baseline
from neuron_responsibility.common import clean_output, load_json
from neuron_responsibility.data import AlignedFeatureDataset
from neuron_responsibility.evaluate import file_signature
from shift_residual_head_tuning.method import configure_score_head_only, score_head_train_mode
from shift_residual_head_tuning.train import add_baseline_arguments, epoch_batches, seed_everything
from shift_u_dual_head_tuning.heads import (
    clone_score_head_parameters, load_score_head_state, relative_score_head_change,
    score_head_state,
)
from shift_u_dual_injection.method import UDualInjector, attach_dual_injector, forward_dual
from shift_u_dual_injection.train import branch_rms, official_frame_metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Shift-Global768 U-dual injection plus score-head tuning.")
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
    parser.add_argument("--hidden-width", type=int, default=1024)
    parser.add_argument("--trunk-depth", type=int, default=2)
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
    if min(args.max_epoch, args.batch_size, args.lr, args.hidden_width, args.trunk_depth) <= 0:
        parser.error("epochs, batch size, lr, hidden width, and trunk depth must be positive")

    out_dir = clean_output(args.out_dir, args.clean)
    checkpoint_path, best_path = out_dir / "checkpoint_last.pth", out_dir / "model_best.pth"
    if checkpoint_path.exists() and not args.resume:
        raise RuntimeError("checkpoint exists; use --resume, --clean, or a new --out-dir")
    seed_everything(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() and args.device.startswith("cuda") else "cpu")
    adapter = build_baseline(args, str(device)).to(device)
    head_parameters, head_names = configure_score_head_only(adapter, args.baseline)
    initial_head_parameters = clone_score_head_parameters(adapter, args.baseline)
    selection = load_json(args.neuron_json)
    if selection.get("method") != "intravideo_paired_shift_global768":
        raise ValueError("U-dual head experiment requires the unchanged Shift-Global768 selection artifact")
    if int(selection.get("neuron_width", 0)) != 768:
        raise ValueError(f"U-dual head experiment requires 768 neurons, got {selection.get('neuron_width')}")
    injector = UDualInjector(
        neuron_width=768, hidden_width=args.hidden_width, trunk_depth=args.trunk_depth,
    ).to(device)
    attach_dual_injector(adapter, injector)
    trainable_baseline_names = [
        name for name, parameter in adapter.named_parameters()
        if parameter.requires_grad
        and not name.startswith("pre_temporal_conditioner.")
        and not name.startswith("feature_modulator.")
    ]
    if set(trainable_baseline_names) != set(head_names):
        raise RuntimeError(
            f"score-head-only audit failed: actual={trainable_baseline_names}, expected={head_names}"
        )
    optimizer = torch.optim.AdamW(
        [
            {"params": injector.parameters(), "lr": args.lr, "name": "u_dual_injector"},
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
        scheduler_rule, scheduler_per_step = "CosineAnnealingLR per epoch", False
    elif args.baseline == "desc":
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[4, 8], gamma=0.1)
        scheduler_rule, scheduler_per_step = "MultiStepLR milestones=[4,8], gamma=0.1 per epoch", False
    else:
        from transformers import get_scheduler
        scheduler = get_scheduler(
            "cosine", optimizer=optimizer, num_warmup_steps=20,
            num_training_steps=args.max_epoch * steps_per_epoch,
        )
        scheduler_rule, scheduler_per_step = "Transformers cosine per step, warmup_steps=20", True

    run_config = {
        "method": "shift_global768_u_dual_score_head_v1",
        "baseline": args.baseline,
        "dataset": args.dataset,
        "baseline_weight": file_signature(args.baseline_weight),
        "sensitivity_weight": file_signature(args.sensitivity_weight),
        "consistency_weight": file_signature(args.consistency_weight),
        "neuron_json": args.neuron_json,
        "neuron_json_sha256": hashlib.sha256(Path(args.neuron_json).read_bytes()).hexdigest(),
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
        "injector_parameters": sum(parameter.numel() for parameter in injector.parameters()),
        "score_head_parameters": sum(parameter.numel() for parameter in head_parameters),
        "clip_trainable_parameters": 0,
        "text_trainable_parameters": 0,
        "temporal_trainable_parameters": 0,
    }
    (out_dir / "parameter_report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False), flush=True)

    start_epoch, best, processed_samples = 0, -float("inf"), 0
    if args.resume:
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        if checkpoint.get("run_config") != run_config:
            raise RuntimeError("resume configuration differs from checkpoint")
        injector.load_state_dict(checkpoint["injector_state_dict"], strict=True)
        load_score_head_state(adapter, args.baseline, checkpoint["score_head_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        start_epoch = int(checkpoint["epoch"]) + 1
        best = float(checkpoint["best_metric"])
        processed_samples = int(checkpoint["processed_samples"])

    def payload(epoch: int, metrics: dict, tag: str) -> dict:
        return {
            "method": run_config["method"],
            "epoch": epoch,
            "best_metric": best,
            "processed_samples": processed_samples,
            "injector_state_dict": injector.state_dict(),
            "score_head_state_dict": score_head_state(adapter, args.baseline),
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
        metrics = official_frame_metrics(adapter, args.val_list, args.gt_path, args.frames_per_snippet, device)
        selected = metrics["frame_auc" if args.dataset == "ucf" else "frame_ap"]
        if selected > best:
            best = selected
            torch.save(payload(epoch, metrics, tag), best_path)
        print(f"validation {tag}: {metrics} | best={best:.6f}", flush=True)
        return metrics

    history_path = out_dir / "history.jsonl"
    next_eval = ((processed_samples // args.dsanet_ucf_eval_samples) + 1) * args.dsanet_ucf_eval_samples
    for epoch in range(start_epoch, args.max_epoch):
        score_head_train_mode(adapter, injector, args.baseline)
        batches, steps = epoch_batches(
            train_set, normal_set, abnormal_set, args.batch_size, args.num_workers,
            args.seed + epoch, paired,
        )
        totals = {"loss": 0.0, "early_rms": 0.0, "late_rms": 0.0}
        progress = tqdm(batches, total=steps, desc=f"U-dual+score-head {epoch + 1}/{args.max_epoch}")
        metrics: dict = {}
        for step, batch in enumerate(progress, 1):
            clip = batch["clip"].to(device, non_blocking=True)
            neurons = batch["neurons"].to(device, non_blocking=True)
            lengths = batch["length"].to(device, non_blocking=True)
            labels = batch["binary_label"].to(device, non_blocking=True)
            output, records = forward_dual(adapter, clip, neurons, lengths)
            loss = adapter.original_loss(output, labels, list(batch["label_text"]), lengths)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            if scheduler_per_step:
                scheduler.step()
            early_rms, late_rms = branch_rms(records["early"]), branch_rms(records["late"])
            processed_samples += int(labels.numel())
            totals["loss"] += float(loss.detach())
            totals["early_rms"] += float(early_rms.detach())
            totals["late_rms"] += float(late_rms.detach())
            progress.set_postfix(
                loss=f"{totals['loss'] / step:.4f}",
                early=f"{totals['early_rms'] / step:.4f}",
                late=f"{totals['late_rms'] / step:.4f}",
            )
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
        early_average = totals["early_rms"] / max(1, steps)
        late_average = totals["late_rms"] / max(1, steps)
        head_change = float(relative_score_head_change(adapter, args.baseline, initial_head_parameters).detach())
        record = {
            "epoch": epoch + 1,
            "loss": totals["loss"] / max(1, steps),
            "early_gate": float(injector.early_gate().detach()),
            "late_gate": float(injector.late_gate().detach()),
            "early_applied_delta_rms": early_average,
            "late_applied_delta_rms": late_average,
            "late_to_early_rms_ratio": late_average / max(early_average, 1e-12),
            "score_head_relative_change": head_change,
            "learning_rate": float(optimizer.param_groups[0]["lr"]),
            "metrics": metrics,
        }
        with history_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        torch.save(payload(epoch, metrics, "epoch_recovery"), checkpoint_path)
        print(json.dumps(record, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
