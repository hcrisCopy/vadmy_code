#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import random
import shutil
import sys
from pathlib import Path

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
from neuron_responsibility.data import AlignedFeatureDataset
from neuron_responsibility.model import (
    NeuronResponsibilityProbe,
    ResponsibilityCorrectionHead,
    partition_responsibility_loss,
)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def add_baseline_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--baseline", choices=["dsanet", "desc", "lagovad"], required=True)
    parser.add_argument("--baseline-root", required=True)
    parser.add_argument("--baseline-weight", default="")
    parser.add_argument("--sensitivity-weight", default="")
    parser.add_argument("--consistency-weight", default="")


def load_probe(path: str, device: torch.device) -> tuple[NeuronResponsibilityProbe, float]:
    checkpoint = torch.load(path, map_location="cpu")
    config = checkpoint["config"]
    model = NeuronResponsibilityProbe(
        config["neuron_width"], config["hidden_width"],
        active_neurons=config.get("active_neurons", config["neuron_width"]),
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device).eval().requires_grad_(False)
    threshold = config.get("normal_threshold")
    if threshold is None:
        raise RuntimeError("probe has no calibrated normal_threshold; retrain Stage A")
    return model, float(threshold)


def pad_chunks(array: np.ndarray, chunk_length: int) -> tuple[torch.Tensor, torch.Tensor]:
    chunks, lengths = [], []
    for start in range(0, len(array), chunk_length):
        part = array[start:start + chunk_length]
        lengths.append(len(part))
        chunks.append(np.pad(part, ((0, chunk_length - len(part)), (0, 0))))
    return torch.from_numpy(np.stack(chunks).astype(np.float32)), torch.tensor(lengths)


def official_frame_metrics(adapter, probe, correction, aligned_csv: str, gt_path: str,
                           frames_per_snippet: int, device: torch.device) -> dict[str, float]:
    """Evaluate the official binary branch at frame level, as the baselines do."""
    adapter.eval()
    probe.eval()
    correction.eval()
    frame = pd.read_csv(aligned_csv)
    if "key" not in frame.columns:
        frame["key"] = frame["clip_path"].map(lambda value: Path(str(value)).stem)
    scores = []
    groups = list(frame.groupby("key", sort=False))
    with torch.no_grad():
        for _, group in tqdm(groups, desc="official frame validation", unit="video", leave=False):
            clip = np.concatenate([np.load(str(path)).astype(np.float32) for path in group["clip_path"]])
            neurons = np.concatenate([np.load(str(path)).astype(np.float32) for path in group["neuron_path"]])
            clip_chunks, lengths = pad_chunks(clip, adapter.visual_length)
            neuron_chunks, neuron_lengths = pad_chunks(neurons, adapter.visual_length)
            if not torch.equal(lengths, neuron_lengths):
                raise RuntimeError("validation CLIP and neuron chunk lengths differ")
            lengths = lengths.to(device)
            output = adapter.forward_baseline(clip_chunks.to(device), lengths)
            neuron_probability = torch.sigmoid(probe(neuron_chunks.to(device), lengths))
            logits = correction(output.binary_logits, neuron_probability, lengths)
            for index, length in enumerate(lengths.tolist()):
                scores.append(torch.sigmoid(logits[index, :length]).cpu())
    frame_scores = np.repeat(torch.cat(scores).numpy(), frames_per_snippet)
    gt = np.load(gt_path).astype(np.int64).reshape(-1)
    usable = min(len(gt), len(frame_scores))
    if usable != len(gt) or usable != len(frame_scores):
        print(f"metric length alignment: gt={len(gt)} prediction={len(frame_scores)} usable={usable}")
    return {
        "frame_auc": float(roc_auc_score(gt[:usable], frame_scores[:usable])),
        "frame_ap": float(average_precision_score(gt[:usable], frame_scores[:usable])),
    }


def is_head_parameter(name: str) -> bool:
    return any(token in name for token in ("classifier", "mlp1", "mlp2", "bin_head", "sim_head"))


def main() -> None:
    parser = argparse.ArgumentParser(description="Stage B: partitioned responsibility with gradual unfreezing.")
    add_baseline_arguments(parser)
    parser.add_argument("--dataset", choices=["ucf", "xd"], required=True)
    parser.add_argument("--train-list", required=True)
    parser.add_argument("--val-list", required=True)
    parser.add_argument("--gt-path", required=True)
    parser.add_argument("--probe-model", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--max-epoch", type=int, default=10)
    parser.add_argument("--warmup-epochs", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--correction-lr", type=float, default=5e-5)
    parser.add_argument("--head-lr", type=float, default=1e-5)
    parser.add_argument("--temporal-lr", type=float, default=1e-6)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--responsibility-weight", type=float, default=1.0)
    parser.add_argument("--anchor-weight", type=float, default=1e-4)
    parser.add_argument("--persistence", type=int, default=3)
    parser.add_argument("--frames-per-snippet", type=int, default=16)
    parser.add_argument("--dsanet-ucf-eval-samples", type=int, default=1280)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=234)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--clean", action="store_true")
    args = parser.parse_args()
    if args.warmup_epochs < 0 or args.warmup_epochs >= args.max_epoch:
        parser.error("--warmup-epochs must be in [0, max_epoch)")

    out_dir = Path(args.out_dir)
    if args.clean and out_dir.exists():
        shutil.rmtree(out_dir.resolve())
    out_dir.mkdir(parents=True, exist_ok=True)
    seed_everything(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() and args.device.startswith("cuda") else "cpu")
    adapter = build_baseline(args, str(device)).to(device)
    adapter.set_train_scope("temporal_heads")
    target_parameters = {name: parameter for name, parameter in adapter.named_parameters() if parameter.requires_grad}
    if any("clip" in name.lower() for name in target_parameters):
        raise RuntimeError("CLIP must remain frozen")
    initial_parameters = {name: parameter.detach().clone() for name, parameter in target_parameters.items()}
    probe, neuron_threshold = load_probe(args.probe_model, device)
    correction = ResponsibilityCorrectionHead().to(device)
    head_parameters = [parameter for name, parameter in target_parameters.items() if is_head_parameter(name)]
    temporal_parameters = [parameter for name, parameter in target_parameters.items() if not is_head_parameter(name)]
    if not head_parameters or not temporal_parameters:
        raise RuntimeError("functional head/last-temporal partition is empty")
    optimizer = torch.optim.AdamW([
        {"params": correction.parameters(), "lr": args.correction_lr, "name": "correction"},
        {"params": head_parameters, "lr": args.head_lr, "name": "heads"},
        {"params": temporal_parameters, "lr": args.temporal_lr, "name": "last_temporal"},
    ], weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.max_epoch)
    train_set = AlignedFeatureDataset(args.train_list, args.dataset, adapter.visual_length)
    train_loader = DataLoader(
        train_set, batch_size=args.batch_size, shuffle=True, drop_last=True,
        num_workers=args.num_workers, pin_memory=device.type == "cuda",
    )

    report = {
        "baseline": args.baseline,
        "policy": "correction warmup, then heads plus final temporal refinement",
        "warmup_epochs": args.warmup_epochs,
        "neuron_threshold": neuron_threshold,
        "head_parameters": int(sum(parameter.numel() for parameter in head_parameters)),
        "last_temporal_parameters": int(sum(parameter.numel() for parameter in temporal_parameters)),
        "correction_parameters": int(sum(parameter.numel() for parameter in correction.parameters())),
        "clip_trainable_parameters": 0,
        "target_tensors": list(target_parameters),
    }
    (out_dir / "parameter_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)

    checkpoint_path = out_dir / "checkpoint_last.pth"
    best_path = out_dir / "model_best.pth"
    start_epoch, best, processed_samples = 0, -float("inf"), 0
    if args.resume and checkpoint_path.exists():
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        expected = {"baseline": args.baseline, "dataset": args.dataset, "probe_model": args.probe_model}
        actual = {key: checkpoint.get(key) for key in expected}
        if actual != expected:
            raise RuntimeError(f"resume configuration mismatch: checkpoint={actual}, command={expected}")
        adapter.load_state_dict(checkpoint["model_state_dict"])
        correction.load_state_dict(checkpoint["correction_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        start_epoch = int(checkpoint["epoch"]) + 1
        best = float(checkpoint["best_metric"])
        processed_samples = int(checkpoint.get("processed_samples", 0))

    history_path = out_dir / "history.jsonl"

    def validate_and_save(epoch: int, tag: str) -> dict[str, float]:
        nonlocal best
        metrics = official_frame_metrics(
            adapter, probe, correction, args.val_list, args.gt_path,
            args.frames_per_snippet, device,
        )
        selection_metric = metrics["frame_auc" if args.dataset == "ucf" else "frame_ap"]
        checkpoint = {
            "epoch": epoch, "best_metric": max(best, selection_metric),
            "model_state_dict": adapter.state_dict(),
            "correction_state_dict": correction.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "baseline": args.baseline, "dataset": args.dataset,
            "probe_model": args.probe_model, "neuron_threshold": neuron_threshold,
            "processed_samples": processed_samples, "metrics": metrics,
            "selection_rule": "UCF frame AUC" if args.dataset == "ucf" else "XD frame AP",
            "validation_tag": tag,
        }
        torch.save(checkpoint, checkpoint_path)
        if selection_metric > best:
            best = selection_metric
            torch.save(checkpoint, best_path)
        print(f"validation {tag}: {metrics} | best={best:.6f}", flush=True)
        return metrics

    next_eval = ((processed_samples // args.dsanet_ucf_eval_samples) + 1) * args.dsanet_ucf_eval_samples
    for epoch in range(start_epoch, args.max_epoch):
        baseline_active = epoch >= args.warmup_epochs
        for parameter in target_parameters.values():
            parameter.requires_grad_(baseline_active)
        adapter.train(baseline_active)
        correction.train()
        probe.eval()
        running = {"total": 0.0, "base": 0.0, "responsibility": 0.0, "anchor": 0.0}
        partition_totals = {key: 0 for key in ("agreement_high", "baseline_only", "neuron_only", "agreement_low", "pure_normal")}
        progress = tqdm(train_loader, desc=f"joint epoch {epoch + 1}/{args.max_epoch}")
        for step, batch in enumerate(progress, 1):
            clip = batch["clip"].to(device, non_blocking=True)
            neurons = batch["neurons"].to(device, non_blocking=True)
            lengths = batch["length"].to(device, non_blocking=True)
            labels = batch["binary_label"].to(device, non_blocking=True)
            output = adapter.forward_baseline(clip, lengths)
            with torch.no_grad():
                neuron_probability = torch.sigmoid(probe(neurons, lengths))
            final_logits = correction(output.binary_logits, neuron_probability, lengths)
            base_loss = adapter.original_loss(output, labels, list(batch["label_text"]), lengths) if baseline_active else final_logits.sum() * 0.0
            responsibility_loss, partitions = partition_responsibility_loss(
                final_logits, output.binary_logits, neuron_probability, labels, lengths,
                neuron_threshold, args.persistence,
            )
            anchor_loss = sum(
                (parameter - initial_parameters[name]).square().mean()
                for name, parameter in target_parameters.items()
            ) if baseline_active else final_logits.sum() * 0.0
            loss = base_loss + args.responsibility_weight * responsibility_loss + args.anchor_weight * anchor_loss
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            processed_samples += int(labels.numel())
            for key, value in partitions.items():
                partition_totals[key] += int(value.item())
            for key, value in (("total", loss), ("base", base_loss), ("responsibility", responsibility_loss), ("anchor", anchor_loss)):
                running[key] += float(value.detach())
            progress.set_postfix(stage="joint" if baseline_active else "warmup", loss=f"{running['total'] / step:.4f}")
            if args.baseline == "dsanet" and args.dataset == "ucf" and processed_samples >= next_eval:
                validate_and_save(epoch, f"sample_{processed_samples}")
                adapter.train(baseline_active)
                correction.train()
                next_eval += args.dsanet_ucf_eval_samples
        scheduler.step()
        if not (args.baseline == "dsanet" and args.dataset == "ucf"):
            metrics = validate_and_save(epoch, f"epoch_{epoch + 1}")
        else:
            metrics = {"selection_deferred_to_fixed_step": True}
            if not best_path.exists():
                metrics = validate_and_save(epoch, f"epoch_{epoch + 1}_bootstrap")
        record = {
            "epoch": epoch + 1, "stage": "joint" if baseline_active else "warmup",
            **{f"{key}_loss": value / max(1, len(train_loader)) for key, value in running.items()},
            "partitions": partition_totals, "metrics": metrics,
        }
        with history_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record) + "\n")
        if args.baseline == "dsanet" and args.dataset == "ucf":
            # Author-style best selection remains fixed-step, while this
            # separate last checkpoint guarantees epoch-boundary recovery.
            torch.save({
                "epoch": epoch, "best_metric": best,
                "model_state_dict": adapter.state_dict(),
                "correction_state_dict": correction.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "baseline": args.baseline, "dataset": args.dataset,
                "probe_model": args.probe_model, "neuron_threshold": neuron_threshold,
                "processed_samples": processed_samples, "metrics": metrics,
                "selection_rule": "UCF frame AUC at fixed training-sample intervals",
                "validation_tag": "epoch_recovery_only",
            }, checkpoint_path)
        print(json.dumps(record), flush=True)


if __name__ == "__main__":
    main()
