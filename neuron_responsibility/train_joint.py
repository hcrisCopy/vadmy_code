#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import random
import shutil
import sys
from pathlib import Path

import numpy as np
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
    responsibility_mil_loss,
    topk_mil_probability,
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


def load_probe(path: str, device: torch.device) -> NeuronResponsibilityProbe:
    checkpoint = torch.load(path, map_location="cpu")
    config = checkpoint["config"]
    model = NeuronResponsibilityProbe(config["neuron_width"], config["hidden_width"])
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device).eval().requires_grad_(False)
    return model


def evaluate_video_level(adapter, probe, loader, device) -> dict[str, float]:
    adapter.eval()
    probe.eval()
    targets, baseline_scores, neuron_scores = [], [], []
    with torch.no_grad():
        for batch in tqdm(loader, desc="joint validation", leave=False):
            clip = batch["clip"].to(device)
            neurons = batch["neurons"].to(device)
            lengths = batch["length"].to(device)
            output = adapter.forward_baseline(clip, lengths)
            neuron_prob = torch.sigmoid(probe(neurons, lengths))
            baseline_scores.extend(topk_mil_probability(torch.sigmoid(output.binary_logits), lengths).cpu().tolist())
            neuron_scores.extend(topk_mil_probability(neuron_prob, lengths).cpu().tolist())
            targets.extend(batch["binary_label"].tolist())
    return {
        "baseline_video_auc": float(roc_auc_score(targets, baseline_scores)),
        "baseline_video_ap": float(average_precision_score(targets, baseline_scores)),
        "neuron_video_auc": float(roc_auc_score(targets, neuron_scores)),
        "neuron_video_ap": float(average_precision_score(targets, neuron_scores)),
    }


def default_lr(adapter_name: str, dataset: str) -> float:
    if adapter_name == "lagovad":
        return 1e-5
    if adapter_name == "dsanet":
        return 7e-5 if dataset == "ucf" else 1e-5
    return 5e-5 if dataset == "ucf" else 1e-5


def main() -> None:
    parser = argparse.ArgumentParser(description="Stage B: joint responsibility-guided baseline training.")
    add_baseline_arguments(parser)
    parser.add_argument("--dataset", choices=["ucf", "xd"], required=True)
    parser.add_argument("--train-list", required=True)
    parser.add_argument("--val-list", required=True)
    parser.add_argument("--probe-model", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--train-scope", choices=["frozen", "heads", "temporal_heads", "all_non_clip"], default="heads")
    parser.add_argument(
        "--training-mode",
        choices=["baseline_only", "responsibility"],
        default="responsibility",
        help="Fair ablation switch: both modes use the same train scope; only responsibility adds the probe-guided loss.",
    )
    parser.add_argument("--max-epoch", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=0.0, help="0 keeps the official dataset/baseline learning rate.")
    parser.add_argument("--sensitivity-lr", type=float, default=1e-3, help="DeSC temporal-sensitivity stream LR.")
    parser.add_argument("--consistency-lr", type=float, default=0.0, help="0 keeps DeSC's dataset option LR.")
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--responsibility-weight", type=float, default=1.0)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=234)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--clean", action="store_true")
    args = parser.parse_args()
    if args.training_mode == "responsibility" and args.responsibility_weight <= 0:
        parser.error("responsibility mode requires --responsibility-weight > 0")
    effective_responsibility_weight = (
        float(args.responsibility_weight) if args.training_mode == "responsibility" else 0.0
    )

    out_dir = Path(args.out_dir)
    if args.clean and out_dir.exists():
        shutil.rmtree(out_dir.resolve())
    out_dir.mkdir(parents=True, exist_ok=True)
    seed_everything(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() and args.device.startswith("cuda") else "cpu")
    adapter = build_baseline(args, str(device)).to(device)
    adapter.set_train_scope(args.train_scope)
    probe = load_probe(args.probe_model, device)
    train_set = AlignedFeatureDataset(args.train_list, args.dataset, adapter.visual_length)
    val_set = AlignedFeatureDataset(args.val_list, args.dataset, adapter.visual_length)
    train_loader = DataLoader(
        train_set, batch_size=args.batch_size, shuffle=True, drop_last=True,
        num_workers=args.num_workers, pin_memory=device.type == "cuda",
    )
    val_loader = DataLoader(
        val_set, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=device.type == "cuda",
    )
    trainable = [parameter for parameter in adapter.parameters() if parameter.requires_grad]
    if not trainable:
        raise RuntimeError("joint training has no trainable baseline parameters; use train_probe.py for frozen diagnostics")
    trainable_names = [name for name, parameter in adapter.named_parameters() if parameter.requires_grad]
    clip_trainable_names = [name for name in trainable_names if "clip" in name.lower()]
    if clip_trainable_names:
        raise RuntimeError(f"CLIP parameters unexpectedly trainable: {clip_trainable_names[:5]}")
    parameter_report = {
        "baseline": args.baseline,
        "training_mode": args.training_mode,
        "responsibility_weight": effective_responsibility_weight,
        "train_scope": args.train_scope,
        "baseline_total_parameters": int(sum(parameter.numel() for parameter in adapter.parameters())),
        "baseline_trainable_parameters": int(sum(parameter.numel() for parameter in trainable)),
        "probe_total_parameters": int(sum(parameter.numel() for parameter in probe.parameters())),
        "probe_trainable_parameters": 0,
        "clip_trainable_parameters": 0,
        "trainable_tensors": trainable_names,
    }
    with (out_dir / "parameter_report.json").open("w", encoding="utf-8") as handle:
        json.dump(parameter_report, handle, indent=2, ensure_ascii=False)
    print(json.dumps(parameter_report, indent=2, ensure_ascii=False), flush=True)
    learning_rate = args.lr if args.lr > 0 else default_lr(args.baseline, args.dataset)
    if args.baseline == "desc":
        sensitivity_parameters = [
            parameter for name, parameter in adapter.named_parameters()
            if name.startswith("sensitivity.") and parameter.requires_grad
        ]
        consistency_parameters = [
            parameter for name, parameter in adapter.named_parameters()
            if name.startswith("consistency.") and parameter.requires_grad
        ]
        consistency_lr = args.consistency_lr if args.consistency_lr > 0 else learning_rate
        optimizer = torch.optim.AdamW(
            [
                {"params": sensitivity_parameters, "lr": args.sensitivity_lr},
                {"params": consistency_parameters, "lr": consistency_lr},
            ],
            weight_decay=args.weight_decay,
        )
    else:
        optimizer = torch.optim.AdamW(trainable, lr=learning_rate, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.max_epoch)
    checkpoint_path = out_dir / "checkpoint_last.pth"
    best_path = out_dir / "model_best.pth"
    start_epoch, best = 0, -float("inf")
    if args.resume and checkpoint_path.exists():
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        expected = {
            "baseline": args.baseline,
            "dataset": args.dataset,
            "train_scope": args.train_scope,
            "training_mode": args.training_mode,
            "responsibility_weight": effective_responsibility_weight,
        }
        actual = {key: checkpoint.get(key) for key in expected}
        if actual != expected:
            raise RuntimeError(
                f"resume configuration mismatch: checkpoint={actual}, command={expected}; "
                "use the matching command or a different --out-dir"
            )
        adapter.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        start_epoch = int(checkpoint["epoch"]) + 1
        best = float(checkpoint["best_metric"])

    history_path = out_dir / "history.jsonl"
    for epoch in range(start_epoch, args.max_epoch):
        adapter.train()
        probe.eval()
        running_total = running_base = running_resp = 0.0
        progress = tqdm(train_loader, desc=f"joint epoch {epoch + 1}/{args.max_epoch}")
        for step, batch in enumerate(progress, 1):
            clip = batch["clip"].to(device, non_blocking=True)
            neurons = batch["neurons"].to(device, non_blocking=True)
            lengths = batch["length"].to(device, non_blocking=True)
            labels = batch["binary_label"].to(device, non_blocking=True)
            texts = list(batch["label_text"])
            output = adapter.forward_baseline(clip, lengths)
            base_loss = adapter.original_loss(output, labels, texts, lengths)
            if args.training_mode == "responsibility":
                with torch.no_grad():
                    neuron_probability = torch.sigmoid(probe(neurons, lengths))
                resp_loss = responsibility_mil_loss(
                    output.binary_logits, neuron_probability, labels, lengths
                )
            else:
                resp_loss = output.binary_logits.sum() * 0.0
            loss = base_loss + effective_responsibility_weight * resp_loss
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            running_total += float(loss.detach())
            running_base += float(base_loss.detach())
            running_resp += float(resp_loss.detach())
            progress.set_postfix(
                loss=f"{running_total / step:.4f}",
                base=f"{running_base / step:.4f}",
                resp=f"{running_resp / step:.4f}",
            )
        scheduler.step()
        metrics = evaluate_video_level(adapter, probe, val_loader, device)
        selection_metric = metrics["baseline_video_auc" if args.dataset == "ucf" else "baseline_video_ap"]
        record = {
            "epoch": epoch + 1,
            "loss": running_total / max(1, len(train_loader)),
            "base_loss": running_base / max(1, len(train_loader)),
            "responsibility_loss": running_resp / max(1, len(train_loader)),
            **metrics,
        }
        with history_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        checkpoint = {
            "epoch": epoch,
            "best_metric": max(best, selection_metric),
            "model_state_dict": adapter.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "baseline": args.baseline,
            "dataset": args.dataset,
            "train_scope": args.train_scope,
            "training_mode": args.training_mode,
            "responsibility_weight": effective_responsibility_weight,
            "probe_model": args.probe_model,
            "metrics": metrics,
        }
        torch.save(checkpoint, checkpoint_path)
        if selection_metric > best:
            best = selection_metric
            torch.save(checkpoint, best_path)
        print(f"epoch {epoch + 1}: {metrics} | best={best:.6f}", flush=True)


if __name__ == "__main__":
    main()
