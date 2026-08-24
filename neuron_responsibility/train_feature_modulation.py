#!/usr/bin/env python3
"""Train score-free sparse-neuron feature modulation with gradual unfreezing."""

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
from neuron_responsibility.common import clean_output
from neuron_responsibility.data import AlignedFeatureDataset
from neuron_responsibility.feature_modulation import (
    SparseNeuronFeatureModulator,
    score_free_modulation_losses,
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


def is_head_parameter(name: str) -> bool:
    return any(token in name for token in ("classifier", "mlp1", "mlp2", "bin_head", "sim_head"))


def pad_chunks(array: np.ndarray, chunk_length: int) -> tuple[torch.Tensor, torch.Tensor]:
    chunks, lengths = [], []
    for start in range(0, max(1, len(array)), chunk_length):
        part = array[start:start + chunk_length]
        if not len(part):
            continue
        lengths.append(len(part))
        chunks.append(np.pad(part, ((0, chunk_length - len(part)), (0, 0))))
    return torch.from_numpy(np.stack(chunks).astype(np.float32)), torch.tensor(lengths, dtype=torch.long)


def load_modulator(
    evidence_config: str,
    feature_width: int,
    context_width: int,
    temporal_kernel: int,
    evidence_cap: float,
) -> SparseNeuronFeatureModulator:
    path = Path(evidence_config)
    metadata_path = path.with_suffix(".json")
    if not metadata_path.exists():
        metadata_path = path.parent / "evidence_config.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    arrays = np.load(path)
    return SparseNeuronFeatureModulator(
        neuron_width=int(metadata["neuron_width"]),
        active_indices=torch.from_numpy(arrays["active_indices"].astype(np.int64)),
        thresholds=torch.from_numpy(arrays["thresholds"].astype(np.float32)),
        feature_width=feature_width,
        context_width=context_width,
        temporal_kernel=temporal_kernel,
        evidence_cap=evidence_cap,
    )


def merge_paired_batches(normal: dict, abnormal: dict) -> dict:
    merged = {}
    for key in ("clip", "neurons", "length", "binary_label"):
        merged[key] = torch.cat([normal[key], abnormal[key]], dim=0)
    for key in ("label_text", "key"):
        merged[key] = list(normal[key]) + list(abnormal[key])
    return merged


def epoch_batches(
    train_set: AlignedFeatureDataset,
    normal_set: AlignedFeatureDataset | None,
    abnormal_set: AlignedFeatureDataset | None,
    batch_size: int,
    num_workers: int,
    pin_memory: bool,
    seed: int,
    paired: bool,
) -> tuple[Iterator[dict], int]:
    generator = torch.Generator().manual_seed(seed)
    common = {
        "batch_size": batch_size,
        "shuffle": True,
        "drop_last": True,
        "num_workers": num_workers,
        "pin_memory": pin_memory,
        "generator": generator,
    }
    if not paired:
        loader = DataLoader(train_set, **common)
        return iter(loader), len(loader)
    if normal_set is None or abnormal_set is None:
        raise RuntimeError("paired training requires normal and abnormal datasets")
    normal_loader = DataLoader(normal_set, **common)
    abnormal_generator = torch.Generator().manual_seed(seed + 100_003)
    abnormal_loader = DataLoader(abnormal_set, **{**common, "generator": abnormal_generator})

    def paired_iterator() -> Iterator[dict]:
        for normal, abnormal in zip(normal_loader, abnormal_loader):
            yield merge_paired_batches(normal, abnormal)

    return paired_iterator(), min(len(normal_loader), len(abnormal_loader))


def relative_parameter_anchor(
    parameters: dict[str, torch.nn.Parameter],
    initial: dict[str, torch.Tensor],
) -> torch.Tensor:
    terms = []
    for name, parameter in parameters.items():
        if parameter.requires_grad:
            denominator = initial[name].square().mean().detach().clamp_min(1e-8)
            terms.append((parameter - initial[name]).square().mean() / denominator)
    if not terms:
        reference = next(iter(parameters.values()))
        return reference.sum() * 0.0
    return torch.stack(terms).mean()


def official_frame_metrics(
    adapter,
    aligned_csv: str,
    gt_path: str,
    frames_per_snippet: int,
    device: torch.device,
) -> dict[str, float]:
    adapter.eval()
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
            output, _ = adapter.forward_modulated(
                clip_chunks.to(device), neuron_chunks.to(device), lengths
            )
            for index, length in enumerate(lengths.tolist()):
                scores.append(torch.sigmoid(output.binary_logits[index, :length]).cpu())
    frame_scores = np.repeat(torch.cat(scores).numpy(), frames_per_snippet)
    gt = np.load(gt_path).astype(np.int64).reshape(-1)
    usable = min(len(gt), len(frame_scores))
    if usable != len(gt) or usable != len(frame_scores):
        print(f"metric length alignment: gt={len(gt)} prediction={len(frame_scores)} usable={usable}")
    return {
        "frame_auc": float(roc_auc_score(gt[:usable], frame_scores[:usable])),
        "frame_ap": float(average_precision_score(gt[:usable], frame_scores[:usable])),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Score-free sparse-neuron feature modulation with gradual unfreezing."
    )
    add_baseline_arguments(parser)
    parser.add_argument("--dataset", choices=["ucf", "xd"], required=True)
    parser.add_argument("--train-list", required=True)
    parser.add_argument("--val-list", required=True)
    parser.add_argument("--gt-path", required=True)
    parser.add_argument("--evidence-config", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--max-epoch", type=int, default=10)
    parser.add_argument("--head-start-epoch", type=int, default=2)
    parser.add_argument("--temporal-start-epoch", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--adapter-lr", type=float, default=5e-5)
    parser.add_argument("--head-lr", type=float, default=1e-5)
    parser.add_argument("--temporal-lr", type=float, default=1e-6)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--auxiliary-weight", type=float, default=0.1)
    parser.add_argument("--normal-weight", type=float, default=0.1)
    parser.add_argument("--smooth-weight", type=float, default=0.01)
    parser.add_argument("--sparse-weight", type=float, default=0.001)
    parser.add_argument("--anchor-weight", type=float, default=0.01)
    parser.add_argument("--context-width", type=int, default=32)
    parser.add_argument("--temporal-kernel", type=int, default=5)
    parser.add_argument("--evidence-cap", type=float, default=6.0)
    parser.add_argument("--frames-per-snippet", type=int, default=16)
    parser.add_argument("--dsanet-ucf-eval-samples", type=int, default=1280)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=234)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--clean", action="store_true")
    args = parser.parse_args()
    if args.clean and args.resume:
        parser.error("--clean and --resume cannot be used together")
    if not 0 <= args.head_start_epoch <= args.temporal_start_epoch < args.max_epoch:
        parser.error("require 0 <= head-start-epoch <= temporal-start-epoch < max-epoch")
    if min(
        args.adapter_lr, args.head_lr, args.temporal_lr, args.auxiliary_weight,
        args.normal_weight, args.smooth_weight, args.sparse_weight, args.anchor_weight,
    ) < 0:
        parser.error("learning rates and loss weights must be non-negative")

    out_dir = clean_output(args.out_dir, args.clean)
    checkpoint_path = out_dir / "checkpoint_last.pth"
    best_path = out_dir / "model_best.pth"
    if checkpoint_path.exists() and not args.resume:
        raise RuntimeError("checkpoint exists; use --resume, --clean, or a new --out-dir")
    seed_everything(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() and args.device.startswith("cuda") else "cpu")

    adapter = build_baseline(args, str(device)).to(device)
    adapter.set_train_scope("temporal_heads")
    baseline_parameters = {
        name: parameter for name, parameter in adapter.named_parameters() if parameter.requires_grad
    }
    if any("clip" in name.lower() for name in baseline_parameters):
        raise RuntimeError("CLIP must remain frozen")
    head_parameters = [
        parameter for name, parameter in baseline_parameters.items() if is_head_parameter(name)
    ]
    temporal_parameters = [
        parameter for name, parameter in baseline_parameters.items() if not is_head_parameter(name)
    ]
    if not head_parameters or not temporal_parameters:
        raise RuntimeError("functional head/last-temporal partition is empty")
    initial_parameters = {
        name: parameter.detach().clone() for name, parameter in baseline_parameters.items()
    }
    adapter.set_train_scope("frozen")
    modulator = load_modulator(
        args.evidence_config, feature_width=512, context_width=args.context_width,
        temporal_kernel=args.temporal_kernel, evidence_cap=args.evidence_cap,
    ).to(device)
    adapter.attach_feature_modulator(modulator)

    optimizer = torch.optim.AdamW([
        {"params": modulator.parameters(), "lr": args.adapter_lr, "name": "neuron_modulator"},
        {"params": head_parameters, "lr": args.head_lr, "name": "heads"},
        {"params": temporal_parameters, "lr": args.temporal_lr, "name": "last_temporal"},
    ], weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.max_epoch)

    train_set = AlignedFeatureDataset(args.train_list, args.dataset, adapter.visual_length)
    paired = args.baseline == "dsanet" and args.dataset == "ucf"
    normal_set = AlignedFeatureDataset(
        args.train_list, args.dataset, adapter.visual_length, split="normal"
    ) if paired else None
    abnormal_set = AlignedFeatureDataset(
        args.train_list, args.dataset, adapter.visual_length, split="abnormal"
    ) if paired else None

    run_config = {
        "method": SparseNeuronFeatureModulator.method_name,
        "baseline": args.baseline,
        "dataset": args.dataset,
        "evidence_config": args.evidence_config,
        "head_start_epoch": args.head_start_epoch,
        "temporal_start_epoch": args.temporal_start_epoch,
        "batch_size": args.batch_size,
        "paired_normal_abnormal_batches": paired,
        "modulator_config": modulator.config(),
    }
    report = {
        **run_config,
        "policy": "modulator, then heads, then final temporal refinement",
        "modulator_parameters": sum(parameter.numel() for parameter in modulator.parameters()),
        "head_parameters": sum(parameter.numel() for parameter in head_parameters),
        "last_temporal_parameters": sum(parameter.numel() for parameter in temporal_parameters),
        "clip_trainable_parameters": 0,
        "target_tensors": list(baseline_parameters),
    }
    (out_dir / "parameter_report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(report, indent=2, ensure_ascii=False), flush=True)

    start_epoch, best, processed_samples = 0, -float("inf"), 0
    if args.resume:
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"resume checkpoint not found: {checkpoint_path}")
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

    def checkpoint_payload(epoch: int, metrics: dict, tag: str) -> dict:
        return {
            "method": SparseNeuronFeatureModulator.method_name,
            "epoch": epoch,
            "best_metric": best,
            "processed_samples": processed_samples,
            "model_state_dict": adapter.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "modulator_config": modulator.config(),
            "run_config": run_config,
            "metrics": metrics,
            "selection_rule": "UCF frame AUC" if args.dataset == "ucf" else "XD frame AP",
            "validation_tag": tag,
        }

    def validate_and_select(epoch: int, tag: str) -> dict[str, float]:
        nonlocal best
        metrics = official_frame_metrics(
            adapter, args.val_list, args.gt_path, args.frames_per_snippet, device
        )
        selection_metric = metrics["frame_auc" if args.dataset == "ucf" else "frame_ap"]
        if selection_metric > best:
            best = selection_metric
            torch.save(checkpoint_payload(epoch, metrics, tag), best_path)
        print(f"validation {tag}: {metrics} | best={best:.6f}", flush=True)
        return metrics

    next_eval = ((processed_samples // args.dsanet_ucf_eval_samples) + 1) * args.dsanet_ucf_eval_samples
    for epoch in range(start_epoch, args.max_epoch):
        if epoch < args.head_start_epoch:
            stage, scope = "modulator", "frozen"
        elif epoch < args.temporal_start_epoch:
            stage, scope = "modulator_heads", "heads"
        else:
            stage, scope = "modulator_temporal_heads", "temporal_heads"
        adapter.set_train_scope(scope)
        modulator.requires_grad_(True)
        adapter.train()
        batches, step_count = epoch_batches(
            train_set, normal_set, abnormal_set, args.batch_size, args.num_workers,
            device.type == "cuda", args.seed + epoch, paired,
        )
        running = {
            key: 0.0 for key in (
                "total", "baseline", "auxiliary", "normal", "smooth", "sparse", "anchor"
            )
        }
        progress = tqdm(batches, total=step_count, desc=f"feature modulation {epoch + 1}/{args.max_epoch}")
        for step, batch in enumerate(progress, 1):
            clip = batch["clip"].to(device, non_blocking=True)
            neurons = batch["neurons"].to(device, non_blocking=True)
            lengths = batch["length"].to(device, non_blocking=True)
            labels = batch["binary_label"].to(device, non_blocking=True)
            output, records = adapter.forward_modulated(clip, neurons, lengths)
            baseline_loss = adapter.original_loss(
                output, labels, list(batch["label_text"]), lengths
            )
            modulation = score_free_modulation_losses(records, labels, lengths)
            anchor = relative_parameter_anchor(baseline_parameters, initial_parameters)
            loss = (
                baseline_loss
                + args.auxiliary_weight * modulation["auxiliary"]
                + args.normal_weight * modulation["normal"]
                + args.smooth_weight * modulation["smooth"]
                + args.sparse_weight * modulation["sparse"]
                + args.anchor_weight * anchor
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            processed_samples += int(labels.numel())
            values = {
                "total": loss,
                "baseline": baseline_loss,
                **modulation,
                "anchor": anchor,
            }
            for key, value in values.items():
                running[key] += float(value.detach())
            progress.set_postfix(
                stage=stage,
                loss=f"{running['total'] / step:.4f}",
                scale=f"{float(modulator.residual_scale.detach()):.4f}",
            )
            if paired and processed_samples >= next_eval:
                validate_and_select(epoch, f"sample_{processed_samples}")
                adapter.train()
                next_eval += args.dsanet_ucf_eval_samples
        scheduler.step()
        if not paired:
            metrics = validate_and_select(epoch, f"epoch_{epoch + 1}")
        else:
            metrics = {"selection_deferred_to_fixed_step": True}
            if not best_path.exists():
                metrics = validate_and_select(epoch, f"epoch_{epoch + 1}_bootstrap")
        record = {
            "epoch": epoch + 1,
            "stage": stage,
            **{f"{key}_loss": value / max(1, step_count) for key, value in running.items()},
            "residual_scale": float(modulator.residual_scale.detach()),
            "metrics": metrics,
        }
        with history_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        torch.save(checkpoint_payload(epoch, metrics, "epoch_recovery"), checkpoint_path)
        print(json.dumps(record, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
