#!/usr/bin/env python3
"""Train pre-temporal neuron conditioning with conflict-controlled unfreezing."""

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
import torch.nn.functional as F
from sklearn.metrics import average_precision_score, roc_auc_score
from torch.utils.data import DataLoader
from tqdm import tqdm


PACKAGE_ROOT = Path(__file__).resolve().parent.parent
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))

from neuron_responsibility.baselines import build_baseline
from neuron_responsibility.boundary_localization import (
    IndependentNeuronLocalizer,
    NeuronBoundaryConditioner,
    boundary_supervision_loss,
    synthesize_boundary_batch,
)
from neuron_responsibility.common import clean_output
from neuron_responsibility.data import AlignedFeatureDataset
from neuron_responsibility.evaluate import pad_chunks
from neuron_responsibility.model import valid_mask


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


def merge_paired_batches(normal: dict, abnormal: dict) -> dict:
    merged = {}
    for key in ("clip", "neurons", "length", "binary_label"):
        merged[key] = torch.cat([normal[key], abnormal[key]], dim=0)
    for key in ("label_text", "key"):
        merged[key] = list(normal[key]) + list(abnormal[key])
    return merged


def epoch_batches(
    normal_set: AlignedFeatureDataset,
    abnormal_set: AlignedFeatureDataset,
    batch_size: int,
    num_workers: int,
    pin_memory: bool,
    seed: int,
) -> tuple[Iterator[tuple[dict, dict, dict]], int]:
    common = {
        "batch_size": batch_size,
        "shuffle": True,
        "drop_last": True,
        "num_workers": num_workers,
        "pin_memory": pin_memory,
    }
    normal_loader = DataLoader(
        normal_set,
        generator=torch.Generator().manual_seed(seed),
        **common,
    )
    abnormal_loader = DataLoader(
        abnormal_set,
        generator=torch.Generator().manual_seed(seed + 100_003),
        **common,
    )

    def iterator() -> Iterator[tuple[dict, dict, dict]]:
        for normal, abnormal in zip(normal_loader, abnormal_loader):
            yield normal, abnormal, merge_paired_batches(normal, abnormal)

    return iterator(), min(len(normal_loader), len(abnormal_loader))


def load_localizer(path: str, device: torch.device) -> IndependentNeuronLocalizer:
    checkpoint = torch.load(path, map_location="cpu")
    if checkpoint.get("method") != IndependentNeuronLocalizer.method_name:
        raise ValueError(f"unsupported localizer checkpoint: {checkpoint.get('method')}")
    model = IndependentNeuronLocalizer.from_config(checkpoint["model_config"])
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.requires_grad_(False)
    return model.to(device).eval()


def relative_parameter_anchor(
    parameters: dict[str, torch.nn.Parameter],
    initial: dict[str, torch.Tensor],
) -> torch.Tensor:
    terms = []
    for name, parameter in parameters.items():
        if parameter.requires_grad:
            denominator = initial[name].square().mean().detach().clamp_min(1e-8)
            terms.append((parameter - initial[name]).square().mean() / denominator)
    if terms:
        return torch.stack(terms).mean()
    reference = next(iter(parameters.values()))
    return reference.sum() * 0.0


def preservation_loss(
    current_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    lengths: torch.Tensor,
) -> torch.Tensor:
    mask = valid_mask(lengths, current_logits.shape[1], current_logits.dtype)
    target = torch.sigmoid(teacher_logits.detach())
    value = F.binary_cross_entropy_with_logits(current_logits, target, reduction="none")
    return (value * mask).sum() / mask.sum().clamp_min(1.0)


def conditioner_regularization(
    records: list[dict[str, torch.Tensor | str]],
    labels: torch.Tensor,
) -> torch.Tensor:
    terms = []
    normal = labels < 0.5
    for record in records:
        applied = record["applied_delta"]
        mask = record["mask"]
        if not isinstance(applied, torch.Tensor) or not isinstance(mask, torch.Tensor):
            raise TypeError("invalid pre-temporal conditioning record")
        if bool(normal.any()):
            selected_delta = applied[normal]
            selected_mask = mask[normal].unsqueeze(-1)
        else:
            selected_delta = applied
            selected_mask = mask.unsqueeze(-1)
        denominator = selected_mask.sum().clamp_min(1.0) * applied.shape[-1]
        terms.append((selected_delta.square() * selected_mask).sum() / denominator)
    return torch.stack(terms).mean()


def conflict_controlled_step(
    base_objective: torch.Tensor,
    boundary_objective: torch.Tensor,
    parameters: list[torch.nn.Parameter],
    boundary_weight: float,
    optimizer: torch.optim.Optimizer,
) -> dict[str, float]:
    trainable = [parameter for parameter in parameters if parameter.requires_grad]
    if not trainable:
        raise RuntimeError("conflict-controlled update has no trainable parameters")
    base_grad = torch.autograd.grad(
        base_objective, trainable, retain_graph=True, allow_unused=True
    )
    boundary_grad = torch.autograd.grad(
        boundary_objective, trainable, allow_unused=True
    )
    dot = torch.zeros((), device=base_objective.device)
    base_norm = torch.zeros_like(dot)
    for first, second in zip(base_grad, boundary_grad):
        if first is not None and second is not None:
            dot = dot + (first * second).sum()
            base_norm = base_norm + first.square().sum()
    coefficient = torch.where(
        dot < 0,
        dot / base_norm.clamp_min(1e-12),
        torch.zeros_like(dot),
    )
    optimizer.zero_grad(set_to_none=True)
    for parameter, first, second in zip(trainable, base_grad, boundary_grad):
        if first is None and second is None:
            continue
        if first is None:
            combined = float(boundary_weight) * second
        elif second is None:
            combined = first
        else:
            projected = second - coefficient * first
            combined = first + float(boundary_weight) * projected
        parameter.grad = combined.detach()
    optimizer.step()
    return {
        "gradient_dot": float(dot.detach()),
        "gradient_conflict": float((dot < 0).detach()),
        "projection_coefficient": float(coefficient.detach()),
    }


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
    with torch.no_grad():
        for _, group in tqdm(
            list(frame.groupby("key", sort=False)),
            desc="official frame validation",
            unit="video",
            leave=False,
        ):
            clip = np.concatenate([
                np.load(str(path)).astype(np.float32) for path in group["clip_path"]
            ])
            neurons = np.concatenate([
                np.load(str(path)).astype(np.float32) for path in group["neuron_path"]
            ])
            clip_chunks, lengths = pad_chunks(clip, adapter.visual_length)
            neuron_chunks, neuron_lengths = pad_chunks(neurons, adapter.visual_length)
            if not torch.equal(lengths, neuron_lengths):
                raise RuntimeError("validation CLIP and neuron chunk lengths differ")
            output, _ = adapter.forward_conditioned(
                clip_chunks.to(device), neuron_chunks.to(device), lengths.to(device)
            )
            for index, length in enumerate(lengths.tolist()):
                scores.append(torch.sigmoid(output.binary_logits[index, :length]).cpu())
    prediction = np.repeat(torch.cat(scores).numpy(), frames_per_snippet)
    truth = np.load(gt_path).astype(np.int64).reshape(-1)
    usable = min(len(prediction), len(truth))
    return {
        "frame_auc": float(roc_auc_score(truth[:usable], prediction[:usable])),
        "frame_ap": float(average_precision_score(truth[:usable], prediction[:usable])),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Pre-temporal neuron conditioning with boundary supervision and PCGrad."
    )
    add_baseline_arguments(parser)
    parser.add_argument("--dataset", choices=["ucf", "xd"], required=True)
    parser.add_argument("--train-list", required=True)
    parser.add_argument("--val-list", required=True)
    parser.add_argument("--gt-path", required=True)
    parser.add_argument("--localizer-model", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--max-epoch", type=int, default=8)
    parser.add_argument("--temporal-start-epoch", type=int, default=2)
    parser.add_argument("--head-start-epoch", type=int, default=6)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--adapter-lr", type=float, default=5e-5)
    parser.add_argument("--temporal-lr", type=float, default=1e-6)
    parser.add_argument("--head-lr", type=float, default=2e-7)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--boundary-objective-weight", type=float, default=1.0)
    parser.add_argument("--synthetic-bce-weight", type=float, default=1.0)
    parser.add_argument("--synthetic-dice-weight", type=float, default=1.0)
    parser.add_argument("--boundary-shape-weight", type=float, default=0.1)
    parser.add_argument("--preservation-weight", type=float, default=0.5)
    parser.add_argument("--normal-delta-weight", type=float, default=0.1)
    parser.add_argument("--anchor-weight", type=float, default=0.01)
    parser.add_argument("--adapter-width", type=int, default=32)
    parser.add_argument("--max-adapter-scale", type=float, default=0.25)
    parser.add_argument("--min-segment", type=int, default=4)
    parser.add_argument("--max-segment", type=int, default=32)
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
    if not 0 <= args.temporal_start_epoch <= args.head_start_epoch < args.max_epoch:
        parser.error("require 0 <= temporal-start-epoch <= head-start-epoch < max-epoch")
    if args.min_segment <= 0 or args.max_segment < args.min_segment:
        parser.error("require 0 < min-segment <= max-segment")
    if min(
        args.adapter_lr, args.temporal_lr, args.head_lr, args.weight_decay,
        args.boundary_objective_weight, args.synthetic_bce_weight,
        args.synthetic_dice_weight, args.boundary_shape_weight,
        args.preservation_weight, args.normal_delta_weight, args.anchor_weight,
    ) < 0:
        parser.error("learning rates and loss weights must be non-negative")

    out_dir = clean_output(args.out_dir, args.clean)
    checkpoint_path = out_dir / "checkpoint_last.pth"
    best_path = out_dir / "model_best.pth"
    if checkpoint_path.exists() and not args.resume:
        raise RuntimeError("checkpoint exists; use --resume, --clean, or another --out-dir")
    seed_everything(args.seed)
    device = torch.device(
        args.device if torch.cuda.is_available() and args.device.startswith("cuda") else "cpu"
    )

    adapter = build_baseline(args, str(device)).to(device)
    teacher = build_baseline(args, str(device)).to(device).eval()
    teacher.requires_grad_(False)
    adapter.set_train_scope("temporal_heads")
    baseline_parameters = {
        name: parameter for name, parameter in adapter.named_parameters() if parameter.requires_grad
    }
    head_parameters = [
        parameter for name, parameter in baseline_parameters.items() if is_head_parameter(name)
    ]
    temporal_parameters = [
        parameter for name, parameter in baseline_parameters.items() if not is_head_parameter(name)
    ]
    if not head_parameters or not temporal_parameters:
        raise RuntimeError("functional head/last-temporal parameter partition is empty")
    initial_parameters = {
        name: parameter.detach().clone() for name, parameter in baseline_parameters.items()
    }
    adapter.set_train_scope("frozen")

    localizer = load_localizer(args.localizer_model, device)
    conditioner = NeuronBoundaryConditioner(
        localizer,
        feature_width=512,
        adapter_width=args.adapter_width,
        max_scale=args.max_adapter_scale,
    ).to(device)
    conditioner.localizer.requires_grad_(False)
    adapter.attach_pre_temporal_conditioner(conditioner)
    conditioner_parameters = [
        parameter
        for name, parameter in conditioner.named_parameters()
        if not name.startswith("localizer.")
    ]
    optimizer = torch.optim.AdamW([
        {"params": conditioner_parameters, "lr": args.adapter_lr, "name": "pre_temporal_adapter"},
        {"params": temporal_parameters, "lr": args.temporal_lr, "name": "last_temporal"},
        {"params": head_parameters, "lr": args.head_lr, "name": "heads"},
    ], weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.max_epoch)

    normal_set = AlignedFeatureDataset(
        args.train_list, args.dataset, adapter.visual_length, split="normal"
    )
    abnormal_set = AlignedFeatureDataset(
        args.train_list, args.dataset, adapter.visual_length, split="abnormal"
    )
    run_config = {
        "method": NeuronBoundaryConditioner.method_name,
        "baseline": args.baseline,
        "dataset": args.dataset,
        "localizer_model": args.localizer_model,
        "temporal_start_epoch": args.temporal_start_epoch,
        "head_start_epoch": args.head_start_epoch,
        "batch_size_per_class": args.batch_size,
        "conditioner": conditioner.config(),
        "policy": "adapter, then final temporal block, then binary/semantic heads",
        "gradient_control": "PCGrad boundary gradient against original objective",
    }
    report = {
        **run_config,
        "localizer_parameters_frozen": sum(p.numel() for p in conditioner.localizer.parameters()),
        "adapter_parameters": sum(p.numel() for p in conditioner_parameters),
        "last_temporal_parameters": sum(p.numel() for p in temporal_parameters),
        "head_parameters": sum(p.numel() for p in head_parameters),
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
            "method": NeuronBoundaryConditioner.method_name,
            "epoch": epoch,
            "best_metric": best,
            "processed_samples": processed_samples,
            "model_state_dict": adapter.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "conditioner_config": conditioner.config(),
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
        selection = metrics["frame_auc" if args.dataset == "ucf" else "frame_ap"]
        if selection > best:
            best = selection
            torch.save(checkpoint_payload(epoch, metrics, tag), best_path)
        print(f"validation {tag}: {metrics} | best={best:.6f}", flush=True)
        return metrics

    next_eval = (
        (processed_samples // args.dsanet_ucf_eval_samples) + 1
    ) * args.dsanet_ucf_eval_samples
    for epoch in range(start_epoch, args.max_epoch):
        if epoch < args.temporal_start_epoch:
            stage, scope = "adapter", "frozen"
        elif epoch < args.head_start_epoch:
            stage, scope = "adapter_temporal", "temporal_heads"
        else:
            stage, scope = "adapter_temporal_heads", "temporal_heads"
        adapter.set_train_scope(scope)
        if stage == "adapter_temporal":
            for parameter in head_parameters:
                parameter.requires_grad_(False)
        conditioner.requires_grad_(True)
        conditioner.localizer.requires_grad_(False)
        conditioner.localizer.eval()
        adapter.train()
        conditioner.localizer.eval()
        batches, steps = epoch_batches(
            normal_set,
            abnormal_set,
            args.batch_size,
            args.num_workers,
            device.type == "cuda",
            args.seed + epoch,
        )
        running = {
            name: 0.0
            for name in (
                "total", "baseline", "preserve", "normal_delta", "anchor",
                "synthetic_bce", "synthetic_dice", "boundary_shape",
                "gradient_dot", "gradient_conflict",
            )
        }
        progress = tqdm(batches, total=steps, desc=f"boundary conditioning {epoch + 1}/{args.max_epoch}")
        for step, (normal, abnormal, real) in enumerate(progress, 1):
            clip = real["clip"].to(device, non_blocking=True)
            neurons = real["neurons"].to(device, non_blocking=True)
            lengths = real["length"].to(device, non_blocking=True)
            labels = real["binary_label"].to(device, non_blocking=True)
            with torch.no_grad():
                teacher_output = teacher.forward_baseline(clip, lengths)
            current_output, records = adapter.forward_conditioned(clip, neurons, lengths)
            baseline_loss = adapter.original_loss(
                current_output, labels, list(real["label_text"]), lengths
            )
            preserve = preservation_loss(
                current_output.binary_logits, teacher_output.binary_logits, lengths
            )
            normal_delta = conditioner_regularization(records, labels)
            anchor = relative_parameter_anchor(baseline_parameters, initial_parameters)
            base_objective = (
                baseline_loss
                + args.preservation_weight * preserve
                + args.normal_delta_weight * normal_delta
                + args.anchor_weight * anchor
            )

            normal_clip = normal["clip"].to(device, non_blocking=True)
            normal_neurons = normal["neurons"].to(device, non_blocking=True)
            normal_lengths = normal["length"].to(device, non_blocking=True)
            abnormal_clip = abnormal["clip"].to(device, non_blocking=True)
            abnormal_neurons = abnormal["neurons"].to(device, non_blocking=True)
            abnormal_lengths = abnormal["length"].to(device, non_blocking=True)
            synthetic = synthesize_boundary_batch(
                conditioner.localizer,
                normal_clip,
                normal_neurons,
                normal_lengths,
                abnormal_clip,
                abnormal_neurons,
                abnormal_lengths,
                args.min_segment,
                args.max_segment,
            )
            synthetic_output, _ = adapter.forward_conditioned(
                synthetic["clip"], synthetic["neurons"], synthetic["lengths"]
            )
            boundary = boundary_supervision_loss(
                synthetic_output.binary_logits,
                synthetic["targets"],
                synthetic["lengths"],
                synthetic["confidence"],
            )
            boundary_objective = (
                args.synthetic_bce_weight * boundary["bce"]
                + args.synthetic_dice_weight * boundary["dice"]
                + args.boundary_shape_weight * boundary["boundary"]
            )
            trainable_parameters = conditioner_parameters + temporal_parameters + head_parameters
            gradient = conflict_controlled_step(
                base_objective,
                boundary_objective,
                trainable_parameters,
                args.boundary_objective_weight,
                optimizer,
            )
            processed_samples += int(labels.numel())
            total = base_objective + args.boundary_objective_weight * boundary_objective
            values = {
                "total": total,
                "baseline": baseline_loss,
                "preserve": preserve,
                "normal_delta": normal_delta,
                "anchor": anchor,
                "synthetic_bce": boundary["bce"],
                "synthetic_dice": boundary["dice"],
                "boundary_shape": boundary["boundary"],
                "gradient_dot": gradient["gradient_dot"],
                "gradient_conflict": gradient["gradient_conflict"],
            }
            for name, value in values.items():
                running[name] += float(value.detach()) if isinstance(value, torch.Tensor) else float(value)
            progress.set_postfix(
                stage=stage,
                loss=f"{running['total'] / step:.4f}",
                conflict=f"{running['gradient_conflict'] / step:.2f}",
                scale=f"{float(torch.tanh(conditioner.residual_logit).detach() * conditioner.max_scale):.4f}",
            )
            if args.baseline == "dsanet" and args.dataset == "ucf" and processed_samples >= next_eval:
                validate_and_select(epoch, f"sample_{processed_samples}")
                adapter.train()
                conditioner.localizer.eval()
                next_eval += args.dsanet_ucf_eval_samples
        scheduler.step()
        if not (args.baseline == "dsanet" and args.dataset == "ucf"):
            metrics = validate_and_select(epoch, f"epoch_{epoch + 1}")
        else:
            metrics = {"selection_deferred_to_fixed_step": True}
            if not best_path.exists():
                metrics = validate_and_select(epoch, f"epoch_{epoch + 1}_bootstrap")
        record = {
            "epoch": epoch + 1,
            "stage": stage,
            **{f"{key}_loss": value / max(1, steps) for key, value in running.items()},
            "metrics": metrics,
        }
        with history_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        torch.save(checkpoint_payload(epoch, metrics, "epoch_recovery"), checkpoint_path)
        print(json.dumps(record, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()

