#!/usr/bin/env python3
"""Train CNCR with counterfactual responsibility and gradual temporal unfreezing."""

from __future__ import annotations

import argparse
import gc
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm


PACKAGE_ROOT = Path(__file__).resolve().parent.parent
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))

from neuron_responsibility.baselines import build_baseline, class_targets
from neuron_responsibility.circuit_routing import load_circuit_router
from neuron_responsibility.common import clean_output
from neuron_responsibility.data import AlignedFeatureDataset
from neuron_responsibility.evaluate_circuit_routing import evaluate_models
from neuron_responsibility.model import valid_mask


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def merge(normal: dict, abnormal: dict) -> dict:
    tensor_keys = ("clip", "neurons", "length", "binary_label")
    result = {key: torch.cat([normal[key], abnormal[key]], dim=0) for key in tensor_keys}
    for key in ("label_text", "sample_id"):
        result[key] = list(normal[key]) + list(abnormal[key])
    return result


def cache_teacher_binary(
    adapter,
    dataset,
    path: Path,
    batch_size: int,
    workers: int,
    device: torch.device,
) -> tuple[torch.Tensor, dict[str, int]]:
    if path.exists():
        value = torch.load(path, map_location="cpu")
        return value["logits"].float(), {
            str(key): index for index, key in enumerate(value["sample_ids"])
        }
    loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=False, num_workers=workers,
        pin_memory=device.type == "cuda",
    )
    values, sample_ids = [], []
    adapter.eval()
    with torch.no_grad():
        for batch in tqdm(loader, desc="cache released DSANet logits", unit="batch"):
            output = adapter.forward_baseline(
                batch["clip"].to(device), batch["length"].to(device)
            )
            values.append(output.binary_logits.cpu().half())
            sample_ids.extend(map(str, batch["sample_id"]))
    logits = torch.cat(values)
    torch.save({"logits": logits, "sample_ids": sample_ids}, path)
    return logits.float(), {key: index for index, key in enumerate(sample_ids)}


def preservation_loss(logits: torch.Tensor, teacher: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
    mask = valid_mask(lengths, logits.shape[1], logits.dtype)
    value = F.binary_cross_entropy_with_logits(
        logits, torch.sigmoid(teacher), reduction="none"
    )
    return (value * mask).sum() / mask.sum().clamp_min(1.0)


def counterfactual_loss(
    plus,
    minus,
    views,
    target_full: torch.Tensor,
    labels: torch.Tensor,
    lengths: torch.Tensor,
    top_fraction: float,
    semantic_margin: float,
    binary_margin: float,
) -> tuple[torch.Tensor, dict[str, float]]:
    valid = valid_mask(lengths, plus.binary_logits.shape[1], plus.binary_logits.dtype)
    anomaly_rows = torch.nonzero(labels > 0.5, as_tuple=False).flatten()
    ranking_terms = []
    binary_terms = []
    selected_count = 0
    for row in anomaly_rows.tolist():
        length = int(lengths[row])
        count = max(1, int(np.ceil(length * top_fraction)))
        selected = torch.topk(views.target_text_effect[row, :length].detach(), count).indices
        class_index = int(target_full[row].argmax())
        semantic_delta = (
            plus.semantic_logits[row, selected, class_index]
            - minus.semantic_logits[row, selected, class_index]
        )
        binary_delta = plus.binary_logits[row, selected] - minus.binary_logits[row, selected]
        ranking_terms.append(F.relu(semantic_margin - semantic_delta).mean())
        binary_terms.append(F.relu(binary_margin - binary_delta).mean())
        selected_count += count
    semantic_rank = torch.stack(ranking_terms).mean() if ranking_terms else plus.binary_logits.sum() * 0.0
    binary_rank = torch.stack(binary_terms).mean() if binary_terms else plus.binary_logits.sum() * 0.0
    normal = (labels < 0.5).unsqueeze(1).to(valid.dtype) * valid
    normal_binary = (
        ((plus.binary_logits - minus.binary_logits).square() * normal).sum()
        / normal.sum().clamp_min(1.0)
    )
    normal_semantic = (
        ((F.softmax(plus.semantic_logits, dim=-1) - F.softmax(minus.semantic_logits, dim=-1)).square().mean(-1) * normal).sum()
        / normal.sum().clamp_min(1.0)
    )
    loss = semantic_rank + 0.25 * binary_rank + normal_binary + normal_semantic
    return loss, {
        "semantic_rank": float(semantic_rank.detach()),
        "binary_rank": float(binary_rank.detach()),
        "normal_invariance": float((normal_binary + normal_semantic).detach()),
        "selected_snippets": float(selected_count),
    }


def pcgrad_backward(
    primary: torch.Tensor,
    auxiliary: torch.Tensor,
    parameters: list[torch.nn.Parameter],
) -> dict[str, float]:
    primary_grad = torch.autograd.grad(
        primary, parameters, retain_graph=True, allow_unused=True
    )
    auxiliary_grad = torch.autograd.grad(
        auxiliary, parameters, allow_unused=True
    )
    dot = primary.new_zeros(())
    primary_norm = primary.new_zeros(())
    for first, second in zip(primary_grad, auxiliary_grad):
        if first is not None:
            primary_norm = primary_norm + first.detach().square().sum()
        if first is not None and second is not None:
            dot = dot + (first.detach() * second.detach()).sum()
    coefficient = torch.where(
        dot < 0, dot / primary_norm.clamp_min(1e-12), dot.new_zeros(())
    )
    for parameter, first, second in zip(parameters, primary_grad, auxiliary_grad):
        if first is None and second is None:
            parameter.grad = None
            continue
        first_value = torch.zeros_like(parameter) if first is None else first
        second_value = torch.zeros_like(parameter) if second is None else second
        if dot < 0 and first is not None and second is not None:
            second_value = second_value - coefficient * first_value
        parameter.grad = (first_value + second_value).detach()
    cosine = float(dot / (primary_norm.sqrt() + 1e-12))
    return {"gradient_dot": float(dot), "gradient_cosine_proxy": cosine, "projected": float(dot < 0)}


def main() -> None:
    parser = argparse.ArgumentParser(description="Train concept-conditioned neuron circuit routing.")
    parser.add_argument("--baseline", choices=["dsanet", "desc", "lagovad"], required=True)
    parser.add_argument("--baseline-root", required=True); parser.add_argument("--baseline-weight", default="")
    parser.add_argument("--sensitivity-weight", default=""); parser.add_argument("--consistency-weight", default="")
    parser.add_argument("--dataset", choices=["ucf", "xd"], required=True)
    parser.add_argument("--train-list", required=True); parser.add_argument("--val-list", required=True)
    parser.add_argument("--atlas", required=True); parser.add_argument("--gt-path", required=True)
    parser.add_argument("--out-dir", required=True); parser.add_argument("--teacher-cache", default="")
    parser.add_argument("--max-epoch", type=int, default=10); parser.add_argument("--warmup-epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=64); parser.add_argument("--lr", type=float, default=7e-5)
    parser.add_argument("--router-lr", type=float, default=7e-5); parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--counterfactual-weight", type=float, default=0.50)
    parser.add_argument("--preservation-weight", type=float, default=0.50); parser.add_argument("--anchor-weight", type=float, default=0.01)
    parser.add_argument("--top-fraction", type=float, default=0.10)
    parser.add_argument("--semantic-margin", type=float, default=0.10); parser.add_argument("--binary-margin", type=float, default=0.05)
    parser.add_argument("--gate-temperature", type=float, default=0.05)
    parser.add_argument("--max-gain", type=float, default=0.50); parser.add_argument("--initial-gain", type=float, default=0.10)
    parser.add_argument("--frames-per-snippet", type=int, default=16)
    parser.add_argument("--dsanet-ucf-eval-samples", type=int, default=1280)
    parser.add_argument("--num-workers", type=int, default=4); parser.add_argument("--seed", type=int, default=234)
    parser.add_argument("--device", default="cuda"); parser.add_argument("--resume", action="store_true")
    parser.add_argument("--clean", action="store_true")
    args = parser.parse_args()
    if args.clean and args.resume:
        parser.error("--clean and --resume cannot be combined")
    if not 0 < args.top_fraction <= 1:
        parser.error("--top-fraction must be in (0,1]")
    output = clean_output(args.out_dir, args.clean)
    last_path = output / "checkpoint_last.pth"; best_path = output / "model_best.pth"
    if last_path.exists() and not args.resume:
        raise RuntimeError("checkpoint exists; use --resume or --clean")
    seed_everything(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() and args.device.startswith("cuda") else "cpu")
    router, atlas = load_circuit_router(
        args.atlas, args.gate_temperature, args.max_gain, args.initial_gain
    )
    if not atlas.get("gate_passed", False):
        raise RuntimeError("circuit falsification gate failed; refusing to train")
    router = router.to(device)

    teacher = build_baseline(args, str(device)).to(device).eval(); teacher.requires_grad_(False)
    teacher_set = AlignedFeatureDataset(args.train_list, args.dataset, teacher.visual_length)
    teacher_path = Path(args.teacher_cache) if args.teacher_cache else output / "author_train_logits.pth"
    teacher_logits, teacher_index = cache_teacher_binary(
        teacher, teacher_set, teacher_path, args.batch_size * 2, args.num_workers, device
    )
    del teacher, teacher_set; gc.collect()
    if device.type == "cuda": torch.cuda.empty_cache()

    adapter = build_baseline(args, str(device)).to(device)
    adapter.set_train_scope("temporal_only")
    baseline_parameters = [value for value in adapter.parameters() if value.requires_grad]
    router_parameters = [value for value in router.parameters() if value.requires_grad]
    if not baseline_parameters or not router_parameters:
        raise RuntimeError("CNCR requires both router and final temporal parameters")
    initial_baseline = [value.detach().clone() for value in baseline_parameters]
    optimizer = torch.optim.AdamW(
        [
            {"params": router_parameters, "lr": args.router_lr, "name": "router"},
            {"params": baseline_parameters, "lr": 0.0, "name": "temporal"},
        ],
        weight_decay=args.weight_decay,
    )
    normal = AlignedFeatureDataset(args.train_list, args.dataset, adapter.visual_length, "normal")
    abnormal = AlignedFeatureDataset(args.train_list, args.dataset, adapter.visual_length, "abnormal")
    report = {
        "method": "concept_conditioned_neuron_circuit_routing_v1",
        "baseline": args.baseline, "dataset": args.dataset,
        "policy": "router warm-up, then final temporal block; CLIP and heads frozen",
        "baseline_trainable_parameters": sum(value.numel() for value in baseline_parameters),
        "router_trainable_parameters": sum(value.numel() for value in router_parameters),
        "atlas_gate": atlas["checks"], "batch_size_per_normality": args.batch_size,
        "learning_rate": args.lr, "router_learning_rate": args.router_lr,
    }
    (output / "parameter_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)

    start_epoch, best, processed = 0, -float("inf"), 0
    if args.resume:
        checkpoint = torch.load(last_path, map_location="cpu")
        adapter.load_state_dict(checkpoint["model_state_dict"], strict=True)
        router.load_state_dict(checkpoint["router_state_dict"], strict=True)
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        start_epoch = int(checkpoint["epoch"]) + 1
        best = float(checkpoint["best_metric"]); processed = int(checkpoint["processed_samples"])

    def payload(epoch: int, metrics: dict) -> dict:
        return {
            "method": report["method"], "epoch": epoch, "best_metric": best,
            "processed_samples": processed, "model_state_dict": adapter.state_dict(),
            "router_state_dict": router.state_dict(), "optimizer_state_dict": optimizer.state_dict(),
            "config": vars(args), "metrics": metrics,
        }

    def validate(epoch: int, tag: str) -> dict:
        nonlocal best
        metrics, _ = evaluate_models(
            adapter, router, args.val_list, args.gt_path, args.dataset,
            args.frames_per_snippet, device, include_author=False,
            description=f"validation {tag}",
        )
        metric_name = "auc" if args.dataset == "ucf" else "ap"
        value = float(metrics["cncr"][metric_name])
        if value > best:
            best = value
            torch.save(payload(epoch, metrics), best_path)
        print(f"validation {tag}: {json.dumps(metrics)} best={best:.6f}", flush=True)
        return metrics

    if not args.resume:
        validate(-1, "initial_router")
    next_eval = (processed // args.dsanet_ucf_eval_samples + 1) * args.dsanet_ucf_eval_samples
    history_path = output / "history.jsonl"
    parameters = router_parameters + baseline_parameters
    for epoch in range(start_epoch, args.max_epoch):
        baseline_lr = 0.0 if epoch < args.warmup_epochs else args.lr * 0.5 * (
            1.0 + np.cos(np.pi * (epoch - args.warmup_epochs) / max(1, args.max_epoch - args.warmup_epochs))
        )
        router_lr = args.router_lr * 0.5 * (1.0 + np.cos(np.pi * epoch / max(1, args.max_epoch)))
        optimizer.param_groups[0]["lr"] = float(router_lr)
        optimizer.param_groups[1]["lr"] = float(baseline_lr)
        common = {
            "batch_size": args.batch_size, "shuffle": True, "drop_last": True,
            "num_workers": args.num_workers, "pin_memory": device.type == "cuda",
        }
        normal_loader = DataLoader(
            normal, generator=torch.Generator().manual_seed(args.seed + epoch), **common
        )
        abnormal_loader = DataLoader(
            abnormal, generator=torch.Generator().manual_seed(args.seed + 100003 + epoch), **common
        )
        adapter.train(); router.train()
        running = {
            "total": 0.0, "original": 0.0, "preserve": 0.0, "counterfactual": 0.0,
            "semantic_rank": 0.0, "binary_rank": 0.0, "normal_invariance": 0.0,
            "gradient_projected": 0.0, "gate": 0.0, "text_effect": 0.0,
        }
        progress = tqdm(
            zip(normal_loader, abnormal_loader), total=min(len(normal_loader), len(abnormal_loader)),
            desc=f"CNCR {epoch + 1}/{args.max_epoch}", unit="batch",
        )
        for step, (normal_batch, abnormal_batch) in enumerate(progress, 1):
            batch = merge(normal_batch, abnormal_batch)
            clip = batch["clip"].to(device); compact = batch["neurons"].to(device)
            lengths = batch["length"].to(device); labels = batch["binary_label"].to(device)
            full_targets = class_targets(batch["label_text"], adapter.label_map, device)
            concept_targets = full_targets[:, 1:]
            views = router(clip, compact, concept_targets)
            plus = adapter.forward_baseline(views.enhanced, lengths)
            minus = adapter.forward_baseline(views.suppressed, lengths)
            original = adapter.original_loss(plus, labels, batch["label_text"], lengths)
            teacher_rows = torch.tensor(
                [teacher_index[str(value)] for value in batch["sample_id"]], dtype=torch.long
            )
            teacher_batch = teacher_logits.index_select(0, teacher_rows).to(device)
            preserve = preservation_loss(plus.binary_logits, teacher_batch, lengths)
            anchor = torch.stack([
                (value - initial).square().mean() / initial.square().mean().clamp_min(1e-8)
                for value, initial in zip(baseline_parameters, initial_baseline)
            ]).mean()
            primary = original + args.preservation_weight * preserve + args.anchor_weight * anchor
            cf, diagnostics = counterfactual_loss(
                plus, minus, views, full_targets, labels, lengths, args.top_fraction,
                args.semantic_margin, args.binary_margin,
            )
            auxiliary = args.counterfactual_weight * cf
            optimizer.zero_grad(set_to_none=True)
            gradient = pcgrad_backward(primary, auxiliary, parameters)
            torch.nn.utils.clip_grad_norm_(parameters, max_norm=5.0)
            optimizer.step(); processed += int(labels.numel())
            values = {
                "total": float((primary + auxiliary).detach()), "original": float(original.detach()),
                "preserve": float(preserve.detach()), "counterfactual": float(cf.detach()),
                "semantic_rank": diagnostics["semantic_rank"], "binary_rank": diagnostics["binary_rank"],
                "normal_invariance": diagnostics["normal_invariance"],
                "gradient_projected": gradient["projected"],
                "gate": float(views.anomaly_gate.mean().detach()),
                "text_effect": float(views.target_text_effect.mean().detach()),
            }
            for key, value in values.items(): running[key] += value
            progress.set_postfix(
                loss=f"{running['total'] / step:.4f}", gate=f"{running['gate'] / step:.3f}",
                pcgrad=f"{running['gradient_projected'] / step:.2f}",
            )
            if args.baseline == "dsanet" and args.dataset == "ucf" and processed >= next_eval:
                validate(epoch, f"sample_{processed}")
                adapter.train(); router.train(); next_eval += args.dsanet_ucf_eval_samples
        if args.baseline == "dsanet" and args.dataset == "ucf":
            metrics = {"selection_deferred_to_fixed_sample_interval": True}
        else:
            metrics = validate(epoch, f"epoch_{epoch + 1}")
        record = {
            "epoch": epoch + 1, "router_lr": router_lr, "baseline_lr": baseline_lr,
            **{key: value / max(1, step) for key, value in running.items()}, "metrics": metrics,
        }
        with history_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record) + "\n")
        torch.save(payload(epoch, metrics), last_path)
        print(json.dumps(record), flush=True)


if __name__ == "__main__":
    main()
