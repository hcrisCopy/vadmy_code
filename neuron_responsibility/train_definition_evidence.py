#!/usr/bin/env python3
"""Train baselines with definition circuits as training-only supervision.

The original loss, batching convention and UCF/XD model-selection metrics are
kept.  The added loss follows relative top/bottom MIL supervision, while test
inference calls the released baseline directly.
"""

from __future__ import annotations

import argparse
import gc
import json
import random
import sys
from pathlib import Path

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

from neuron_responsibility.baselines import build_baseline, class_targets
from neuron_responsibility.common import clean_output
from neuron_responsibility.data import AlignedFeatureDataset
from neuron_responsibility.definition_evidence import DefinitionEvidence, definition_losses
from neuron_responsibility.evaluate import pad_chunks
from neuron_responsibility.model import valid_mask


def seed_everything(seed: int) -> None:
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)


def merge(normal: dict, abnormal: dict) -> dict:
    result = {}
    for key in ("clip", "neurons", "length", "binary_label"):
        result[key] = torch.cat([normal[key], abnormal[key]], dim=0)
    for key in ("label_text", "key", "sample_id"):
        result[key] = list(normal[key]) + list(abnormal[key])
    return result


def relative_anchor(parameters: dict[str, torch.nn.Parameter], initial: dict[str, torch.Tensor]) -> torch.Tensor:
    terms = []
    for name, parameter in parameters.items():
        if parameter.requires_grad:
            denominator = initial[name].square().mean().clamp_min(1e-8)
            terms.append((parameter - initial[name]).square().mean() / denominator)
    return torch.stack(terms).mean() if terms else next(iter(parameters.values())).sum() * 0.0


def frame_metrics(adapter, csv_path: str, gt_path: str, repeat: int, device: torch.device, description: str) -> dict[str, float]:
    adapter.eval(); frame = pd.read_csv(csv_path)
    scores = []
    with torch.no_grad():
        for _, group in tqdm(list(frame.groupby("key", sort=False)), desc=description, unit="video", leave=False):
            clip = np.concatenate([np.load(str(path)).astype(np.float32) for path in group["clip_path"]])
            chunks, lengths = pad_chunks(clip, adapter.visual_length)
            output = adapter.forward_baseline(chunks.to(device), lengths.to(device))
            scores.extend([
                torch.sigmoid(output.binary_logits[index, :length]).cpu()
                for index, length in enumerate(lengths.tolist())
            ])
    prediction = np.repeat(torch.cat(scores).numpy(), repeat)
    truth = np.load(gt_path).astype(np.int64).reshape(-1)
    usable = min(len(truth), len(prediction))
    return {
        "frame_auc": float(roc_auc_score(truth[:usable], prediction[:usable])),
        "frame_ap": float(average_precision_score(truth[:usable], prediction[:usable])),
        "frames": int(usable),
    }


def parameter_group(name: str) -> str:
    if "video_anomaly_refiner" in name:
        return "dnp"
    if any(token in name for token in ("classifier", "mlp1", "mlp2", "bin_head", "sim_head")):
        return "heads"
    return "temporal"


def cache_teacher_binary(adapter, dataset, path: Path, batch_size: int, workers: int, device: torch.device) -> tuple[torch.Tensor, dict[str, int]]:
    if path.exists():
        value = torch.load(path, map_location="cpu")
        return value["logits"].float(), {str(key): index for index, key in enumerate(value["sample_ids"])}
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=workers, pin_memory=device.type == "cuda")
    values, sample_ids = [], []
    adapter.eval()
    with torch.no_grad():
        for batch in tqdm(loader, desc="cache released baseline logits", unit="batch"):
            output = adapter.forward_baseline(batch["clip"].to(device), batch["length"].to(device))
            values.append(output.binary_logits.cpu().half()); sample_ids.extend(map(str, batch["sample_id"]))
    logits = torch.cat(values)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"logits": logits, "sample_ids": sample_ids}, path)
    return logits.float(), {key: index for index, key in enumerate(sample_ids)}


def preservation_loss(logits: torch.Tensor, teacher: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
    mask = valid_mask(lengths, logits.shape[1], logits.dtype)
    value = F.binary_cross_entropy_with_logits(logits, torch.sigmoid(teacher), reduction="none")
    return (value * mask).sum() / mask.sum().clamp_min(1.0)


def main() -> None:
    parser = argparse.ArgumentParser(description="Training-only definition evidence adaptation.")
    parser.add_argument("--baseline", choices=["dsanet", "desc", "lagovad"], required=True)
    parser.add_argument("--baseline-root", required=True); parser.add_argument("--baseline-weight", default="")
    parser.add_argument("--sensitivity-weight", default=""); parser.add_argument("--consistency-weight", default="")
    parser.add_argument("--dataset", choices=["ucf", "xd"], required=True)
    parser.add_argument("--train-list", required=True); parser.add_argument("--val-list", required=True)
    parser.add_argument("--atlas", required=True); parser.add_argument("--gt-path", required=True)
    parser.add_argument("--teacher-cache", default="")
    parser.add_argument("--out-dir", required=True); parser.add_argument("--max-epoch", type=int, default=10)
    parser.add_argument("--temporal-start-epoch", type=int, default=2); parser.add_argument("--reference-start-epoch", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=64); parser.add_argument("--head-lr", type=float, default=7e-5)
    parser.add_argument("--temporal-lr", type=float, default=7e-6); parser.add_argument("--reference-lr", type=float, default=7e-6)
    parser.add_argument("--weight-decay", type=float, default=0.0); parser.add_argument("--top-fraction", type=float, default=0.10)
    parser.add_argument("--binary-margin", type=float, default=0.20); parser.add_argument("--semantic-margin", type=float, default=0.20)
    parser.add_argument("--binary-weight", type=float, default=0.25); parser.add_argument("--semantic-weight", type=float, default=0.25)
    parser.add_argument("--normal-weight", type=float, default=0.10); parser.add_argument("--dnp-weight", type=float, default=0.20)
    parser.add_argument("--anchor-weight", type=float, default=0.01); parser.add_argument("--frames-per-snippet", type=int, default=16)
    parser.add_argument("--preservation-weight", type=float, default=0.50)
    parser.add_argument("--dsanet-ucf-eval-samples", type=int, default=1280)
    parser.add_argument("--num-workers", type=int, default=4); parser.add_argument("--seed", type=int, default=234)
    parser.add_argument("--device", default="cuda"); parser.add_argument("--resume", action="store_true")
    parser.add_argument("--clean", action="store_true")
    args = parser.parse_args()
    if args.clean and args.resume:
        parser.error("--clean and --resume cannot be combined")
    if not 0 <= args.temporal_start_epoch <= args.reference_start_epoch < args.max_epoch:
        parser.error("invalid progressive-unfreezing epochs")
    output = clean_output(args.out_dir, args.clean)
    last_path, best_path = output / "checkpoint_last.pth", output / "model_best.pth"
    if last_path.exists() and not args.resume:
        raise RuntimeError("checkpoint exists; use --resume or --clean")
    seed_everything(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() and args.device.startswith("cuda") else "cpu")
    teacher = build_baseline(args, str(device)).to(device).eval(); teacher.requires_grad_(False)
    teacher_set = AlignedFeatureDataset(args.train_list, args.dataset, teacher.visual_length)
    teacher_path = Path(args.teacher_cache) if args.teacher_cache else output / "author_train_logits.pth"
    teacher_logits, teacher_index = cache_teacher_binary(
        teacher, teacher_set, teacher_path, args.batch_size * 2, args.num_workers, device
    )
    missing_teacher = sorted(set(map(str, teacher_set.frame["clip_path"])) - set(teacher_index))
    if missing_teacher:
        raise RuntimeError(f"teacher cache misses {len(missing_teacher)} training samples; first={missing_teacher[0]}")
    del teacher, teacher_set; gc.collect()
    if device.type == "cuda": torch.cuda.empty_cache()
    adapter = build_baseline(args, str(device)).to(device)
    maximum_scope = "evidence_adaptation" if args.baseline == "dsanet" else "temporal_heads"
    adapter.set_train_scope(maximum_scope)
    parameters = {name: value for name, value in adapter.named_parameters() if value.requires_grad}
    if any("clip" in name.lower() for name in parameters):
        raise RuntimeError("CLIP must remain frozen")
    grouped = {name: [] for name in ("heads", "temporal", "dnp")}
    for name, parameter in parameters.items():
        grouped[parameter_group(name)].append(parameter)
    optimizer_groups = [
        {"params": grouped["heads"], "lr": args.head_lr, "name": "heads"},
        {"params": grouped["temporal"], "lr": args.temporal_lr, "name": "temporal"},
    ]
    if grouped["dnp"]:
        optimizer_groups.append({"params": grouped["dnp"], "lr": args.reference_lr, "name": "dnp"})
    optimizer = torch.optim.AdamW(optimizer_groups, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.max_epoch)
    initial = {name: value.detach().clone() for name, value in parameters.items()}
    evidence_model = DefinitionEvidence(args.atlas).to(device).eval(); evidence_model.requires_grad_(False)
    normal_set = AlignedFeatureDataset(args.train_list, args.dataset, adapter.visual_length, "normal")
    abnormal_set = AlignedFeatureDataset(args.train_list, args.dataset, adapter.visual_length, "abnormal")
    report = {
        "method": "definition_sensitive_training_evidence_v1", "baseline": args.baseline, "dataset": args.dataset,
        "inference": "released baseline path only; circuits are absent", "selection": "UCF frame AUC / XD frame AP",
        "circuit_source": evidence_model.atlas["selection_source"], "compact_width": evidence_model.width,
        "trainable": {key: sum(value.numel() for value in values) for key, values in grouped.items()},
        "clip_trainable_parameters": 0, "target_tensors": list(parameters),
    }
    (output / "parameter_report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False), flush=True)
    start_epoch, best, processed = 0, -float("inf"), 0
    run_config = vars(args).copy()
    if args.resume:
        checkpoint = torch.load(last_path, map_location="cpu")
        if checkpoint.get("config") != run_config:
            raise RuntimeError("resume configuration differs from checkpoint")
        adapter.load_state_dict(checkpoint["model_state_dict"], strict=True)
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"]); scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        start_epoch = int(checkpoint["epoch"]) + 1; best = float(checkpoint["best_metric"]); processed = int(checkpoint["processed_samples"])

    def payload(epoch: int, metrics: dict, tag: str) -> dict:
        return {"method": report["method"], "epoch": epoch, "best_metric": best, "processed_samples": processed,
                "model_state_dict": adapter.state_dict(), "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(), "config": run_config, "metrics": metrics,
                "validation_tag": tag, "selection_rule": report["selection"]}

    def validate(epoch: int, tag: str) -> dict:
        nonlocal best
        metrics = frame_metrics(adapter, args.val_list, args.gt_path, args.frames_per_snippet, device, f"validation {tag}")
        value = metrics["frame_auc" if args.dataset == "ucf" else "frame_ap"]
        if value > best:
            best = value; torch.save(payload(epoch, metrics, tag), best_path)
        print(f"validation {tag}: {json.dumps(metrics)} best={best:.6f}", flush=True)
        return metrics

    if not args.resume:
        validate(-1, "released_author")
    next_eval = (processed // args.dsanet_ucf_eval_samples + 1) * args.dsanet_ucf_eval_samples
    history_path = output / "history.jsonl"
    for epoch in range(start_epoch, args.max_epoch):
        if epoch < args.temporal_start_epoch:
            stage, scope = "heads", "heads"
        elif epoch < args.reference_start_epoch or args.baseline != "dsanet":
            stage, scope = "temporal_heads", "temporal_heads"
        else:
            stage, scope = "definition_reference", "evidence_adaptation"
        adapter.set_train_scope(scope); adapter.train()
        common = {"batch_size": args.batch_size, "shuffle": True, "drop_last": True,
                  "num_workers": args.num_workers, "pin_memory": device.type == "cuda"}
        normal_loader = DataLoader(normal_set, generator=torch.Generator().manual_seed(args.seed + epoch), **common)
        abnormal_loader = DataLoader(abnormal_set, generator=torch.Generator().manual_seed(args.seed + 100003 + epoch), **common)
        running = {key: 0.0 for key in ("total", "original", "preservation", "binary_rank", "semantic_hard_negative", "normal_suppression", "dnp_rank", "anchor")}
        progress = tqdm(zip(normal_loader, abnormal_loader), total=min(len(normal_loader), len(abnormal_loader)), desc=f"definition evidence {epoch + 1}/{args.max_epoch}", unit="batch")
        step = 0
        for step, (normal, abnormal) in enumerate(progress, 1):
            batch = merge(normal, abnormal)
            clip = batch["clip"].to(device, non_blocking=True); compact = batch["neurons"].to(device, non_blocking=True)
            lengths = batch["length"].to(device, non_blocking=True); labels = batch["binary_label"].to(device, non_blocking=True)
            output_value = adapter.forward_baseline(clip, lengths)
            original = adapter.original_loss(output_value, labels, batch["label_text"], lengths)
            teacher_rows = torch.tensor([teacher_index[str(value)] for value in batch["sample_id"]], dtype=torch.long)
            teacher_batch = teacher_logits.index_select(0, teacher_rows).to(device)
            preserve = preservation_loss(output_value.binary_logits, teacher_batch, lengths)
            with torch.no_grad():
                evidence = evidence_model(compact)
            targets = class_targets(batch["label_text"], adapter.label_map, device)
            extra = definition_losses(output_value, evidence, targets, labels, lengths, args.top_fraction, args.binary_margin, args.semantic_margin)
            anchor = relative_anchor(parameters, initial)
            loss = (original + args.preservation_weight * preserve
                    + args.binary_weight * extra["binary_rank"] + args.semantic_weight * extra["semantic_hard_negative"]
                    + args.normal_weight * extra["normal_suppression"] + args.dnp_weight * extra["dnp_rank"] + args.anchor_weight * anchor)
            optimizer.zero_grad(set_to_none=True); loss.backward()
            torch.nn.utils.clip_grad_norm_([value for value in parameters.values() if value.requires_grad], 5.0)
            optimizer.step(); processed += int(labels.numel())
            values = {"total": loss, "original": original, "preservation": preserve, **extra, "anchor": anchor}
            for key, value in values.items(): running[key] += float(value.detach())
            progress.set_postfix(stage=stage, loss=f"{running['total'] / step:.4f}", rank=f"{running['binary_rank'] / step:.3f}")
            if args.baseline == "dsanet" and args.dataset == "ucf" and processed >= next_eval:
                validate(epoch, f"sample_{processed}"); adapter.train(); next_eval += args.dsanet_ucf_eval_samples
        scheduler.step()
        metrics = {"selection_deferred_to_fixed_sample_interval": True}
        if args.baseline != "dsanet" or args.dataset != "ucf" or not best_path.exists():
            metrics = validate(epoch, f"epoch_{epoch + 1}")
        record = {"epoch": epoch + 1, "stage": stage, **{f"{key}_loss": value / max(1, step) for key, value in running.items()}, "metrics": metrics}
        with history_path.open("a", encoding="utf-8") as handle: handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        torch.save(payload(epoch, metrics, "epoch_recovery"), last_path)
        print(json.dumps(record, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
