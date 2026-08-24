#!/usr/bin/env python3
"""Fine-tune only the last temporal block with a confidence-masked neuron prior."""

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

from neuron_responsibility.baselines import build_baseline
from neuron_responsibility.common import clean_output, load_json
from neuron_responsibility.data import AlignedFeatureDataset
from neuron_responsibility.evaluate import pad_chunks
from neuron_responsibility.model import valid_mask


def seed_everything(seed: int) -> None:
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)


def merge(normal: dict, abnormal: dict) -> dict:
    result = {key: torch.cat([normal[key], abnormal[key]], 0) for key in ("clip", "neurons", "length", "binary_label")}
    for key in ("label_text", "sample_id"):
        result[key] = list(normal[key]) + list(abnormal[key])
    return result


def cache_teacher(adapter, dataset, path: Path, batch_size: int, workers: int, device):
    if path.exists():
        value = torch.load(path, map_location="cpu")
        return value["logits"].float(), {key: index for index, key in enumerate(value["sample_ids"])}
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=workers, pin_memory=device.type == "cuda")
    values, ids = [], []
    adapter.eval()
    with torch.no_grad():
        for batch in tqdm(loader, desc="cache released-model logits", unit="batch"):
            output = adapter.forward_baseline(batch["clip"].to(device), batch["length"].to(device))
            values.append(output.binary_logits.cpu().float()); ids.extend(map(str, batch["sample_id"]))
    logits = torch.cat(values)
    torch.save({"logits": logits, "sample_ids": ids}, path)
    return logits, {key: index for index, key in enumerate(ids)}


def prior_loss(logits, prior_features, labels, lengths, low: float, high: float):
    prior = prior_features[..., 0].clamp(0, 1)
    valid = valid_mask(lengths, logits.shape[1], logits.dtype)
    normal = (labels < 0.5).unsqueeze(1)
    mask = ((prior <= low) | (prior >= high) | normal).to(logits.dtype) * valid
    target = torch.where(normal, torch.zeros_like(prior), (prior >= high).to(prior.dtype))
    loss = F.binary_cross_entropy_with_logits(logits, target, reduction="none")
    return (loss * mask).sum() / mask.sum().clamp_min(1.0), float(mask.sum().detach() / valid.sum().clamp_min(1.0))


def preservation_loss(logits, teacher, lengths):
    mask = valid_mask(lengths, logits.shape[1], logits.dtype)
    loss = F.binary_cross_entropy_with_logits(logits, torch.sigmoid(teacher), reduction="none")
    return (loss * mask).sum() / mask.sum().clamp_min(1.0)


def official_metrics(adapter, csv_path: str, gt_path: str, repeat: int, device):
    frame = pd.read_csv(csv_path); scores = []
    adapter.eval()
    with torch.no_grad():
        for _, group in tqdm(list(frame.groupby("key", sort=False)), desc="official validation", unit="video", leave=False):
            clip = np.concatenate([np.load(str(path)).astype(np.float32) for path in group["clip_path"]])
            chunks, lengths = pad_chunks(clip, adapter.visual_length)
            output = adapter.forward_baseline(chunks.to(device), lengths.to(device))
            scores.extend(torch.sigmoid(output.binary_logits[i, :length]).cpu() for i, length in enumerate(lengths.tolist()))
    prediction = np.repeat(torch.cat(scores).numpy(), repeat)
    truth = np.load(gt_path).astype(np.int64).reshape(-1); usable = min(len(truth), len(prediction))
    return {"frame_auc": float(roc_auc_score(truth[:usable], prediction[:usable])), "frame_ap": float(average_precision_score(truth[:usable], prediction[:usable]))}


def main() -> None:
    parser = argparse.ArgumentParser(description="Last-temporal training from a text neuron prior.")
    parser.add_argument("--baseline", choices=["dsanet", "desc", "lagovad"], required=True)
    parser.add_argument("--baseline-root", required=True)
    parser.add_argument("--baseline-weight", default="")
    parser.add_argument("--sensitivity-weight", default="")
    parser.add_argument("--consistency-weight", default="")
    parser.add_argument("--dataset", choices=["ucf", "xd"], required=True)
    parser.add_argument("--train-list", required=True); parser.add_argument("--val-list", required=True)
    parser.add_argument("--gt-path", required=True); parser.add_argument("--gate-metrics", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--max-epoch", type=int, default=4); parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-6); parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--prior-weight", type=float, default=0.5); parser.add_argument("--preservation-weight", type=float, default=0.5)
    parser.add_argument("--anchor-weight", type=float, default=0.01)
    parser.add_argument("--low-confidence", type=float, default=0.20); parser.add_argument("--high-confidence", type=float, default=0.80)
    parser.add_argument("--frames-per-snippet", type=int, default=16); parser.add_argument("--dsanet-ucf-eval-samples", type=int, default=1280)
    parser.add_argument("--num-workers", type=int, default=4); parser.add_argument("--seed", type=int, default=234); parser.add_argument("--device", default="cuda")
    parser.add_argument("--resume", action="store_true"); parser.add_argument("--clean", action="store_true")
    args = parser.parse_args()
    if args.clean and args.resume: parser.error("--clean and --resume cannot be combined")
    gate = load_json(args.gate_metrics)
    if not gate.get("gate_passed", False):
        raise RuntimeError("falsification gate did not pass; do not train the baseline")
    output = clean_output(args.out_dir, args.clean); last_path = output / "checkpoint_last.pth"; best_path = output / "model_best.pth"
    if last_path.exists() and not args.resume: raise RuntimeError("checkpoint exists; use --resume or --clean")
    seed_everything(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() and args.device.startswith("cuda") else "cpu")
    teacher = build_baseline(args, str(device)).to(device).eval(); teacher.requires_grad_(False)
    teacher_set = AlignedFeatureDataset(args.train_list, args.dataset, teacher.visual_length)
    teacher_logits, teacher_index = cache_teacher(teacher, teacher_set, output / "author_train_logits.pth", args.batch_size * 2, args.num_workers, device)
    del teacher, teacher_set; gc.collect()
    if device.type == "cuda": torch.cuda.empty_cache()
    adapter = build_baseline(args, str(device)).to(device); adapter.set_train_scope("temporal_only")
    trainable = {name: parameter for name, parameter in adapter.named_parameters() if parameter.requires_grad}
    if not trainable: raise RuntimeError("temporal-only scope contains no parameters")
    initial = {name: parameter.detach().clone() for name, parameter in trainable.items()}
    optimizer = torch.optim.AdamW(trainable.values(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.max_epoch)
    normal = AlignedFeatureDataset(args.train_list, args.dataset, adapter.visual_length, "normal")
    abnormal = AlignedFeatureDataset(args.train_list, args.dataset, adapter.visual_length, "abnormal")
    config = {"method": "clip_text_neuron_responsibility_v1", "baseline": args.baseline, "dataset": args.dataset, "scope": "last temporal block; all heads and CLIP frozen", "lr": args.lr, "prior_weight": args.prior_weight}
    report = {**config, "trainable_parameters": sum(value.numel() for value in trainable.values()), "trainable_tensors": list(trainable), "gate": gate}
    (output / "parameter_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)
    start, best, processed = 0, -float("inf"), 0
    if args.resume:
        checkpoint = torch.load(last_path, map_location="cpu"); adapter.load_state_dict(checkpoint["model_state_dict"], strict=True)
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"]); scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        start, best, processed = checkpoint["epoch"] + 1, checkpoint["best_metric"], checkpoint["processed_samples"]

    def payload(epoch, metrics):
        return {"method": config["method"], "epoch": epoch, "best_metric": best, "processed_samples": processed, "model_state_dict": adapter.state_dict(), "optimizer_state_dict": optimizer.state_dict(), "scheduler_state_dict": scheduler.state_dict(), "config": config, "metrics": metrics}

    def validate(epoch, tag):
        nonlocal best
        metrics = official_metrics(adapter, args.val_list, args.gt_path, args.frames_per_snippet, device)
        value = metrics["frame_auc" if args.dataset == "ucf" else "frame_ap"]
        if value > best: best = value; torch.save(payload(epoch, metrics), best_path)
        print(f"validation {tag}: {metrics} best={best:.6f}", flush=True); return metrics

    next_eval = (processed // args.dsanet_ucf_eval_samples + 1) * args.dsanet_ucf_eval_samples
    history = output / "history.jsonl"
    for epoch in range(start, args.max_epoch):
        common = {"batch_size": args.batch_size, "shuffle": True, "drop_last": True, "num_workers": args.num_workers, "pin_memory": device.type == "cuda"}
        normal_loader = DataLoader(normal, generator=torch.Generator().manual_seed(args.seed + epoch), **common)
        abnormal_loader = DataLoader(abnormal, generator=torch.Generator().manual_seed(args.seed + 100003 + epoch), **common)
        adapter.train(); running = {"total": 0.0, "original": 0.0, "prior": 0.0, "preserve": 0.0, "anchor": 0.0, "coverage": 0.0}
        progress = tqdm(zip(normal_loader, abnormal_loader), total=min(len(normal_loader), len(abnormal_loader)), desc=f"text responsibility {epoch + 1}/{args.max_epoch}", unit="batch")
        for step, (normal_batch, abnormal_batch) in enumerate(progress, 1):
            batch = merge(normal_batch, abnormal_batch); clip = batch["clip"].to(device); lengths = batch["length"].to(device); labels = batch["binary_label"].to(device); priors = batch["neurons"].to(device)
            teacher_rows = torch.tensor([teacher_index[str(value)] for value in batch["sample_id"]], dtype=torch.long)
            teacher_batch = teacher_logits.index_select(0, teacher_rows).to(device)
            output_value = adapter.forward_baseline(clip, lengths)
            original = adapter.original_loss(output_value, labels, batch["label_text"], lengths)
            prior, coverage = prior_loss(output_value.binary_logits, priors, labels, lengths, args.low_confidence, args.high_confidence)
            preserve = preservation_loss(output_value.binary_logits, teacher_batch, lengths)
            anchor = torch.stack([((parameter - initial[name]).square().mean() / initial[name].square().mean().clamp_min(1e-8)) for name, parameter in trainable.items()]).mean()
            total = original + args.prior_weight * prior + args.preservation_weight * preserve + args.anchor_weight * anchor
            optimizer.zero_grad(set_to_none=True); total.backward(); optimizer.step(); processed += int(labels.numel())
            values = {"total": float(total.detach()), "original": float(original.detach()), "prior": float(prior.detach()), "preserve": float(preserve.detach()), "anchor": float(anchor.detach()), "coverage": coverage}
            for key, value in values.items(): running[key] += value
            progress.set_postfix(loss=f"{running['total']/step:.4f}", coverage=f"{running['coverage']/step:.2f}")
            if args.baseline == "dsanet" and args.dataset == "ucf" and processed >= next_eval:
                validate(epoch, f"sample_{processed}"); adapter.train(); next_eval += args.dsanet_ucf_eval_samples
        scheduler.step()
        metrics = {"selection_deferred_to_fixed_step": True} if args.baseline == "dsanet" and args.dataset == "ucf" else validate(epoch, f"epoch_{epoch+1}")
        if not best_path.exists(): metrics = validate(epoch, f"epoch_{epoch+1}_bootstrap")
        record = {"epoch": epoch + 1, **{key: value / max(1, step) for key, value in running.items()}, "metrics": metrics}
        with history.open("a", encoding="utf-8") as handle: handle.write(json.dumps(record) + "\n")
        torch.save(payload(epoch, metrics), last_path); print(json.dumps(record), flush=True)


if __name__ == "__main__":
    main()
