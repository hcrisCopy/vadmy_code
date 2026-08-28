#!/usr/bin/env python3
"""Partially adapt a copied baseline with LaGoVAD-style pseudo supervision."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.metrics import average_precision_score, roc_auc_score
from torch.utils.data import DataLoader
from tqdm import tqdm

from semantic_knn_splicing.baselines import build_baseline
from semantic_knn_splicing.common import base_key, clean_output, seed_everything
from semantic_knn_splicing.data import BaselineTrainDataset


def merge_original_batches(normal: dict, abnormal: dict) -> dict:
    merged = {}
    for key in ("clip", "length", "binary_label", "dense_label", "synthetic"):
        merged[key] = torch.cat([normal[key], abnormal[key]], dim=0)
    merged["label_text"] = list(normal["label_text"]) + list(abnormal["label_text"])
    return merged


def pad_chunks(array: np.ndarray, chunk_length: int) -> tuple[torch.Tensor, torch.Tensor]:
    chunks, lengths = [], []
    for start in range(0, max(1, len(array)), chunk_length):
        part = array[start:start + chunk_length]
        if not len(part):
            continue
        lengths.append(len(part))
        if len(part) < chunk_length:
            part = np.pad(part, ((0, chunk_length - len(part)), (0, 0)), mode="constant")
        chunks.append(part)
    return torch.from_numpy(np.stack(chunks).astype(np.float32)), torch.tensor(lengths, dtype=torch.long)


def pseudo_supervision_losses(
    logits: torch.Tensor,
    frame_labels: torch.Tensor,
    lengths: torch.Tensor,
    selected: torch.Tensor,
    topk_ratio: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """LaGoVAD dense BCE and pseudo-MIL, restricted to synthesized examples."""
    selected_indices = torch.where(selected > 0.5)[0]
    if not len(selected_indices):
        zero = logits.sum() * 0.0
        return zero, zero
    selected_logits = logits.index_select(0, selected_indices)
    selected_labels = frame_labels.index_select(0, selected_indices).float()
    selected_lengths = lengths.index_select(0, selected_indices)
    mask = (
        torch.arange(logits.shape[1], device=logits.device).unsqueeze(0)
        < selected_lengths.unsqueeze(1)
    ).float()
    dense = F.binary_cross_entropy_with_logits(
        selected_logits, selected_labels, reduction="none"
    )
    dense = (dense * mask).sum() / mask.sum().clamp_min(1.0)

    # This follows LaGoVAD's pseudo_sup_mil_loss: for an abnormal synthesized
    # sequence, select top-k only inside the annotated inserted span.
    running_logits, running_labels = [], []
    for index in range(len(selected_indices)):
        length = int(selected_lengths[index].item())
        labels = selected_labels[index, :length]
        positive = torch.where(labels > 0.5)[0]
        if not len(positive):
            continue
        k = max(1, len(positive) // topk_ratio)
        running_logits.append(
            selected_logits[index].index_select(0, positive).topk(k).values.mean()
        )
        running_labels.append(1.0)
    if not running_logits:
        return dense, logits.sum() * 0.0
    values = torch.stack(running_logits)
    targets = torch.tensor(running_labels, device=values.device, dtype=values.dtype)
    return dense, F.binary_cross_entropy_with_logits(values, targets)


def official_frame_metrics(
    adapter,
    test_list: str,
    gt_path: str,
    frames_per_snippet: int,
    device: torch.device,
    score_cache: Path | None = None,
) -> dict[str, float]:
    adapter.eval()
    frame = pd.read_csv(test_list)
    if not {"path", "label"}.issubset(frame.columns):
        raise ValueError(f"{test_list}: expected path/label columns")
    frame["key"] = frame["path"].map(base_key)
    scores = []
    if score_cache is not None:
        score_cache.mkdir(parents=True, exist_ok=True)
    with torch.no_grad():
        for key, group in tqdm(frame.groupby("key", sort=False), desc="official frame validation", unit="video", leave=False):
            cache_path = score_cache / f"{key}.npy" if score_cache is not None else None
            if cache_path is not None and cache_path.exists():
                scores.append(torch.from_numpy(np.load(cache_path).astype(np.float32)))
                continue
            clip = np.concatenate([np.load(str(path)).astype(np.float32) for path in group["path"]])
            chunks, lengths = pad_chunks(clip, adapter.visual_length)
            output = adapter.forward_baseline(chunks.to(device), lengths.to(device))
            video_scores = torch.cat([
                torch.sigmoid(output.binary_logits[index, :length]).cpu()
                for index, length in enumerate(lengths.tolist())
            ])
            if cache_path is not None:
                np.save(cache_path, video_scores.numpy())
            scores.append(video_scores)
    prediction = np.repeat(torch.cat(scores).numpy(), frames_per_snippet)
    truth = np.load(gt_path).astype(np.int64).reshape(-1)
    usable = min(len(truth), len(prediction))
    return {
        "frame_auc": float(roc_auc_score(truth[:usable], prediction[:usable])),
        "frame_ap": float(average_precision_score(truth[:usable], prediction[:usable])),
        "frames": int(usable),
    }


def parameter_groups(adapter, lr: float) -> tuple[list[dict], list[str]]:
    """Train only native visual-text interaction and score-head parameters."""
    adapter.set_train_scope("heads")
    parameters, names = [], []
    for name, parameter in adapter.named_parameters():
        if not parameter.requires_grad:
            continue
        lowered = name.lower()
        if "clipmodel." in lowered or "clip_text_model.model" in lowered:
            raise RuntimeError(f"CLIP backbone unexpectedly trainable: {name}")
        names.append(name)
        parameters.append(parameter)
    if not parameters:
        raise RuntimeError("visual-text interaction/head parameter group is empty")
    adapter.set_train_scope("frozen")
    return [{"params": parameters, "lr": lr, "name": "interaction_and_heads"}], names


def build_scheduler(optimizer, baseline: str, dataset: str, max_epoch: int):
    if baseline == "desc":
        milestones = [4, 8] if dataset == "ucf" else [6, 8]
        return torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=milestones, gamma=0.1)
    return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max_epoch)


def keep_clip_eval(adapter, baseline: str) -> None:
    if baseline == "dsanet":
        adapter.base.clipmodel.eval()
    elif baseline == "desc":
        adapter.sensitivity.clipmodel.eval()
        adapter.consistency.clipmodel.eval()
    else:
        adapter.base.clip_text_model.model.eval()


def main() -> None:
    parser = argparse.ArgumentParser(description="Train semantic KNN splicing with a copied baseline.")
    parser.add_argument("--baseline", choices=["dsanet", "desc", "lagovad"], required=True)
    parser.add_argument("--baseline-root", required=True)
    parser.add_argument("--baseline-weight", default="")
    parser.add_argument("--sensitivity-weight", default="")
    parser.add_argument("--consistency-weight", default="")
    parser.add_argument("--dataset", choices=["ucf", "xd"], required=True)
    parser.add_argument("--train-list", required=True)
    parser.add_argument("--synthetic-list", required=True)
    parser.add_argument("--test-list", required=True)
    parser.add_argument("--gt-path", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--max-epoch", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, required=True)
    parser.add_argument("--weight-decay", type=float, required=True)
    parser.add_argument("--pseudo-dense-weight", type=float, default=1.0)
    parser.add_argument("--pseudo-mil-weight", type=float, default=1.0)
    parser.add_argument("--pseudo-topk-ratio", type=int, default=4)
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
    output = clean_output(args.out_dir, args.clean)
    checkpoint_path, best_path = output / "checkpoint_last.pth", output / "model_best.pth"
    resume = args.resume or checkpoint_path.exists()
    if args.resume and not checkpoint_path.exists():
        raise FileNotFoundError(f"resume checkpoint does not exist: {checkpoint_path}")
    seed_everything(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() and args.device.startswith("cuda") else "cpu")
    adapter = build_baseline(args, str(device)).to(device)
    groups, trainable_names = parameter_groups(adapter, args.lr)
    optimizer = torch.optim.AdamW(groups, weight_decay=args.weight_decay)
    scheduler = build_scheduler(optimizer, args.baseline, args.dataset, args.max_epoch)
    dataset_args = (
        args.train_list, args.synthetic_list, args.dataset,
        adapter.visual_length, args.baseline,
    )
    loader_options = dict(
        batch_size=args.batch_size, shuffle=True, drop_last=True,
        num_workers=args.num_workers, pin_memory=device.type == "cuda",
    )
    normal_loader = DataLoader(BaselineTrainDataset(*dataset_args, split="normal"), **loader_options)
    abnormal_loader = DataLoader(BaselineTrainDataset(*dataset_args, split="abnormal"), **loader_options)
    synthetic_loader = DataLoader(BaselineTrainDataset(*dataset_args, split="synthetic"), **loader_options)
    steps_per_epoch = min(len(normal_loader), len(abnormal_loader))
    if steps_per_epoch == 0 or len(synthetic_loader) == 0:
        raise RuntimeError("normal, abnormal, and synthetic training loaders must all be non-empty")
    run_config = {
        "method": "semantic_knn_splicing_partial_baseline_adaptation_v1",
        "baseline": args.baseline, "dataset": args.dataset, "lr": args.lr,
        "weight_decay": args.weight_decay,
        "pseudo_dense_weight": args.pseudo_dense_weight,
        "pseudo_mil_weight": args.pseudo_mil_weight,
        "pseudo_topk_ratio": args.pseudo_topk_ratio,
        "train_scope": "native visual-text interaction and scoring heads only",
    }
    (output / "parameter_report.json").write_text(json.dumps({
        **run_config, "trainable_tensors": trainable_names,
        "trainable_parameters": sum(parameter.numel() for group in groups for parameter in group["params"]),
        "clip_trainable_parameters": 0,
    }, indent=2, ensure_ascii=False), encoding="utf-8")
    start_epoch, best, processed = 0, -float("inf"), 0
    if resume:
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        if checkpoint["run_config"] != run_config:
            raise RuntimeError("resume configuration differs from checkpoint")
        adapter.load_state_dict(checkpoint["model_state_dict"], strict=True)
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        start_epoch, best, processed = int(checkpoint["epoch"]) + 1, float(checkpoint["best_metric"]), int(checkpoint["processed_samples"])
    history_path = output / "history.jsonl"

    def payload(epoch: int, metrics: dict, tag: str) -> dict:
        return {
            "epoch": epoch, "best_metric": best, "processed_samples": processed,
            "run_config": run_config, "model_state_dict": adapter.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(), "scheduler_state_dict": scheduler.state_dict(),
            "metrics": metrics, "validation_tag": tag,
            "selection_rule": "UCF frame AUC" if args.dataset == "ucf" else "XD frame AP",
        }

    def validate(epoch: int, tag: str) -> dict:
        nonlocal best
        metrics = official_frame_metrics(adapter, args.test_list, args.gt_path, args.frames_per_snippet, device)
        value = metrics["frame_auc" if args.dataset == "ucf" else "frame_ap"]
        if value > best:
            best = value
            torch.save(payload(epoch, metrics, tag), best_path)
        print(f"validation {tag}: {metrics} | best={best:.6f}", flush=True)
        return metrics

    if not resume:
        adapter.set_train_scope("frozen")
        validate(-1, "author_initialization")
    if start_epoch >= args.max_epoch:
        print(f"baseline training already completed {args.max_epoch} epochs", flush=True)
        return
    next_eval = ((processed // args.dsanet_ucf_eval_samples) + 1) * args.dsanet_ucf_eval_samples
    for epoch in range(start_epoch, args.max_epoch):
        scope = "interaction_heads"
        adapter.set_train_scope("heads")
        adapter.train()
        keep_clip_eval(adapter, args.baseline)
        running = {"total": 0.0, "author": 0.0, "pseudo_dense": 0.0, "pseudo_mil": 0.0}
        synthetic_iterator = iter(synthetic_loader)
        paired = zip(normal_loader, abnormal_loader)
        progress = tqdm(paired, total=steps_per_epoch, desc=f"{args.baseline} {epoch + 1}/{args.max_epoch}", unit="batch")
        for step, (normal_batch, abnormal_batch) in enumerate(progress, 1):
            try:
                synthetic_batch = next(synthetic_iterator)
            except StopIteration:
                synthetic_iterator = iter(synthetic_loader)
                synthetic_batch = next(synthetic_iterator)
            batch = merge_original_batches(normal_batch, abnormal_batch)
            clip = batch["clip"].to(device, non_blocking=True)
            lengths = batch["length"].to(device, non_blocking=True)
            labels = batch["binary_label"].to(device, non_blocking=True)
            output_value = adapter.forward_baseline(clip, lengths)
            author = adapter.original_loss(output_value, labels, list(batch["label_text"]), lengths)

            optimizer.zero_grad(set_to_none=True)
            author.backward()
            synthetic_clip = synthetic_batch["clip"].to(device, non_blocking=True)
            synthetic_lengths = synthetic_batch["length"].to(device, non_blocking=True)
            synthetic_output = adapter.forward_baseline(synthetic_clip, synthetic_lengths)
            pseudo_dense, pseudo_mil = pseudo_supervision_losses(
                synthetic_output.binary_logits, synthetic_batch["dense_label"].to(device),
                synthetic_lengths, synthetic_batch["synthetic"].to(device), args.pseudo_topk_ratio,
            )
            pseudo_loss = (
                args.pseudo_dense_weight * pseudo_dense
                + args.pseudo_mil_weight * pseudo_mil
            )
            pseudo_loss.backward()
            optimizer.step()
            processed += int(labels.numel())
            running["total"] += float(author.detach() + pseudo_loss.detach())
            running["author"] += float(author.detach())
            running["pseudo_dense"] += float(pseudo_dense.detach())
            running["pseudo_mil"] += float(pseudo_mil.detach())
            progress.set_postfix(
                scope=scope,
                loss=f"{running['total'] / step:.4f}",
                dense=f"{running['pseudo_dense'] / step:.4f}",
                mil=f"{running['pseudo_mil'] / step:.4f}",
            )
            if args.baseline == "dsanet" and args.dataset == "ucf" and processed >= next_eval:
                validate(epoch, f"sample_{processed}")
                adapter.train()
                keep_clip_eval(adapter, args.baseline)
                next_eval += args.dsanet_ucf_eval_samples
        scheduler.step()
        metrics = {"selection_deferred_to_fixed_step": True}
        if not (args.baseline == "dsanet" and args.dataset == "ucf"):
            metrics = validate(epoch, f"epoch_{epoch + 1}")
        record = {"epoch": epoch + 1, "scope": scope, **{f"{key}_loss": value / max(1, steps_per_epoch) for key, value in running.items()}, "metrics": metrics, "best_metric": best}
        with history_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        torch.save(payload(epoch, metrics, "epoch_recovery"), checkpoint_path)
        print(json.dumps(record, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
