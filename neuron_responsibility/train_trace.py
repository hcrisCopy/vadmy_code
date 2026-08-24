#!/usr/bin/env python3
"""Train TRACE evidence, then partially unfreeze a released VAD baseline."""

from __future__ import annotations

import argparse
import gc
import json
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
from neuron_responsibility.cacc_data import CACCFeatureDataset, VideoGroupedSampler
from neuron_responsibility.common import clean_output, load_hidden
from neuron_responsibility.trace import (
    TraceNeuronEvidence, TraceThresholds, responsibility_sets,
    trace_pretraining_losses, trace_student_losses,
)
from neuron_responsibility.train_cacc import (
    artifact_tensors, cache_teacher, group_features, merge, pad_chunks,
    preservation_loss, relative_anchor, seed_everything,
)


_VALIDATION_CACHE: dict[str, list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]]] = {}


def metrics(truth: np.ndarray, prediction: np.ndarray) -> dict[str, float]:
    usable = min(len(truth), len(prediction))
    truth, prediction = truth[:usable], prediction[:usable]
    return {
        "frame_auc": float(roc_auc_score(truth, prediction)),
        "frame_ap": float(average_precision_score(truth, prediction)),
        "frames": int(usable),
    }


def validation_cache(csv_path: str, visual_length: int) -> list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
    key = str(Path(csv_path).resolve())
    if key not in _VALIDATION_CACHE:
        frame = pd.read_csv(csv_path); groups = []
        for _, group in tqdm(list(frame.groupby("key", sort=False)), desc="cache TRACE validation", unit="video", leave=False):
            clip, hidden = group_features(group)
            clip_chunks, lengths = pad_chunks(clip, visual_length)
            hidden_chunks, hidden_lengths = pad_chunks(hidden, visual_length)
            if not torch.equal(lengths, hidden_lengths):
                raise RuntimeError("validation CLIP and hidden lengths differ")
            groups.append((clip_chunks, hidden_chunks.half(), lengths))
        _VALIDATION_CACHE[key] = groups
    return _VALIDATION_CACHE[key]


@torch.no_grad()
def validate(
    adapter, evidence: TraceNeuronEvidence, thresholds: TraceThresholds | None,
    csv_path: str, gt_path: str, repeat: int, device: torch.device, description: str,
) -> dict[str, float]:
    adapter.eval(); evidence.eval()
    baseline_scores, semantic_scores, temporal_scores, combined_scores = [], [], [], []
    for clip_chunks, hidden_chunks, lengths in tqdm(
        validation_cache(csv_path, adapter.visual_length), desc=description, unit="video", leave=False
    ):
        length_device = lengths.to(device)
        output = adapter.forward_baseline(clip_chunks.to(device), length_device)
        record = evidence(hidden_chunks.to(device), length_device)
        semantic = torch.sigmoid(record["semantic_logits"])
        temporal = record["temporal_score"]
        if thresholds is None:
            combined = 0.5 * semantic + 0.5 * torch.sigmoid(temporal - temporal.mean())
        else:
            semantic_scaled = torch.sigmoid(
                (semantic - thresholds.semantic_high) / max(0.02, thresholds.semantic_high - thresholds.semantic_low)
            )
            temporal_scaled = torch.sigmoid(
                (temporal - thresholds.temporal_high) / max(0.02, thresholds.temporal_high - thresholds.temporal_low)
            )
            combined = 0.5 * semantic_scaled + 0.5 * temporal_scaled
        for index, length in enumerate(lengths.tolist()):
            baseline_scores.append(torch.sigmoid(output.binary_logits[index, :length]).cpu())
            semantic_scores.append(semantic[index, :length].cpu())
            temporal_scores.append(temporal[index, :length].cpu())
            combined_scores.append(combined[index, :length].cpu())
    truth = np.load(gt_path).astype(np.int64).reshape(-1)
    result = {}
    for name, values in (
        ("baseline", baseline_scores), ("trace_semantic", semantic_scores),
        ("trace_temporal", temporal_scores), ("trace_combined", combined_scores),
    ):
        score = np.repeat(torch.cat(values).numpy(), repeat)
        for key, value in metrics(truth, score).items():
            result[f"{name}_{key}"] = value
    return result


@torch.no_grad()
def calibrate(
    evidence: TraceNeuronEvidence, dataset: CACCFeatureDataset,
    low_quantile: float, high_quantile: float, device: torch.device,
) -> TraceThresholds:
    evidence.eval(); semantic, temporal = [], []
    representatives = dataset.frame.groupby("key", sort=True).first().reset_index()
    for _, row in tqdm(representatives.iterrows(), total=len(representatives), desc="calibrate normal TRACE evidence", unit="video"):
        hidden, _ = load_hidden(str(row["hidden_path"]))
        chunks, lengths = pad_chunks(hidden, dataset.visual_length)
        record = evidence(chunks.to(device), lengths.to(device))
        for index, length in enumerate(lengths.tolist()):
            semantic.append(torch.sigmoid(record["semantic_logits"][index, :length]).cpu())
            temporal.append(record["temporal_score"][index, :length].cpu())
    semantic_np = torch.cat(semantic).numpy(); temporal_np = torch.cat(temporal).numpy()
    return TraceThresholds(
        semantic_low=float(np.quantile(semantic_np, low_quantile)),
        semantic_high=float(np.quantile(semantic_np, high_quantile)),
        temporal_low=float(np.quantile(temporal_np, low_quantile)),
        temporal_high=float(np.quantile(temporal_np, high_quantile)),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="TRACE neuron evidence and partial baseline adaptation.")
    parser.add_argument("--baseline", choices=["dsanet", "desc", "lagovad"], required=True)
    parser.add_argument("--baseline-root", required=True); parser.add_argument("--baseline-weight", default="")
    parser.add_argument("--sensitivity-weight", default=""); parser.add_argument("--consistency-weight", default="")
    parser.add_argument("--dataset", choices=["ucf", "xd"], required=True)
    parser.add_argument("--train-list", required=True); parser.add_argument("--val-list", required=True)
    parser.add_argument("--artifact", required=True); parser.add_argument("--gt-path", required=True)
    parser.add_argument("--teacher-cache", default=""); parser.add_argument("--out-dir", required=True)
    parser.add_argument("--max-epoch", type=int, default=6); parser.add_argument("--evidence-epochs", type=int, default=2)
    parser.add_argument("--temporal-start-epoch", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=64); parser.add_argument("--hidden-width", type=int, default=128)
    parser.add_argument("--active-neurons", type=int, default=96); parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--evidence-lr", type=float, default=5e-5); parser.add_argument("--head-lr", type=float, default=2e-6)
    parser.add_argument("--temporal-lr", type=float, default=5e-7); parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--evidence-normal-weight", type=float, default=0.10)
    parser.add_argument("--evidence-agreement-weight", type=float, default=0.20)
    parser.add_argument("--evidence-smooth-weight", type=float, default=0.001)
    parser.add_argument("--evidence-sparse-weight", type=float, default=0.01)
    parser.add_argument("--pseudo-weight", type=float, default=0.50); parser.add_argument("--ranking-weight", type=float, default=0.20)
    parser.add_argument("--ap-weight", type=float, default=0.10); parser.add_argument("--semantic-weight", type=float, default=0.10)
    parser.add_argument("--event-weight", type=float, default=0.001); parser.add_argument("--preservation-weight", type=float, default=0.20)
    parser.add_argument("--baseline-anchor-weight", type=float, default=0.005)
    parser.add_argument("--normal-low-quantile", type=float, default=0.50); parser.add_argument("--normal-high-quantile", type=float, default=0.95)
    parser.add_argument("--hard-normal-fraction", type=float, default=0.05); parser.add_argument("--grow-steps", type=int, default=8)
    parser.add_argument("--frames-per-snippet", type=int, default=16); parser.add_argument("--eval-samples", type=int, default=1280)
    parser.add_argument("--num-workers", type=int, default=4); parser.add_argument("--cache-videos", type=int, default=8)
    parser.add_argument("--seed", type=int, default=234); parser.add_argument("--device", default="cuda")
    parser.add_argument("--resume", action="store_true"); parser.add_argument("--clean", action="store_true")
    args = parser.parse_args()
    if args.clean and args.resume:
        parser.error("--clean and --resume cannot be combined")
    if not 0 < args.evidence_epochs < args.temporal_start_epoch < args.max_epoch:
        parser.error("require 0 < evidence-epochs < temporal-start-epoch < max-epoch")
    output_dir = clean_output(args.out_dir, args.clean)
    last_path, best_path = output_dir / "checkpoint_last.pth", output_dir / "model_best.pth"
    if last_path.exists() and not args.resume:
        raise RuntimeError("checkpoint exists; use --resume or --clean")
    seed_everything(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() and args.device.startswith("cuda") else "cpu")
    adapter = build_baseline(args, str(device)).to(device)
    evidence = TraceNeuronEvidence.from_artifact(
        artifact_tensors(args.artifact), args.hidden_width, args.active_neurons, args.dropout
    ).to(device)
    adapter.set_train_scope("temporal_heads")
    baseline_parameters = {name: value for name, value in adapter.named_parameters() if value.requires_grad}
    if any("clip" in name.lower() for name in baseline_parameters):
        raise RuntimeError("CLIP must remain frozen")
    initial_baseline = {name: value.detach().clone() for name, value in baseline_parameters.items()}
    heads, temporal = [], []
    for name, parameter in baseline_parameters.items():
        if any(token in name for token in ("classifier", "mlp1", "mlp2", "bin_head", "sim_head")):
            heads.append(parameter)
        else:
            temporal.append(parameter)
    optimizer = torch.optim.AdamW([
        {"params": evidence.parameters(), "lr": args.evidence_lr, "name": "trace_evidence"},
        {"params": heads, "lr": args.head_lr, "name": "baseline_heads"},
        {"params": temporal, "lr": args.temporal_lr, "name": "baseline_temporal"},
    ], weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.max_epoch)
    train_all = CACCFeatureDataset(args.train_list, args.dataset, adapter.visual_length, cache_videos=args.cache_videos)
    normal_set = CACCFeatureDataset(args.train_list, args.dataset, adapter.visual_length, "normal", args.cache_videos)
    abnormal_set = CACCFeatureDataset(args.train_list, args.dataset, adapter.visual_length, "abnormal", args.cache_videos)
    teacher_path = Path(args.teacher_cache) if args.teacher_cache else output_dir / "author_train_logits.pth"
    teacher_adapter = build_baseline(args, str(device)).to(device).eval(); teacher_adapter.requires_grad_(False)
    teacher_logits, teacher_index = cache_teacher(
        teacher_adapter, train_all, teacher_path, args.batch_size * 2, args.num_workers, device
    )
    del teacher_adapter; gc.collect()
    if device.type == "cuda": torch.cuda.empty_cache()
    run_config = {key: value for key, value in vars(args).items() if key not in {"resume", "clean"}}
    parameter_report = {
        "method": evidence.method_name, "baseline": args.baseline, "dataset": args.dataset,
        "score_dependency": "TRACE discovery and teachers do not use baseline anomaly scores",
        "selected_neurons": evidence.neuron_width, "active_neurons": evidence.active_neurons,
        "trace_parameters": sum(value.numel() for value in evidence.parameters()),
        "baseline_trainable_parameters": sum(value.numel() for value in baseline_parameters.values()),
        "clip_trainable_parameters": 0,
    }
    (output_dir / "parameter_report.json").write_text(json.dumps(parameter_report, indent=2), encoding="utf-8")
    print(json.dumps(parameter_report, indent=2), flush=True)

    start_epoch = resume_step = processed = joint_processed = 0
    best = -float("inf"); thresholds: TraceThresholds | None = None
    if args.resume:
        checkpoint = torch.load(last_path, map_location="cpu")
        if checkpoint.get("run_config") != run_config:
            raise RuntimeError("resume configuration differs from checkpoint")
        adapter.load_state_dict(checkpoint["baseline_state_dict"], strict=True)
        evidence.load_state_dict(checkpoint["evidence_state_dict"], strict=True)
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"]); scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        start_epoch = int(checkpoint["epoch"]); resume_step = int(checkpoint.get("next_step", 0))
        processed = int(checkpoint["processed_samples"]); best = float(checkpoint["best_metric"])
        joint_processed = int(checkpoint.get("joint_processed_samples", 0))
        if checkpoint.get("thresholds"):
            thresholds = TraceThresholds.from_dict(checkpoint["thresholds"])

    def payload(epoch: int, next_step: int, validation: dict, tag: str) -> dict:
        return {
            "method": evidence.method_name, "epoch": epoch, "next_step": next_step,
            "processed_samples": processed, "joint_processed_samples": joint_processed,
            "best_metric": best,
            "baseline_state_dict": adapter.state_dict(), "evidence_state_dict": evidence.state_dict(),
            "evidence_config": evidence.config(), "thresholds": thresholds.as_dict() if thresholds else None,
            "optimizer_state_dict": optimizer.state_dict(), "scheduler_state_dict": scheduler.state_dict(),
            "run_config": run_config, "metrics": validation, "validation_tag": tag,
        }

    def run_validation(epoch: int, next_step: int, tag: str) -> dict:
        nonlocal best
        result = validate(
            adapter, evidence, thresholds, args.val_list, args.gt_path,
            args.frames_per_snippet, device, f"TRACE validation {tag}",
        )
        selection = result["baseline_frame_auc" if args.dataset == "ucf" else "baseline_frame_ap"]
        if selection >= best - 1e-12:
            best = selection; torch.save(payload(epoch, next_step, result, tag), best_path)
        torch.save(payload(epoch, next_step, result, tag), last_path)
        with (output_dir / "history.jsonl").open("a", encoding="utf-8") as handle:
            handle.write(json.dumps({"tag": tag, "epoch": epoch, "step": next_step,
                                     "processed": processed, "thresholds": thresholds.as_dict() if thresholds else None,
                                     "joint_processed": joint_processed,
                                     **result}) + "\n")
        print(f"validation {tag}: {json.dumps(result)} best={best:.6f}", flush=True)
        return result

    if not args.resume:
        run_validation(0, 0, "released_author")
    next_eval = (joint_processed // args.eval_samples + 1) * args.eval_samples
    for epoch in range(start_epoch, args.max_epoch):
        evidence_stage = epoch < args.evidence_epochs
        if evidence_stage:
            stage = "score_free_evidence"
            evidence.requires_grad_(True); evidence.train(); adapter.set_train_scope("frozen"); adapter.eval()
        else:
            if thresholds is None:
                thresholds = calibrate(
                    evidence, normal_set, args.normal_low_quantile,
                    args.normal_high_quantile, device,
                )
                print(f"TRACE thresholds: {json.dumps(thresholds.as_dict())}", flush=True)
            evidence.requires_grad_(False); evidence.eval()
            scope = "heads" if epoch < args.temporal_start_epoch else "temporal_heads"
            stage = scope; adapter.set_train_scope(scope); adapter.train()
        common = {"batch_size": args.batch_size, "drop_last": True, "num_workers": args.num_workers,
                  "pin_memory": device.type == "cuda"}
        normal_loader = DataLoader(normal_set, sampler=VideoGroupedSampler(normal_set, args.seed + epoch), **common)
        abnormal_loader = DataLoader(abnormal_set, sampler=VideoGroupedSampler(abnormal_set, args.seed + 100003 + epoch), **common)
        total_steps = min(len(normal_loader), len(abnormal_loader))
        progress = tqdm(zip(normal_loader, abnormal_loader), total=total_steps,
                        desc=f"TRACE {epoch + 1}/{args.max_epoch} {stage}", unit="batch")
        for step, (normal, abnormal) in enumerate(progress, 1):
            if epoch == start_epoch and step <= resume_step:
                continue
            batch = merge(normal, abnormal)
            hidden = batch["hidden"].to(device); lengths = batch["length"].to(device)
            labels = batch["binary_label"].to(device)
            record = evidence(hidden, lengths)
            if evidence_stage:
                losses = trace_pretraining_losses(evidence, record, labels, lengths)
                loss = losses["mil"] + args.evidence_normal_weight * losses["normal"]
                loss = loss + args.evidence_agreement_weight * losses["agreement"]
                loss = loss + args.evidence_smooth_weight * losses["smooth"]
                loss = loss + args.evidence_sparse_weight * losses["sparse"]
                counts = ""
            else:
                clip = batch["clip"].to(device)
                output_value = adapter.forward_baseline(clip, lengths)
                original = adapter.original_loss(output_value, labels, batch["label_text"], lengths)
                targets = class_targets(batch["label_text"], adapter.label_map, device)
                sets = responsibility_sets(record, labels, thresholds, args.grow_steps)
                losses = trace_student_losses(
                    output_value.binary_logits, output_value.semantic_logits, sets,
                    labels, targets, lengths, args.hard_normal_fraction,
                )
                rows = torch.tensor([teacher_index[str(value)] for value in batch["sample_id"]], dtype=torch.long)
                teacher = teacher_logits.index_select(0, rows).to(device)
                preserve = preservation_loss(output_value.binary_logits, teacher, lengths)
                anchor = relative_anchor(baseline_parameters, initial_baseline)
                loss = original + args.pseudo_weight * losses["pseudo_binary"]
                loss = loss + args.ranking_weight * losses["ranking"] + args.ap_weight * losses["ap"]
                loss = loss + args.semantic_weight * losses["semantic"] + args.event_weight * losses["event"]
                loss = loss + args.preservation_weight * preserve + args.baseline_anchor_weight * anchor
                counts = f" seeds={int(losses['seed_count'])} pos={int(losses['positive_count'])} hard={int(losses['hard_normal_count'])}"
            optimizer.zero_grad(set_to_none=True); loss.backward()
            torch.nn.utils.clip_grad_norm_([value for group in optimizer.param_groups for value in group["params"] if value.grad is not None], 5.0)
            optimizer.step(); processed += len(labels)
            if not evidence_stage:
                joint_processed += len(labels)
            progress.set_postfix_str(f"loss={float(loss.detach()):.4f}{counts}")
            if not evidence_stage:
                while args.baseline == "dsanet" and args.dataset == "ucf" and joint_processed >= next_eval:
                    run_validation(epoch, step, f"samples_{next_eval}")
                    adapter.train(); adapter.set_train_scope("heads" if epoch < args.temporal_start_epoch else "temporal_heads")
                    next_eval += args.eval_samples
        if evidence_stage:
            thresholds = calibrate(
                evidence, normal_set, args.normal_low_quantile,
                args.normal_high_quantile, device,
            )
            run_validation(epoch + 1, 0, f"evidence_epoch_{epoch + 1}")
        scheduler.step(); resume_step = 0
        torch.save(payload(epoch + 1, 0, {}, f"epoch_{epoch + 1}_end"), last_path)
    print(f"TRACE training complete; best={best:.6f}; checkpoint={best_path}", flush=True)


if __name__ == "__main__":
    main()
