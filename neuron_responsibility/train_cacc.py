#!/usr/bin/env python3
"""Train CACC with author losses, fixed-sample validation and gradual unfreezing."""

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
from neuron_responsibility.cacc import CrossLayerAnomalyConceptCircuit, cacc_losses
from neuron_responsibility.cacc_data import CACCFeatureDataset, VideoGroupedSampler
from neuron_responsibility.common import clean_output, load_hidden, resample_feature
from neuron_responsibility.model import valid_mask


_VALIDATION_CACHE: dict[str, list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]]] = {}


def seed_everything(seed: int) -> None:
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)


def artifact_tensors(path: str) -> dict[str, torch.Tensor]:
    arrays = np.load(path)
    return {key: torch.from_numpy(arrays[key].astype(np.float32)) for key in arrays.files}


def merge(normal: dict, abnormal: dict) -> dict:
    result = {key: torch.cat([normal[key], abnormal[key]], dim=0) for key in ("clip", "hidden", "length", "binary_label")}
    for key in ("label_text", "key", "sample_id"):
        result[key] = list(normal[key]) + list(abnormal[key])
    return result


def pad_chunks(array: np.ndarray, chunk_length: int) -> tuple[torch.Tensor, torch.Tensor]:
    chunks, lengths = [], []
    for start in range(0, max(1, len(array)), chunk_length):
        part = array[start:start + chunk_length]
        if not len(part):
            continue
        pad = [(0, chunk_length - len(part))] + [(0, 0)] * (part.ndim - 1)
        chunks.append(np.pad(part, pad)); lengths.append(len(part))
    return torch.from_numpy(np.stack(chunks).astype(np.float32)), torch.tensor(lengths, dtype=torch.long)


def group_features(group: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    first = group.iloc[0]
    hidden_native, _ = load_hidden(str(first["hidden_path"]))
    clips, hidden = [], []
    for _, row in group.iterrows():
        clip = np.load(str(row["clip_path"])).astype(np.float32)
        aligned = resample_feature(hidden_native.reshape(len(hidden_native), -1), len(clip)).reshape(
            len(clip), hidden_native.shape[1], hidden_native.shape[2]
        )
        clips.append(clip); hidden.append(aligned)
    return np.concatenate(clips), np.concatenate(hidden)


def frame_metrics(adapter, csv_path: str, gt_path: str, repeat: int, device: torch.device, description: str, conditioned: bool = True) -> dict[str, float]:
    adapter.eval(); scores = []
    cache_key = str(Path(csv_path).resolve())
    if cache_key not in _VALIDATION_CACHE:
        frame = pd.read_csv(csv_path); cached_groups = []
        for _, group in tqdm(list(frame.groupby("key", sort=False)), desc="cache validation features", unit="video", leave=False):
            clip, hidden = group_features(group)
            clip_chunks, lengths = pad_chunks(clip, adapter.visual_length)
            hidden_chunks, hidden_lengths = pad_chunks(hidden, adapter.visual_length)
            if not torch.equal(lengths, hidden_lengths):
                raise RuntimeError("validation modalities have different chunk lengths")
            # The source hidden states are float16. Keeping the cache in that
            # dtype avoids a 2.4 GB duplicate; CACC promotes them to float32.
            cached_groups.append((clip_chunks, hidden_chunks.half(), lengths))
        _VALIDATION_CACHE[cache_key] = cached_groups
    with torch.no_grad():
        for clip_chunks, hidden_chunks, lengths in tqdm(_VALIDATION_CACHE[cache_key], desc=description, unit="video", leave=False):
            lengths_device = lengths.to(device)
            if conditioned:
                output, _ = adapter.forward_conditioned(clip_chunks.to(device), hidden_chunks.to(device), lengths_device)
            else:
                output = adapter.forward_baseline(clip_chunks.to(device), lengths_device)
            scores.extend(torch.sigmoid(output.binary_logits[i, :length]).cpu() for i, length in enumerate(lengths.tolist()))
    prediction = np.repeat(torch.cat(scores).numpy(), repeat)
    truth = np.load(gt_path).astype(np.int64).reshape(-1)
    usable = min(len(truth), len(prediction))
    return {"frame_auc": float(roc_auc_score(truth[:usable], prediction[:usable])),
            "frame_ap": float(average_precision_score(truth[:usable], prediction[:usable])), "frames": int(usable)}


def cache_teacher(adapter, dataset: CACCFeatureDataset, path: Path, batch_size: int, workers: int, device: torch.device) -> tuple[torch.Tensor, dict[str, int]]:
    if path.exists():
        value = torch.load(path, map_location="cpu")
        return value["logits"].float(), {str(key): i for i, key in enumerate(value["sample_ids"])}
    loader = DataLoader(dataset, batch_size=batch_size, sampler=VideoGroupedSampler(dataset, 0), num_workers=workers, pin_memory=device.type == "cuda")
    logits, sample_ids = [], []
    adapter.eval()
    with torch.no_grad():
        for batch in tqdm(loader, desc="cache released baseline logits", unit="batch"):
            output = adapter.forward_baseline(batch["clip"].to(device), batch["length"].to(device))
            logits.append(output.binary_logits.cpu().half()); sample_ids.extend(map(str, batch["sample_id"]))
    result = {"logits": torch.cat(logits), "sample_ids": sample_ids}
    path.parent.mkdir(parents=True, exist_ok=True); torch.save(result, path)
    return result["logits"].float(), {key: i for i, key in enumerate(sample_ids)}


def preservation_loss(logits: torch.Tensor, teacher: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
    mask = valid_mask(lengths, logits.shape[1], logits.dtype)
    loss = F.binary_cross_entropy_with_logits(logits, torch.sigmoid(teacher), reduction="none")
    return (loss * mask).sum() / mask.sum().clamp_min(1.0)


def relative_anchor(parameters: dict[str, torch.nn.Parameter], initial: dict[str, torch.Tensor]) -> torch.Tensor:
    terms = []
    for name, parameter in parameters.items():
        if parameter.requires_grad:
            denominator = initial[name].square().mean().clamp_min(1e-8)
            terms.append((parameter - initial[name]).square().mean() / denominator)
    return torch.stack(terms).mean() if terms else next(iter(parameters.values())).sum() * 0.0


def main() -> None:
    parser = argparse.ArgumentParser(description="Cross-layer anomaly concept circuit training.")
    parser.add_argument("--baseline", choices=["dsanet", "desc", "lagovad"], required=True)
    parser.add_argument("--baseline-root", required=True); parser.add_argument("--baseline-weight", default="")
    parser.add_argument("--sensitivity-weight", default=""); parser.add_argument("--consistency-weight", default="")
    parser.add_argument("--dataset", choices=["ucf", "xd"], required=True)
    parser.add_argument("--train-list", required=True); parser.add_argument("--val-list", required=True)
    parser.add_argument("--artifact", required=True); parser.add_argument("--gt-path", required=True)
    parser.add_argument("--teacher-cache", default=""); parser.add_argument("--out-dir", required=True)
    parser.add_argument("--max-epoch", type=int, default=10)
    parser.add_argument("--temporal-start-epoch", type=int, default=2); parser.add_argument("--reference-start-epoch", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=64); parser.add_argument("--concept-width", type=int, default=64)
    parser.add_argument("--temporal-kernel", type=int, default=5); parser.add_argument("--semantic-temperature", type=float, default=0.07)
    parser.add_argument("--max-residual-scale", type=float, default=0.25)
    parser.add_argument("--circuit-lr", type=float, default=5e-5); parser.add_argument("--anchor-lr", type=float, default=1e-5)
    parser.add_argument("--head-lr", type=float, default=1e-5); parser.add_argument("--temporal-lr", type=float, default=1e-6)
    parser.add_argument("--reference-lr", type=float, default=1e-6); parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--mil-weight", type=float, default=0.10); parser.add_argument("--normal-weight", type=float, default=0.10)
    parser.add_argument("--compact-weight", type=float, default=0.01); parser.add_argument("--smooth-weight", type=float, default=0.01)
    parser.add_argument("--layer-sparse-weight", type=float, default=0.001); parser.add_argument("--semantic-anchor-weight", type=float, default=0.01)
    parser.add_argument("--preservation-weight", type=float, default=0.50); parser.add_argument("--baseline-anchor-weight", type=float, default=0.01)
    parser.add_argument("--frames-per-snippet", type=int, default=16); parser.add_argument("--dsanet-ucf-eval-samples", type=int, default=1280)
    parser.add_argument("--num-workers", type=int, default=4); parser.add_argument("--cache-videos", type=int, default=8)
    parser.add_argument("--seed", type=int, default=234); parser.add_argument("--device", default="cuda")
    parser.add_argument("--resume", action="store_true"); parser.add_argument("--clean", action="store_true")
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
    adapter = build_baseline(args, str(device)).to(device)
    circuit = CrossLayerAnomalyConceptCircuit.from_artifact(
        artifact_tensors(args.artifact), args.concept_width, args.temporal_kernel,
        args.semantic_temperature, args.max_residual_scale,
    ).to(device)
    adapter.attach_pre_temporal_conditioner(circuit)
    maximum_scope = "evidence_adaptation" if args.baseline == "dsanet" else "temporal_heads"
    adapter.set_train_scope(maximum_scope)
    baseline_parameters = {name: value for name, value in adapter.named_parameters() if value.requires_grad and not name.startswith("pre_temporal_conditioner.")}
    if any("clip" in name.lower() for name in baseline_parameters):
        raise RuntimeError("CLIP must remain frozen")
    initial_baseline = {name: value.detach().clone() for name, value in baseline_parameters.items()}
    heads, temporal, reference = [], [], []
    for name, parameter in baseline_parameters.items():
        if "video_anomaly_refiner" in name:
            reference.append(parameter)
        elif any(token in name for token in ("classifier", "mlp1", "mlp2", "bin_head", "sim_head")):
            heads.append(parameter)
        else:
            temporal.append(parameter)
    anchor_parameters = [circuit.normal_anchor_delta, circuit.abnormal_anchor_delta]
    anchor_ids = {id(value) for value in anchor_parameters}
    circuit_parameters = [value for value in circuit.parameters() if id(value) not in anchor_ids]
    groups = [{"params": circuit_parameters, "lr": args.circuit_lr, "name": "circuit"},
              {"params": anchor_parameters, "lr": args.anchor_lr, "name": "semantic_anchors"},
              {"params": heads, "lr": args.head_lr, "name": "heads"},
              {"params": temporal, "lr": args.temporal_lr, "name": "temporal"}]
    if reference:
        groups.append({"params": reference, "lr": args.reference_lr, "name": "reference"})
    optimizer = torch.optim.AdamW(groups, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.max_epoch)

    train_all = CACCFeatureDataset(args.train_list, args.dataset, adapter.visual_length, cache_videos=args.cache_videos)
    teacher_path = Path(args.teacher_cache) if args.teacher_cache else output / "author_train_logits.pth"
    teacher_adapter = build_baseline(args, str(device)).to(device).eval(); teacher_adapter.requires_grad_(False)
    teacher_logits, teacher_index = cache_teacher(teacher_adapter, train_all, teacher_path, args.batch_size * 2, args.num_workers, device)
    del teacher_adapter; gc.collect()
    if device.type == "cuda": torch.cuda.empty_cache()
    normal_set = CACCFeatureDataset(args.train_list, args.dataset, adapter.visual_length, "normal", args.cache_videos)
    abnormal_set = CACCFeatureDataset(args.train_list, args.dataset, adapter.visual_length, "abnormal", args.cache_videos)
    missing_teacher = sorted(set(map(str, train_all.frame["clip_path"])) - set(teacher_index))
    if missing_teacher:
        raise RuntimeError(f"teacher cache misses {len(missing_teacher)} samples; first={missing_teacher[0]}")
    run_config = {key: value for key, value in vars(args).items() if key not in {"resume", "clean"}}
    report = {"method": circuit.method_name, "baseline": args.baseline, "dataset": args.dataset,
              "selection": "UCF frame AUC / XD frame AP at author fixed-sample intervals",
              "score_dependency": "none: normal statistics + CLIP text anchors + video labels",
              "parameters": {group["name"]: sum(value.numel() for value in group["params"]) for group in groups},
              "clip_trainable_parameters": 0}
    (output / "parameter_report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False), flush=True)

    start_epoch, resume_step, best, processed = 0, 0, -float("inf"), 0
    if args.resume:
        checkpoint = torch.load(last_path, map_location="cpu")
        if checkpoint.get("run_config") != run_config:
            raise RuntimeError("resume configuration differs from checkpoint")
        adapter.load_state_dict(checkpoint["model_state_dict"], strict=True)
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"]); scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        start_epoch, resume_step = int(checkpoint["epoch"]), int(checkpoint.get("next_step", 0))
        best, processed = float(checkpoint["best_metric"]), int(checkpoint["processed_samples"])

    def payload(epoch: int, next_step: int, metrics: dict, tag: str) -> dict:
        return {"method": circuit.method_name, "epoch": epoch, "next_step": next_step,
                "best_metric": best, "processed_samples": processed,
                "model_state_dict": adapter.state_dict(), "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(), "run_config": run_config,
                "circuit_config": circuit.config(), "metrics": metrics, "validation_tag": tag}

    def validate(epoch: int, next_step: int, tag: str) -> dict:
        nonlocal best
        metrics = frame_metrics(adapter, args.val_list, args.gt_path, args.frames_per_snippet, device, f"validation {tag}")
        value = metrics["frame_auc" if args.dataset == "ucf" else "frame_ap"]
        if value > best:
            best = value
            torch.save(payload(epoch, next_step, metrics, tag), best_path)
        torch.save(payload(epoch, next_step, metrics, tag), last_path)
        with (output / "history.jsonl").open("a", encoding="utf-8") as handle:
            handle.write(json.dumps({"tag": tag, "epoch": epoch, "step": next_step, "processed": processed, **metrics}) + "\n")
        print(f"validation {tag}: {json.dumps(metrics)} best={best:.6f}", flush=True)
        return metrics

    if not args.resume:
        validate(0, 0, "released_author")
    next_eval = (processed // args.dsanet_ucf_eval_samples + 1) * args.dsanet_ucf_eval_samples
    weights = {"mil": args.mil_weight, "normal": args.normal_weight, "compact": args.compact_weight,
               "smooth": args.smooth_weight, "layer_sparse": args.layer_sparse_weight}
    for epoch in range(start_epoch, args.max_epoch):
        if epoch < args.temporal_start_epoch:
            stage, scope = "heads", "heads"
        elif epoch < args.reference_start_epoch or args.baseline != "dsanet":
            stage, scope = "temporal_heads", "temporal_heads"
        else:
            stage, scope = "normal_reference", "evidence_adaptation"
        adapter.set_train_scope(scope); adapter.train()
        common = {"batch_size": args.batch_size, "drop_last": True, "num_workers": args.num_workers, "pin_memory": device.type == "cuda"}
        normal_loader = DataLoader(normal_set, sampler=VideoGroupedSampler(normal_set, args.seed + epoch), **common)
        abnormal_loader = DataLoader(abnormal_set, sampler=VideoGroupedSampler(abnormal_set, args.seed + 100003 + epoch), **common)
        total_steps = min(len(normal_loader), len(abnormal_loader))
        progress = tqdm(zip(normal_loader, abnormal_loader), total=total_steps, desc=f"CACC {epoch + 1}/{args.max_epoch} {stage}", unit="batch")
        for step, (normal, abnormal) in enumerate(progress, 1):
            if epoch == start_epoch and step <= resume_step:
                continue
            batch = merge(normal, abnormal)
            clip = batch["clip"].to(device); hidden = batch["hidden"].to(device)
            lengths = batch["length"].to(device); labels = batch["binary_label"].to(device)
            output_value, records = adapter.forward_conditioned(clip, hidden, lengths)
            original = adapter.original_loss(output_value, labels, batch["label_text"], lengths)
            aux = cacc_losses(records, labels, lengths)
            teacher_rows = torch.tensor([teacher_index[str(value)] for value in batch["sample_id"]], dtype=torch.long)
            teacher = teacher_logits.index_select(0, teacher_rows).to(device)
            preserve = preservation_loss(output_value.binary_logits, teacher, lengths)
            semantic_anchor = circuit.normal_anchor_delta.square().mean() + circuit.abnormal_anchor_delta.square().mean()
            baseline_anchor = relative_anchor(baseline_parameters, initial_baseline)
            loss = original + args.preservation_weight * preserve + args.semantic_anchor_weight * semantic_anchor + args.baseline_anchor_weight * baseline_anchor
            for name, weight in weights.items():
                if name in aux:
                    loss = loss + weight * aux[name]
            optimizer.zero_grad(set_to_none=True); loss.backward()
            torch.nn.utils.clip_grad_norm_(adapter.parameters(), 5.0); optimizer.step()
            processed += len(labels)
            progress.set_postfix(loss=f"{float(loss.detach()):.4f}", gate=f"{float(records[0]['gate'].mean().detach()):.3f}", gain=f"{float(records[0]['gain'].detach()):.4f}")
            while args.baseline == "dsanet" and args.dataset == "ucf" and processed >= next_eval:
                validate(epoch, step, f"samples_{next_eval}"); adapter.train(); adapter.set_train_scope(scope)
                next_eval += args.dsanet_ucf_eval_samples
        scheduler.step(); resume_step = 0
        torch.save(payload(epoch + 1, 0, {}, f"epoch_{epoch + 1}_end"), last_path)
    print(f"training complete; best={best:.6f}; checkpoint={best_path}", flush=True)


if __name__ == "__main__":
    main()
