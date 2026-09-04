from __future__ import annotations

import argparse
import hashlib
import json
import random
import subprocess
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

from vin_vad.data import (
    AuditorTrainingDataset,
    HostScoreTrainingDataset,
    collate_auditor_training,
    collate_host_score_training,
)
from vin_vad.witness_losses import variant_objective
from vin_vad.witness_model import build_witness_variant


@torch.no_grad()
def fit_role_disentangled_reference(
    model: torch.nn.Module,
    dataset: AuditorTrainingDataset,
    normal_indices: list[int],
    device: torch.device,
) -> dict[str, int]:
    """Fit role definitions and their absolute normal calibration on training bags."""
    neurons = model.expert.neurons
    total = torch.zeros(neurons.layers, neurons.dimensions, dtype=torch.float64, device=device)
    square = torch.zeros_like(total)
    snippet_count = 0
    for index in tqdm(normal_indices, desc="fit normal neuron moments", unit="video"):
        hidden = dataset[index]["hidden"].to(device, non_blocking=True)
        normalized = torch.nn.functional.layer_norm(hidden, (neurons.dimensions,)).double()
        total += normalized.sum(dim=0)
        square += normalized.square().sum(dim=0)
        snippet_count += len(normalized)
    mean = total / max(snippet_count, 1)
    variance = (square / max(snippet_count, 1) - mean.square()).clamp_min(1e-4)
    standard_deviation = variance.sqrt()

    class_sum = torch.zeros(
        2, 2, neurons.layers, neurons.dimensions, dtype=torch.float64, device=device
    )
    class_square = torch.zeros_like(class_sum)
    class_count = torch.zeros(2, dtype=torch.float64, device=device)
    residual_sum = torch.zeros_like(class_sum)
    residual_square = torch.zeros_like(class_sum)
    residual_count = torch.zeros(2, dtype=torch.float64, device=device)
    for index in tqdm(range(len(dataset)), desc="rank role neurons", unit="video"):
        item = dataset[index]
        hidden = item["hidden"].to(device, non_blocking=True)
        normalized = torch.nn.functional.layer_norm(hidden, (neurons.dimensions,)).double()
        deviation = (normalized - mean) / standard_deviation
        tail_count = min(len(deviation), max(1, len(deviation) // 16 + 1))
        summary = torch.stack(
            [
                torch.topk(deviation, tail_count, dim=0).values.mean(dim=0),
                torch.topk(-deviation, tail_count, dim=0).values.mean(dim=0),
            ]
        )
        label = int(item["label"])
        class_sum[label] += summary
        class_square[label] += summary.square()
        class_count[label] += 1
        host_score = item["host_score"].to(device, non_blocking=True).double()
        point_residual = (
            1.0 - host_score if label == 1 else host_score
        ).clamp(0.0, 1.0)
        residual_deviation = deviation * point_residual[:, None, None]
        residual_summary = torch.stack(
            [
                torch.topk(residual_deviation, tail_count, dim=0).values.mean(dim=0),
                torch.topk(-residual_deviation, tail_count, dim=0).values.mean(dim=0),
            ]
        )
        residual_sum[label] += residual_summary
        residual_square[label] += residual_summary.square()
        residual_count[label] += 1

    def class_effect(
        total: torch.Tensor,
        square_total: torch.Tensor,
        count: torch.Tensor,
    ) -> torch.Tensor:
        mean_value = total / count[:, None, None, None].clamp_min(1e-6)
        variance_value = (
            square_total / count[:, None, None, None].clamp_min(1e-6)
            - mean_value.square()
        ).clamp_min(1e-6)
        return torch.relu(
            (mean_value[1] - mean_value[0])
            / torch.sqrt(variance_value[0] + variance_value[1])
        )

    def role_definition(effect: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        best_effect, best_direction = effect.max(dim=0)
        selected = torch.topk(best_effect, active_per_layer, dim=-1).indices
        mask = torch.zeros_like(best_effect).scatter_(-1, selected, 1.0)
        direction = torch.where(best_direction == 0, 1.0, -1.0)
        weight = best_effect * mask
        weight = weight / (
            weight.sum(dim=-1, keepdim=True) / active_per_layer
        ).clamp_min(1e-6)
        return mask, direction, weight

    active_per_layer = min(neurons.active, neurons.dimensions)
    normal_mask, normal_direction, normal_weight = role_definition(
        class_effect(class_sum, class_square, class_count)
    )
    primary_mask, primary_direction, primary_weight = role_definition(
        class_effect(residual_sum, residual_square, residual_count)
    )

    role_weight = normal_mask * normal_weight
    normal_scores = []
    for index in tqdm(normal_indices, desc="calibrate normality score", unit="video"):
        hidden = dataset[index]["hidden"].to(device, non_blocking=True)
        normalized = torch.nn.functional.layer_norm(hidden, (neurons.dimensions,)).double()
        deviation = (normalized - mean) / standard_deviation
        directional = torch.relu(deviation * normal_direction)
        layer_score = (directional * role_weight).sum(dim=-1) / role_weight.sum(
            dim=-1
        ).clamp_min(1e-6)
        score = layer_score.mean(dim=-1)
        normal_scores.append(score)
    normal_score = torch.cat(normal_scores)
    score_threshold = torch.quantile(normal_score, 0.95)
    score_std = normal_score.std(unbiased=False).clamp_min(1e-2)
    neurons.set_normal_role(
        mean.float(),
        standard_deviation.float(),
        normal_mask.float(),
        normal_direction.float(),
        normal_weight.float(),
        score_threshold.float(),
        score_std.float(),
    )
    neurons.set_primary_role(
        primary_mask.float(),
        primary_direction.float(),
        primary_weight.float(),
    )
    return {
        "normal_reference_snippets": snippet_count,
        "normal_role_neurons_per_layer": active_per_layer,
        "primary_role_neurons_per_layer": active_per_layer,
    }


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def git_commit() -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"], text=True, encoding="utf-8"
    ).strip()


def atomic_torch_save(value: dict[str, object], path: Path) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(value, temporary)
    temporary.replace(path)


def ensure_training_paths(manifest: Path, output: Path) -> None:
    if "test" in manifest.name.lower():
        raise ValueError("training refuses test manifests")
    if "vadmy_data" not in output.resolve().parts:
        raise ValueError("output must be inside sibling vadmy_data")


def comparable_configuration(config: dict[str, object]) -> dict[str, object]:
    """Git provenance is recorded but is not a training hyperparameter."""
    derived = {
        "git_commit",
        "normal_reference_snippets",
        "normal_role_neurons_per_layer",
        "primary_role_neurons_per_layer",
    }
    comparable = {key: value for key, value in config.items() if key not in derived}
    comparable.setdefault("variant", "w6")
    comparable.setdefault("num_workers", 0)
    comparable.setdefault("cache_training_data", False)
    comparable.setdefault("retain_epoch_checkpoints", False)
    return comparable


def balanced_indices(frame: pd.DataFrame, per_class: int) -> tuple[list[int], list[int]]:
    labels = frame["binary_label"].astype(int)
    normal = frame.index[labels == 0].tolist()
    abnormal = frame.index[labels == 1].tolist()
    if per_class > 0:
        normal = normal[:per_class]
        abnormal = abnormal[:per_class]
    if not normal or not abnormal:
        raise ValueError("training needs both normal and abnormal videos")
    return normal, abnormal


def class_loader(
    dataset: AuditorTrainingDataset | HostScoreTrainingDataset,
    indices: list[int],
    batch_size: int,
    seed: int,
    num_workers: int,
) -> DataLoader:
    generator = torch.Generator().manual_seed(seed)
    return DataLoader(
        Subset(dataset, indices),
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=num_workers,
        pin_memory=True,
        generator=generator,
        collate_fn=(
            collate_host_score_training
            if isinstance(dataset, HostScoreTrainingDataset)
            else collate_auditor_training
        ),
    )


def merge_balanced_batches(
    normal: dict[str, object], abnormal: dict[str, object]
) -> dict[str, object]:
    maximum = max(normal["host_score"].shape[1], abnormal["host_score"].shape[1])

    def pad(value: torch.Tensor, target: int) -> torch.Tensor:
        if value.shape[1] == target:
            return value
        shape = list(value.shape)
        shape[1] = target - value.shape[1]
        return torch.cat([value, value.new_zeros(shape)], dim=1)

    merged = {
        "host_score": torch.cat(
            [pad(normal["host_score"], maximum), pad(abnormal["host_score"], maximum)]
        ),
        "mask": torch.cat([pad(normal["mask"], maximum), pad(abnormal["mask"], maximum)]),
        "labels": torch.cat([normal["labels"], abnormal["labels"]]),
    }
    if "hidden" in normal and "hidden" in abnormal:
        merged["hidden"] = torch.cat(
            [pad(normal["hidden"], maximum), pad(abnormal["hidden"], maximum)]
        )
    return merged


def checkpoint_payload(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    epoch: int,
    history: list[dict[str, float | int]],
    config: dict[str, object],
) -> dict[str, object]:
    return {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "epoch": epoch,
        "history": history,
        "config": config,
        "python_rng": random.getstate(),
        "numpy_rng": np.random.get_state(),
        "torch_rng": torch.get_rng_state(),
        "cuda_rng": torch.cuda.get_rng_state_all(),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the single-graph Witness-VAD model")
    parser.add_argument("--dataset", required=True, choices=("ucf", "xd"))
    parser.add_argument("--variant", required=True, choices=("w1", "w2", "w6"))
    parser.add_argument("--train-manifest", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--epochs", type=int, required=True)
    parser.add_argument("--stop-after-epoch", type=int, default=0)
    parser.add_argument("--batch-size", type=int, required=True)
    parser.add_argument("--maximum-length", type=int, required=True)
    parser.add_argument("--videos-per-class", type=int, default=0)
    parser.add_argument("--num-workers", type=int, required=True)
    parser.add_argument("--cache-training-data", action="store_true")
    parser.add_argument("--active-neurons", type=int, required=True)
    parser.add_argument("--temporal-width", type=int, required=True)
    parser.add_argument("--eta-normal", type=float, required=True)
    parser.add_argument("--eta-anomaly", type=float, required=True)
    parser.add_argument("--lambda-witness", type=float, required=True)
    parser.add_argument("--lambda-final", type=float, required=True)
    parser.add_argument("--lambda-normal", type=float, required=True)
    parser.add_argument("--lambda-sparse", type=float, required=True)
    parser.add_argument("--rank-margin", type=float, required=True)
    parser.add_argument("--rank-weight", type=float, required=True)
    parser.add_argument("--smooth-weight", type=float, required=True)
    parser.add_argument("--learning-rate", type=float, required=True)
    parser.add_argument("--weight-decay", type=float, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--device", required=True)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--retain-epoch-checkpoints", action="store_true")
    args = parser.parse_args()

    if args.batch_size < 2 or args.batch_size % 2:
        raise ValueError("batch-size must be an even number >=2")
    if args.epochs < 1 or not 0 <= args.stop_after_epoch <= args.epochs:
        raise ValueError("invalid epoch configuration")
    manifest = Path(args.train_manifest)
    output = Path(args.out_dir)
    ensure_training_paths(manifest, output)
    output.mkdir(parents=True, exist_ok=True)
    checkpoints = output / "checkpoints"
    checkpoints.mkdir(exist_ok=True)
    seed_everything(args.seed)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA training requested but CUDA is unavailable")

    configuration = {
        key: value
        for key, value in vars(args).items()
        if key not in {"resume", "stop_after_epoch", "out_dir"}
    }
    configuration.update(
        {
            "manifest_sha256": file_sha256(manifest),
            "git_commit": git_commit(),
            "selection_policy": (
                "external_test_primary_metric_best"
                if args.retain_epoch_checkpoints
                else "last_checkpoint_only"
            ),
            "test_data_used": False,
            "optimizer_count": 1,
        }
    )
    config_path = output / "config.json"
    last_path = checkpoints / "last.pt"
    if last_path.exists() and not args.resume:
        raise RuntimeError("checkpoint exists; use --resume or clean the exact run directory")
    if config_path.exists() and args.resume:
        saved_configuration = json.loads(config_path.read_text(encoding="utf-8"))
        if comparable_configuration(saved_configuration) != comparable_configuration(configuration):
            raise RuntimeError("resume configuration differs from the saved run")
        configuration = saved_configuration
    else:
        config_path.write_text(json.dumps(configuration, indent=2), encoding="utf-8")

    dataset = (
        HostScoreTrainingDataset(str(manifest), maximum_length=args.maximum_length)
        if args.variant == "w1"
        else AuditorTrainingDataset(str(manifest), maximum_length=args.maximum_length)
    )
    if args.cache_training_data and isinstance(dataset, AuditorTrainingDataset):
        dataset.preload(
            tqdm(
                range(len(dataset)),
                desc=f"{args.dataset} preload hidden states",
                unit="video",
            )
        )
    normal_indices, abnormal_indices = balanced_indices(
        dataset.frame, args.videos_per_class
    )
    half_batch = args.batch_size // 2
    if min(len(normal_indices), len(abnormal_indices)) < half_batch:
        raise ValueError("not enough videos per class for one balanced batch")

    model = build_witness_variant(
        args.variant,
        active=args.active_neurons,
        temporal_width=args.temporal_width,
        eta_normal=args.eta_normal,
        eta_anomaly=args.eta_anomaly,
    ).to(device)
    if (
        args.variant == "w6"
        and not (args.resume and last_path.exists())
        and isinstance(dataset, AuditorTrainingDataset)
    ):
        configuration.update(
            fit_role_disentangled_reference(model, dataset, normal_indices, device)
        )
        config_path.write_text(json.dumps(configuration, indent=2), encoding="utf-8")
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs
    )
    history: list[dict[str, float | int]] = []
    start_epoch = 0
    if args.resume and last_path.exists():
        saved = torch.load(last_path, map_location=device, weights_only=False)
        if comparable_configuration(saved["config"]) != comparable_configuration(configuration):
            raise RuntimeError("checkpoint configuration differs from this run")
        model.load_state_dict(saved["model"])
        optimizer.load_state_dict(saved["optimizer"])
        scheduler.load_state_dict(saved["scheduler"])
        history = list(saved["history"])
        start_epoch = int(saved["epoch"])
        random.setstate(saved["python_rng"])
        np.random.set_state(saved["numpy_rng"])
        torch.set_rng_state(saved["torch_rng"].cpu())
        torch.cuda.set_rng_state_all([state.cpu() for state in saved["cuda_rng"]])
        print(f"resume from completed epoch {start_epoch}", flush=True)

    for epoch in range(start_epoch, args.epochs):
        if args.stop_after_epoch and epoch >= args.stop_after_epoch:
            break
        normal_loader = class_loader(
            dataset,
            normal_indices,
            half_batch,
            args.seed + 1000 * epoch,
            args.num_workers,
        )
        abnormal_loader = class_loader(
            dataset,
            abnormal_indices,
            half_batch,
            args.seed + 1000 * epoch + 1,
            args.num_workers,
        )
        steps = min(len(normal_loader), len(abnormal_loader))
        if steps == 0:
            raise RuntimeError("balanced data loaders produced zero batches")
        model.train()
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        started = time.perf_counter()
        totals = {
            "total": 0.0,
            "video": 0.0,
            "witness_mil": 0.0,
            "final_mil": 0.0,
            "dense_normal": 0.0,
            "sparse": 0.0,
        }
        lr_start = float(optimizer.param_groups[0]["lr"])
        progress = tqdm(
            zip(normal_loader, abnormal_loader),
            total=steps,
            desc=f"{args.dataset} Witness epoch {epoch + 1}/{args.epochs}",
            unit="batch",
        )
        for step, (normal, abnormal) in enumerate(progress, start=1):
            batch = merge_balanced_batches(normal, abnormal)
            hidden = (
                None
                if args.variant == "w1"
                else batch["hidden"].to(device, non_blocking=True)
            )
            host_score = batch["host_score"].to(device, non_blocking=True)
            validity = batch["mask"].to(device, non_blocking=True)
            labels = batch["labels"].to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            result = (
                model(host_score, validity)
                if args.variant == "w1"
                else model(hidden, host_score, validity)
            )
            expert = getattr(model, "expert", None)
            sparsity = (
                None
                if expert is None
                else expert.neurons.sparsity_surrogate()
            )
            losses = variant_objective(
                args.variant,
                result,
                host_score,
                validity,
                labels,
                sparsity,
                lambda_witness=args.lambda_witness,
                lambda_final=args.lambda_final,
                lambda_normal=args.lambda_normal,
                lambda_sparse=args.lambda_sparse,
                rank_weight=args.rank_weight,
                rank_margin=args.rank_margin,
                smooth_weight=args.smooth_weight,
            )
            losses["total"].backward()
            optimizer.step()
            for name in totals:
                totals[name] += float(losses[name].detach())
            progress.set_postfix(
                total=f"{totals['total'] / step:.4f}",
                video=f"{totals['video'] / step:.3f}",
                witness=f"{totals['witness_mil'] / step:.3f}",
                final=f"{totals['final_mil'] / step:.3f}",
                normal=f"{totals['dense_normal'] / step:.3f}",
                sparse=f"{totals['sparse'] / step:.3f}",
            )
        scheduler.step()
        elapsed = time.perf_counter() - started
        record: dict[str, float | int] = {
            "epoch": epoch + 1,
            **{name: value / steps for name, value in totals.items()},
            "lr_start": lr_start,
            "lr_end": float(optimizer.param_groups[0]["lr"]),
            "seconds": elapsed,
            "peak_gpu_memory_mb": (
                torch.cuda.max_memory_allocated(device) / (1024 * 1024)
                if device.type == "cuda"
                else 0.0
            ),
            "steps": steps,
        }
        history.append(record)
        (output / "history.json").write_text(
            json.dumps(history, indent=2), encoding="utf-8"
        )
        atomic_torch_save(
            checkpoint_payload(
                model, optimizer, scheduler, epoch + 1, history, configuration
            ),
            last_path,
        )
        if args.retain_epoch_checkpoints:
            atomic_torch_save(
                checkpoint_payload(
                    model, optimizer, scheduler, epoch + 1, history, configuration
                ),
                checkpoints / f"epoch_{epoch + 1:03d}.pt",
            )
        print(json.dumps(record, indent=2), flush=True)

    completed_epochs = len(history)
    complete = completed_epochs == args.epochs
    metrics = {
        "status": "complete" if complete else "planned_stop",
        "completed_epochs": completed_epochs,
        "target_epochs": args.epochs,
        "checkpoint": str(last_path),
        "optimizer_count": 1,
        "test_data_used": False,
        "selection_policy": configuration["selection_policy"],
        "epoch_checkpoints_retained": args.retain_epoch_checkpoints,
        "history": history,
    }
    (output / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    checkpoint_note = (
        "all epoch checkpoints retained for external baseline-compatible selection"
        if args.retain_epoch_checkpoints
        else "last checkpoint only"
    )
    summary = f"""# Witness-VAD training run

- Status: **{metrics['status']}**
- Completed epochs: {completed_epochs}/{args.epochs}
- Dataset: {args.dataset} training split only
- Checkpoint: `checkpoints/last.pt`
- Training output: {checkpoint_note}; test data used during training: false
"""
    (output / "summary.md").write_text(summary, encoding="utf-8")
    print(json.dumps(metrics, indent=2), flush=True)


if __name__ == "__main__":
    main()
