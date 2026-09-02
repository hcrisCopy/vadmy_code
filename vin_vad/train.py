from __future__ import annotations

import argparse
import hashlib
import json
import random
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

from vin_vad.context_predictor import MaskedContextPredictor, gaussian_nll
from vin_vad.data import AuditorTrainingDataset, collate_auditor_training
from vin_vad.host_auditor import NormalQCalibrator, TwoAxisHostAuditor
from vin_vad.losses import asymmetric_mil_loss, correction_budget_loss
from vin_vad.model import CVAVADCorrectionModel
from vin_vad.violation_field import ViolationField


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def ensure_output_path(path: Path) -> None:
    if "vadmy_data" not in path.resolve().parts:
        raise ValueError("out-dir must be inside the sibling vadmy_data directory")


def atomic_torch_save(value: dict[str, object], path: Path) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(value, temporary)
    temporary.replace(path)


def build_predictor(config: dict[str, object]) -> MaskedContextPredictor:
    return MaskedContextPredictor(
        model_width=int(config["model_width"]),
        input_rank=int(config["input_rank"]),
        head_rank=int(config["head_rank"]),
        attention_heads=int(config["attention_heads"]),
        attention_layers=int(config["attention_layers"]),
        guard_radius=int(config["guard_radius"]),
        dropout=float(config["dropout"]),
        sigma_min=float(config["sigma_min"]),
        sigma_max=float(config["sigma_max"]),
    )


def build_model(args: argparse.Namespace) -> tuple[CVAVADCorrectionModel, dict[str, str]]:
    b1_dir = Path(args.b1_dir)
    b1_config_path = b1_dir / "config.json"
    b1_checkpoint_path = b1_dir / "context_predictor_best.pt"
    b2_checkpoint_path = Path(args.b2_checkpoint)
    predictor_config = json.loads(b1_config_path.read_text(encoding="utf-8"))
    predictor = build_predictor(predictor_config)
    predictor_checkpoint = torch.load(
        b1_checkpoint_path, map_location="cpu", weights_only=False
    )
    predictor.load_state_dict(predictor_checkpoint["model"])

    field = ViolationField(
        delta=args.delta,
        statistics_momentum=args.statistics_momentum,
    )
    field_checkpoint = torch.load(
        b2_checkpoint_path, map_location="cpu", weights_only=False
    )
    saved_field_config = field_checkpoint["config"]
    if float(saved_field_config["delta"]) != args.delta or float(
        saved_field_config["statistics_momentum"]
    ) != args.statistics_momentum:
        raise RuntimeError("B4 delta/statistics momentum must match the B2 checkpoint")
    field.load_state_dict(field_checkpoint["field"])

    auditor = TwoAxisHostAuditor(
        alpha_cross=args.alpha_cross,
        alpha_within=args.alpha_within,
    )
    q_calibrator = NormalQCalibrator(
        capacity=args.q_reservoir_capacity,
        normal_quantile=args.normal_quantile,
    )
    sources = {
        "b1_config": file_sha256(b1_config_path),
        "b1_checkpoint": file_sha256(b1_checkpoint_path),
        "b2_checkpoint": file_sha256(b2_checkpoint_path),
    }
    return CVAVADCorrectionModel(
        predictor=predictor,
        field=field,
        auditor=auditor,
        q_calibrator=q_calibrator,
    ), sources


def merge_class_batches(
    normal: dict[str, object], abnormal: dict[str, object]
) -> dict[str, torch.Tensor]:
    """Pad normal and abnormal batches to one shared temporal length."""
    maximum = max(normal["hidden"].shape[1], abnormal["hidden"].shape[1])

    def pad(batch: dict[str, object]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        difference = maximum - batch["hidden"].shape[1]
        hidden = F.pad(batch["hidden"], (0, 0, 0, 0, 0, difference))
        host_score = F.pad(batch["host_score"], (0, difference))
        mask = F.pad(batch["mask"], (0, difference))
        return hidden, host_score, mask

    normal_hidden, normal_host, normal_mask = pad(normal)
    abnormal_hidden, abnormal_host, abnormal_mask = pad(abnormal)
    return {
        "hidden": torch.cat((normal_hidden, abnormal_hidden), dim=0),
        "host_score": torch.cat((normal_host, abnormal_host), dim=0),
        "mask": torch.cat((normal_mask, abnormal_mask), dim=0),
        "labels": torch.cat((normal["labels"], abnormal["labels"]), dim=0),
    }


def make_class_loaders(
    dataset: AuditorTrainingDataset,
    normal_indices: list[int],
    abnormal_indices: list[int],
    args: argparse.Namespace,
    epoch: int,
) -> tuple[DataLoader, DataLoader]:
    common = {
        "batch_size": args.batch_size,
        "shuffle": True,
        "drop_last": True,
        "num_workers": args.num_workers,
        "pin_memory": args.device.startswith("cuda"),
        "collate_fn": collate_auditor_training,
    }
    normal_generator = torch.Generator().manual_seed(args.seed + 2 * epoch)
    abnormal_generator = torch.Generator().manual_seed(args.seed + 2 * epoch + 1)
    return (
        DataLoader(
            Subset(dataset, normal_indices), generator=normal_generator, **common
        ),
        DataLoader(
            Subset(dataset, abnormal_indices), generator=abnormal_generator, **common
        ),
    )


def checkpoint_payload(
    model: CVAVADCorrectionModel,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    epoch: int,
    history: list[dict[str, float | int]],
    config: dict[str, object],
) -> dict[str, object]:
    return {
        "epoch": epoch,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "history": history,
        "config": config,
        "python_rng": random.getstate(),
        "numpy_rng": np.random.get_state(),
        "torch_rng": torch.get_rng_state(),
        "cuda_rng": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train the CVA-VAD B4 joint correction model around frozen DSANet scores."
    )
    parser.add_argument("--dataset", choices=["ucf", "xd"], required=True)
    parser.add_argument("--train-manifest", required=True)
    parser.add_argument("--b1-dir", required=True)
    parser.add_argument("--b2-checkpoint", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--epochs", type=int, required=True)
    parser.add_argument("--batch-size", type=int, required=True, help="Per-class batch size.")
    parser.add_argument("--lr", type=float, required=True)
    parser.add_argument("--weight-decay", type=float, required=True)
    parser.add_argument("--maximum-length", type=int, required=True)
    parser.add_argument("--delta", type=float, required=True)
    parser.add_argument("--statistics-momentum", type=float, required=True)
    parser.add_argument("--alpha-cross", type=float, required=True)
    parser.add_argument("--alpha-within", type=float, required=True)
    parser.add_argument("--correction-budget", type=float, required=True)
    parser.add_argument("--lambda-context", type=float, required=True)
    parser.add_argument("--lambda-budget", type=float, required=True)
    parser.add_argument("--q-reservoir-capacity", type=int, required=True)
    parser.add_argument("--normal-quantile", type=float, required=True)
    parser.add_argument("--gradient-clip", type=float, required=True)
    parser.add_argument("--num-workers", type=int, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--device", required=True)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--clean", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output = Path(args.out_dir)
    ensure_output_path(output)
    if args.clean and output.exists():
        shutil.rmtree(output)
    output.mkdir(parents=True, exist_ok=True)
    seed_everything(args.seed)

    model, source_hashes = build_model(args)
    config = vars(args).copy()
    config.pop("clean")
    config.pop("resume")
    config.update(
        {
            "train_manifest_sha256": file_sha256(args.train_manifest),
            "source_sha256": source_hashes,
            "host_policy": "frozen cached DSANet score; never optimized",
            "long_video_policy": "DSANet uniform temporal bin averaging",
            "selection_policy": "fixed final epoch; validation/test never used",
            "optimizer_count": 1,
        }
    )
    config_path = output / "config.json"
    if config_path.exists() and json.loads(config_path.read_text(encoding="utf-8")) != config:
        raise RuntimeError("B4 configuration changed; use --clean or a new output directory")
    config_path.write_text(json.dumps(config, indent=2), encoding="utf-8")
    summary_path = output / "summary.json"
    final_path = output / "model_final.pt"
    if args.resume and summary_path.exists() and final_path.exists():
        print(f"reusing completed B4 run: {summary_path}", flush=True)
        print(summary_path.read_text(encoding="utf-8"), flush=True)
        return

    dataset = AuditorTrainingDataset(args.train_manifest, args.maximum_length)
    labels = dataset.frame["binary_label"].astype(int).to_numpy()
    normal_indices = np.flatnonzero(labels == 0).tolist()
    abnormal_indices = np.flatnonzero(labels == 1).tolist()
    if min(len(normal_indices), len(abnormal_indices)) < args.batch_size:
        raise RuntimeError("B4 needs at least one full normal and abnormal batch")

    device = torch.device(args.device)
    model = model.to(device)
    projected_parameters = [model.auditor.kappa_cross, model.auditor.kappa_within]
    projected_ids = {id(parameter) for parameter in projected_parameters}
    regular_parameters = [
        parameter
        for parameter in model.parameters()
        if parameter.requires_grad and id(parameter) not in projected_ids
    ]
    optimizer = torch.optim.AdamW(
        [
            {"params": regular_parameters, "weight_decay": args.weight_decay},
            {"params": projected_parameters, "weight_decay": 0.0},
        ],
        lr=args.lr,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    latest_path = output / "checkpoint_latest.pt"
    history_path = output / "history.json"
    history: list[dict[str, float | int]] = []
    start_epoch = 0
    maximum_cross_gradient = 0.0
    maximum_within_gradient = 0.0
    if args.resume and latest_path.exists():
        checkpoint = torch.load(latest_path, map_location=device, weights_only=False)
        if checkpoint["config"] != config:
            raise RuntimeError("checkpoint configuration does not match this B4 run")
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        scheduler.load_state_dict(checkpoint["scheduler"])
        history = list(checkpoint["history"])
        start_epoch = int(checkpoint["epoch"]) + 1
        random.setstate(checkpoint["python_rng"])
        np.random.set_state(checkpoint["numpy_rng"])
        torch.set_rng_state(checkpoint["torch_rng"])
        if device.type == "cuda":
            torch.cuda.set_rng_state_all(checkpoint["cuda_rng"])
        if history:
            maximum_cross_gradient = max(float(row["max_cross_gradient_abs"]) for row in history)
            maximum_within_gradient = max(float(row["max_within_gradient_abs"]) for row in history)
        print(f"resume {args.dataset} B4 from epoch {start_epoch + 1}", flush=True)

    for epoch in range(start_epoch, args.epochs):
        normal_loader, abnormal_loader = make_class_loaders(
            dataset, normal_indices, abnormal_indices, args, epoch
        )
        steps = min(len(normal_loader), len(abnormal_loader))
        model.train()
        totals = {
            "total_loss": 0.0,
            "weak_loss": 0.0,
            "context_loss": 0.0,
            "budget_penalty": 0.0,
            "correction_size": 0.0,
            "normal_evidence": 0.0,
            "abnormal_evidence": 0.0,
        }
        epoch_cross_gradient = 0.0
        epoch_within_gradient = 0.0
        progress = tqdm(
            zip(normal_loader, abnormal_loader),
            total=steps,
            desc=f"{args.dataset} B4 epoch {epoch + 1}/{args.epochs}",
            unit="batch",
        )
        for step, (normal_batch, abnormal_batch) in enumerate(progress, start=1):
            batch = merge_class_batches(normal_batch, abnormal_batch)
            hidden = batch["hidden"].to(device, non_blocking=True)
            host_score = batch["host_score"].to(device, non_blocking=True)
            mask = batch["mask"].to(device, non_blocking=True)
            batch_labels = batch["labels"].to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            result = model(hidden, host_score, mask, batch_labels, update_statistics=True)
            normal_prediction_mask = result["distribution"]["prediction_mask"] & (
                batch_labels <= 0.5
            ).unsqueeze(1)
            context_loss = gaussian_nll(
                result["distribution"]["normalized_hidden"],
                result["distribution"]["mean"],
                result["distribution"]["sigma"],
                normal_prediction_mask,
            )
            weak_loss, _ = asymmetric_mil_loss(
                result["corrected_score"], mask, batch_labels
            )
            budget_penalty, correction_size = correction_budget_loss(
                result["correction_size"], args.correction_budget
            )
            loss = (
                weak_loss
                + args.lambda_context * context_loss
                + args.lambda_budget * budget_penalty
            )
            loss.backward()
            cross_gradient = float(model.auditor.kappa_cross.grad.abs())
            within_gradient = float(model.auditor.kappa_within.grad.abs())
            epoch_cross_gradient = max(epoch_cross_gradient, cross_gradient)
            epoch_within_gradient = max(epoch_within_gradient, within_gradient)
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.gradient_clip)
            optimizer.step()
            model.auditor.project_parameters()

            values = {
                "total_loss": loss,
                "weak_loss": weak_loss,
                "context_loss": context_loss,
                "budget_penalty": budget_penalty,
                "correction_size": correction_size,
                "normal_evidence": result["evidence_video"][batch_labels <= 0.5].mean(),
                "abnormal_evidence": result["evidence_video"][batch_labels > 0.5].mean(),
            }
            for name, value in values.items():
                totals[name] += float(value.detach())
            progress.set_postfix(
                loss=f"{totals['total_loss'] / step:.4f}",
                kx=f"{float(model.auditor.kappa_cross):.3f}",
                kw=f"{float(model.auditor.kappa_within):.3f}",
                corr=f"{totals['correction_size'] / step:.3f}",
            )
        scheduler.step()
        maximum_cross_gradient = max(maximum_cross_gradient, epoch_cross_gradient)
        maximum_within_gradient = max(maximum_within_gradient, epoch_within_gradient)
        probability = model.field.probabilities().detach()
        record: dict[str, float | int] = {
            "epoch": epoch + 1,
            **{name: value / steps for name, value in totals.items()},
            "kappa_cross": float(model.auditor.kappa_cross.detach()),
            "kappa_within": float(model.auditor.kappa_within.detach()),
            "max_cross_gradient_abs": epoch_cross_gradient,
            "max_within_gradient_abs": epoch_within_gradient,
            "field_support_size": int((probability > 0.0).sum()),
            "normal_q_median": float(model.q_calibrator.median),
            "normal_q_mad": float(model.q_calibrator.mad),
            "tau_normal": float(model.q_calibrator.tau_normal),
            "lr": float(optimizer.param_groups[0]["lr"]),
        }
        history.append(record)
        history_path.write_text(json.dumps(history, indent=2), encoding="utf-8")
        atomic_torch_save(
            checkpoint_payload(model, optimizer, scheduler, epoch, history, config),
            latest_path,
        )
        print(json.dumps(record, indent=2), flush=True)

    final_checkpoint = checkpoint_payload(
        model, optimizer, scheduler, args.epochs - 1, history, config
    )
    atomic_torch_save(final_checkpoint, final_path)
    finite_history = all(
        np.isfinite(float(value))
        for row in history
        for value in row.values()
        if isinstance(value, (float, int))
    )
    status = "pass"
    if not finite_history or int(model.q_calibrator.count) < 2:
        status = "fail"
    if maximum_cross_gradient == 0.0 or maximum_within_gradient == 0.0:
        status = "fail"
    summary = {
        "status": status,
        "dataset": args.dataset,
        "scope": "B4 training audit; no validation/test metric or model selection",
        "epochs": args.epochs,
        "normal_training_videos": len(normal_indices),
        "abnormal_training_videos": len(abnormal_indices),
        "steps_per_epoch": min(
            len(normal_indices) // args.batch_size,
            len(abnormal_indices) // args.batch_size,
        ),
        "optimizer_count": 1,
        "trainable_parameters": sum(
            parameter.numel() for parameter in model.parameters() if parameter.requires_grad
        ),
        "predictor_parameters": sum(
            parameter.numel()
            for parameter in model.predictor.parameters()
            if parameter.requires_grad
        ),
        "field_support_size": int((model.field.probabilities() > 0.0).sum()),
        "kappa_cross": float(model.auditor.kappa_cross.detach()),
        "kappa_within": float(model.auditor.kappa_within.detach()),
        "maximum_cross_gradient_abs": maximum_cross_gradient,
        "maximum_within_gradient_abs": maximum_within_gradient,
        "normal_activation_statistics_updates": int(model.field.statistics_updates),
        "normal_q_reservoir_count": int(model.q_calibrator.count),
        "normal_q_statistics_updates": int(model.q_calibrator.updates),
        "normal_q_median": float(model.q_calibrator.median),
        "normal_q_mad": float(model.q_calibrator.mad),
        "tau_normal": float(model.q_calibrator.tau_normal),
        "final_epoch": history[-1],
        "checkpoint_contains": [
            "predictor",
            "omega",
            "kappa_cross",
            "kappa_within",
            "normal_activation_statistics",
            "normal_q_reservoir_and_statistics",
            "optimizer",
            "scheduler",
            "configuration",
            "data_manifest_hash",
            "random_states",
        ],
        "test_split_used": False,
        "selection_policy": config["selection_policy"],
    }
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)
    if status != "pass":
        raise RuntimeError("B4 failed; inspect gradient and normal-statistics diagnostics")


if __name__ == "__main__":
    main()
