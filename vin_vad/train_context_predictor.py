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
from torch.utils.data import DataLoader
from tqdm import tqdm

from vin_vad.context_predictor import MaskedContextPredictor, gaussian_nll
from vin_vad.data import NormalContextWindowDataset, collate_context_windows


def file_sha256(path: str) -> str:
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


def clean_output(path: Path) -> None:
    ensure_output_path(path)
    if path.exists():
        shutil.rmtree(path)


def prepare_normal_split(
    source_manifest: str,
    output: Path,
    validation_fraction: float,
    seed: int,
) -> tuple[Path, Path]:
    data_dir = output / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    train_path = data_dir / "train_normal.csv"
    validation_path = data_dir / "validation_normal.csv"
    audit_path = data_dir / "split_audit.json"
    signature = {
        "source_manifest": source_manifest,
        "source_sha256": file_sha256(source_manifest),
        "validation_fraction": validation_fraction,
        "seed": seed,
    }
    signature_path = data_dir / "signature.json"
    if train_path.exists() and validation_path.exists() and audit_path.exists():
        if json.loads(signature_path.read_text(encoding="utf-8")) != signature:
            raise RuntimeError("normal split inputs changed; rerun with --clean")
        return train_path, validation_path
    frame = pd.read_csv(source_manifest)
    required = {"key", "binary_label", "hidden_path", "valid_snippets"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"{source_manifest}: missing columns {sorted(missing)}")
    normal = frame[frame["binary_label"].astype(int) == 0].reset_index(drop=True)
    if len(normal) < 10:
        raise RuntimeError("B1 needs at least ten normal training videos")
    rng = np.random.default_rng(seed)
    indices = rng.permutation(len(normal))
    validation_count = max(1, round(len(normal) * validation_fraction))
    validation_indices = set(indices[:validation_count].tolist())
    train = normal[
        [index not in validation_indices for index in range(len(normal))]
    ].reset_index(drop=True)
    validation = normal[
        [index in validation_indices for index in range(len(normal))]
    ].reset_index(drop=True)
    train.to_csv(train_path, index=False)
    validation.to_csv(validation_path, index=False)
    signature_path.write_text(json.dumps(signature, indent=2), encoding="utf-8")
    audit = {
        "status": "pass",
        "policy": "normal videos from the official training split only",
        "seed": seed,
        "train_normal_videos": len(train),
        "validation_normal_videos": len(validation),
        "overlap": len(set(train["key"]) & set(validation["key"])),
    }
    audit_path.write_text(json.dumps(audit, indent=2), encoding="utf-8")
    return train_path, validation_path


def _layer_normalize(array: np.ndarray, epsilon: float = 1e-5) -> np.ndarray:
    mean = array.mean(axis=-1, keepdims=True)
    variance = array.var(axis=-1, keepdims=True)
    return (array - mean) / np.sqrt(variance + epsilon)


def fit_global_statistics(manifest: Path, output: Path) -> tuple[np.ndarray, np.ndarray]:
    final_path = output / "global_normal_statistics.npz"
    if final_path.exists():
        with np.load(final_path, allow_pickle=False) as archive:
            return np.asarray(archive["mean"]), np.asarray(archive["sigma"])
    frame = pd.read_csv(manifest)
    partial_path = output / "global_normal_statistics.partial.npz"
    sums = np.zeros((12, 768), dtype=np.float64)
    squares = np.zeros_like(sums)
    count = 0
    start = 0
    if partial_path.exists():
        with np.load(partial_path, allow_pickle=False) as archive:
            sums = np.asarray(archive["sums"], dtype=np.float64)
            squares = np.asarray(archive["squares"], dtype=np.float64)
            count = int(archive["count"])
            start = int(archive["next_index"])
    for index in tqdm(
        range(start, len(frame)),
        initial=start,
        total=len(frame),
        desc="fit global normal",
        unit="video",
    ):
        row = frame.iloc[index]
        with np.load(str(row.hidden_path), allow_pickle=False) as archive:
            hidden = np.asarray(
                archive["hidden"][: int(row.valid_snippets)], dtype=np.float32
            )
        normalized = _layer_normalize(hidden)
        sums += normalized.sum(axis=0, dtype=np.float64)
        squares += np.square(normalized, dtype=np.float64).sum(axis=0)
        count += len(normalized)
        if (index + 1) % 25 == 0:
            np.savez_compressed(
                partial_path,
                sums=sums,
                squares=squares,
                count=np.asarray(count),
                next_index=np.asarray(index + 1),
            )
    mean = sums / count
    variance = np.maximum(squares / count - np.square(mean), 1e-6)
    sigma = np.sqrt(variance)
    np.savez_compressed(final_path, mean=mean.astype(np.float32), sigma=sigma.astype(np.float32), count=np.asarray(count))
    partial_path.unlink(missing_ok=True)
    return mean.astype(np.float32), sigma.astype(np.float32)


def build_model(args: argparse.Namespace) -> MaskedContextPredictor:
    return MaskedContextPredictor(
        model_width=args.model_width,
        input_rank=args.input_rank,
        head_rank=args.head_rank,
        attention_heads=args.attention_heads,
        attention_layers=args.attention_layers,
        guard_radius=args.guard_radius,
        dropout=args.dropout,
        sigma_min=args.sigma_min,
        sigma_max=args.sigma_max,
    )


def evaluate(
    model: MaskedContextPredictor,
    loader: DataLoader,
    global_mean: torch.Tensor,
    global_sigma: torch.Tensor,
    device: torch.device,
    description: str,
) -> tuple[float, float]:
    model.eval()
    conditional_sum = 0.0
    global_sum = 0.0
    targets = 0
    with torch.no_grad():
        for batch in tqdm(loader, desc=description, unit="batch", leave=False):
            hidden = batch["hidden"].to(device, non_blocking=True)
            mask = batch["mask"].to(device, non_blocking=True)
            result = model(hidden, mask)
            conditional = gaussian_nll(
                result["normalized_hidden"],
                result["mean"],
                result["sigma"],
                result["prediction_mask"],
            )
            baseline = gaussian_nll(
                result["normalized_hidden"],
                global_mean.expand_as(result["normalized_hidden"]),
                global_sigma.expand_as(result["normalized_hidden"]),
                result["prediction_mask"],
            )
            batch_targets = int(result["prediction_mask"].sum().item())
            conditional_sum += float(conditional) * batch_targets
            global_sum += float(baseline) * batch_targets
            targets += batch_targets
    return conditional_sum / targets, global_sum / targets


def atomic_torch_save(value: dict[str, object], path: Path) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(value, temporary)
    temporary.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train and validate the CVA-VAD B1 normal context predictor.")
    parser.add_argument("--dataset", choices=["ucf", "xd"], required=True)
    parser.add_argument("--source-manifest", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--epochs", type=int, required=True)
    parser.add_argument("--patience", type=int, required=True)
    parser.add_argument("--batch-size", type=int, required=True)
    parser.add_argument("--lr", type=float, required=True)
    parser.add_argument("--weight-decay", type=float, required=True)
    parser.add_argument("--maximum-length", type=int, required=True)
    parser.add_argument("--window-overlap", type=int, required=True)
    parser.add_argument("--model-width", type=int, required=True)
    parser.add_argument("--input-rank", type=int, required=True)
    parser.add_argument("--head-rank", type=int, required=True)
    parser.add_argument("--attention-heads", type=int, required=True)
    parser.add_argument("--attention-layers", type=int, required=True)
    parser.add_argument("--guard-radius", type=int, required=True)
    parser.add_argument("--dropout", type=float, required=True)
    parser.add_argument("--sigma-min", type=float, required=True)
    parser.add_argument("--sigma-max", type=float, required=True)
    parser.add_argument("--validation-fraction", type=float, required=True)
    parser.add_argument("--num-workers", type=int, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--device", required=True)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--clean", action="store_true")
    args = parser.parse_args()

    output = Path(args.out_dir)
    ensure_output_path(output)
    if args.clean:
        clean_output(output)
    output.mkdir(parents=True, exist_ok=True)
    seed_everything(args.seed)
    config = vars(args).copy()
    config.pop("clean")
    config.pop("resume")
    config["source_manifest_sha256"] = file_sha256(args.source_manifest)
    config_path = output / "config.json"
    if config_path.exists() and json.loads(config_path.read_text(encoding="utf-8")) != config:
        raise RuntimeError("B1 configuration changed; rerun with --clean")
    config_path.write_text(json.dumps(config, indent=2), encoding="utf-8")
    summary_path = output / "summary.json"
    if args.resume and summary_path.exists():
        print(f"reusing completed B1 run: {summary_path}", flush=True)
        print(summary_path.read_text(encoding="utf-8"), flush=True)
        return

    train_manifest, validation_manifest = prepare_normal_split(
        args.source_manifest, output, args.validation_fraction, args.seed
    )
    global_mean_np, global_sigma_np = fit_global_statistics(train_manifest, output)
    train_dataset = NormalContextWindowDataset(
        str(train_manifest), args.maximum_length, args.window_overlap
    )
    validation_dataset = NormalContextWindowDataset(
        str(validation_manifest), args.maximum_length, args.window_overlap
    )
    generator = torch.Generator().manual_seed(args.seed)
    loader_arguments = {
        "batch_size": args.batch_size,
        "num_workers": args.num_workers,
        "pin_memory": args.device.startswith("cuda"),
        "collate_fn": collate_context_windows,
    }
    train_loader = DataLoader(
        train_dataset, shuffle=True, generator=generator, **loader_arguments
    )
    validation_loader = DataLoader(
        validation_dataset, shuffle=False, **loader_arguments
    )
    device = torch.device(args.device)
    model = build_model(args).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    global_mean = torch.from_numpy(global_mean_np).to(device)[None, None]
    global_sigma = torch.from_numpy(global_sigma_np).to(device)[None, None]
    latest_path = output / "checkpoint_latest.pt"
    best_path = output / "context_predictor_best.pt"
    history: list[dict[str, float | int]] = []
    start_epoch = 0
    best_validation = float("inf")
    best_epoch = -1
    stale_epochs = 0
    if args.resume and latest_path.exists():
        checkpoint = torch.load(latest_path, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        history = checkpoint["history"]
        start_epoch = int(checkpoint["epoch"]) + 1
        best_validation = float(checkpoint["best_validation"])
        best_epoch = int(checkpoint["best_epoch"])
        stale_epochs = int(checkpoint["stale_epochs"])

    for epoch in range(start_epoch, args.epochs):
        model.train()
        train_sum = 0.0
        batches = 0
        progress = tqdm(train_loader, desc=f"{args.dataset} B1 epoch {epoch + 1}/{args.epochs}", unit="batch")
        for batch in progress:
            hidden = batch["hidden"].to(device, non_blocking=True)
            mask = batch["mask"].to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            result = model(hidden, mask)
            loss = gaussian_nll(
                result["normalized_hidden"],
                result["mean"],
                result["sigma"],
                result["prediction_mask"],
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()
            train_sum += float(loss.detach())
            batches += 1
            progress.set_postfix(nll=f"{train_sum / batches:.5f}")
        validation_nll, global_nll = evaluate(
            model,
            validation_loader,
            global_mean,
            global_sigma,
            device,
            f"{args.dataset} validation",
        )
        record = {
            "epoch": epoch,
            "train_nll": train_sum / batches,
            "validation_conditional_nll": validation_nll,
            "validation_global_nll": global_nll,
        }
        history.append(record)
        improved = validation_nll < best_validation
        if improved:
            best_validation = validation_nll
            best_epoch = epoch
            stale_epochs = 0
            atomic_torch_save(
                {"model": model.state_dict(), "config": config, "epoch": epoch}, best_path
            )
        else:
            stale_epochs += 1
        atomic_torch_save(
            {
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "epoch": epoch,
                "history": history,
                "best_validation": best_validation,
                "best_epoch": best_epoch,
                "stale_epochs": stale_epochs,
            },
            latest_path,
        )
        (output / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
        print(json.dumps(record, indent=2), flush=True)
        if stale_epochs >= args.patience:
            print(f"early stop after {stale_epochs} stale epochs", flush=True)
            break

    best = torch.load(best_path, map_location=device, weights_only=False)
    model.load_state_dict(best["model"])
    final_conditional, final_global = evaluate(
        model,
        validation_loader,
        global_mean,
        global_sigma,
        device,
        f"{args.dataset} final validation",
    )
    summary = {
        "status": "pass" if final_conditional < final_global else "fail",
        "dataset": args.dataset,
        "criterion": "held-out normal conditional NLL must be lower than per-neuron global Gaussian NLL",
        "best_epoch": int(best["epoch"]),
        "validation_conditional_nll": final_conditional,
        "validation_global_nll": final_global,
        "relative_nll_improvement": (final_global - final_conditional) / abs(final_global),
        "train_normal_videos": len(pd.read_csv(train_manifest)),
        "validation_normal_videos": len(pd.read_csv(validation_manifest)),
        "train_windows": len(train_dataset),
        "validation_windows": len(validation_dataset),
        "trainable_parameters": sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad),
        "test_split_used": False,
    }
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)
    if summary["status"] != "pass":
        raise RuntimeError("B1 failed: conditional NLL did not beat the global baseline")


if __name__ == "__main__":
    main()
