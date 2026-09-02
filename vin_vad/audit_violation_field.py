from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from vin_vad.context_predictor import MaskedContextPredictor, detached_distribution
from vin_vad.data import NormalContextWindowDataset, collate_context_windows
from vin_vad.violation_field import ViolationField


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def ensure_output_path(path: Path) -> None:
    if "vadmy_data" not in path.resolve().parts:
        raise ValueError("out-dir must be inside the sibling vadmy_data directory")


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


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Audit CVA-VAD B2 directional violation fields on training normals."
    )
    parser.add_argument("--dataset", choices=["ucf", "xd"], required=True)
    parser.add_argument("--b1-dir", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--delta", type=float, required=True)
    parser.add_argument("--statistics-momentum", type=float, required=True)
    parser.add_argument("--batch-size", type=int, required=True)
    parser.add_argument("--maximum-length", type=int, required=True)
    parser.add_argument("--num-workers", type=int, required=True)
    parser.add_argument("--device", required=True)
    parser.add_argument("--clean", action="store_true")
    args = parser.parse_args()

    b1_dir = Path(args.b1_dir)
    output = Path(args.out_dir)
    ensure_output_path(output)
    if args.clean and output.exists():
        shutil.rmtree(output)
    output.mkdir(parents=True, exist_ok=True)

    predictor_config_path = b1_dir / "config.json"
    predictor_checkpoint_path = b1_dir / "context_predictor_best.pt"
    manifest_path = b1_dir / "data" / "train_normal.csv"
    split_audit_path = b1_dir / "data" / "split_audit.json"
    run_config = {
        "dataset": args.dataset,
        "b1_dir": str(b1_dir),
        "delta": args.delta,
        "statistics_momentum": args.statistics_momentum,
        "batch_size": args.batch_size,
        "maximum_length": args.maximum_length,
        "num_workers": args.num_workers,
        "device": args.device,
        "predictor_checkpoint_sha256": file_sha256(predictor_checkpoint_path),
        "normal_manifest_sha256": file_sha256(manifest_path),
    }
    config_path = output / "config.json"
    if config_path.exists():
        previous_config = json.loads(config_path.read_text(encoding="utf-8"))
        if previous_config != run_config:
            raise RuntimeError("B2 configuration changed; rerun with --clean")
    config_path.write_text(json.dumps(run_config, indent=2), encoding="utf-8")
    summary_path = output / "summary.json"
    if summary_path.exists():
        print(f"reusing completed B2 audit: {summary_path}", flush=True)
        print(summary_path.read_text(encoding="utf-8"), flush=True)
        return

    manifest = pd.read_csv(manifest_path)
    if (manifest["binary_label"].astype(int) != 0).any():
        raise RuntimeError("B2 statistics manifest contains a non-normal video")
    split_audit = json.loads(split_audit_path.read_text(encoding="utf-8"))
    if split_audit.get("overlap") != 0:
        raise RuntimeError("B1 train/validation split overlap is non-zero")

    predictor_config = json.loads(predictor_config_path.read_text(encoding="utf-8"))
    device = torch.device(args.device)
    predictor = build_predictor(predictor_config).to(device)
    checkpoint = torch.load(
        predictor_checkpoint_path, map_location=device, weights_only=False
    )
    predictor.load_state_dict(checkpoint["model"])
    predictor.eval()
    field = ViolationField(
        delta=args.delta,
        statistics_momentum=args.statistics_momentum,
    ).to(device)
    field.eval()

    dataset = NormalContextWindowDataset(
        str(manifest_path),
        maximum_length=args.maximum_length,
        overlap=0,
        training=False,
        exhaustive=False,
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        collate_fn=collate_context_windows,
    )
    direction_overlap = 0
    with torch.no_grad():
        for batch in tqdm(loader, desc=f"{args.dataset} B2 normal stats", unit="batch"):
            hidden = batch["hidden"].to(device, non_blocking=True)
            validity = batch["mask"].to(device, non_blocking=True)
            distribution = predictor(hidden, validity)
            mean, sigma = detached_distribution(distribution)
            result = field(
                distribution["normalized_hidden"],
                mean,
                sigma,
                validity,
                torch.zeros(len(hidden), device=device),
                update_statistics=True,
            )
            direction_overlap += int(
                (
                    (result["directional"][..., 0] > 0.0)
                    & (result["directional"][..., 1] > 0.0)
                ).sum()
            )

    # Save one fixed held-in example after statistics are locked. Its complete
    # residual chain is small enough to inspect and independently recompute.
    sample_batch = next(iter(loader))
    with torch.no_grad():
        hidden = sample_batch["hidden"][:1].to(device)
        validity = sample_batch["mask"][:1].to(device)
        distribution = predictor(hidden, validity)
        mean, sigma = detached_distribution(distribution)
        sample = field(
            distribution["normalized_hidden"], mean, sigma, validity
        )
    keep = min(16, int(validity[0].sum()))
    directional = sample["directional"][0, :keep].cpu().numpy()
    probability = sample["probability"].cpu().numpy()
    activation = sample["activation"][0, :keep].cpu().numpy()
    evidence = sample["evidence"][0, :keep].cpu().numpy()
    residual = sample["residual"][0, :keep].cpu().numpy()
    recomputed_activation = np.einsum("tldq,ldq->t", directional, probability)
    recomputed_evidence = (recomputed_activation - float(field.running_median)) / (
        1.4826 * float(field.running_mad) + field.epsilon
    )
    activation_error = float(np.max(np.abs(activation - recomputed_activation)))
    evidence_error = float(np.max(np.abs(evidence - recomputed_evidence)))
    np.savez_compressed(
        output / "recompute_sample.npz",
        key=np.asarray(sample_batch["keys"][0]),
        residual=residual,
        directional=directional,
        probability=probability,
        activation=activation,
        evidence=evidence,
        running_median=field.running_median.cpu().numpy(),
        running_mad=field.running_mad.cpu().numpy(),
        epsilon=np.asarray(field.epsilon),
    )
    torch.save(
        {
            "field": field.state_dict(),
            "config": {
                "delta": args.delta,
                "statistics_momentum": args.statistics_momentum,
                "epsilon": field.epsilon,
            },
            "predictor_checkpoint_sha256": file_sha256(predictor_checkpoint_path),
            "normal_manifest_sha256": file_sha256(manifest_path),
        },
        output / "violation_field_initial.pt",
    )
    probability_tensor = sample["probability"]
    summary = {
        "status": "pass",
        "dataset": args.dataset,
        "scope": "B2 mechanism audit; no detector training or test evaluation",
        "delta": args.delta,
        "statistics_policy": "stop-gradient EMA of per-batch normal median/MAD",
        "statistics_momentum": args.statistics_momentum,
        "statistics_source": "B1 train_normal fixed center window per video",
        "normal_training_videos": len(dataset),
        "normal_snippets_seen": int(field.normal_snippets_seen),
        "statistics_updates": int(field.statistics_updates),
        "running_median": float(field.running_median),
        "running_mad": float(field.running_mad),
        "direction_overlap_count": direction_overlap,
        "probability_min": float(probability_tensor.min()),
        "probability_sum_error": abs(float(probability_tensor.sum()) - 1.0),
        "initial_support_size": int((probability_tensor > 0.0).sum()),
        "recompute_activation_max_abs_error": activation_error,
        "recompute_evidence_max_abs_error": evidence_error,
        "abnormal_labels_used_for_statistics": False,
        "padding_used_for_statistics": False,
        "validation_split_used": False,
        "test_split_used": False,
        "predictor_checkpoint_sha256": file_sha256(predictor_checkpoint_path),
        "normal_manifest_sha256": file_sha256(manifest_path),
    }
    if direction_overlap != 0:
        summary["status"] = "fail"
    if summary["probability_min"] < 0.0 or summary["probability_sum_error"] > 1e-5:
        summary["status"] = "fail"
    if max(activation_error, evidence_error) > 1e-5:
        summary["status"] = "fail"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)
    if summary["status"] != "pass":
        raise RuntimeError("B2 violation-field audit failed")


if __name__ == "__main__":
    main()
