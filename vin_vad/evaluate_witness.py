from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import average_precision_score, roc_auc_score
from tqdm import tqdm

from vin_vad.evaluate import expand_snippet_scores, score_curve_metrics
from vin_vad.witness_model import build_witness_variant


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_savez(path: Path, **arrays: object) -> None:
    temporary = path.with_suffix(".tmp.npz")
    np.savez_compressed(temporary, **arrays)
    temporary.replace(path)


def build_from_checkpoint(
    checkpoint: dict[str, object], device: torch.device
) -> tuple[torch.nn.Module, dict[str, object]]:
    config = checkpoint["config"]
    model = build_witness_variant(
        str(config["variant"]),
        active=int(config["active_neurons"]),
        temporal_width=int(config["temporal_width"]),
        eta_normal=float(config["eta_normal"]),
        eta_anomaly=float(config["eta_anomaly"]),
    )
    model.load_state_dict(checkpoint["model"])
    return model.to(device).eval(), config


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate one final Witness-VAD checkpoint")
    parser.add_argument("--dataset", required=True, choices=("ucf", "xd"))
    parser.add_argument("--variant", required=True, choices=("w0", "w1", "w2", "w6"))
    parser.add_argument("--test-manifest", required=True)
    parser.add_argument("--checkpoint", default="")
    parser.add_argument("--host-metrics", required=True)
    parser.add_argument("--gt-path", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--target-tpr", type=float, required=True)
    parser.add_argument("--device", required=True)
    parser.add_argument(
        "--selection-policy",
        choices=("last_checkpoint_only", "test_primary_metric_best"),
        default="last_checkpoint_only",
    )
    parser.add_argument("--no-curve-cache", action="store_true")
    parser.add_argument("--eta-anomaly-override", type=float, default=None)
    args = parser.parse_args()

    output = Path(args.out_dir)
    if "vadmy_data" not in output.resolve().parts:
        raise ValueError("out-dir must be inside sibling vadmy_data")
    output.mkdir(parents=True, exist_ok=True)
    if args.variant == "w0" and args.checkpoint:
        raise ValueError("w0 must not receive a checkpoint")
    if args.variant != "w0" and not args.checkpoint:
        raise ValueError(f"{args.variant} requires a checkpoint")
    if args.eta_anomaly_override is not None:
        if args.variant not in {"w2", "w6"}:
            raise ValueError("eta-anomaly-override is only valid for w2 and w6")
        if args.eta_anomaly_override <= 0.0:
            raise ValueError("eta-anomaly-override must be positive")
    signature = {
        "dataset": args.dataset,
        "variant": args.variant,
        "test_manifest": {"path": args.test_manifest, "sha256": file_sha256(args.test_manifest)},
        "host_metrics": {"path": args.host_metrics, "sha256": file_sha256(args.host_metrics)},
        "gt": {"path": args.gt_path, "sha256": file_sha256(args.gt_path)},
        "checkpoint": (
            None
            if args.variant == "w0"
            else {"path": args.checkpoint, "sha256": file_sha256(args.checkpoint)}
        ),
        "target_tpr": args.target_tpr,
        "post_processing": "none",
        "selection_policy": args.selection_policy,
        "eta_anomaly_override": args.eta_anomaly_override,
    }
    signature_path = output / "signature.json"
    if signature_path.exists() and json.loads(signature_path.read_text(encoding="utf-8")) != signature:
        raise RuntimeError("evaluation inputs changed; clean the exact F3 directory")
    signature_path.write_text(json.dumps(signature, indent=2), encoding="utf-8")
    metrics_path = output / "metrics.json"
    if metrics_path.exists() and (output / "per_video.csv").exists():
        print(f"reuse completed {args.dataset}/{args.variant} evaluation", flush=True)
        print(metrics_path.read_text(encoding="utf-8"), flush=True)
        return

    device = torch.device(args.device)
    model = None
    training_config = None
    if args.variant != "w0":
        checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
        model, training_config = build_from_checkpoint(checkpoint, device)
        if (
            args.selection_policy == "last_checkpoint_only"
            and int(checkpoint["epoch"]) != int(training_config["epochs"])
        ):
            raise RuntimeError("evaluation requires the final training checkpoint")

    manifest = pd.read_csv(args.test_manifest)
    required = {
        "key",
        "binary_label",
        "hidden_path",
        "host_score_path",
        "valid_snippets",
        "evaluation_frames",
    }
    missing = required - set(manifest.columns)
    if missing:
        raise ValueError(f"test manifest is missing {sorted(missing)}")
    gt = np.asarray(np.load(args.gt_path, allow_pickle=False), dtype=np.int8).reshape(-1)
    curves_dir = output / "curves"
    curves_dir.mkdir(exist_ok=True)
    frame_scores: list[np.ndarray] = []
    frame_labels: list[np.ndarray] = []
    abnormal_scores: list[np.ndarray] = []
    abnormal_labels: list[np.ndarray] = []
    rows: list[dict[str, object]] = []
    offset = 0
    for row in tqdm(
        manifest.itertuples(index=False),
        total=len(manifest),
        desc=f"{args.dataset}/{args.variant} final evaluation",
        unit="video",
    ):
        curve_path = curves_dir / f"{row.key}.npz"
        if curve_path.exists() and not args.no_curve_cache:
            with np.load(curve_path, allow_pickle=False) as archive:
                corrected = np.asarray(archive["corrected_score"], dtype=np.float32)
                host = np.asarray(archive["host_score"], dtype=np.float32)
                frame_indices = np.asarray(archive["frame_indices"], dtype=np.int64)
                video_probability = float(archive["video_probability"])
        else:
            length = int(row.valid_snippets)
            host = np.asarray(
                np.load(str(row.host_score_path), allow_pickle=False), dtype=np.float32
            ).reshape(-1)[:length]
            with np.load(str(row.hidden_path), allow_pickle=False) as archive:
                frame_indices = np.asarray(archive["frame_indices"][:length], dtype=np.int64)
                hidden_array = (
                    None
                    if args.variant in {"w0", "w1"}
                    else np.asarray(archive["hidden"][:length], dtype=np.float32).copy()
                )
            if args.variant == "w0":
                corrected = host.copy()
                video_probability = float("nan")
                evidence = np.empty(0, dtype=np.float32)
                delta_normal = np.zeros_like(host)
                delta_anomaly = np.zeros_like(host)
            else:
                host_tensor = torch.from_numpy(host.copy()).unsqueeze(0).to(device)
                validity = torch.ones(1, length, dtype=torch.bool, device=device)
                with torch.no_grad():
                    result = (
                        model(host_tensor, validity)
                        if args.variant == "w1"
                        else model(
                            torch.from_numpy(hidden_array).unsqueeze(0).to(device),
                            host_tensor,
                            validity,
                            eta_anomaly_override=args.eta_anomaly_override,
                        )
                    )
                corrected = result["corrected_score"][0].cpu().numpy().astype(np.float32)
                video_probability = float(result["video_probability"][0].cpu())
                evidence = (
                    np.empty(0, dtype=np.float32)
                    if "evidence" not in result
                    else result["evidence"][0].cpu().numpy().astype(np.float32)
                )
                delta_normal = result["delta_normal"][0].cpu().numpy().astype(np.float32)
                delta_anomaly = result["delta_anomaly"][0].cpu().numpy().astype(np.float32)
            if not args.no_curve_cache:
                atomic_savez(
                    curve_path,
                    host_score=host,
                    corrected_score=corrected,
                    evidence=evidence,
                    delta_normal=delta_normal,
                    delta_anomaly=delta_anomaly,
                    video_probability=np.asarray(video_probability, dtype=np.float32),
                    frame_indices=frame_indices,
                )

        frame_count = int(row.evaluation_frames)
        labels = gt[offset : offset + frame_count]
        if len(labels) != frame_count:
            raise RuntimeError(f"{row.key}: GT ended before the audited video boundary")
        offset += frame_count
        frame_score = expand_snippet_scores(corrected, frame_indices, frame_count)
        frame_scores.append(frame_score)
        frame_labels.append(labels)
        if int(row.binary_label) == 1:
            abnormal_scores.append(frame_score)
            abnormal_labels.append(labels)
        rows.append(
            {
                "key": str(row.key),
                "binary_label": int(row.binary_label),
                "snippets": len(corrected),
                "video_probability": video_probability,
                "mean_absolute_correction": float(np.mean(np.abs(corrected - host))),
                "curve_path": str(curve_path),
            }
        )
    if offset != len(gt):
        raise RuntimeError(f"evaluated frames {offset} != GT frames {len(gt)}")

    metrics = score_curve_metrics(frame_scores, frame_labels, args.target_tpr)
    abnormal_score = np.concatenate(abnormal_scores)
    abnormal_label = np.concatenate(abnormal_labels)
    host_metrics = json.loads(Path(args.host_metrics).read_text(encoding="utf-8"))
    primary = "pooled_auc" if args.dataset == "ucf" else "pooled_ap"
    metrics.update(
        {
            "status": "pass",
            "dataset": args.dataset,
            "variant": args.variant,
            "primary_metric": primary,
            "primary_gain": float(metrics[primary]) - float(host_metrics[primary]),
            "pooled_auc_gain": float(metrics["pooled_auc"]) - float(host_metrics["pooled_auc"]),
            "pooled_ap_gain": float(metrics["pooled_ap"]) - float(host_metrics["pooled_ap"]),
            "cross_auc_gain": float(metrics["cross_auc"]) - float(host_metrics["cross_auc"]),
            "within_auc_gain": float(metrics["within_auc"]) - float(host_metrics["within_auc"]),
            "macro_within_auc_gain": float(metrics["macro_within_auc"]) - float(host_metrics["macro_within_auc"]),
            "abnormal_only_auc": float(roc_auc_score(abnormal_label, abnormal_score)),
            "abnormal_only_ap": float(average_precision_score(abnormal_label, abnormal_score)),
            "normal_frame_fpr_change": float(metrics["normal_fpr"]["normal_video_frame_fpr"])
            - float(host_metrics["normal_fpr"]["normal_video_frame_fpr"]),
            "mean_absolute_correction": float(
                np.mean([item["mean_absolute_correction"] for item in rows])
            ),
            "test_videos": len(rows),
            "test_frames": len(gt),
            "post_processing": "none",
            "eta_anomaly_override": args.eta_anomaly_override,
            "test_used_for_selection": args.selection_policy == "test_primary_metric_best",
            "checkpoint_epoch": (
                None if args.variant == "w0" else int(checkpoint["epoch"])
            ),
            "training_config": training_config,
        }
    )
    pd.DataFrame(rows).to_csv(output / "per_video.csv", index=False)
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(json.dumps(metrics, indent=2), flush=True)


if __name__ == "__main__":
    main()
