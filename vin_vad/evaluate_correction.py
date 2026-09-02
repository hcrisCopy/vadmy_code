from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from vin_vad.evaluate import expand_snippet_scores, score_curve_metrics
from vin_vad.train import build_model


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def ensure_output_path(path: Path) -> None:
    if "vadmy_data" not in path.resolve().parts:
        raise ValueError("out-dir must be inside the sibling vadmy_data directory")


def atomic_savez(path: Path, **arrays: object) -> None:
    temporary = path.with_suffix(".tmp.npz")
    np.savez_compressed(temporary, **arrays)
    temporary.replace(path)


def window_starts(length: int, maximum_length: int, overlap: int) -> list[int]:
    if length < 1 or maximum_length < 2 or not 0 <= overlap < maximum_length:
        raise ValueError("invalid sequence/window/overlap")
    if length <= maximum_length:
        return [0]
    stride = maximum_length - overlap
    starts = list(range(0, length - maximum_length + 1, stride))
    if starts[-1] != length - maximum_length:
        starts.append(length - maximum_length)
    return starts


def infer_evidence(
    model: torch.nn.Module,
    hidden: torch.Tensor,
    maximum_length: int,
    overlap: int,
    device: torch.device,
) -> torch.Tensor:
    """Average overlapping context predictions into one full-length curve."""
    length = len(hidden)
    evidence_sum = torch.zeros(length, dtype=torch.float32)
    evidence_count = torch.zeros(length, dtype=torch.float32)
    contextual = model.evidence_id in {"c2", "c3", "c4"}
    effective_overlap = overlap if contextual else 0
    with torch.no_grad():
        for start in window_starts(length, maximum_length, effective_overlap):
            end = min(length, start + maximum_length)
            chunk = hidden[start:end].unsqueeze(0).to(device)
            validity = torch.ones(1, end - start, dtype=torch.bool, device=device)
            result = model.evidence_forward(
                chunk,
                validity,
                update_statistics=False,
                require_context_loss=False,
            )
            values = result["field"]["evidence"][0].detach().cpu()
            evidence_sum[start:end] += values
            evidence_count[start:end] += 1.0
    if torch.any(evidence_count == 0):
        raise RuntimeError("window inference left an uncovered snippet")
    return evidence_sum / evidence_count


def metric_delta(
    corrected: dict[str, object], host: dict[str, object], name: str
) -> float:
    return float(corrected[name]) - float(host[name])


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate one fixed E1 correction checkpoint on the B0 test split."
    )
    parser.add_argument("--dataset", choices=["ucf", "xd"], required=True)
    parser.add_argument("--test-manifest", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--host-metrics", required=True)
    parser.add_argument("--gt-path", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--maximum-length", type=int, required=True)
    parser.add_argument("--window-overlap", type=int, required=True)
    parser.add_argument("--target-tpr", type=float, required=True)
    parser.add_argument("--device", required=True)
    parser.add_argument("--clean", action="store_true")
    args = parser.parse_args()

    output = Path(args.out_dir)
    ensure_output_path(output)
    if args.clean and output.exists():
        shutil.rmtree(output)
    output.mkdir(parents=True, exist_ok=True)
    signature = {
        key: value
        for key, value in vars(args).items()
        if key not in {"clean", "out_dir"}
    }
    signature["inputs_sha256"] = {
        "test_manifest": file_sha256(args.test_manifest),
        "checkpoint": file_sha256(args.checkpoint),
        "host_metrics": file_sha256(args.host_metrics),
        "gt": file_sha256(args.gt_path),
    }
    signature["post_processing"] = "none"
    signature_path = output / "signature.json"
    if signature_path.exists() and json.loads(signature_path.read_text(encoding="utf-8")) != signature:
        raise RuntimeError("E1 evaluation inputs changed; rerun with --clean")
    signature_path.write_text(json.dumps(signature, indent=2), encoding="utf-8")
    metrics_path = output / "metrics.json"
    if metrics_path.exists() and (output / "per_video.csv").exists():
        print(f"reusing completed E1 evaluation: {metrics_path}", flush=True)
        print(metrics_path.read_text(encoding="utf-8"), flush=True)
        return

    device = torch.device(args.device)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    training_config = checkpoint["config"]
    model, _ = build_model(SimpleNamespace(**training_config))
    model.load_state_dict(checkpoint["model"])
    model = model.to(device).eval()
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
    curves_dir.mkdir(parents=True, exist_ok=True)
    frame_scores: list[np.ndarray] = []
    frame_labels: list[np.ndarray] = []
    rows: list[dict[str, object]] = []
    offset = 0
    for row in tqdm(
        manifest.itertuples(index=False),
        total=len(manifest),
        desc=f"{args.dataset}/{model.evidence_id} evaluation",
        unit="video",
    ):
        curve_path = curves_dir / f"{row.key}.npz"
        if curve_path.exists():
            with np.load(curve_path, allow_pickle=False) as archive:
                corrected_score = np.asarray(archive["corrected_score"], dtype=np.float32)
                correction_size = float(archive["correction_size"])
                frame_indices = np.asarray(archive["frame_indices"], dtype=np.int64)
        else:
            length = int(row.valid_snippets)
            with np.load(str(row.hidden_path), allow_pickle=False) as archive:
                hidden = torch.from_numpy(
                    np.asarray(archive["hidden"][:length], dtype=np.float32).copy()
                )
                frame_indices = np.asarray(
                    archive["frame_indices"][:length], dtype=np.int64
                )
            host_score = torch.from_numpy(
                np.asarray(
                    np.load(str(row.host_score_path), allow_pickle=False),
                    dtype=np.float32,
                ).reshape(-1)[:length].copy()
            )
            evidence = infer_evidence(
                model,
                hidden,
                args.maximum_length,
                args.window_overlap,
                device,
            )
            with torch.no_grad():
                result = model.auditor(
                    host_score.unsqueeze(0).to(device),
                    evidence.unsqueeze(0).to(device),
                    torch.ones(1, length, dtype=torch.bool, device=device),
                )
            corrected_score = (
                result["corrected_score"][0].detach().cpu().numpy().astype(np.float32)
            )
            correction_size = float(result["correction_size"].detach().cpu())
            atomic_savez(
                curve_path,
                corrected_score=corrected_score,
                evidence=evidence.numpy(),
                delta_cross=result["delta_cross"][0].detach().cpu().numpy(),
                delta_within=result["delta_within"][0].detach().cpu().numpy(),
                correction_size=np.asarray(correction_size, dtype=np.float32),
                frame_indices=frame_indices,
            )
        frame_count = int(row.evaluation_frames)
        labels = gt[offset : offset + frame_count]
        if len(labels) != frame_count:
            raise RuntimeError(f"{row.key}: GT ended before the audited boundary")
        offset += frame_count
        frame_score = expand_snippet_scores(
            corrected_score, frame_indices, frame_count
        )
        frame_scores.append(frame_score)
        frame_labels.append(labels)
        rows.append(
            {
                "key": str(row.key),
                "binary_label": int(row.binary_label),
                "snippets": len(corrected_score),
                "correction_size": correction_size,
                "corrected_top_score": float(np.max(corrected_score)),
                "curve_path": str(curve_path),
            }
        )
    if offset != len(gt):
        raise RuntimeError(f"evaluated frames {offset} != GT frames {len(gt)}")

    metrics = score_curve_metrics(frame_scores, frame_labels, args.target_tpr)
    host_metrics = json.loads(Path(args.host_metrics).read_text(encoding="utf-8"))
    metrics.update(
        {
            "status": "pass",
            "dataset": args.dataset,
            "evidence": model.evidence_id,
            "primary_metric": "pooled_auc" if args.dataset == "ucf" else "pooled_ap",
            "primary_gain": metric_delta(
                metrics, host_metrics, "pooled_auc" if args.dataset == "ucf" else "pooled_ap"
            ),
            "pooled_auc_gain": metric_delta(metrics, host_metrics, "pooled_auc"),
            "pooled_ap_gain": metric_delta(metrics, host_metrics, "pooled_ap"),
            "cross_auc_gain": metric_delta(metrics, host_metrics, "cross_auc"),
            "within_auc_gain": metric_delta(metrics, host_metrics, "within_auc"),
            "macro_within_auc_gain": metric_delta(
                metrics, host_metrics, "macro_within_auc"
            ),
            "normal_frame_fpr_change": float(metrics["normal_fpr"]["normal_video_frame_fpr"])
            - float(host_metrics["normal_fpr"]["normal_video_frame_fpr"]),
            "mean_correction_size": float(
                np.mean([item["correction_size"] for item in rows])
            ),
            "test_videos": len(rows),
            "test_frames": len(gt),
            "window_policy": {
                "maximum_length": args.maximum_length,
                "overlap": args.window_overlap,
                "aggregation": "mean on overlapping contextual predictions",
            },
            "post_processing": "none",
            "test_used_for_selection": False,
        }
    )
    pd.DataFrame(rows).to_csv(output / "per_video.csv", index=False)
    metrics_path.write_text(
        json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(metrics, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
