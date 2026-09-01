from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from tqdm import tqdm

from vin_vad.evaluate import expand_snippet_scores, score_curve_metrics


def file_sha256(path: str) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def ensure_output_path(path: Path) -> None:
    if "vadmy_data" not in path.resolve().parts:
        raise ValueError("out-dir must be inside the sibling vadmy_data directory")


def clean_output(path: Path) -> None:
    ensure_output_path(path)
    if path.exists():
        shutil.rmtree(path)


def _manifest_issues(
    audited: pd.DataFrame,
    host: pd.DataFrame,
    split: str,
) -> dict[str, object]:
    audited_keys = audited["key"].astype(str)
    host_keys = host["key"].astype(str)
    duplicate_audited = sorted(audited_keys[audited_keys.duplicated()].unique().tolist())
    duplicate_host = sorted(host_keys[host_keys.duplicated()].unique().tolist())
    audited_set = set(audited_keys)
    host_set = set(host_keys)
    return {
        "split": split,
        "duplicate_audited_keys": duplicate_audited,
        "duplicate_host_keys": duplicate_host,
        "missing_host_keys": sorted(audited_set - host_set),
        "unexpected_host_keys": sorted(host_set - audited_set),
        "length_mismatches": [],
        "label_mismatches": [],
        "missing_score_files": [],
    }


def align_host_manifest(
    audited_manifest: str,
    host_manifest: str,
    split: str,
) -> tuple[pd.DataFrame, dict[str, object]]:
    """Join one audited split to frozen DSANet scores without resampling."""
    audited = pd.read_csv(audited_manifest)
    host = pd.read_csv(host_manifest)
    audited_required = {
        "key",
        "binary_label",
        "valid_snippets",
        "evaluation_frames",
        "hidden_path",
    }
    host_required = {"key", "binary_label", "baseline_score_path", "snippets"}
    if not audited_required.issubset(audited.columns):
        raise ValueError(
            f"{audited_manifest}: missing {sorted(audited_required - set(audited.columns))}"
        )
    if not host_required.issubset(host.columns):
        raise ValueError(
            f"{host_manifest}: missing {sorted(host_required - set(host.columns))}"
        )
    audited = audited.copy()
    host = host.copy()
    audited["key"] = audited["key"].astype(str)
    host["key"] = host["key"].astype(str)
    issues = _manifest_issues(audited, host, split)
    if issues["duplicate_audited_keys"] or issues["duplicate_host_keys"]:
        return pd.DataFrame(), issues

    host_by_key = host.set_index("key", drop=False)
    rows: list[dict[str, object]] = []
    for row in audited.itertuples(index=False):
        key = str(row.key)
        if key not in host_by_key.index:
            continue
        host_row = host_by_key.loc[key]
        score_path = Path(str(host_row.baseline_score_path))
        if not score_path.is_file():
            issues["missing_score_files"].append(str(score_path))
            continue
        score = np.asarray(np.load(score_path, allow_pickle=False), dtype=np.float32).reshape(-1)
        expected = int(row.valid_snippets)
        declared = int(host_row.snippets)
        if len(score) != expected or declared != expected:
            issues["length_mismatches"].append(
                {
                    "key": key,
                    "hidden_and_feature_snippets": expected,
                    "score_file_snippets": len(score),
                    "host_manifest_snippets": declared,
                }
            )
            continue
        if int(row.binary_label) != int(host_row.binary_label):
            issues["label_mismatches"].append(
                {
                    "key": key,
                    "audited": int(row.binary_label),
                    "host": int(host_row.binary_label),
                }
            )
            continue
        output = row._asdict()
        output["host_score_path"] = str(score_path)
        output["host_score_sha256"] = file_sha256(str(score_path))
        rows.append(output)
    return pd.DataFrame(rows), issues


def has_issues(issues: dict[str, object]) -> bool:
    return any(bool(value) for key, value in issues.items() if key != "split")


def evaluate_identity(
    manifest: pd.DataFrame,
    gt_path: str,
    output: Path,
    target_tpr: float,
) -> tuple[dict[str, object], pd.DataFrame]:
    gt = np.asarray(np.load(gt_path, allow_pickle=False), dtype=np.int8).reshape(-1)
    curves = output / "curves"
    curves.mkdir(parents=True, exist_ok=True)
    frame_scores: list[np.ndarray] = []
    frame_labels: list[np.ndarray] = []
    rows: list[dict[str, object]] = []
    offset = 0
    maximum_identity_error = 0.0
    for row in tqdm(
        manifest.itertuples(index=False),
        total=len(manifest),
        desc="B0 host identity",
        unit="video",
    ):
        host_score = np.asarray(
            np.load(str(row.host_score_path), allow_pickle=False), dtype=np.float32
        ).reshape(-1)
        corrected_score = host_score.copy()
        identity_error = float(np.max(np.abs(corrected_score - host_score)))
        maximum_identity_error = max(maximum_identity_error, identity_error)
        with np.load(str(row.hidden_path), allow_pickle=False) as archive:
            frame_indices = np.asarray(archive["frame_indices"], dtype=np.int64)[
                : int(row.valid_snippets)
            ]
        frame_count = int(row.evaluation_frames)
        frame_score = expand_snippet_scores(host_score, frame_indices, frame_count)
        label = gt[offset : offset + frame_count]
        if len(label) != frame_count:
            raise RuntimeError(f"{row.key}: GT ended before the audited video boundary")
        offset += frame_count
        frame_scores.append(frame_score)
        frame_labels.append(label)
        curve_path = curves / f"{row.key}.npz"
        np.savez_compressed(
            curve_path,
            host_score=host_score,
            corrected_score=corrected_score,
            frame_indices=frame_indices,
        )
        mixed = bool(np.any(label) and np.any(label == 0))
        rows.append(
            {
                "key": str(row.key),
                "binary_label": int(row.binary_label),
                "snippets": len(host_score),
                "evaluation_frames": frame_count,
                "host_top_score": float(np.max(host_score)),
                "host_mean_score": float(np.mean(host_score)),
                "within_auc": float(roc_auc_score(label, frame_score)) if mixed else np.nan,
                "identity_max_abs_error": identity_error,
                "curve_path": str(curve_path),
            }
        )
    if offset != len(gt):
        raise RuntimeError(f"audited video frames {offset} != GT frames {len(gt)}")
    metrics = score_curve_metrics(frame_scores, frame_labels, target_tpr)
    metrics.update(
        {
            "status": "pass" if maximum_identity_error == 0.0 else "fail",
            "host_identity_max_abs_error": maximum_identity_error,
            "test_videos": len(manifest),
            "test_frames": len(gt),
            "score_space": "official DSANet sigmoid(S_det)",
            "frame_expansion": "audited CLIP stride-16 boundaries; no smoothing",
        }
    )
    return metrics, pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Audit frozen DSANet host identity and establish the CVA-VAD B0 evaluator."
    )
    parser.add_argument("--dataset", choices=["ucf", "xd"], required=True)
    parser.add_argument("--train-manifest", required=True)
    parser.add_argument("--test-manifest", required=True)
    parser.add_argument("--train-host-manifest", required=True)
    parser.add_argument("--test-host-manifest", required=True)
    parser.add_argument("--gt-path", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--target-tpr", type=float, required=True)
    parser.add_argument("--clean", action="store_true")
    args = parser.parse_args()

    output = Path(args.out_dir)
    ensure_output_path(output)
    if args.clean:
        clean_output(output)
    output.mkdir(parents=True, exist_ok=True)
    signature = {
        "dataset": args.dataset,
        "target_tpr": args.target_tpr,
        "inputs": {
            name: {"path": value, "sha256": file_sha256(value)}
            for name, value in (
                ("train_manifest", args.train_manifest),
                ("test_manifest", args.test_manifest),
                ("train_host_manifest", args.train_host_manifest),
                ("test_host_manifest", args.test_host_manifest),
                ("gt_path", args.gt_path),
            )
        },
    }
    signature_path = output / "signature.json"
    if signature_path.exists() and json.loads(signature_path.read_text(encoding="utf-8")) != signature:
        raise RuntimeError("B0 inputs changed; rerun with --clean")
    signature_path.write_text(json.dumps(signature, indent=2), encoding="utf-8")
    metrics_path = output / "metrics.json"
    if metrics_path.exists() and (output / "per_video.csv").exists():
        print(f"reusing completed B0 evaluation: {metrics_path}", flush=True)
        print(metrics_path.read_text(encoding="utf-8"), flush=True)
        return

    aligned: dict[str, pd.DataFrame] = {}
    split_issues: dict[str, dict[str, object]] = {}
    for split, audited_path, host_path in (
        ("train", args.train_manifest, args.train_host_manifest),
        ("test", args.test_manifest, args.test_host_manifest),
    ):
        aligned[split], split_issues[split] = align_host_manifest(
            audited_path, host_path, split
        )
    audit = {
        "status": "fail" if any(has_issues(value) for value in split_issues.values()) else "pass",
        "dataset": args.dataset,
        "train_videos": len(aligned["train"]),
        "test_videos": len(aligned["test"]),
        "issues": split_issues,
    }
    (output / "audit.json").write_text(
        json.dumps(audit, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    if audit["status"] != "pass":
        raise RuntimeError(f"B0 alignment failed; inspect {output / 'audit.json'}")
    aligned["train"].to_csv(output / "train_aligned.csv", index=False)
    aligned["test"].to_csv(output / "test_aligned.csv", index=False)

    metrics, per_video = evaluate_identity(
        aligned["test"], args.gt_path, output, args.target_tpr
    )
    metrics["dataset"] = args.dataset
    per_video.to_csv(output / "per_video.csv", index=False)
    metrics_path.write_text(
        json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(metrics, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
