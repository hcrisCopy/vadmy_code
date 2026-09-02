from __future__ import annotations

import argparse
import hashlib
import json
import shlex
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

from vin_vad.evaluate import score_curve_metrics


DATASETS = ("ucf", "xd")
EXPECTED_TEST_VIDEOS = {"ucf": 290, "xd": 800}
PRIMARY_METRIC = {"ucf": "pooled_auc", "xd": "pooled_ap"}
VARIANTS = {
    "u0_host": (
        "--disable-correction",
        "--disable-agreement",
        "--disable-event-gate",
        "--disable-video-suppression",
        "--disable-temporal",
    ),
    "u1_full": (),
    "u2_no_video_suppression": ("--disable-video-suppression",),
    "u3_no_local_neuron_correction": (
        "--disable-correction",
        "--disable-agreement",
        "--disable-event-gate",
    ),
    "u4_no_temporal_rules": ("--disable-temporal",),
}
COMPONENT_VARIANT = {
    "video_suppression": "u2_no_video_suppression",
    "local_neuron_correction": "u3_no_local_neuron_correction",
    "temporal_rules": "u4_no_temporal_rules",
}
LEGACY_SEED = 234
LEGACY_EVALUATOR_ARGUMENTS = (
    "--frames-per-snippet",
    "16",
    "--correction-weight",
    "0.2",
    "--neuron-weight",
    "0.1",
    "--event-width",
    "41",
    "--event-weight",
    "1.0",
    "--normality-gate-weight",
    "0.5",
    "--normality-smoothing-blend",
    "0.25",
    "--agreement-residual-weight",
    "0.0",
    "--triple-agreement-weight",
    "0.0",
    "--normal-suppression-weight",
    "1.0",
    "--persistence-weight",
    "1.0",
    "--gaussian-sigma",
    "0.5",
    "--advance-snippets",
    "0.5",
)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def require_data_path(value: str, label: str) -> Path:
    path = Path(value)
    if path.is_absolute():
        raise ValueError(f"{label} must be a relative path: {value}")
    data_root = (Path.cwd().parent / "vadmy_data").resolve()
    resolved = path.resolve()
    if resolved != data_root and data_root not in resolved.parents:
        raise ValueError(f"{label} must stay below ../vadmy_data: {value}")
    return path


def source_files(args: argparse.Namespace, dataset: str) -> dict[str, Path]:
    source = args.source_run / dataset
    normality = args.normality_cache_root / dataset / args.normality_cache_name
    context = args.context_cache_root / dataset / args.context_cache_name
    return {
        "baseline_train": source / "dsanet/baseline_train/baseline_scores.csv",
        "baseline_test": source / "dsanet/baseline_test/baseline_scores.csv",
        "expert_train": source / "expert/train/expert_scores.csv",
        "expert_test": source / "expert/test/expert_scores.csv",
        "normality_train": normality / "train/expert3_scores.csv",
        "normality_test": normality / "test/expert3_scores.csv",
        "context_train": context / "train/student_scores.csv",
        "context_test": context / "test/student_scores.csv",
        "correction": source / "dsanet/correction/model_best.pth",
        "ground_truth": Path(f"baseline/DSANet/list/gt_{dataset}.npy")
        if dataset == "ucf"
        else Path("baseline/DSANet/list/gt.npy"),
        "historical_metrics": (
            args.historical_run / dataset / "dsanet/evaluation/metrics.json"
        ),
    }


def audit_sources(files: dict[str, Path], dataset: str) -> dict[str, object]:
    missing = [str(path) for path in files.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError("missing F0 source files:\n  " + "\n  ".join(missing))
    row_counts = {
        name: len(pd.read_csv(path))
        for name, path in files.items()
        if path.suffix == ".csv"
    }
    test_counts = {
        row_counts[name]
        for name in ("baseline_test", "expert_test", "normality_test", "context_test")
    }
    if test_counts != {EXPECTED_TEST_VIDEOS[dataset]}:
        raise ValueError(
            f"{dataset} test manifests disagree or are incomplete: {row_counts}"
        )
    return {
        "dataset": dataset,
        "row_counts": row_counts,
        "inputs": {
            name: {"path": str(path), "sha256": file_sha256(path)}
            for name, path in files.items()
        },
    }


def evaluation_command(
    files: dict[str, Path],
    dataset: str,
    output: Path,
    disabled: tuple[str, ...],
    device: str,
) -> list[str]:
    return [
        sys.executable,
        "-m",
        "universal_neuron_adapter.evaluate",
        "--baseline-train-manifest",
        str(files["baseline_train"]),
        "--baseline-manifest",
        str(files["baseline_test"]),
        "--expert-train-manifest",
        str(files["expert_train"]),
        "--expert-manifest",
        str(files["expert_test"]),
        "--expert3-train-manifest",
        str(files["normality_train"]),
        "--expert3-manifest",
        str(files["normality_test"]),
        "--student-train-manifest",
        str(files["context_train"]),
        "--student-manifest",
        str(files["context_test"]),
        "--correction-model",
        str(files["correction"]),
        "--gt-path",
        str(files["ground_truth"]),
        "--baseline",
        "dsanet",
        "--dataset",
        dataset,
        "--out-dir",
        str(output),
        *LEGACY_EVALUATOR_ARGUMENTS,
        "--device",
        device,
        *disabled,
    ]


def load_frame_curves(
    evaluation_dir: Path,
    ground_truth: Path,
    frames_per_snippet: int,
) -> tuple[list[np.ndarray], list[np.ndarray]]:
    rows = pd.read_csv(evaluation_dir / "per_video.csv")
    truth = np.asarray(np.load(ground_truth, allow_pickle=False), dtype=np.int8).reshape(-1)
    frame_curves: list[np.ndarray] = []
    frame_labels: list[np.ndarray] = []
    offset = 0
    for row in rows.itertuples(index=False):
        with np.load(str(row.curve_path), allow_pickle=False) as archive:
            curve = np.asarray(archive["corrected"], dtype=np.float32).reshape(-1)
        frame_curve = np.repeat(curve, frames_per_snippet)
        end = offset + len(frame_curve)
        label = truth[offset:end]
        if len(label) != len(frame_curve):
            raise RuntimeError(f"{row.key}: ground truth ended before the score curve")
        frame_curves.append(frame_curve)
        frame_labels.append(label)
        offset = end
    if offset != len(truth):
        raise RuntimeError(f"score frames {offset} != ground-truth frames {len(truth)}")
    return frame_curves, frame_labels


def diagnostic_metrics(
    frame_curves: list[np.ndarray],
    frame_labels: list[np.ndarray],
    target_tpr: float,
) -> dict[str, object]:
    metrics = score_curve_metrics(frame_curves, frame_labels, target_tpr)
    abnormal = [
        (score, label)
        for score, label in zip(frame_curves, frame_labels)
        if np.any(label)
    ]
    abnormal_score = np.concatenate([item[0] for item in abnormal])
    abnormal_label = np.concatenate([item[1] for item in abnormal])
    metrics["abnormal_only_auc"] = float(
        roc_auc_score(abnormal_label, abnormal_score)
    )
    metrics["abnormal_only_ap"] = float(
        average_precision_score(abnormal_label, abnormal_score)
    )
    metrics["abnormal_videos"] = len(abnormal)
    return metrics


def dominant_source(component_drops: dict[str, float]) -> str:
    positive = {name: value for name, value in component_drops.items() if value > 0.0}
    if not positive:
        return "none_positive"
    ordered = sorted(positive.items(), key=lambda item: item[1], reverse=True)
    if len(ordered) > 1 and ordered[0][1] < 2.0 * ordered[1][1]:
        return "combination"
    return ordered[0][0]


def run_variant(
    args: argparse.Namespace,
    dataset: str,
    name: str,
    disabled: tuple[str, ...],
    files: dict[str, Path],
    source_audit: dict[str, object],
) -> dict[str, object]:
    output = args.output_dir / dataset / name
    output.mkdir(parents=True, exist_ok=True)
    signature = {
        "dataset": dataset,
        "variant": name,
        "disabled_components": [
            item.removeprefix("--disable-").replace("-", "_") for item in disabled
        ],
        "source_inputs": source_audit["inputs"],
    }
    signature_path = output / "signature.json"
    if signature_path.is_file():
        if json.loads(signature_path.read_text(encoding="utf-8")) != signature:
            raise RuntimeError(f"stale F0 output at {output}; rerun with --clean")
    else:
        signature_path.write_text(
            json.dumps(signature, indent=2), encoding="utf-8"
        )
    metrics_path = output / "metrics.json"
    legacy_metrics_path = output / "legacy_metrics.json"
    per_video_path = output / "per_video.csv"
    if (
        args.resume
        and metrics_path.is_file()
        and legacy_metrics_path.is_file()
        and per_video_path.is_file()
    ):
        print(f"[reuse] {dataset}/{name}", flush=True)
        raw = json.loads(legacy_metrics_path.read_text(encoding="utf-8"))
    else:
        command = evaluation_command(files, dataset, output, disabled, args.device)
        print(f"[run] {dataset}/{name}", flush=True)
        print(shlex.join(command), flush=True)
        subprocess.run(command, check=True)
        raw = json.loads(metrics_path.read_text(encoding="utf-8"))
        legacy_metrics_path.write_text(
            json.dumps(raw, indent=2), encoding="utf-8"
        )
    expected_disabled = set(signature["disabled_components"])
    actual_disabled = set(raw["configuration"]["disabled_components"])
    if actual_disabled != expected_disabled:
        raise RuntimeError(
            f"{metrics_path}: disabled={sorted(actual_disabled)}, "
            f"expected={sorted(expected_disabled)}"
        )
    curves, labels = load_frame_curves(
        output, files["ground_truth"], frames_per_snippet=16
    )
    diagnostic = diagnostic_metrics(curves, labels, args.target_tpr)
    diagnostic["legacy_evaluator"] = raw
    metrics_path.write_text(
        json.dumps(diagnostic, indent=2), encoding="utf-8"
    )
    return diagnostic


def write_summary(
    output: Path,
    rows: list[dict[str, object]],
    information: dict[str, object],
) -> None:
    frame = pd.DataFrame(rows)
    frame.to_csv(output / "comparison.csv", index=False)
    (output / "metrics.json").write_text(
        json.dumps(
            {
                f"{row['dataset']}/{row['variant']}": row
                for row in rows
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    (output / "information_source.json").write_text(
        json.dumps(information, indent=2), encoding="utf-8"
    )
    lines = [
        "# F0 Universal 提点验尸",
        "",
        f"状态：**{information['status']}**。",
        "",
        "| 数据集 | 变体 | 主指标(%) | 相对 host(pp) | 相对 Full 下降(pp) | "
        "Cross-AUC(%) | Macro-Within-AUC(%) | 异常视频 AUC/AP(%) | Normal FPR(%) |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        abnormal = (
            row["abnormal_only_auc"]
            if row["dataset"] == "ucf"
            else row["abnormal_only_ap"]
        )
        lines.append(
            f"| {row['dataset']} | {row['variant']} | "
            f"{row['primary_percent']:.3f} | {row['gain_vs_host_pp']:+.3f} | "
            f"{row['drop_from_full_pp']:+.3f} | {row['cross_auc_percent']:.3f} | "
            f"{row['macro_within_auc_percent']:.3f} | {abnormal:.3f} | "
            f"{row['normal_fpr_percent']:.3f} |"
        )
    lines.extend(["", "## 裁决", ""])
    for dataset in DATASETS:
        decision = information["datasets"][dataset]
        lines.append(
            f"- {dataset.upper()}：历史复现误差 "
            f"{decision['historical_reproduction_abs_error_pp']:.4f} pp；"
            f"最大信息来源为 **{decision['dominant_source']}**；"
            f"三项 leave-one-out 下降为 {decision['component_drop_from_full_pp']}。"
        )
    lines.extend(
        [
            "",
            "说明：leave-one-out 下降不能相加；dominant=combination 表示没有单一模块"
            "以两倍差距主导，增益依赖组合。",
        ]
    )
    (output / "summary.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Reproduce and dissect the cached DSANet Universal adapter."
    )
    parser.add_argument("--source-run", required=True)
    parser.add_argument("--normality-cache-root", required=True)
    parser.add_argument("--normality-cache-name", required=True)
    parser.add_argument("--context-cache-root", required=True)
    parser.add_argument("--context-cache-name", required=True)
    parser.add_argument("--historical-run", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--target-tpr", type=float, required=True)
    parser.add_argument("--device", required=True)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    args.source_run = require_data_path(args.source_run, "source-run")
    args.normality_cache_root = require_data_path(
        args.normality_cache_root, "normality-cache-root"
    )
    args.context_cache_root = require_data_path(
        args.context_cache_root, "context-cache-root"
    )
    args.historical_run = require_data_path(args.historical_run, "historical-run")
    args.output_dir = require_data_path(args.output_dir, "output-dir")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    config = {
        "source_run": str(args.source_run),
        "normality_cache_root": str(args.normality_cache_root),
        "normality_cache_name": args.normality_cache_name,
        "context_cache_root": str(args.context_cache_root),
        "context_cache_name": args.context_cache_name,
        "historical_run": str(args.historical_run),
        "output_dir": str(args.output_dir),
        "target_tpr": args.target_tpr,
        "device": args.device,
        "legacy_seed": LEGACY_SEED,
        "legacy_evaluator_arguments": list(LEGACY_EVALUATOR_ARGUMENTS),
        "git_revision": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True
        ).strip(),
        "variants": {name: list(value) for name, value in VARIANTS.items()},
    }
    (args.output_dir / "config.json").write_text(
        json.dumps(config, indent=2), encoding="utf-8"
    )

    all_metrics: dict[str, dict[str, dict[str, object]]] = {}
    audits = {}
    for dataset in DATASETS:
        files = source_files(args, dataset)
        audits[dataset] = audit_sources(files, dataset)
        all_metrics[dataset] = {}
        for name, disabled in VARIANTS.items():
            all_metrics[dataset][name] = run_variant(
                args, dataset, name, disabled, files, audits[dataset]
            )
    (args.output_dir / "source_audit.json").write_text(
        json.dumps(audits, indent=2), encoding="utf-8"
    )

    rows: list[dict[str, object]] = []
    information: dict[str, object] = {"status": "pass", "datasets": {}}
    for dataset in DATASETS:
        primary = PRIMARY_METRIC[dataset]
        host = float(all_metrics[dataset]["u0_host"][primary])
        full = float(all_metrics[dataset]["u1_full"][primary])
        history = json.loads(
            source_files(args, dataset)["historical_metrics"].read_text(
                encoding="utf-8"
            )
        )
        historical = float(
            history["corrected"]["auc" if dataset == "ucf" else "ap"]
        )
        reproduction_error = 100.0 * abs(full - historical)
        drops = {
            component: 100.0
            * (full - float(all_metrics[dataset][variant][primary]))
            for component, variant in COMPONENT_VARIANT.items()
        }
        information["datasets"][dataset] = {
            "primary_metric": primary,
            "historical_reference_percent": 100.0 * historical,
            "rerun_full_percent": 100.0 * full,
            "historical_reproduction_abs_error_pp": reproduction_error,
            "reproduction_pass_at_0.1pp": reproduction_error <= 0.1,
            "component_drop_from_full_pp": drops,
            "dominant_source": dominant_source(drops),
        }
        if reproduction_error > 0.1:
            information["status"] = "fail"
        for name, metrics in all_metrics[dataset].items():
            value = float(metrics[primary])
            rows.append(
                {
                    "dataset": dataset,
                    "variant": name,
                    "primary_metric": primary,
                    "primary_percent": 100.0 * value,
                    "gain_vs_host_pp": 100.0 * (value - host),
                    "drop_from_full_pp": 100.0 * (full - value),
                    "pooled_auc_percent": 100.0 * float(metrics["pooled_auc"]),
                    "pooled_ap_percent": 100.0 * float(metrics["pooled_ap"]),
                    "cross_auc_percent": 100.0 * float(metrics["cross_auc"]),
                    "macro_within_auc_percent": 100.0
                    * float(metrics["macro_within_auc"]),
                    "abnormal_only_auc": 100.0
                    * float(metrics["abnormal_only_auc"]),
                    "abnormal_only_ap": 100.0
                    * float(metrics["abnormal_only_ap"]),
                    "normal_fpr_percent": 100.0
                    * float(metrics["normal_fpr"]["normal_video_frame_fpr"]),
                }
            )
    write_summary(args.output_dir, rows, information)
    print((args.output_dir / "summary.md").read_text(encoding="utf-8"), flush=True)
    if information["status"] != "pass":
        raise SystemExit("F0 failed: Universal full did not reproduce within 0.1 pp")


if __name__ == "__main__":
    main()
