from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

import pandas as pd


BASELINES = ("lagovad", "desc", "dsanet", "vadclip")
DATASETS = ("ucf", "xd")
FORMAL_SEED = 234
EXPECTED_TEST_VIDEOS = {"ucf": 290, "xd": 800}


@dataclass(frozen=True)
class Stage:
    index: int
    name: str
    added_module: str
    disabled: tuple[str, ...]


STAGES = (
    Stage(0, "baseline", "frozen baseline", (
        "--disable-correction", "--disable-agreement", "--disable-event-gate",
        "--disable-video-suppression", "--disable-temporal",
    )),
    Stage(1, "neuron_correction", "primary-neuron score correction", (
        "--disable-agreement", "--disable-event-gate",
        "--disable-video-suppression", "--disable-temporal",
    )),
    Stage(2, "agreement_conflict", "multi-source agreement and conflict suppression", (
        "--disable-event-gate", "--disable-video-suppression", "--disable-temporal",
    )),
    Stage(3, "event_gate", "multi-scale neuron event gate", (
        "--disable-video-suppression", "--disable-temporal",
    )),
    Stage(4, "normal_suppression", "normal-video suppression", (
        "--disable-temporal",
    )),
    Stage(5, "full_temporal", "temporal persistence and boundary recovery", ()),
)


def data_path(value: str, label: str) -> Path:
    path = Path(value)
    if path.is_absolute():
        raise ValueError(f"{label} must be a relative path: {value}")
    data_root = (Path.cwd().parent / "vadmy_data").resolve()
    resolved = path.resolve()
    if resolved != data_root and data_root not in resolved.parents:
        raise ValueError(f"{label} must stay below ../vadmy_data: {value}")
    return path


def required_inputs(source: Path, dataset: str, baseline: str) -> dict[str, Path]:
    dataset_root = source / dataset
    baseline_root = dataset_root / baseline
    return {
        "baseline_train": baseline_root / "baseline_train/baseline_scores.csv",
        "baseline_test": baseline_root / "baseline_test/baseline_scores.csv",
        "primary_train": dataset_root / "expert/train/expert_scores.csv",
        "primary_test": dataset_root / "expert/test/expert_scores.csv",
        "normality_train": dataset_root / "normality_expert/train/expert3_scores.csv",
        "normality_test": dataset_root / "normality_expert/test/expert3_scores.csv",
        "context_train": dataset_root / "context_student/train/student_scores.csv",
        "context_test": dataset_root / "context_student/test/student_scores.csv",
        "correction": baseline_root / "correction/model_best.pth",
        "formal_metrics": baseline_root / "evaluation/metrics.json",
        "ground_truth": Path(f"../vadmy_data/annotations/{dataset}/gt.npy"),
    }


def audit_source(source: Path, dataset: str, baseline: str) -> dict[str, Path]:
    files = required_inputs(source, dataset, baseline)
    missing = [f"{name}: {path}" for name, path in files.items() if not path.is_file()]
    if missing:
        details = "\n  ".join(missing)
        raise FileNotFoundError(
            f"source run is incomplete for {baseline}/{dataset}:\n  {details}\n"
            "Run that single formal baseline first; do not combine artifacts from other runs."
        )

    experiment_path = source / "experiment.json"
    if experiment_path.is_file():
        experiment = json.loads(experiment_path.read_text(encoding="utf-8"))
        if int(experiment.get("seed", -1)) != FORMAL_SEED:
            raise ValueError(
                f"source run seed is {experiment.get('seed')}, expected {FORMAL_SEED}"
            )

    test_rows = len(pd.read_csv(files["baseline_test"]))
    if test_rows != EXPECTED_TEST_VIDEOS[dataset]:
        raise ValueError(
            f"{baseline}/{dataset} has {test_rows} test videos; "
            f"expected {EXPECTED_TEST_VIDEOS[dataset]}"
        )

    formal = json.loads(files["formal_metrics"].read_text(encoding="utf-8"))
    configuration = formal.get("configuration", {})
    if "context_diverse_weight" in configuration:
        raise ValueError(
            f"{files['formal_metrics']} belongs to the removed Top-64 diverse-expert protocol"
        )
    return files


def safe_clean(target: Path, output_root: Path) -> None:
    if not target.exists():
        return
    resolved_target = target.resolve()
    resolved_root = output_root.resolve()
    if resolved_target == resolved_root or resolved_root not in resolved_target.parents:
        raise ValueError(f"refusing to clean unsafe path: {resolved_target}")
    shutil.rmtree(resolved_target)


def evaluation_command(
    files: dict[str, Path], dataset: str, baseline: str, output: Path, stage: Stage,
) -> list[str]:
    return [
        sys.executable, "-m", "universal_neuron_adapter.evaluate",
        "--baseline-train-manifest", str(files["baseline_train"]),
        "--baseline-manifest", str(files["baseline_test"]),
        "--expert-train-manifest", str(files["primary_train"]),
        "--expert-manifest", str(files["primary_test"]),
        "--expert3-train-manifest", str(files["normality_train"]),
        "--expert3-manifest", str(files["normality_test"]),
        "--student-train-manifest", str(files["context_train"]),
        "--student-manifest", str(files["context_test"]),
        "--correction-model", str(files["correction"]),
        "--gt-path", str(files["ground_truth"]),
        "--baseline", baseline,
        "--dataset", dataset,
        "--out-dir", str(output),
        "--frames-per-snippet", "16",
        "--event-width", "41",
        "--event-weight", "1.0",
        "--normality-smoothing-blend", "0.25",
        "--persistence-weight", "1.0",
        "--gaussian-sigma", "0.5",
        "--advance-snippets", "0.5",
        "--device", "cuda",
        *stage.disabled,
    ]


def read_stage(stage: Stage, metrics_path: Path, dataset: str, baseline: str) -> dict[str, object]:
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    expected_disabled = {
        value.removeprefix("--disable-").replace("-", "_") for value in stage.disabled
    }
    actual_disabled = set(metrics.get("configuration", {}).get("disabled_components", []))
    if actual_disabled != expected_disabled:
        raise ValueError(
            f"stale stage at {metrics_path}: disabled={sorted(actual_disabled)}, "
            f"expected={sorted(expected_disabled)}; rerun with --clean"
        )
    metric_name = "auc" if dataset == "ucf" else "ap"
    baseline_value = 100.0 * float(metrics["baseline"][metric_name])
    value = 100.0 * float(metrics["corrected"][metric_name])
    return {
        "baseline": baseline,
        "dataset": dataset,
        "metric": metric_name.upper(),
        "stage_index": stage.index,
        "stage": stage.name,
        "added_module": stage.added_module,
        "baseline_value": baseline_value,
        "value": value,
        "gain_vs_baseline_pp": value - baseline_value,
    }


def run_dataset(
    source: Path, output_root: Path, dataset: str, baseline: str, clean: bool,
) -> list[dict[str, object]]:
    files = audit_source(source, dataset, baseline)
    target = output_root / dataset / baseline
    if clean:
        safe_clean(target, output_root)

    rows: list[dict[str, object]] = []
    previous: float | None = None
    for stage in STAGES:
        stage_output = target / f"m{stage.index}_{stage.name}"
        metrics_path = stage_output / "metrics.json"
        if metrics_path.is_file():
            print(f"[reuse] {baseline}/{dataset}/{stage.name}", flush=True)
        else:
            print(f"[run] {baseline}/{dataset}/{stage.name}", flush=True)
            subprocess.run(
                evaluation_command(files, dataset, baseline, stage_output, stage),
                check=True,
            )
        row = read_stage(stage, metrics_path, dataset, baseline)
        value = float(row["value"])
        row["gain_vs_previous_pp"] = 0.0 if previous is None else value - previous
        previous = value
        rows.append(row)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the current M0-M5 cumulative ablation for one frozen baseline."
    )
    parser.add_argument("--baseline", choices=BASELINES, required=True)
    parser.add_argument("--dataset", choices=(*DATASETS, "both"), default="both")
    parser.add_argument("--source-run", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--clean", action="store_true")
    args = parser.parse_args()

    source = data_path(args.source_run, "source-run")
    output = data_path(args.output_dir, "output-dir")
    datasets = DATASETS if args.dataset == "both" else (args.dataset,)
    rows: list[dict[str, object]] = []
    for dataset in datasets:
        rows.extend(run_dataset(source, output, dataset, args.baseline, args.clean))

    output.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(rows).sort_values(["dataset", "stage_index"])
    csv_path = output / f"{args.baseline}_cumulative.csv"
    json_path = output / f"{args.baseline}_cumulative.json"
    frame.to_csv(csv_path, index=False)
    json_path.write_text(
        json.dumps(frame.to_dict(orient="records"), indent=2), encoding="utf-8"
    )
    print(frame.to_string(index=False), flush=True)
    print(f"wrote {csv_path} and {json_path}", flush=True)


if __name__ == "__main__":
    main()
