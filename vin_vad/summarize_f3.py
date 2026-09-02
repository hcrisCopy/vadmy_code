from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


VARIANTS = ("w0", "w1", "w2", "w6")
DATASETS = ("ucf", "xd")


def main() -> None:
    parser = argparse.ArgumentParser(description="Apply the pre-registered Witness-VAD F3 gates")
    parser.add_argument("--root", required=True)
    args = parser.parse_args()
    root = Path(args.root)
    if "vadmy_data" not in root.resolve().parts or root.resolve().name != "f3_performance":
        raise ValueError("root must be the exact F3 directory inside sibling vadmy_data")

    all_metrics: dict[str, dict[str, dict[str, object]]] = {}
    rows: list[dict[str, object]] = []
    gates: dict[str, dict[str, bool | float]] = {}
    for dataset in DATASETS:
        all_metrics[dataset] = {}
        primary = "pooled_auc" if dataset == "ucf" else "pooled_ap"
        for variant in VARIANTS:
            path = root / dataset / variant / "evaluation" / "metrics.json"
            metrics = json.loads(path.read_text(encoding="utf-8"))
            all_metrics[dataset][variant] = metrics
            rows.append(
                {
                    "dataset": dataset,
                    "variant": variant,
                    "primary_metric": primary,
                    "primary_percent": 100.0 * float(metrics[primary]),
                    "gain_over_host_pp": 100.0 * float(metrics["primary_gain"]),
                    "pooled_auc_percent": 100.0 * float(metrics["pooled_auc"]),
                    "pooled_ap_percent": 100.0 * float(metrics["pooled_ap"]),
                    "cross_auc_percent": 100.0 * float(metrics["cross_auc"]),
                    "macro_within_auc_percent": 100.0 * float(metrics["macro_within_auc"]),
                    "abnormal_only_auc_percent": 100.0 * float(metrics["abnormal_only_auc"]),
                    "abnormal_only_ap_percent": 100.0 * float(metrics["abnormal_only_ap"]),
                    "normal_fpr_percent": 100.0
                    * float(metrics["normal_fpr"]["normal_video_frame_fpr"]),
                    "mean_absolute_correction": float(metrics["mean_absolute_correction"]),
                }
            )
        host = all_metrics[dataset]["w0"]
        video = all_metrics[dataset]["w1"]
        neuron = all_metrics[dataset]["w2"]
        full = all_metrics[dataset]["w6"]
        full_gain = float(full[primary]) - float(host[primary])
        full_over_video = float(full[primary]) - float(video[primary])
        neuron_gain = float(neuron[primary]) - float(host[primary])
        normal_fpr_change = float(full["normal_fpr"]["normal_video_frame_fpr"]) - float(
            host["normal_fpr"]["normal_video_frame_fpr"]
        )
        macro_within_gain = float(full["macro_within_auc"]) - float(host["macro_within_auc"])
        gates[dataset] = {
            "host_identity": abs(float(host["primary_gain"])) <= 1e-12,
            "full_gain_pp": 100.0 * full_gain,
            "full_gain_at_least_1pp": full_gain >= 0.01,
            "full_over_video_pp": 100.0 * full_over_video,
            "full_over_video_at_least_0_2pp": full_over_video >= 0.002,
            "neuron_gain_pp": 100.0 * neuron_gain,
            "neuron_gain_positive": neuron_gain > 0.0,
            "normal_fpr_change_pp": 100.0 * normal_fpr_change,
            "macro_within_gain_pp": 100.0 * macro_within_gain,
            "no_fpr_within_tradeoff": not (
                normal_fpr_change < 0.0 and macro_within_gain < -0.002
            ),
        }

    gate_names = (
        "host_identity",
        "full_gain_at_least_1pp",
        "full_over_video_at_least_0_2pp",
        "neuron_gain_positive",
        "no_fpr_within_tradeoff",
    )
    passed = all(bool(gates[dataset][name]) for dataset in DATASETS for name in gate_names)
    table = pd.DataFrame(rows)
    table.to_csv(root / "main_results.csv", index=False)
    decomposition = {
        "status": "go" if passed else "no_go",
        "gates": gates,
        "metrics": all_metrics,
        "tradeoff_rule": (
            "if Full lowers normal FPR, Macro-Within-AUC may not fall by more than 0.2 pp"
        ),
    }
    (root / "error_decomposition.json").write_text(
        json.dumps(decomposition, indent=2), encoding="utf-8"
    )
    (root / "metrics.json").write_text(
        json.dumps({"status": decomposition["status"], "gates": gates}, indent=2),
        encoding="utf-8",
    )
    configuration = {
        "datasets": list(DATASETS),
        "variants": list(VARIANTS),
        "epochs": 20,
        "seed": 42,
        "selection_policy": "last_checkpoint_only",
        "post_processing": "none",
        "primary_metrics": {"ucf": "pooled_auc", "xd": "pooled_ap"},
        "gates": {
            "full_over_host_pp": 1.0,
            "full_over_video_pp": 0.2,
            "neuron_over_host_pp": "strictly_positive",
            "maximum_macro_within_drop_when_fpr_improves_pp": 0.2,
        },
    }
    (root / "config.json").write_text(
        json.dumps(configuration, indent=2), encoding="utf-8"
    )
    summary_lines = [
        "# Witness-VAD F3 performance gate",
        "",
        f"- Decision: **{'GO' if passed else 'NO-GO'}**",
        "- All values below use the final epoch and the shared B0 evaluator; no post-processing.",
        "",
        "| Dataset | W0 host | W1 video | W2 neuron | W6 full | Full gain | Full-W1 | W2 gain |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for dataset in DATASETS:
        primary = "pooled_auc" if dataset == "ucf" else "pooled_ap"
        values = all_metrics[dataset]
        summary_lines.append(
            "| {dataset} {metric} | {w0:.3f} | {w1:.3f} | {w2:.3f} | {w6:.3f} | {gain:+.3f} | {over:+.3f} | {ngain:+.3f} |".format(
                dataset=dataset.upper(),
                metric="AUC" if dataset == "ucf" else "AP",
                w0=100 * float(values["w0"][primary]),
                w1=100 * float(values["w1"][primary]),
                w2=100 * float(values["w2"][primary]),
                w6=100 * float(values["w6"][primary]),
                gain=float(gates[dataset]["full_gain_pp"]),
                over=float(gates[dataset]["full_over_video_pp"]),
                ngain=float(gates[dataset]["neuron_gain_pp"]),
            )
        )
    summary_lines.extend(
        [
            "",
            "Read `main_results.csv` for the full table and `error_decomposition.json` for Cross/Within/FPR gates.",
        ]
    )
    (root / "summary.md").write_text("\n".join(summary_lines) + "\n", encoding="utf-8")
    print(json.dumps({"status": decomposition["status"], "gates": gates}, indent=2), flush=True)
    if not passed:
        raise RuntimeError("F3 is NO-GO; do not proceed to F4")


if __name__ == "__main__":
    main()
