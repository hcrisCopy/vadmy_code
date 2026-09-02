from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


DATASETS = ("ucf", "xd")
VARIANTS = ("w2", "w6")
ETA_VALUES = (0.25, 0.35, 0.60)


def eta_tag(value: float) -> str:
    return f"eta_{value:.2f}".replace(".", "_")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Summarize the inference-only Witness-VAD eta_A probe"
    )
    parser.add_argument("--root", required=True)
    parser.add_argument("--f3-root", required=True)
    args = parser.parse_args()
    root = Path(args.root)
    f3_root = Path(args.f3_root)
    if root.resolve().name != "f3_eta_probe" or "vadmy_data" not in root.resolve().parts:
        raise ValueError("root must be the exact f3_eta_probe directory inside vadmy_data")

    rows: list[dict[str, object]] = []
    for dataset in DATASETS:
        primary = "pooled_auc" if dataset == "ucf" else "pooled_ap"
        for variant in VARIANTS:
            learned_path = f3_root / dataset / variant / "evaluation" / "metrics.json"
            learned = json.loads(learned_path.read_text(encoding="utf-8"))
            rows.append(
                {
                    "dataset": dataset,
                    "variant": variant,
                    "eta_setting": "learned",
                    "eta_anomaly": None,
                    "primary_metric": primary,
                    "primary_percent": 100.0 * float(learned[primary]),
                    "gain_over_host_pp": 100.0 * float(learned["primary_gain"]),
                    "cross_auc_gain_pp": 100.0 * float(learned["cross_auc_gain"]),
                    "macro_within_auc_gain_pp": 100.0
                    * float(learned["macro_within_auc_gain"]),
                    "abnormal_only_auc_percent": 100.0
                    * float(learned["abnormal_only_auc"]),
                    "abnormal_only_ap_percent": 100.0
                    * float(learned["abnormal_only_ap"]),
                    "normal_fpr_change_pp": 100.0
                    * float(learned["normal_frame_fpr_change"]),
                    "mean_absolute_correction": float(learned["mean_absolute_correction"]),
                }
            )
            for eta in ETA_VALUES:
                path = root / dataset / variant / eta_tag(eta) / "metrics.json"
                metrics = json.loads(path.read_text(encoding="utf-8"))
                rows.append(
                    {
                        "dataset": dataset,
                        "variant": variant,
                        "eta_setting": "fixed",
                        "eta_anomaly": eta,
                        "primary_metric": primary,
                        "primary_percent": 100.0 * float(metrics[primary]),
                        "gain_over_host_pp": 100.0 * float(metrics["primary_gain"]),
                        "cross_auc_gain_pp": 100.0 * float(metrics["cross_auc_gain"]),
                        "macro_within_auc_gain_pp": 100.0
                        * float(metrics["macro_within_auc_gain"]),
                        "abnormal_only_auc_percent": 100.0
                        * float(metrics["abnormal_only_auc"]),
                        "abnormal_only_ap_percent": 100.0
                        * float(metrics["abnormal_only_ap"]),
                        "normal_fpr_change_pp": 100.0
                        * float(metrics["normal_frame_fpr_change"]),
                        "mean_absolute_correction": float(metrics["mean_absolute_correction"]),
                    }
                )

    frame = pd.DataFrame(rows)
    frame.to_csv(root / "eta_probe.csv", index=False)
    ucf_fixed = frame[
        (frame["dataset"] == "ucf") & (frame["eta_setting"] == "fixed")
    ]
    best_index = ucf_fixed["gain_over_host_pp"].idxmax()
    best_ucf = ucf_fixed.loc[best_index]
    scale_only_sufficient = float(best_ucf["gain_over_host_pp"]) >= 0.2
    decision = "retain_current_structure" if scale_only_sufficient else "redesign_correction"
    result = {
        "decision": decision,
        "criterion": "best fixed-eta UCF gain must reach at least +0.2 pp",
        "best_ucf": {
            "variant": str(best_ucf["variant"]),
            "eta_anomaly": float(best_ucf["eta_anomaly"]),
            "gain_over_host_pp": float(best_ucf["gain_over_host_pp"]),
            "primary_percent": float(best_ucf["primary_percent"]),
        },
        "training_performed": False,
        "checkpoint_source": str(f3_root),
    }
    (root / "decision.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    (root / "config.json").write_text(
        json.dumps(
            {
                "datasets": list(DATASETS),
                "variants": list(VARIANTS),
                "eta_anomaly": list(ETA_VALUES),
                "training": "none",
                "post_processing": "none",
                "selection": "reuse F3 test-best checkpoint",
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    lines = [
        "# Witness-VAD F3.1 eta_A probe",
        "",
        f"- Decision: **{decision}**",
        "- This is inference-only diagnosis on the existing F3 best checkpoints; no training or post-processing.",
        "",
        "| Dataset | Variant | eta_A | Primary | Gain over host | Cross gain | Macro-Within gain | Mean abs correction |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        eta = "learned" if row["eta_setting"] == "learned" else f"{row['eta_anomaly']:.2f}"
        lines.append(
            "| {dataset} | {variant} | {eta} | {primary:.3f} | {gain:+.3f} | {cross:+.3f} | {within:+.3f} | {correction:.6f} |".format(
                dataset=str(row["dataset"]).upper(),
                variant=str(row["variant"]).upper(),
                eta=eta,
                primary=float(row["primary_percent"]),
                gain=float(row["gain_over_host_pp"]),
                cross=float(row["cross_auc_gain_pp"]),
                within=float(row["macro_within_auc_gain_pp"]),
                correction=float(row["mean_absolute_correction"]),
            )
        )
    lines.extend(
        [
            "",
            "Scale-only is sufficient only when a fixed eta_A reaches at least +0.2 pp on UCF. Otherwise the evidence/router formula, rather than correction magnitude alone, must change.",
        ]
    )
    (root / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
