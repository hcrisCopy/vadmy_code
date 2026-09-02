from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize the frozen DSANet E1 decision.")
    parser.add_argument("--root", required=True)
    args = parser.parse_args()
    root = Path(args.root)
    if "vadmy_data" not in root.resolve().parts:
        raise ValueError("root must be inside the sibling vadmy_data directory")

    rows: list[dict[str, object]] = []
    training_parameter_counts: set[int] = set()
    for dataset in ("ucf", "xd"):
        for evidence in ("c0", "c1", "c2", "c3", "c4"):
            training = json.loads(
                (root / dataset / evidence / "training" / "summary.json").read_text(
                    encoding="utf-8"
                )
            )
            metrics = json.loads(
                (root / dataset / evidence / "evaluation" / "metrics.json").read_text(
                    encoding="utf-8"
                )
            )
            training_parameter_counts.add(int(training["trainable_parameters"]))
            rows.append(
                {
                    "dataset": dataset,
                    "evidence": evidence,
                    "primary_metric": metrics["primary_metric"],
                    "primary_value": metrics[metrics["primary_metric"]],
                    "primary_gain": metrics["primary_gain"],
                    "pooled_auc": metrics["pooled_auc"],
                    "pooled_ap": metrics["pooled_ap"],
                    "cross_auc": metrics["cross_auc"],
                    "within_auc": metrics["within_auc"],
                    "macro_within_auc": metrics["macro_within_auc"],
                    "normal_fpr_change": metrics["normal_frame_fpr_change"],
                    "mean_correction_size": metrics["mean_correction_size"],
                    "kappa_cross": training["kappa_cross"],
                    "kappa_within": training["kappa_within"],
                    "field_support_size": training["field_support_size"],
                    "trainable_parameters": training["trainable_parameters"],
                }
            )
    table = pd.DataFrame(rows)
    table.to_csv(root / "comparison.csv", index=False)

    c3_detection_dominance: dict[str, bool] = {}
    replacement_status: dict[str, str] = {}
    conditional_nll_status: dict[str, bool] = {}
    for dataset in ("ucf", "xd"):
        subset = table[table["dataset"] == dataset].set_index("evidence")
        c3_detection_dominance[dataset] = float(subset.loc["c3", "primary_value"]) > float(
            subset.drop(index="c3")["primary_value"].max()
        )
        replacement = json.loads(
            (root / dataset / "c3" / "context_replacement" / "summary.json").read_text(
                encoding="utf-8"
            )
        )
        replacement_status[dataset] = str(replacement["status"])
        b1 = json.loads(
            (
                root.parent / "b1" / dataset / "summary.json"
            ).read_text(encoding="utf-8")
        )
        conditional_nll_status[dataset] = float(b1["validation_conditional_nll"]) < float(
            b1["validation_global_nll"]
        )

    checks = {
        "same_trainable_parameter_count": len(training_parameter_counts) == 1,
        "conditional_nll_beats_global_on_both_datasets": all(
            conditional_nll_status.values()
        ),
        "c3_primary_metric_beats_c0_c1_c2_c4_on_both_datasets": all(
            c3_detection_dominance.values()
        ),
        "c3_context_replacement_passes_on_both_datasets": all(
            value == "pass" for value in replacement_status.values()
        ),
    }
    summary = {
        "status": "pass" if all(checks.values()) else "no-go",
        "decision": (
            "retain the contextual-directional evidence claim"
            if all(checks.values())
            else "shrink the contextual-directional evidence claim before E2"
        ),
        "checks": checks,
        "per_dataset": {
            dataset: {
                "conditional_nll_beats_global": conditional_nll_status[dataset],
                "c3_detection_dominance": c3_detection_dominance[dataset],
                "context_replacement": replacement_status[dataset],
            }
            for dataset in ("ucf", "xd")
        },
        "primary_metrics": {"ucf": "pooled_auc", "xd": "pooled_ap"},
        "selection_policy": "fixed final epoch; no test-driven checkpoint or hyperparameter selection",
        "post_processing": "none",
    }
    (root / "summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(table.to_string(index=False), flush=True)
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
