from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import torch


def parameter_max_difference(
    first: dict[str, torch.Tensor], second: dict[str, torch.Tensor]
) -> float:
    return max(
        float((first[name].cpu() - second[name].cpu()).abs().max())
        for name in first
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit Witness-VAD F2 determinism and resume")
    parser.add_argument("--reference-dir", required=True)
    parser.add_argument("--resumed-dir", required=True)
    parser.add_argument("--interruption-record", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--relative-loss-tolerance", type=float, required=True)
    parser.add_argument("--parameter-tolerance", type=float, required=True)
    args = parser.parse_args()

    output = Path(args.out_dir)
    if "vadmy_data" not in output.resolve().parts or output.resolve().name != "f2_train_contract":
        raise ValueError("out-dir must be the exact F2 directory inside sibling vadmy_data")
    reference = Path(args.reference_dir)
    resumed = Path(args.resumed_dir)
    interruption = json.loads(Path(args.interruption_record).read_text(encoding="utf-8"))
    reference_history = json.loads((reference / "history.json").read_text(encoding="utf-8"))
    resumed_history = json.loads((resumed / "history.json").read_text(encoding="utf-8"))
    reference_checkpoint = torch.load(
        reference / "checkpoints" / "last.pt", map_location="cpu", weights_only=False
    )
    resumed_checkpoint = torch.load(
        resumed / "checkpoints" / "last.pt", map_location="cpu", weights_only=False
    )
    first_loss_reference = float(reference_history[0]["total"])
    first_loss_resumed = float(resumed_history[0]["total"])
    relative_error = abs(first_loss_reference - first_loss_resumed) / max(
        abs(first_loss_reference), 1e-12
    )
    lr_continuity_error = abs(
        float(resumed_history[1]["lr_start"]) - float(resumed_history[0]["lr_end"])
    )
    parameter_difference = parameter_max_difference(
        reference_checkpoint["model"], resumed_checkpoint["model"]
    )
    required_checkpoint_keys = {
        "model",
        "optimizer",
        "scheduler",
        "epoch",
        "history",
        "config",
        "python_rng",
        "numpy_rng",
        "torch_rng",
        "cuda_rng",
    }
    checkpoint_complete = required_checkpoint_keys <= set(resumed_checkpoint)
    passed = (
        len(reference_history) == 2
        and len(resumed_history) == 2
        and int(resumed_checkpoint["epoch"]) == 2
        and interruption["status"] == "planned_stop"
        and int(interruption["completed_epochs"]) == 1
        and relative_error <= args.relative_loss_tolerance
        and lr_continuity_error <= 1e-12
        and parameter_difference <= args.parameter_tolerance
        and checkpoint_complete
        and not bool(resumed_checkpoint["config"]["test_data_used"])
        and resumed_checkpoint["config"]["selection_policy"] == "last_checkpoint_only"
    )
    report = {
        "status": "pass" if passed else "fail",
        "reference_completed_epochs": len(reference_history),
        "resumed_completed_epochs": len(resumed_history),
        "checkpoint_epoch": int(resumed_checkpoint["epoch"]),
        "interruption_status": interruption["status"],
        "interruption_completed_epochs": int(interruption["completed_epochs"]),
        "first_epoch_total_loss_reference": first_loss_reference,
        "first_epoch_total_loss_interrupted_run": first_loss_resumed,
        "first_epoch_relative_error": relative_error,
        "relative_loss_tolerance": args.relative_loss_tolerance,
        "lr_end_epoch_1": float(resumed_history[0]["lr_end"]),
        "lr_start_epoch_2": float(resumed_history[1]["lr_start"]),
        "lr_continuity_error": lr_continuity_error,
        "final_parameter_max_abs_difference": parameter_difference,
        "parameter_tolerance": args.parameter_tolerance,
        "checkpoint_complete": checkpoint_complete,
        "checkpoint_keys": sorted(resumed_checkpoint),
        "test_data_used": bool(resumed_checkpoint["config"]["test_data_used"]),
        "selection_policy": resumed_checkpoint["config"]["selection_policy"],
        "maximum_peak_gpu_memory_mb": max(
            float(row["peak_gpu_memory_mb"])
            for row in reference_history + resumed_history
        ),
        "mean_epoch_seconds": sum(
            float(row["seconds"]) for row in reference_history + resumed_history
        ) / (len(reference_history) + len(resumed_history)),
        "git_commit": resumed_checkpoint["config"]["git_commit"],
    }
    (output / "resume_report.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    canonical_checkpoints = output / "checkpoints"
    canonical_checkpoints.mkdir(exist_ok=True)
    shutil.copy2(
        resumed / "checkpoints" / "last.pt", canonical_checkpoints / "last.pt"
    )
    (output / "metrics.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    root_config = {
        "stage": "f2_train_contract",
        "reference_dir": str(reference),
        "resumed_dir": str(resumed),
        "training": resumed_checkpoint["config"],
        "relative_loss_tolerance": args.relative_loss_tolerance,
        "parameter_tolerance": args.parameter_tolerance,
    }
    (output / "config.json").write_text(json.dumps(root_config, indent=2), encoding="utf-8")
    summary = f"""# Witness-VAD F2 training-contract summary

- Status: **{report['status'].upper()}**
- Resume: epoch 1 checkpoint resumed to epoch 2 with LR continuity error `{lr_continuity_error:.3e}`.
- Determinism: two same-seed first-epoch losses have relative error `{relative_error:.3e}`.
- Final reference/resumed parameter max difference: `{parameter_difference:.3e}`.
- Peak GPU memory: `{report['maximum_peak_gpu_memory_mb']:.1f} MiB`.
- Mean epoch time: `{report['mean_epoch_seconds']:.2f} s`.
- Checkpoint: `checkpoints/last.pt` (the verified resumed checkpoint).
- Training read test data: `{report['test_data_used']}`; selection is `{report['selection_policy']}`.
"""
    (output / "summary.md").write_text(summary, encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)
    if not passed:
        raise RuntimeError("F2 contract failed; inspect resume_report.json")


if __name__ == "__main__":
    main()
