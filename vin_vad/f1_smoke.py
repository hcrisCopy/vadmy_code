from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from vin_vad.data import AuditorTrainingDataset
from vin_vad.witness_losses import witness_objective
from vin_vad.witness_model import WitnessVAD
from vin_vad.witness_router import masked_mean


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def ensure_output_path(path: Path) -> None:
    resolved = path.resolve()
    if "vadmy_data" not in resolved.parts or resolved.name != "f1_smoke":
        raise ValueError("out-dir must be the exact F1 directory inside sibling vadmy_data")


def audit_real_contract(b0_root: Path, maximum_length: int) -> dict[str, object]:
    report: dict[str, object] = {}
    for dataset_name in tqdm(("ucf", "xd"), desc="F1 B0 contract", unit="dataset"):
        manifest = b0_root / dataset_name / "evaluation" / "train_aligned.csv"
        frame = pd.read_csv(manifest)
        selected_indices = []
        for label in (0, 1):
            matches = frame.index[frame["binary_label"].astype(int) == label]
            if len(matches) == 0:
                raise RuntimeError(f"{dataset_name}: missing label {label} in B0 training manifest")
            selected_indices.append(int(matches[0]))
        dataset = AuditorTrainingDataset(str(manifest), maximum_length=maximum_length)
        items = [dataset[index] for index in selected_indices]
        for item in items:
            if item["hidden"].shape[1:] != (12, 768):
                raise RuntimeError(f"{dataset_name}: invalid hidden-state contract")
            if len(item["host_score"]) != len(item["hidden"]):
                raise RuntimeError(f"{dataset_name}: host/hidden length mismatch")
        report[dataset_name] = {
            "manifest": str(manifest),
            "checked_keys": [str(item["key"]) for item in items],
            "checked_labels": [int(item["label"]) for item in items],
            "checked_lengths": [len(item["host_score"]) for item in items],
            "hidden_tail_shape": [12, 768],
            "host_alignment": "pass",
        }
    return report


def gradient_audit(model: WitnessVAD, device: torch.device) -> dict[str, object]:
    generator = torch.Generator(device="cpu").manual_seed(73)
    hidden = torch.randn(2, 8, 12, 768, generator=generator).to(device)
    host = torch.tensor(
        [[0.75, 0.60, 0.55, 0.40, 0.30, 0.20, 0.0, 0.0],
         [0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45]],
        device=device,
    )
    validity = torch.tensor(
        [[True] * 6 + [False] * 2, [True] * 8], device=device
    )
    labels = torch.tensor([0.0, 1.0], device=device)
    components: dict[str, object] = {}
    for name in ("video", "witness_mil", "final_mil", "dense_normal", "sparse"):
        model.zero_grad(set_to_none=True)
        result = model(hidden, host, validity)
        losses = witness_objective(
            result,
            host,
            validity,
            labels,
            model.expert.neurons.sparsity_surrogate(),
        )
        losses[name].backward()
        groups = {
            "gate": model.expert.neurons.gate_logits,
            "signed_weight": model.expert.neurons.signed_weights,
        }
        gradient = {
            group: 0.0 if parameter.grad is None else float(parameter.grad.abs().sum())
            for group, parameter in groups.items()
        }
        gradient["witness_total"] = float(
            sum(
                parameter.grad.abs().sum().item()
                for parameter in model.expert.parameters()
                if parameter.grad is not None
            )
        )
        gradient["pass"] = gradient["witness_total"] > 0.0
        components[name] = gradient

    model.zero_grad(set_to_none=True)
    result = model(hidden, host, validity)
    losses = witness_objective(
        result,
        host,
        validity,
        labels,
        model.expert.neurons.sparsity_surrogate(),
    )
    losses["total"].backward()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    optimizer.step()
    identity = model(hidden, host, validity, eta_normal_override=0.0, eta_anomaly_override=0.0)
    return {
        "components": components,
        "all_components_reach_witness": all(bool(row["pass"]) for row in components.values()),
        "single_optimizer": optimizer.__class__.__name__,
        "joint_step_finite": bool(torch.isfinite(losses["total"]).item()),
        "identity_max_abs_error": float((identity["corrected_score"][validity] - host[validity]).abs().max()),
        "normal_max_delta": float(result["delta_normal"][validity].max()),
        "anomaly_mean_abs": float(masked_mean(result["delta_anomaly"], validity).abs().max()),
        "active_neurons_per_layer": model.expert.neurons.active_counts().tolist(),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Witness-VAD F1 structural and gradient smoke test")
    parser.add_argument("--b0-root", default="../vadmy_data/vin_vad/dsanet/b0")
    parser.add_argument("--out-dir", default="../vadmy_data/witness_vad/dsanet/f1_smoke")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--active-neurons", type=int, default=32)
    parser.add_argument("--temporal-width", type=int, default=64)
    parser.add_argument("--maximum-length", type=int, default=256)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    output = Path(args.out_dir)
    ensure_output_path(output)
    output.mkdir(parents=True, exist_ok=True)
    seed_everything(args.seed)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("F1 formal smoke requires CUDA")

    config = vars(args).copy()
    (output / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    data_contract = audit_real_contract(Path(args.b0_root), args.maximum_length)
    model = WitnessVAD(
        active=args.active_neurons,
        temporal_width=args.temporal_width,
        eta_normal=1.0,
        eta_anomaly=0.25,
    ).to(device)
    gradients = gradient_audit(model, device)
    gradients["device"] = str(device)
    gradients["data_contract"] = data_contract
    gradient_path = output / "gradient_report.json"
    gradient_path.write_text(json.dumps(gradients, indent=2), encoding="utf-8")

    active_ok = gradients["active_neurons_per_layer"] == [args.active_neurons] * 12
    status = "pass" if (
        gradients["all_components_reach_witness"]
        and gradients["joint_step_finite"]
        and gradients["identity_max_abs_error"] == 0.0
        and gradients["normal_max_delta"] <= 0.0
        and gradients["anomaly_mean_abs"] <= 1e-6
        and active_ok
    ) else "fail"
    metrics = {
        "status": status,
        "unit_test_report": "test_report.txt",
        "gradient_report": "gradient_report.json",
        "optimizer_count": 1,
        "trainable_parameters": sum(parameter.numel() for parameter in model.parameters()),
        "real_b0_datasets_checked": 2,
        **{key: gradients[key] for key in (
            "all_components_reach_witness",
            "joint_step_finite",
            "identity_max_abs_error",
            "normal_max_delta",
            "anomaly_mean_abs",
            "active_neurons_per_layer",
        )},
    }
    (output / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    summary = f"""# Witness-VAD F1 smoke summary

- Status: **{status.upper()}**
- Structure: signed top-32 neurons/layer + fixed d=1/d=2 temporal readout + 10-D video state + one routed residual.
- Training graph: one forward, one backward, one AdamW optimizer.
- Frozen-host boundary: the neuron-only expert API receives hidden states and masks only.
- Identity: max absolute error at eta_N=eta_A=0 is `{gradients['identity_max_abs_error']:.3e}`.
- Route constraints: normal maximum delta `{gradients['normal_max_delta']:.3e}`; anomaly video-mean absolute delta `{gradients['anomaly_mean_abs']:.3e}`.
- Gradient coverage: all five objective components reach witness parameters = `{gradients['all_components_reach_witness']}`.
- Real-data contract: UCF and XD B0 manifests each checked with one normal and one abnormal video.

Read `test_report.txt` for unit tests and `gradient_report.json` for per-loss gradients.
"""
    (output / "summary.md").write_text(summary, encoding="utf-8")
    print(json.dumps(metrics, indent=2), flush=True)
    if status != "pass":
        raise RuntimeError("F1 smoke failed; inspect gradient_report.json")


if __name__ == "__main__":
    main()
