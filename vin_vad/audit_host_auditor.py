from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from vin_vad.host_auditor import TwoAxisHostAuditor, masked_mean


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def ensure_output_path(path: Path) -> None:
    if "vadmy_data" not in path.resolve().parts:
        raise ValueError("out-dir must be inside the sibling vadmy_data directory")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Audit B3 two-axis host correction on a saved B2 training sample."
    )
    parser.add_argument("--dataset", choices=["ucf", "xd"], required=True)
    parser.add_argument("--train-manifest", required=True)
    parser.add_argument("--b2-sample", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--alpha-cross", type=float, required=True)
    parser.add_argument("--alpha-within", type=float, required=True)
    parser.add_argument("--audit-kappa-cross", type=float, required=True)
    parser.add_argument("--audit-kappa-within", type=float, required=True)
    parser.add_argument("--clean", action="store_true")
    args = parser.parse_args()

    output = Path(args.out_dir)
    ensure_output_path(output)
    if args.clean and output.exists():
        shutil.rmtree(output)
    output.mkdir(parents=True, exist_ok=True)
    config = {
        key: value
        for key, value in vars(args).items()
        if key not in {"clean", "out_dir"}
    }
    config["train_manifest_sha256"] = file_sha256(args.train_manifest)
    config["b2_sample_sha256"] = file_sha256(args.b2_sample)
    config_path = output / "config.json"
    if config_path.exists() and json.loads(config_path.read_text()) != config:
        raise RuntimeError("B3 configuration changed; rerun with --clean")
    config_path.write_text(json.dumps(config, indent=2), encoding="utf-8")
    summary_path = output / "summary.json"
    if summary_path.exists():
        print(f"reusing completed B3 audit: {summary_path}", flush=True)
        print(summary_path.read_text(encoding="utf-8"), flush=True)
        return

    with np.load(args.b2_sample, allow_pickle=False) as archive:
        key = str(archive["key"].item())
        evidence_np = np.asarray(archive["evidence"], dtype=np.float32)
    manifest = pd.read_csv(args.train_manifest)
    selected = manifest[manifest["key"].astype(str) == key]
    if len(selected) != 1:
        raise RuntimeError(f"B2 sample key {key!r} is not unique in B0 train manifest")
    row = selected.iloc[0]
    if int(row.binary_label) != 0:
        raise RuntimeError("B3 formal audit sample must be a normal training video")
    host_np = np.asarray(
        np.load(str(row.host_score_path), allow_pickle=False), dtype=np.float32
    ).reshape(-1)
    length = min(len(host_np), len(evidence_np))
    if length < 2:
        raise RuntimeError("B3 formal audit needs at least two aligned snippets")
    host = torch.from_numpy(host_np[:length]).unsqueeze(0)
    evidence = torch.from_numpy(evidence_np[:length]).unsqueeze(0)
    validity = torch.ones_like(host, dtype=torch.bool)

    auditor = TwoAxisHostAuditor(
        alpha_cross=args.alpha_cross,
        alpha_within=args.alpha_within,
        # B4 will estimate q statistics from normal training videos. These
        # neutral values are only for the B3 algebra audit, not model selection.
        normal_q_median=0.0,
        normal_q_mad=1.0,
        tau_normal=0.0,
    )
    identity = auditor(host, evidence, validity)
    identity_error = float((identity["corrected_score"] - host).abs().max())
    weights = torch.linspace(0.25, 1.25, length).unsqueeze(0)
    gradient_probe = (identity["corrected_score"] * weights).sum()
    gradient_probe.backward()
    cross_gradient = float(auditor.kappa_cross.grad.abs())
    within_gradient = float(auditor.kappa_within.grad.abs())
    auditor.zero_grad(set_to_none=True)

    auditor.kappa_cross.data.fill_(args.audit_kappa_cross)
    auditor.kappa_within.data.fill_(args.audit_kappa_within)
    full = auditor(host, evidence, validity)
    cross_only = auditor(host, evidence, validity, enable_within=False)
    within_only = auditor(host, evidence, validity, enable_cross=False)
    cross_constant_error = float(
        (full["delta_cross"] - full["delta_cross_video"].unsqueeze(1)).abs().max()
    )
    within_mean_error = float(masked_mean(full["delta_within"], validity).abs().max())
    cross_bound_violation = max(
        0.0, float(full["delta_cross"].abs().max()) - args.alpha_cross
    )
    within_bound_violation = max(
        0.0, float(full["delta_within"].abs().max()) - 2 * args.alpha_within
    )
    cross_positive_max = max(0.0, float(full["delta_cross"].max()))
    cross_independence_error = float(
        (full["delta_cross"] - cross_only["delta_cross"]).abs().max()
    )
    within_independence_error = float(
        (full["delta_within"] - within_only["delta_within"]).abs().max()
    )

    padded_host = torch.cat((host, torch.tensor([[0.0, 1.0]])), dim=1)
    padded_evidence = torch.cat((evidence, torch.tensor([[1e6, -1e6]])), dim=1)
    padded_validity = torch.cat(
        (validity, torch.zeros(1, 2, dtype=torch.bool)), dim=1
    )
    padded = auditor(padded_host, padded_evidence, padded_validity)
    padding_output_error = float(
        (full["corrected_score"] - padded["corrected_score"][:, :length]).abs().max()
    )
    padding_budget_error = abs(
        float(full["correction_size"]) - float(padded["correction_size"])
    )

    np.savez_compressed(
        output / "audit_arrays.npz",
        key=np.asarray(key),
        host_score=host.numpy(),
        evidence=evidence.numpy(),
        delta_cross=full["delta_cross"].detach().numpy(),
        delta_within=full["delta_within"].detach().numpy(),
        corrected_score=full["corrected_score"].detach().numpy(),
    )
    summary = {
        "status": "pass",
        "dataset": args.dataset,
        "scope": "B3 algebra and safety audit; no training or test evaluation",
        "sample_key": key,
        "normal_training_sample": True,
        "valid_snippets": length,
        "identity_max_abs_error": identity_error,
        "zero_point_cross_gradient_abs": cross_gradient,
        "zero_point_within_gradient_abs": within_gradient,
        "cross_positive_violation": cross_positive_max,
        "cross_constant_max_abs_error": cross_constant_error,
        "within_masked_mean_abs_error": within_mean_error,
        "cross_bound_violation": cross_bound_violation,
        "within_bound_violation": within_bound_violation,
        "cross_branch_independence_error": cross_independence_error,
        "within_branch_independence_error": within_independence_error,
        "padding_output_max_abs_error": padding_output_error,
        "padding_budget_abs_error": padding_budget_error,
        "test_split_used": False,
        "audit_alpha_cross": args.alpha_cross,
        "audit_alpha_within": args.alpha_within,
        "audit_kappa_cross": args.audit_kappa_cross,
        "audit_kappa_within": args.audit_kappa_within,
        "audit_constants_used_for_model_selection": False,
    }
    exact_checks = (
        identity_error,
        cross_positive_max,
        cross_constant_error,
        cross_bound_violation,
        within_bound_violation,
        cross_independence_error,
        within_independence_error,
        padding_output_error,
        padding_budget_error,
    )
    if max(exact_checks) > 1e-6 or within_mean_error > 1e-6:
        summary["status"] = "fail"
    if cross_gradient == 0.0 or within_gradient == 0.0:
        summary["status"] = "fail"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)
    if summary["status"] != "pass":
        raise RuntimeError("B3 host-auditor audit failed")


if __name__ == "__main__":
    main()
