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


def centered_hidden(row: pd.Series, length: int) -> tuple[torch.Tensor, int]:
    valid = int(row.valid_snippets)
    start = max(0, (valid - length) // 2)
    with np.load(str(row.hidden_path), allow_pickle=False) as archive:
        hidden = np.asarray(archive["hidden"][start : start + length], dtype=np.float32)
    if hidden.shape != (length, 12, 768):
        raise ValueError(f"{row.key}: invalid centered hidden shape {hidden.shape}")
    return torch.from_numpy(hidden.copy()), start


def model_chain(
    model: torch.nn.Module,
    hidden: torch.Tensor,
    host_score: torch.Tensor,
    target: int,
) -> dict[str, torch.Tensor]:
    device = next(model.parameters()).device
    hidden = hidden.unsqueeze(0).to(device)
    host_score = host_score.unsqueeze(0).to(device)
    validity = torch.ones(1, hidden.shape[1], dtype=torch.bool, device=device)
    labels = torch.zeros(1, dtype=torch.float32, device=device)
    with torch.no_grad():
        result = model(hidden, host_score, validity, labels, update_statistics=False)
    distribution = result["distribution"]
    field = result["field"]
    return {
        "raw": distribution["normalized_hidden"][0, target].cpu(),
        "mu": distribution["mean"][0, target].cpu(),
        "residual": field["residual"][0, target].cpu(),
        "evidence": field["evidence"][0, target].cpu(),
        "correction": result["total_delta"][0, target].cpu(),
        "corrected_score": result["corrected_score"][0, target].cpu(),
    }


def mean_absolute_change(left: torch.Tensor, right: torch.Tensor) -> float:
    return float((left - right).abs().mean())


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Audit the fixed-target context replacement chain for E1 C3."
    )
    parser.add_argument("--dataset", choices=["ucf", "xd"], required=True)
    parser.add_argument("--validation-manifest", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--pairs", type=int, required=True)
    parser.add_argument("--maximum-length", type=int, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--device", required=True)
    parser.add_argument("--clean", action="store_true")
    args = parser.parse_args()

    output = Path(args.out_dir)
    ensure_output_path(output)
    if args.clean and output.exists():
        shutil.rmtree(output)
    output.mkdir(parents=True, exist_ok=True)
    signature = {
        key: value for key, value in vars(args).items() if key not in {"clean", "out_dir"}
    }
    signature["inputs_sha256"] = {
        "validation_manifest": file_sha256(args.validation_manifest),
        "checkpoint": file_sha256(args.checkpoint),
    }
    signature["intervention"] = (
        "keep the receiver target and its guard region fixed; replace every other "
        "context token with a different normal validation video"
    )
    signature_path = output / "signature.json"
    if signature_path.exists() and json.loads(signature_path.read_text(encoding="utf-8")) != signature:
        raise RuntimeError("replacement audit inputs changed; rerun with --clean")
    signature_path.write_text(json.dumps(signature, indent=2), encoding="utf-8")
    summary_path = output / "summary.json"
    if summary_path.exists() and (output / "per_pair.csv").exists():
        print(summary_path.read_text(encoding="utf-8"), flush=True)
        return

    if args.pairs < 1 or args.maximum_length < 7:
        raise ValueError("pairs must be positive and maximum-length must be at least seven")
    frame = pd.read_csv(args.validation_manifest)
    required = {"key", "binary_label", "hidden_path", "host_score_path", "valid_snippets"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"validation manifest is missing {sorted(missing)}")
    frame = frame[frame["binary_label"].astype(int) == 0].reset_index(drop=True)
    if len(frame) < 2:
        raise RuntimeError("replacement audit needs at least two normal validation videos")

    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model, _ = build_model(SimpleNamespace(**checkpoint["config"]))
    model.load_state_dict(checkpoint["model"])
    if model.evidence_id != "c3":
        raise ValueError("context replacement is the E1 C3 mechanism audit")
    model = model.to(torch.device(args.device)).eval()
    guard = int(model.predictor.guard_radius)
    rng = np.random.default_rng(args.seed)
    receivers = rng.integers(0, len(frame), size=args.pairs)
    donor_offsets = rng.integers(1, len(frame), size=args.pairs)

    rows: list[dict[str, object]] = []
    example: dict[str, np.ndarray] | None = None
    for pair_index, (receiver_index, donor_offset) in enumerate(
        tqdm(
            zip(receivers, donor_offsets),
            total=args.pairs,
            desc=f"{args.dataset}/c3 context replacement",
            unit="pair",
        )
    ):
        donor_index = (int(receiver_index) + int(donor_offset)) % len(frame)
        receiver = frame.iloc[int(receiver_index)]
        donor = frame.iloc[donor_index]
        length = min(
            args.maximum_length,
            int(receiver.valid_snippets),
            int(donor.valid_snippets),
        )
        if length <= 2 * guard + 2:
            raise RuntimeError("validation sequence is too short for guarded replacement")
        receiver_hidden, receiver_start = centered_hidden(receiver, length)
        donor_hidden, _ = centered_hidden(donor, length)
        target = length // 2
        replaced_hidden = receiver_hidden.clone()
        replace_mask = torch.ones(length, dtype=torch.bool)
        replace_mask[max(0, target - guard) : min(length, target + guard + 1)] = False
        replaced_hidden[replace_mask] = donor_hidden[replace_mask]
        host = np.asarray(
            np.load(str(receiver.host_score_path), allow_pickle=False), dtype=np.float32
        ).reshape(-1)[receiver_start : receiver_start + length]
        if len(host) != length:
            raise RuntimeError(f"{receiver.key}: host score ended before the audit window")
        host_score = torch.from_numpy(host.copy())
        original = model_chain(model, receiver_hidden, host_score, target)
        replaced = model_chain(model, replaced_hidden, host_score, target)
        record = {
            "pair": pair_index,
            "receiver": str(receiver.key),
            "donor": str(donor.key),
            "length": length,
            "target": target,
            "raw_max_error": float((original["raw"] - replaced["raw"]).abs().max()),
            "mu_change": mean_absolute_change(original["mu"], replaced["mu"]),
            "residual_change": mean_absolute_change(
                original["residual"], replaced["residual"]
            ),
            "evidence_change": mean_absolute_change(
                original["evidence"], replaced["evidence"]
            ),
            "correction_change": mean_absolute_change(
                original["correction"], replaced["correction"]
            ),
            "corrected_score_change": mean_absolute_change(
                original["corrected_score"], replaced["corrected_score"]
            ),
        }
        rows.append(record)
        if example is None:
            example = {
                "original_raw": original["raw"].numpy(),
                "replaced_raw": replaced["raw"].numpy(),
                "original_mu": original["mu"].numpy(),
                "replaced_mu": replaced["mu"].numpy(),
                "original_residual": original["residual"].numpy(),
                "replaced_residual": replaced["residual"].numpy(),
                "original_evidence": original["evidence"].numpy(),
                "replaced_evidence": replaced["evidence"].numpy(),
                "original_correction": original["correction"].numpy(),
                "replaced_correction": replaced["correction"].numpy(),
                "original_corrected_score": original["corrected_score"].numpy(),
                "replaced_corrected_score": replaced["corrected_score"].numpy(),
            }

    result_frame = pd.DataFrame(rows)
    result_frame.to_csv(output / "per_pair.csv", index=False)
    if example is not None:
        np.savez_compressed(output / "example_chain.npz", **example)
    change_names = [
        "mu_change",
        "residual_change",
        "evidence_change",
        "correction_change",
        "corrected_score_change",
    ]
    stable_fraction = {
        name: float((result_frame[name] > 1e-8).mean()) for name in change_names
    }
    summary = {
        "status": "pass"
        if float(result_frame["raw_max_error"].max()) <= 1e-7
        and min(stable_fraction.values()) >= 0.8
        else "fail",
        "dataset": args.dataset,
        "evidence": "c3",
        "pairs": len(result_frame),
        "target_raw_max_error": float(result_frame["raw_max_error"].max()),
        "mean_changes": {
            name: float(result_frame[name].mean()) for name in change_names
        },
        "changed_pair_fraction": stable_fraction,
        "acceptance": "raw max error <= 1e-7 and every downstream change occurs in >=80% of pairs",
        "scope": "mechanism-chain audit only; matched/random donor controls remain E4",
    }
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
