from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from neuron_responsibility.baselines import build_baseline
from neuron_responsibility.desc_inference import desc_official_probabilities, desc_primary_anomaly_probability
from neuron_responsibility.evaluate import pad_chunks
from universal_neuron_adapter.data import resample_curve


def add_baseline_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--baseline", choices=["dsanet", "desc", "lagovad"], required=True)
    parser.add_argument("--baseline-root", required=True)
    parser.add_argument("--baseline-weight", default="")
    parser.add_argument("--sensitivity-weight", default="")
    parser.add_argument("--consistency-weight", default="")


def infer(adapter, baseline: str, dataset: str, clip: np.ndarray, device: torch.device) -> np.ndarray:
    if baseline == "desc":
        probabilities = desc_official_probabilities(adapter, clip, device)
        return desc_primary_anomaly_probability(probabilities, dataset).numpy().astype(np.float32)
    chunks, lengths = pad_chunks(clip, adapter.visual_length)
    output = adapter.forward_baseline(chunks.to(device), lengths.to(device))
    values = [
        torch.sigmoid(output.binary_logits[index, :length]).cpu()
        for index, length in enumerate(lengths.tolist())
    ]
    return torch.cat(values).numpy().astype(np.float32)


def main() -> None:
    parser = argparse.ArgumentParser(description="Cache exact frozen-baseline score curves for correction training.")
    add_baseline_arguments(parser)
    parser.add_argument("--dataset", choices=["ucf", "xd"], required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--split", choices=["train", "test"], required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    output = Path(args.out_dir)
    score_dir = output / "scores"
    score_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    adapter = build_baseline(args, str(device)).to(device).eval().requires_grad_(False)
    frame = pd.read_csv(args.manifest)
    rows = []
    with torch.no_grad():
        for row in tqdm(frame.itertuples(index=False), total=len(frame), desc=f"cache {args.baseline} {args.split}"):
            paths = str(row.clip_paths).split("|")
            # Training lists contain ten CLIP crops of the same video. The
            # hidden-state cache is crop-independent, so use the deterministic
            # centre crop instead of paying tenfold inference cost.
            paths = [paths[len(paths) // 2]]
            curves = []
            for clip_path in paths:
                clip = np.load(clip_path).astype(np.float32)
                curves.append(infer(adapter, args.baseline, args.dataset, clip, device))
            target_length = len(curves[0])
            score = np.mean([resample_curve(curve, target_length) for curve in curves], axis=0).astype(np.float32)
            target = score_dir / f"{row.key}.npy"
            np.save(target, score)
            rows.append({
                "key": str(row.key), "label": str(row.label), "binary_label": int(row.binary_label),
                "baseline_score_path": str(target), "snippets": len(score),
            })
    pd.DataFrame(rows).to_csv(output / "baseline_scores.csv", index=False)
    print(f"wrote {len(rows)} frozen-baseline curves to {output}", flush=True)


if __name__ == "__main__":
    main()
