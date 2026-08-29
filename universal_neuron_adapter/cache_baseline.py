from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as functional
from tqdm import tqdm

from universal_neuron_adapter.baseline_adapters import build_baseline
from universal_neuron_adapter.desc_inference import desc_official_probabilities, desc_primary_anomaly_probability
from universal_neuron_adapter.data import resample_curve


def pad_chunks(clip: np.ndarray, length: int) -> tuple[torch.Tensor, torch.Tensor]:
    chunks, lengths = [], []
    for start in range(0, len(clip), length):
        value = torch.from_numpy(clip[start:start + length])
        lengths.append(len(value))
        if len(value) < length:
            value = functional.pad(value, (0, 0, 0, length - len(value)))
        chunks.append(value)
    return torch.stack(chunks), torch.tensor(lengths, dtype=torch.long)


def infer(adapter, baseline: str, dataset: str, clip: np.ndarray, device: torch.device) -> tuple[np.ndarray, np.ndarray]:
    if baseline == "desc":
        probabilities = desc_official_probabilities(adapter, clip, device)
        binary = desc_primary_anomaly_probability(probabilities, dataset)
        return binary.numpy().astype(np.float32), probabilities["semantic"].numpy().astype(np.float32)
    chunks, lengths = pad_chunks(clip, adapter.visual_length)
    output = adapter.forward_baseline(chunks.to(device), lengths.to(device))
    binary_parts, semantic_parts = [], []
    for index, valid in enumerate(lengths.tolist()):
        binary_logits = output.binary_logits[index, :valid]
        semantic_logits = output.semantic_logits[index, :valid]
        binary_parts.append(torch.sigmoid(binary_logits).cpu())
        if baseline == "dsanet":
            temperature = float(adapter.options.temp)
            abnormal_mass = torch.sigmoid(binary_logits / temperature).unsqueeze(1)
            aligned = functional.softmax(semantic_logits / temperature, dim=-1)[:, 1:]
            conditional = aligned / aligned.sum(dim=1, keepdim=True).clamp_min(1e-12)
            semantic_parts.append(torch.cat([1.0 - abnormal_mass, abnormal_mass * conditional], dim=1).cpu())
        else:
            semantic_parts.append(functional.softmax(semantic_logits, dim=-1).cpu())
    return torch.cat(binary_parts).numpy().astype(np.float32), torch.cat(semantic_parts).numpy().astype(np.float32)


def main() -> None:
    parser = argparse.ArgumentParser(description="Cache frozen single-baseline binary and semantic probabilities.")
    parser.add_argument("--baseline", choices=["dsanet", "desc", "lagovad"], required=True)
    parser.add_argument("--baseline-root", required=True)
    parser.add_argument("--baseline-weight", default="")
    parser.add_argument("--sensitivity-weight", default="")
    parser.add_argument("--consistency-weight", default="")
    parser.add_argument("--dataset", choices=["ucf", "xd"], required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--split", choices=["train", "test"], required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    output = Path(args.out_dir)
    binary_dir, semantic_dir = output / "scores", output / "semantic_scores"
    binary_dir.mkdir(parents=True, exist_ok=True); semantic_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    adapter = build_baseline(args, str(device)).to(device).eval().requires_grad_(False)
    rows = []
    frame = pd.read_csv(args.manifest)
    with torch.no_grad():
        for row in tqdm(frame.itertuples(index=False), total=len(frame), desc=f"cache {args.baseline} {args.split}"):
            paths = str(row.clip_paths).split("|")
            # Official training features contain ten same-length spatial views
            # (`__0` ... `__9`), not temporal chunks.  Infer every view and
            # average below; selecting one arbitrary middle view is not the
            # official multi-view protocol.  Test manifests contain one path.
            binary_curves, semantic_curves = [], []
            for clip_path in paths:
                binary, semantic = infer(adapter, args.baseline, args.dataset, np.load(clip_path).astype(np.float32), device)
                binary_curves.append(binary)
                semantic_curves.append(semantic)
            length = len(binary_curves[0])
            binary = np.mean([resample_curve(curve, length) for curve in binary_curves], axis=0).astype(np.float32)
            semantic = np.mean(semantic_curves, axis=0).astype(np.float32)
            binary_path, semantic_path = binary_dir / f"{row.key}.npy", semantic_dir / f"{row.key}.npy"
            np.save(binary_path, binary); np.save(semantic_path, semantic)
            rows.append({"key": str(row.key), "label": str(row.label), "binary_label": int(row.binary_label), "baseline_score_path": str(binary_path), "semantic_score_path": str(semantic_path), "snippets": len(binary)})
    pd.DataFrame(rows).to_csv(output / "baseline_scores.csv", index=False)
    print(f"wrote {len(rows)} frozen-baseline curves to {output}", flush=True)


if __name__ == "__main__":
    main()
