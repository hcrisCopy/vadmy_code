#!/usr/bin/env python3
"""Official frame evaluation and causal residual ablation for CACC."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import average_precision_score, roc_auc_score
from tqdm import tqdm

PACKAGE_ROOT = Path(__file__).resolve().parent.parent
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))

from neuron_responsibility.baselines import build_baseline
from neuron_responsibility.cacc import CrossLayerAnomalyConceptCircuit
from neuron_responsibility.common import clean_output
from neuron_responsibility.train_cacc import group_features, pad_chunks


def metrics(truth: np.ndarray, snippet_scores: list[np.ndarray], repeat: int) -> dict[str, float]:
    prediction = np.repeat(np.concatenate(snippet_scores), repeat)
    usable = min(len(truth), len(prediction))
    return {"frame_auc": float(roc_auc_score(truth[:usable], prediction[:usable])),
            "frame_ap": float(average_precision_score(truth[:usable], prediction[:usable])),
            "frames": int(usable)}


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate CACC and released author baseline.")
    parser.add_argument("--baseline", choices=["dsanet", "desc", "lagovad"], required=True)
    parser.add_argument("--baseline-root", required=True); parser.add_argument("--baseline-weight", default="")
    parser.add_argument("--sensitivity-weight", default=""); parser.add_argument("--consistency-weight", default="")
    parser.add_argument("--dataset", choices=["ucf", "xd"], required=True)
    parser.add_argument("--test-list", required=True); parser.add_argument("--gt-path", required=True)
    parser.add_argument("--model-path", required=True); parser.add_argument("--out-dir", required=True)
    parser.add_argument("--frames-per-snippet", type=int, default=16)
    parser.add_argument("--device", default="cuda"); parser.add_argument("--clean", action="store_true")
    args = parser.parse_args()
    checkpoint = torch.load(args.model_path, map_location="cpu")
    if checkpoint.get("method") != CrossLayerAnomalyConceptCircuit.method_name:
        raise ValueError(f"unsupported checkpoint method: {checkpoint.get('method')}")
    config = checkpoint["run_config"]
    if config["baseline"] != args.baseline or config["dataset"] != args.dataset:
        raise ValueError("checkpoint baseline or dataset differs from command")
    output = clean_output(args.out_dir, args.clean); score_dir = output / "scores"; score_dir.mkdir(exist_ok=True)
    device = torch.device(args.device if torch.cuda.is_available() and args.device.startswith("cuda") else "cpu")
    author = build_baseline(args, str(device)).to(device).eval()
    adapted = build_baseline(args, str(device)).to(device)
    circuit = CrossLayerAnomalyConceptCircuit.from_config(checkpoint["circuit_config"]).to(device)
    adapted.attach_pre_temporal_conditioner(circuit)
    adapted.load_state_dict(checkpoint["model_state_dict"], strict=True); adapted.eval()
    frame = pd.read_csv(args.test_list)
    truth = np.load(args.gt_path).astype(np.int64).reshape(-1)
    all_scores = {"author": [], "adapted_full": [], "adapted_without_cacc_residual": [],
                  "cacc_gate": [], "cacc_deviation": [], "cacc_semantic": []}
    layer_sum = np.zeros(circuit.layers, dtype=np.float64); layer_count = 0; rows = []
    with torch.no_grad():
        for key_value, group in tqdm(list(frame.groupby("key", sort=False)), desc="evaluate CACC", unit="video"):
            key = str(key_value); cache_path = score_dir / f"{key}.npz"
            if cache_path.exists() and not args.clean:
                cached = np.load(cache_path)
                values = {name: cached[name] for name in all_scores}
                layer_mean = cached["layer_mean"]
            else:
                clip, hidden = group_features(group)
                clip_chunks, lengths = pad_chunks(clip, author.visual_length)
                hidden_chunks, hidden_lengths = pad_chunks(hidden, author.visual_length)
                if not torch.equal(lengths, hidden_lengths):
                    raise RuntimeError(f"{key}: modality lengths differ")
                clip_device = clip_chunks.to(device); hidden_device = hidden_chunks.to(device); lengths_device = lengths.to(device)
                author_output = author.forward_baseline(clip_device, lengths_device)
                full_output, records = adapted.forward_conditioned(clip_device, hidden_device, lengths_device)
                learned_gain = circuit.gain_logit.detach().clone(); circuit.gain_logit.zero_()
                no_residual_output, _ = adapted.forward_conditioned(clip_device, hidden_device, lengths_device)
                circuit.gain_logit.copy_(learned_gain)
                record = records[0]
                values = {name: [] for name in all_scores}; layer_parts = []
                for index, length in enumerate(lengths.tolist()):
                    values["author"].append(torch.sigmoid(author_output.binary_logits[index, :length]).cpu())
                    values["adapted_full"].append(torch.sigmoid(full_output.binary_logits[index, :length]).cpu())
                    values["adapted_without_cacc_residual"].append(torch.sigmoid(no_residual_output.binary_logits[index, :length]).cpu())
                    values["cacc_gate"].append(record["gate"][index, :length].cpu())
                    values["cacc_deviation"].append(record["deviation_gate"][index, :length].cpu())
                    values["cacc_semantic"].append(record["semantic_gate"][index, :length].cpu())
                    layer_parts.append(record["layer_weights"][index, :length].mean(dim=0).cpu())
                values = {name: torch.cat(parts).numpy().astype(np.float32) for name, parts in values.items()}
                layer_mean = torch.stack(layer_parts).mean(dim=0).numpy().astype(np.float32)
                np.savez_compressed(cache_path, **values, layer_mean=layer_mean)
            for name, value in values.items():
                all_scores[name].append(value)
            layer_sum += layer_mean; layer_count += 1
            rows.append([key, str(group.iloc[0]["label"]), len(values["author"])] + [float(values[name].mean()) for name in all_scores])
    result = {name: metrics(truth, values, args.frames_per_snippet) for name, values in all_scores.items()}
    result["layer_weights"] = {f"layer_{index + 1}": float(value) for index, value in enumerate(layer_sum / max(layer_count, 1))}
    result["learned_residual_scale"] = float((circuit.max_residual_scale * torch.tanh(circuit.gain_logit)).cpu())
    (output / "metrics.json").write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    with (output / "per_video.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle); writer.writerow(["key", "label", "snippets"] + [f"{name}_mean" for name in all_scores]); writer.writerows(rows)
    print(json.dumps(result, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
