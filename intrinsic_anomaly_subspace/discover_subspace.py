#!/usr/bin/env python3
"""Localize critical layers and latent anomaly neurons with V-FIND equations."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

from .common import clean_output, load_pair, read_pair_manifest, save_json, seed_everything, write_csv


def discovery_video_means(pair_manifest: str) -> tuple[np.ndarray, np.ndarray]:
    frame = read_pair_manifest(pair_manifest, "discovery")
    positive, negative = [], []
    for path in tqdm(frame["pair_path"], desc="load discovery videos", unit="video"):
        pos, neg = load_pair(str(path))
        positive.append(pos.mean(axis=0))
        negative.append(neg.mean(axis=0))
    return np.stack(positive).astype(np.float32), np.stack(negative).astype(np.float32)


def layer_statistics(positive: np.ndarray, negative: np.ndarray, epsilon: float) -> tuple[np.ndarray, np.ndarray]:
    """V-FIND Eq. (5), (7), and Appendix Eq. (17)."""
    positive_centroid = positive.mean(axis=0)
    negative_centroid = negative.mean(axis=0)
    numerator = np.sum(positive_centroid * negative_centroid, axis=1)
    denominator = np.linalg.norm(positive_centroid, axis=1) * np.linalg.norm(negative_centroid, axis=1)
    d_cos = 1.0 - numerator / (denominator + epsilon)
    within_variance = 0.5 * (
        positive.var(axis=0, ddof=1).mean(axis=1) + negative.var(axis=0, ddof=1).mean(axis=1)
    )
    hidden_dim = positive.shape[-1]
    d_shift = np.linalg.norm(positive_centroid - negative_centroid, axis=1) / np.sqrt(
        hidden_dim * within_variance + epsilon
    )
    return d_cos.astype(np.float32), d_shift.astype(np.float32)


def train_layer_probe(
    features: np.ndarray,
    labels: np.ndarray,
    output: Path,
    epochs: int,
    lr: float,
    weight_decay: float,
    checkpoint_interval: int,
    device: torch.device,
) -> tuple[np.ndarray, float]:
    checkpoint = output / "checkpoint_last.pth"
    model = torch.nn.Linear(features.shape[1], 1).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    start = 0
    if checkpoint.exists():
        state = torch.load(checkpoint, map_location="cpu")
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        start = int(state["epoch"]) + 1
    x = torch.from_numpy(features).to(device)
    y = torch.from_numpy(labels).to(device)
    progress = tqdm(range(start, epochs), desc=f"probe layer {output.name}", unit="epoch", leave=False)
    for epoch in progress:
        logits = model(x).squeeze(1)
        loss = F.binary_cross_entropy_with_logits(logits, y)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        progress.set_postfix(loss=f"{float(loss.detach()):.4f}")
        if (epoch + 1) % checkpoint_interval == 0 or epoch + 1 == epochs:
            torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict(), "epoch": epoch}, checkpoint)
    return model.weight.detach().cpu().numpy()[0].astype(np.float32), float(model.bias.detach().cpu())


def neuron_effect_size(
    positive: np.ndarray,
    negative: np.ndarray,
    weight: np.ndarray,
    epsilon: float,
) -> np.ndarray:
    """V-FIND Eq. (12)-(15) and Appendix Eq. (20)-(21)."""
    absolute_weight = np.abs(weight)
    positive_response = np.abs(positive) * absolute_weight
    negative_response = np.abs(negative) * absolute_weight
    positive_mean = positive_response.mean(axis=0)
    negative_mean = negative_response.mean(axis=0)
    positive_var = np.sum(
        ((positive_response - positive_mean) / (absolute_weight + epsilon)) ** 2, axis=0
    ) / max(1, len(positive) - 1)
    negative_var = np.sum(
        ((negative_response - negative_mean) / (absolute_weight + epsilon)) ** 2, axis=0
    ) / max(1, len(negative) - 1)
    pooled = np.sqrt(
        ((len(positive) - 1) * positive_var + (len(negative) - 1) * negative_var)
        / max(1, len(positive) + len(negative) - 2)
    )
    return (np.abs(positive_mean - negative_mean) / (pooled + epsilon)).astype(np.float32)


def main() -> None:
    parser = argparse.ArgumentParser(description="V-FIND-style critical-layer and anomaly-neuron discovery.")
    parser.add_argument("--pair-manifest", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument(
        "--layer-rule",
        choices=["intersection", "threshold_union"],
        default="threshold_union",
        help=(
            "V-FIND uses intersection. threshold_union is the explicit VAD adaptation: "
            "retain layers highlighted by either complementary signal, then let neuron effect size filter them."
        ),
    )
    parser.add_argument("--effect-threshold", type=float, default=1.5)
    parser.add_argument("--probe-epochs", type=int, default=300)
    parser.add_argument("--probe-lr", type=float, default=1e-2)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--checkpoint-interval", type=int, default=10)
    parser.add_argument("--epsilon", type=float, default=1e-8)
    parser.add_argument("--seed", type=int, default=234)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--clean", action="store_true")
    args = parser.parse_args()
    if args.effect_threshold <= 0 or args.probe_epochs <= 0 or args.probe_lr <= 0 or args.epsilon <= 0:
        parser.error("threshold, epochs, learning rate, and epsilon must be positive")
    output = clean_output(args.out_dir, args.clean)
    result_path = output / "selected_subspace.json"
    if result_path.exists() and not args.clean:
        print(f"reuse completed subspace: {result_path}", flush=True)
        return
    seed_everything(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() and args.device.startswith("cuda") else "cpu")
    positive, negative = discovery_video_means(args.pair_manifest)
    if positive.shape != negative.shape or positive.shape[0] < 3:
        raise RuntimeError(f"need at least three paired discovery videos, got {positive.shape}, {negative.shape}")
    d_cos, d_shift = layer_statistics(positive, negative, args.epsilon)
    tau_cos = float(d_cos.mean() + d_cos.std())
    tau_shift = float(d_shift.mean() + d_shift.std())
    above_cos = d_cos > tau_cos
    above_shift = d_shift > tau_shift
    intersection = np.flatnonzero(above_cos & above_shift).astype(int).tolist()
    threshold_union = np.flatnonzero(above_cos | above_shift).astype(int).tolist()
    critical = intersection if args.layer_rule == "intersection" else threshold_union
    write_csv(
        output / "layer_metrics.csv",
        [
            "layer_index", "d_cos", "d_shift", "tau_cos", "tau_shift",
            "above_cos", "above_shift", "vfind_intersection", "selected_by_layer_rule",
        ],
        [[
            i, float(d_cos[i]), float(d_shift[i]), tau_cos, tau_shift,
            bool(above_cos[i]), bool(above_shift[i]), i in intersection, i in critical,
        ] for i in range(len(d_cos))],
    )
    if not critical:
        raise RuntimeError(f"layer rule {args.layer_rule!r} selected no layer; layer_metrics.csv was saved")

    probe_root = output / "layer_probes"
    probe_root.mkdir(parents=True, exist_ok=True)
    labels = np.concatenate([np.ones(len(positive)), np.zeros(len(negative))]).astype(np.float32)
    selected, neuron_rows = [], []
    all_effect = np.zeros((positive.shape[1], positive.shape[2]), dtype=np.float32)
    for layer in critical:
        layer_dir = probe_root / f"layer_{layer:02d}"
        layer_dir.mkdir(parents=True, exist_ok=True)
        features = np.concatenate([positive[:, layer], negative[:, layer]], axis=0)
        weight, bias = train_layer_probe(
            features, labels, layer_dir, args.probe_epochs, args.probe_lr,
            args.weight_decay, args.checkpoint_interval, device,
        )
        effect = neuron_effect_size(positive[:, layer], negative[:, layer], weight, args.epsilon)
        all_effect[layer] = effect
        dims = np.flatnonzero(effect >= args.effect_threshold)
        if len(dims):
            order = dims[np.argsort(-effect[dims], kind="mergesort")]
            selected.append({
                "layer_index": layer,
                "dims": order.astype(int).tolist(),
                "effect_sizes": effect[order].astype(float).tolist(),
                "probe_weights": weight[order].astype(float).tolist(),
                "probe_bias": bias,
            })
        neuron_rows.extend(
            [layer, dim, float(effect[dim]), float(weight[dim]), bool(effect[dim] >= args.effect_threshold)]
            for dim in range(len(effect))
        )
    np.save(output / "neuron_effect_sizes.npy", all_effect)
    write_csv(output / "neuron_metrics.csv", ["layer_index", "dimension", "effect_size", "probe_weight", "selected"], neuron_rows)
    selected_width = sum(len(item["dims"]) for item in selected)
    if selected_width == 0:
        raise RuntimeError("no neuron met V-FIND's default effect threshold; neuron metrics were saved")
    save_json(result_path, {
        "method": "shift_vfind_intrinsic_anomaly_subspace_v1",
        "pair_manifest": args.pair_manifest,
        "sample_unit_for_discovery": "per-video mean of top/bottom snippet tails",
        "layer_rule": args.layer_rule,
        "critical_layer_rule": (
            "V-FIND threshold intersection"
            if args.layer_rule == "intersection"
            else "explicit VAD adaptation: threshold union, followed by unchanged neuron effect-size filtering"
        ),
        "d_cos": d_cos.astype(float).tolist(),
        "d_shift": d_shift.astype(float).tolist(),
        "tau_cos": tau_cos,
        "tau_shift": tau_shift,
        "vfind_intersection_layers": intersection,
        "threshold_union_layers": threshold_union,
        "critical_layers": critical,
        "effect_threshold": args.effect_threshold,
        "num_layers": int(positive.shape[1]),
        "hidden_dim": int(positive.shape[2]),
        "selected_width": selected_width,
        "selected": selected,
        "probe_hyperparameters": {"epochs": args.probe_epochs, "lr": args.probe_lr, "weight_decay": args.weight_decay},
        "backbone_trained": False,
        "baseline_trained": False,
    })
    print(f"critical layers={critical} | selected neurons={selected_width} | wrote {result_path}", flush=True)


if __name__ == "__main__":
    main()
