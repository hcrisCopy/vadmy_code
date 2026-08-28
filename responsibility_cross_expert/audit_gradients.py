#!/usr/bin/env python3
"""Audit which baseline parameters each loss can actually reach."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from .baselines import build_baseline
from .common import clean_output
from .data import HeadTrainingDataset
from .train_binary_head import consensus_loss


def one_item(dataset: HeadTrainingDataset) -> dict:
    item = dataset[0]
    result = {}
    for key in ("clip", "target", "length", "binary_label"):
        result[key] = item[key].unsqueeze(0)
    result["label_text"] = [item["label_text"]]
    return result


def merge(first: dict, second: dict) -> dict:
    output = {}
    for key in ("clip", "target", "length", "binary_label"):
        output[key] = torch.cat([first[key], second[key]], dim=0)
    output["label_text"] = first["label_text"] + second["label_text"]
    return output


def group_name(parameter_name: str) -> str:
    parts = parameter_name.split(".")
    return ".".join(parts[:3]) if len(parts) >= 3 else parameter_name


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit author/consensus gradient paths.")
    parser.add_argument("--baseline", choices=["dsanet", "desc", "lagovad"], required=True)
    parser.add_argument("--baseline-root", required=True)
    parser.add_argument("--baseline-weight", default="")
    parser.add_argument("--sensitivity-weight", default="")
    parser.add_argument("--consistency-weight", default="")
    parser.add_argument("--dataset", choices=["ucf", "xd"], required=True)
    parser.add_argument("--consensus-csv", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--clean", action="store_true")
    args = parser.parse_args()
    output_dir = clean_output(args.out_dir, args.clean)
    device = torch.device(
        args.device if torch.cuda.is_available() and args.device.startswith("cuda") else "cpu"
    )
    adapter = build_baseline(args, str(device)).to(device)
    adapter.set_train_scope("binary_head")
    audited = [
        (name, parameter)
        for name, parameter in adapter.named_parameters()
        if parameter.requires_grad
    ]
    if not audited:
        raise RuntimeError("binary-head audit found no trainable parameters")
    normal = HeadTrainingDataset(
        args.consensus_csv, args.dataset, adapter.visual_length, args.baseline, "normal"
    )
    abnormal = HeadTrainingDataset(
        args.consensus_csv, args.dataset, adapter.visual_length, args.baseline, "abnormal"
    )
    batch = merge(one_item(normal), one_item(abnormal))
    clip = batch["clip"].to(device)
    target = batch["target"].to(device)
    lengths = batch["length"].to(device)
    labels = batch["binary_label"].to(device)
    output = adapter.forward_baseline(clip, lengths)
    author_loss = adapter.original_loss(output, labels, batch["label_text"], lengths)
    dense_loss = consensus_loss(adapter.binary_training_logits(output), target, lengths)
    parameters = [value for _, value in audited]
    author_gradients = torch.autograd.grad(
        author_loss, parameters, retain_graph=True, allow_unused=True
    )
    dense_gradients = torch.autograd.grad(
        dense_loss, parameters, retain_graph=False, allow_unused=True
    )
    rows, groups = [], defaultdict(lambda: [0.0, 0.0, 0, 0])
    for (name, parameter), author, dense in zip(audited, author_gradients, dense_gradients):
        author_norm = float(author.norm()) if author is not None else 0.0
        dense_norm = float(dense.norm()) if dense is not None else 0.0
        rows.append([
            name, parameter.numel(), author_norm, dense_norm,
            int(author is not None), int(dense is not None),
        ])
        group = groups[group_name(name)]
        group[0] += author_norm**2
        group[1] += dense_norm**2
        group[2] += parameter.numel()
        group[3] += int(dense is not None)
    csv_path = output_dir / "gradient_paths.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow([
            "parameter", "parameter_count", "author_grad_norm", "consensus_grad_norm",
            "author_reachable", "consensus_reachable",
        ])
        writer.writerows(rows)
    ordered = sorted(groups)
    author_values = [np.sqrt(groups[name][0]) for name in ordered]
    dense_values = [np.sqrt(groups[name][1]) for name in ordered]
    positions = np.arange(len(ordered))
    figure, axis = plt.subplots(figsize=(max(9, 0.55 * len(ordered)), 5.5))
    width = 0.38
    axis.bar(positions - width / 2, np.log10(np.asarray(author_values) + 1e-12), width, label="author loss")
    axis.bar(positions + width / 2, np.log10(np.asarray(dense_values) + 1e-12), width, label="consensus loss")
    axis.set_xticks(positions, ordered, rotation=60, ha="right")
    axis.set_ylabel("log10 gradient norm")
    axis.set_title("Loss-to-module gradient reachability")
    axis.grid(axis="y", alpha=0.25)
    axis.legend(frameon=False)
    figure.tight_layout()
    figure_path = output_dir / "gradient_path_audit.png"
    figure.savefig(figure_path, dpi=180)
    plt.close(figure)
    report = {
        "baseline": args.baseline,
        "dataset": args.dataset,
        "author_loss": float(author_loss.detach()),
        "consensus_loss": float(dense_loss.detach()),
        "audited_trainable_parameters": sum(value.numel() for _, value in audited),
        "consensus_reachable_parameters": sum(
            value.numel() for (_, value), gradient in zip(audited, dense_gradients)
            if gradient is not None
        ),
        "table": str(csv_path),
        "figure": str(figure_path),
        "train_scope": "final binary head only",
        "purpose": "verify that both losses reach the intended trainable head and no other parameter is trainable",
    }
    (output_dir / "gradient_report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(report, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
