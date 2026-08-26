#!/usr/bin/env python3
"""Build a reusable whole-layer normal visual prototype from pure normal videos."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from .common import clean_output, is_normal, stable_validation_key
from .semantic_model import load_clip_resources, project_hidden, selected_layer_spec


def main() -> None:
    parser = argparse.ArgumentParser(description="Build pure-normal visual prototypes for selected layers.")
    parser.add_argument("--dataset", choices=["ucf", "xd"], required=True)
    parser.add_argument("--train-csv", required=True)
    parser.add_argument("--layer-atlas", required=True)
    parser.add_argument("--clip-weight", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--validation-fraction", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=234)
    parser.add_argument("--save-every", type=int, default=50)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--clean", action="store_true")
    args = parser.parse_args()
    if args.save_every <= 0:
        parser.error("--save-every must be positive")
    output = clean_output(args.out_dir, args.clean)
    final_path = output / "normal_prototype.npz"
    if final_path.exists() and not args.clean:
        print(json.dumps({"reused": str(final_path)}, indent=2), flush=True)
        return
    frame = pd.read_csv(args.train_csv)
    required = {"hidden_path", "label", "key"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"{args.train_csv}: missing columns {sorted(missing)}")
    frame = frame[frame["label"].map(lambda value: is_normal(args.dataset, str(value)))]
    frame = frame[
        ~frame["key"].map(
            lambda key: stable_validation_key(
                str(key), args.seed, args.validation_fraction
            )
        )
    ]
    layers, _ = selected_layer_spec(args.layer_atlas)
    device = torch.device(
        args.device if torch.cuda.is_available() and args.device.startswith("cuda") else "cpu"
    )
    resources = load_clip_resources(args.clip_weight, args.dataset, device)
    checkpoint_path = output / "prototype_accumulator.npz"
    sums = np.zeros((len(layers), 512), dtype=np.float64)
    count = 0
    processed: set[str] = set()
    if checkpoint_path.exists() and not args.clean:
        stored = np.load(checkpoint_path, allow_pickle=True)
        sums = stored["sums"].astype(np.float64)
        count = int(stored["count"])
        processed = set(stored["processed"].astype(str).tolist())
    iterator = tqdm(frame.iterrows(), total=len(frame), desc="normal prototype", unit="crop")
    with torch.no_grad():
        for index, (_, row) in enumerate(iterator, 1):
            identity = str(row["hidden_path"])
            if identity in processed:
                continue
            hidden = np.load(str(row["hidden_path"]))["hidden"].astype(np.float32)
            value = project_hidden(
                torch.from_numpy(hidden).to(device),
                resources.ln_weight.to(device),
                resources.ln_bias.to(device),
                resources.projection.to(device),
            )
            sums += value.sum(dim=0).cpu().numpy().astype(np.float64)
            count += len(value)
            processed.add(identity)
            if index % args.save_every == 0:
                np.savez_compressed(
                    checkpoint_path,
                    sums=sums,
                    count=np.int64(count),
                    processed=np.asarray(sorted(processed)),
                    layers=np.asarray(layers),
                )
    if count == 0:
        raise RuntimeError("no pure-normal snippets were available")
    prototype = sums / count
    prototype /= np.maximum(np.linalg.norm(prototype, axis=1, keepdims=True), 1e-12)
    np.savez_compressed(
        final_path,
        prototype=prototype.astype(np.float32),
        layers=np.asarray(layers),
        snippet_count=np.int64(count),
    )
    report = {
        "method": "pure_normal_whole_layer_mean_prototype_v1",
        "dataset": args.dataset,
        "selected_layers_zero_based": layers,
        "normal_crops": len(processed),
        "normal_snippets": count,
        "validation_fraction_excluded": args.validation_fraction,
        "split_seed": args.seed,
        "prototype": str(final_path),
    }
    (output / "prototype_report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(report, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
