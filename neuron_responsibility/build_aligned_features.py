#!/usr/bin/env python3
"""Build selected-neuron features while reusing original 512D CLIP files."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
from tqdm import tqdm


PACKAGE_ROOT = Path(__file__).resolve().parent.parent
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))

from neuron_responsibility.common import (
    base_key,
    clean_output,
    load_hidden,
    load_json,
    read_feature_csv,
    read_hidden_manifest,
    resample_feature,
    resolve_artifact_path,
    write_csv,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create neuron-only features and an aligned clip_path,neuron_path,label CSV."
    )
    parser.add_argument("--source-csv", required=True)
    parser.add_argument("--hidden-manifest", required=True)
    parser.add_argument("--neuron-json", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--out-csv", required=True)
    parser.add_argument("--clean", action="store_true")
    args = parser.parse_args()

    out_dir = clean_output(args.out_dir, args.clean)
    source = read_feature_csv(args.source_csv)
    hidden_by_key, token_pool = read_hidden_manifest(args.hidden_manifest)
    config = load_json(args.neuron_json)
    if config.get("token_pool", "cls") != token_pool:
        raise ValueError("hidden manifest and selected neurons use different token pools")
    mean = np.load(resolve_artifact_path(config["normal_mean_path"], args.neuron_json)).astype(np.float32)
    std = np.load(resolve_artifact_path(config["normal_std_path"], args.neuron_json)).astype(np.float32)
    selected = [
        (int(group["layer_index"]), np.asarray(group["dims"], dtype=np.int64),
         np.asarray(group.get("directions", [1] * len(group["dims"])), dtype=np.float32))
        for group in config["selected"]
    ]
    expected_width = int(config["neuron_width"])
    if sum(len(dims) for _, dims, _ in selected) != expected_width:
        raise ValueError("selected_neurons.json neuron_width does not match selected dimensions")

    rows = []
    hidden_cache: dict[str, np.ndarray] = {}
    for _, row in tqdm(source.iterrows(), total=len(source), desc="aligned neuron features", unit="row"):
        clip_path = str(row["path"])
        key = base_key(clip_path)
        if key not in hidden_by_key:
            raise FileNotFoundError(f"missing hidden state for {key}")
        if key not in hidden_cache:
            hidden, _ = load_hidden(hidden_by_key[key])
            if hidden.shape[1:] != mean.shape:
                raise ValueError(f"{key}: hidden shape {hidden.shape[1:]} != normal statistics {mean.shape}")
            z_hidden = (hidden - mean) / (std + float(config.get("sigma_min", 1e-6)))
            pieces = [z_hidden[:, layer, dims] * direction[None, :] for layer, dims, direction in selected]
            hidden_cache[key] = np.concatenate(pieces, axis=1).astype(np.float32)

        clip = np.load(clip_path, mmap_mode="r")
        if clip.ndim != 2 or clip.shape[1] != 512:
            raise ValueError(f"{clip_path}: expected official CLIP [T,512], got {clip.shape}")
        neuron = resample_feature(hidden_cache[key], int(clip.shape[0]))
        output = out_dir / f"{Path(clip_path).stem}.npy"
        if not output.exists() or args.clean:
            np.save(output, neuron)
        else:
            existing = np.load(output, mmap_mode="r")
            if existing.shape != neuron.shape:
                raise ValueError(f"stale output {output}: {existing.shape} != {neuron.shape}; rerun with --clean")
        rows.append([clip_path, str(output), str(row["label"]), key, int(clip.shape[0])])

    write_csv(args.out_csv, ["clip_path", "neuron_path", "label", "key", "length"], rows)
    print(f"wrote {args.out_csv} with {len(rows)} rows; original 512D CLIP files were not copied", flush=True)


if __name__ == "__main__":
    main()
