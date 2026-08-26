#!/usr/bin/env python3
"""Build aligned 768D selected-neuron features without copying CLIP files."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
from tqdm import tqdm


PACKAGE_ROOT = Path(__file__).resolve().parent.parent
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))

from neuron_responsibility.common import (
    base_key, clean_output, load_hidden, load_json, read_feature_csv,
    read_hidden_manifest, resample_feature, resolve_artifact_path, write_csv,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build CLIP/neuron aligned CSV for residual injection.")
    parser.add_argument("--source-csv", required=True)
    parser.add_argument("--hidden-manifest", required=True)
    parser.add_argument("--neuron-json", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--out-csv", required=True)
    parser.add_argument("--skip-missing-hidden", action="store_true")
    parser.add_argument("--clean", action="store_true")
    args = parser.parse_args()

    out_dir = clean_output(args.out_dir, args.clean)
    signature = {
        "source_csv": hashlib.sha256(Path(args.source_csv).read_bytes()).hexdigest(),
        "hidden_manifest": hashlib.sha256(Path(args.hidden_manifest).read_bytes()).hexdigest(),
        "neuron_json": hashlib.sha256(Path(args.neuron_json).read_bytes()).hexdigest(),
    }
    signature_path = out_dir / "build_signature.json"
    reuse = signature_path.exists() and json.loads(signature_path.read_text(encoding="utf-8")) == signature
    source = read_feature_csv(args.source_csv)
    hidden_by_key, token_pool = read_hidden_manifest(args.hidden_manifest)
    config = load_json(args.neuron_json)
    if config.get("token_pool", "cls") != token_pool:
        raise ValueError("hidden manifest and selected-neuron token pools differ")
    mean = np.load(resolve_artifact_path(config["normal_mean_path"], args.neuron_json)).astype(np.float32)
    std = np.load(resolve_artifact_path(config["normal_std_path"], args.neuron_json)).astype(np.float32)
    selected = [(int(item["layer_index"]), np.asarray(item["dims"], dtype=np.int64)) for item in config["selected"]]
    neuron_width = int(config["neuron_width"])
    if sum(len(dims) for _, dims in selected) != neuron_width:
        raise ValueError("selected dimension count differs from neuron_width")

    rows, skipped, cache = [], [], {}
    for _, row in tqdm(source.iterrows(), total=len(source), desc="build aligned features", unit="row"):
        clip_path, label = str(row["path"]), str(row["label"])
        key = base_key(clip_path)
        if key not in hidden_by_key:
            if not args.skip_missing_hidden:
                raise FileNotFoundError(f"missing hidden state for {key}")
            skipped.append([key, label, clip_path, "missing_hidden"])
            continue
        clip = np.load(clip_path, mmap_mode="r")
        if clip.ndim != 2 or clip.shape[1] != 512:
            raise ValueError(f"{clip_path}: expected [T,512], got {clip.shape}")
        output_path = out_dir / f"{Path(clip_path).stem}.npy"
        if output_path.exists() and reuse:
            existing = np.load(output_path, mmap_mode="r")
            if existing.shape != (len(clip), neuron_width):
                raise ValueError(f"stale aligned feature: {output_path}")
        else:
            if key not in cache:
                hidden, _ = load_hidden(hidden_by_key[key])
                if hidden.shape[1:] != mean.shape:
                    raise ValueError(f"{key}: hidden/statistics shapes differ")
                z_hidden = (hidden - mean) / (std + float(config.get("sigma_min", 1e-6)))
                cache[key] = np.concatenate([z_hidden[:, layer, dims] for layer, dims in selected], axis=1).astype(np.float32)
            neuron = resample_feature(cache[key], len(clip))
            if neuron.shape != (len(clip), neuron_width):
                raise RuntimeError(f"{key}: aligned neuron shape is {neuron.shape}")
            np.save(output_path, neuron)
        rows.append([clip_path, str(output_path), label, key, int(len(clip))])
    write_csv(args.out_csv, ["clip_path", "neuron_path", "label", "key", "length"], rows)
    write_csv(out_dir / "skipped_rows.csv", ["key", "label", "clip_path", "reason"], skipped)
    signature_path.write_text(json.dumps(signature, indent=2), encoding="utf-8")
    print(f"wrote {args.out_csv}: rows={len(rows)}, skipped={len(skipped)}", flush=True)


if __name__ == "__main__":
    main()
