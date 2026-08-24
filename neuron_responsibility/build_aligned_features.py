#!/usr/bin/env python3
"""Build selected-neuron features while reusing original 512D CLIP files."""

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
    parser.add_argument(
        "--skip-missing-hidden",
        action="store_true",
        help="Skip source rows whose base video is absent from the hidden manifest and record them.",
    )
    parser.add_argument("--clean", action="store_true")
    args = parser.parse_args()

    out_dir = clean_output(args.out_dir, args.clean)
    signature = {
        "neuron_json_sha256": hashlib.sha256(Path(args.neuron_json).read_bytes()).hexdigest(),
        "source_csv_sha256": hashlib.sha256(Path(args.source_csv).read_bytes()).hexdigest(),
        "hidden_manifest_sha256": hashlib.sha256(Path(args.hidden_manifest).read_bytes()).hexdigest(),
    }
    signature_path = out_dir / "build_signature.json"
    reuse_existing = not args.clean
    if signature_path.exists() and not args.clean:
        previous = json.loads(signature_path.read_text(encoding="utf-8"))
        if previous != signature:
            reuse_existing = False
            print("selection/input signature changed; rebuilding aligned neuron files", flush=True)
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
    skipped_rows = []
    skipped_keys: set[str] = set()
    hidden_cache: dict[str, np.ndarray] = {}
    for _, row in tqdm(source.iterrows(), total=len(source), desc="aligned neuron features", unit="row"):
        clip_path = str(row["path"])
        key = base_key(clip_path)
        if key not in hidden_by_key:
            if not args.skip_missing_hidden:
                raise FileNotFoundError(
                    f"missing hidden state for {key}; rerun with --skip-missing-hidden only if these videos "
                    "are intentionally excluded"
                )
            skipped_keys.add(key)
            skipped_rows.append([key, str(row["label"]), clip_path, "missing_hidden"])
            continue

        clip = np.load(clip_path, mmap_mode="r")
        if clip.ndim != 2 or clip.shape[1] != 512:
            raise ValueError(f"{clip_path}: expected official CLIP [T,512], got {clip.shape}")
        output = out_dir / f"{Path(clip_path).stem}.npy"
        if output.exists() and reuse_existing:
            existing = np.load(output, mmap_mode="r")
            expected_shape = (int(clip.shape[0]), expected_width)
            if existing.shape != expected_shape:
                raise ValueError(
                    f"stale output {output}: {existing.shape} != {expected_shape}; rerun with --clean"
                )
            rows.append([clip_path, str(output), str(row["label"]), key, int(clip.shape[0])])
            continue

        if key not in hidden_cache:
            hidden, _ = load_hidden(hidden_by_key[key])
            if hidden.shape[1:] != mean.shape:
                raise ValueError(f"{key}: hidden shape {hidden.shape[1:]} != normal statistics {mean.shape}")
            z_hidden = (hidden - mean) / (std + float(config.get("sigma_min", 1e-6)))
            pieces = [z_hidden[:, layer, dims] * direction[None, :] for layer, dims, direction in selected]
            hidden_cache[key] = np.concatenate(pieces, axis=1).astype(np.float32)

        neuron = resample_feature(hidden_cache[key], int(clip.shape[0]))
        np.save(output, neuron)
        rows.append([clip_path, str(output), str(row["label"]), key, int(clip.shape[0])])

    write_csv(args.out_csv, ["clip_path", "neuron_path", "label", "key", "length"], rows)
    signature_path.write_text(json.dumps(signature, indent=2), encoding="utf-8")
    write_csv(
        out_dir / "skipped_rows.csv",
        ["key", "label", "clip_path", "reason"],
        skipped_rows,
    )
    print(
        f"wrote {args.out_csv} with {len(rows)} rows; skipped "
        f"{len(skipped_keys)} videos/{len(skipped_rows)} rows; original 512D CLIP files were not copied",
        flush=True,
    )


if __name__ == "__main__":
    main()
