#!/usr/bin/env python3
"""Extract complete CLS layers chosen by the score-free circuit atlas."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from tqdm import tqdm

from semantic_knn_splicing.common import (
    base_key,
    clean_output,
    load_hidden,
    read_hidden_manifest,
    read_source_csv,
    resample,
    save_json,
    write_csv,
)


def selected_layers(atlas_path: str) -> list[int]:
    atlas = json.loads(Path(atlas_path).read_text(encoding="utf-8"))
    if "blocks" not in atlas:
        raise ValueError(f"{atlas_path}: circuit atlas has no blocks")
    layers = [int(block["layer_zero_based"]) for block in atlas["blocks"]]
    if not layers:
        raise ValueError("circuit atlas selected no layer")
    return layers


def build_split(
    name: str,
    source_csv: str,
    hidden_manifest: str,
    output: Path,
    layers: list[int],
    clean: bool,
    skip_missing: bool,
) -> dict:
    frame = read_source_csv(source_csv)
    mapping = read_hidden_manifest(hidden_manifest)
    feature_dir = output / name / "full_layers"
    feature_dir.mkdir(parents=True, exist_ok=True)
    rows, missing, reused = [], [], 0
    cache: dict[str, np.ndarray] = {}
    for _, row in tqdm(frame.iterrows(), total=len(frame), desc=f"extract {name} full layers", unit="crop"):
        clip_path = str(row["path"])
        key = base_key(clip_path)
        if key not in mapping:
            if not skip_missing:
                raise FileNotFoundError(f"missing hidden state for {key}")
            missing.append([key, str(row["label"]), clip_path, "missing_hidden"])
            continue
        clip = np.load(clip_path, mmap_mode="r")
        target = feature_dir / f"{Path(clip_path).stem}.npz"
        if target.exists() and not clean:
            stored = np.load(target)
            if stored["hidden"].shape != (len(clip), len(layers), 768):
                raise ValueError(f"stale full-layer feature: {target}")
            if "layers" not in stored.files or not np.array_equal(
                stored["layers"], np.asarray(layers)
            ):
                raise ValueError(f"stale layer identities in {target}; rerun with --clean")
            reused += 1
        else:
            if key not in cache:
                hidden = load_hidden(mapping[key])
                if max(layers) >= hidden.shape[1]:
                    raise IndexError(f"{key}: selected layer outside hidden shape {hidden.shape}")
                cache[key] = hidden[:, layers, :]
            value = resample(cache[key], len(clip))
            np.savez_compressed(target, hidden=value.astype(np.float16), layers=np.asarray(layers))
        rows.append([clip_path, str(target), str(row["label"]), key, len(clip)])
    csv_path = output / f"{name}.csv"
    write_csv(csv_path, ["clip_path", "hidden_path", "label", "key", "length"], rows)
    write_csv(output / f"{name}_skipped.csv", ["key", "label", "clip_path", "reason"], missing)
    return {"csv": str(csv_path), "rows": len(rows), "missing": len(missing), "reused": reused}


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract complete CLS layers selected by a circuit atlas.")
    parser.add_argument("--train-list", required=True)
    parser.add_argument("--test-list", required=True)
    parser.add_argument("--train-hidden-manifest", required=True)
    parser.add_argument("--test-hidden-manifest", required=True)
    parser.add_argument("--layer-atlas", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--skip-missing-hidden", action="store_true")
    parser.add_argument("--clean", action="store_true")
    args = parser.parse_args()
    output = clean_output(args.out_dir, args.clean)
    layers = selected_layers(args.layer_atlas)
    report = {
        "method": "whole_layers_from_score_free_circuit_atlas_v1",
        "selected_layers_zero_based": layers,
        "train": build_split(
            "train", args.train_list, args.train_hidden_manifest, output,
            layers, args.clean, args.skip_missing_hidden,
        ),
        "test": build_split(
            "test", args.test_list, args.test_hidden_manifest, output,
            layers, args.clean, args.skip_missing_hidden,
        ),
    }
    save_json(output / "extract_report.json", report)
    print(json.dumps(report, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
