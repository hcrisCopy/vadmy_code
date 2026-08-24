#!/usr/bin/env python3
"""Discover the smallest sufficient CLIP CLS circuits and choose their layers.

This is a multi-layer extension of this repository's ``build_circuit_atlas.py``.
It uses video labels, frozen CLIP text gradients, and normal statistics only;
released baseline anomaly scores are never read.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

PACKAGE_ROOT = Path(__file__).resolve().parent.parent
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))

from neuron_responsibility.build_circuit_atlas import (
    class_names_for,
    collect_statistics,
    concept_gradients,
    select_circuits,
)
from neuron_responsibility.common import (
    base_key,
    clean_output,
    grouped_rows,
    load_hidden,
    read_feature_csv,
    read_hidden_manifest,
    resample_feature,
    save_json,
    write_csv,
)
from neuron_responsibility.text_responsibility import (
    NORMAL_CONCEPTS,
    encode_concepts,
    load_reference_clip,
)


def parse_grid(value: str) -> list[int]:
    grid = sorted(set(int(item) for item in value.split(",") if item.strip()))
    if not grid or grid[0] < 1 or grid[-1] > 768:
        raise argparse.ArgumentTypeError("k-grid must contain integers in [1,768]")
    return grid


def smallest_sufficient_k(class_scores: np.ndarray, ordered: np.ndarray, grid: list[int], ratio: float) -> int:
    maximum = min(grid[-1], len(ordered))
    full = float(class_scores[ordered[:maximum]].clip(min=0).sum())
    if full <= 0:
        return maximum
    for k in grid:
        captured = float(class_scores[ordered[:min(k, maximum)]].clip(min=0).sum())
        if captured >= ratio * full:
            return min(k, maximum)
    return maximum


def block_from_layer(result: dict, ratio: float, grid: list[int]) -> dict:
    scores = result["scores"]
    ordered = result["selections"]
    ks = [smallest_sufficient_k(scores[index], ordered[index], grid, ratio) for index in range(len(ordered))]
    selected = [ordered[index][:ks[index]] for index in range(len(ordered))]
    union = np.unique(np.concatenate(selected))
    lookup = {int(value): index for index, value in enumerate(union)}
    class_mask = np.zeros((len(selected), len(union)), dtype=np.float32)
    directions = np.zeros_like(class_mask)
    weights = np.zeros_like(class_mask)
    raw_direction = np.where(result["positive"] >= result["negative"], 1.0, -1.0)
    for class_index, dims in enumerate(selected):
        columns = np.asarray([lookup[int(value)] for value in dims])
        class_mask[class_index, columns] = 1.0
        directions[class_index, columns] = raw_direction[class_index, dims]
        weights[class_index, columns] = scores[class_index, dims].clip(min=0)
    full_mass = float(np.mean([
        scores[index, ordered[index][:grid[-1]]].clip(min=0).sum()
        for index in range(len(ordered))
    ]))
    direction_gap = np.abs(result["positive"] - result["negative"])
    direction_scale = np.abs(result["positive"]) + np.abs(result["negative"]) + 1e-8
    stability = float(np.mean(direction_gap[:, union] / direction_scale[:, union]))
    quality = full_mass * stability / np.log2(2.0 + len(union))
    return {
        "layer": int(result["layer"]), "union": union, "selected": selected,
        "center": result["center"][union], "scale": result["scale"][union],
        "class_mask": class_mask, "directions": directions, "weights": weights,
        "k_per_class": ks, "union_width": int(len(union)), "full_mass": full_mass,
        "direction_stability": stability, "quality": float(quality),
    }


def serializable_block(block: dict, offset: int) -> dict:
    return {
        "layer_zero_based": block["layer"], "offset": offset,
        "width": block["union_width"], "union_indices": block["union"].tolist(),
        "k_per_class": block["k_per_class"], "center": block["center"].tolist(),
        "scale": block["scale"].tolist(), "class_mask": block["class_mask"].tolist(),
        "directions": block["directions"].tolist(), "weights": block["weights"].tolist(),
        "quality": block["quality"], "full_mass": block["full_mass"],
        "direction_stability": block["direction_stability"],
    }


def runtime_block(value: dict) -> dict:
    return {
        "layer": int(value["layer_zero_based"]),
        "union": np.asarray(value["union_indices"], dtype=np.int64),
        "union_width": int(value["width"]),
        "center": np.asarray(value["center"], dtype=np.float32),
        "scale": np.asarray(value["scale"], dtype=np.float32),
        "class_mask": np.asarray(value["class_mask"], dtype=np.float32),
        "directions": np.asarray(value["directions"], dtype=np.float32),
        "weights": np.asarray(value["weights"], dtype=np.float32),
    }


def build_split(name: str, source_csv: str, manifest: str, output: Path, blocks: list[dict], clean: bool, skip_missing: bool) -> dict:
    frame = read_feature_csv(source_csv)
    mapping, _ = read_hidden_manifest(manifest)
    feature_dir = output / name / "features"
    feature_dir.mkdir(parents=True, exist_ok=True)
    rows, skipped, reused, cache = [], [], 0, {}
    width = sum(block["union_width"] for block in blocks)
    for _, row in tqdm(frame.iterrows(), total=len(frame), desc=f"build {name} definition circuits", unit="crop"):
        clip_path = str(row["path"]); key = base_key(clip_path)
        if key not in mapping:
            if not skip_missing:
                raise FileNotFoundError(f"missing hidden state for {key}")
            skipped.append([key, str(row["label"]), clip_path, "missing_hidden"]); continue
        clip = np.load(clip_path, mmap_mode="r")
        target = feature_dir / f"{Path(clip_path).stem}.npy"
        if target.exists() and not clean:
            compact = np.load(target, mmap_mode="r")
            if compact.shape != (len(clip), width):
                raise ValueError(f"stale compact feature {target}: {compact.shape}")
            reused += 1
        else:
            if key not in cache:
                hidden, _ = load_hidden(mapping[key])
                cache[key] = np.concatenate([
                    hidden[:, block["layer"], :][:, block["union"]] for block in blocks
                ], axis=1).astype(np.float32)
            np.save(target, resample_feature(cache[key], len(clip)).astype(np.float32))
        rows.append([clip_path, str(target), str(row["label"]), key, len(clip)])
    csv_path = output / f"{name}.csv"
    write_csv(csv_path, ["clip_path", "neuron_path", "label", "key", "length"], rows)
    write_csv(output / f"{name}_skipped.csv", ["key", "label", "clip_path", "reason"], skipped)
    return {"csv": str(csv_path), "rows": len(rows), "skipped": len(skipped), "reused": reused}


def main() -> None:
    parser = argparse.ArgumentParser(description="Baseline-score-free multi-layer circuit discovery.")
    parser.add_argument("--dataset", choices=["ucf", "xd"], required=True)
    parser.add_argument("--train-list", required=True); parser.add_argument("--test-list", required=True)
    parser.add_argument("--train-hidden-manifest", required=True); parser.add_argument("--test-hidden-manifest", required=True)
    parser.add_argument("--clip-root", required=True); parser.add_argument("--clip-weight", required=True)
    parser.add_argument("--out-dir", required=True); parser.add_argument("--layers", default="all")
    parser.add_argument("--selected-layers", type=int, default=2); parser.add_argument("--k-grid", type=parse_grid, default=parse_grid("8,16,32,64,128"))
    parser.add_argument("--sufficiency-ratio", type=float, default=0.95)
    parser.add_argument("--tail-fraction", type=float, default=0.10); parser.add_argument("--snippets-per-video", type=int, default=256)
    parser.add_argument("--specificity-weight", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=234); parser.add_argument("--device", default="cuda")
    parser.add_argument("--skip-missing-hidden", action="store_true"); parser.add_argument("--clean", action="store_true")
    args = parser.parse_args()
    if not 0 < args.sufficiency_ratio <= 1 or args.selected_layers < 1:
        parser.error("invalid sufficiency ratio or selected layer count")
    output = clean_output(args.out_dir, args.clean)
    atlas_path = output / "definition_circuits.json"
    if atlas_path.exists() and not args.clean:
        atlas = json.loads(atlas_path.read_text(encoding="utf-8"))
        if (output / "train.csv").exists() and (output / "test.csv").exists():
            print(json.dumps(atlas, indent=2, ensure_ascii=False)); return
        print("circuit discovery is complete; resuming compact feature construction", flush=True)
        chosen = [runtime_block(value) for value in atlas["blocks"]]
        report = {
            "atlas": str(atlas_path),
            "train": build_split("train", args.train_list, args.train_hidden_manifest, output, chosen, False, args.skip_missing_hidden),
            "test": build_split("test", args.test_list, args.test_hidden_manifest, output, chosen, False, args.skip_missing_hidden),
        }
        save_json(output / "build_report.json", report)
        print(json.dumps({"atlas": atlas, "build": report}, indent=2, ensure_ascii=False)); return
    device = torch.device(args.device if torch.cuda.is_available() and args.device.startswith("cuda") else "cpu")
    model, tokenize = load_reference_clip(args.clip_root, args.clip_weight, device)
    normal_text = encode_concepts(model, tokenize, NORMAL_CONCEPTS, device)
    class_names = class_names_for(args.dataset)
    abnormal_text = encode_concepts(model, tokenize, class_names, device)
    groups = grouped_rows(read_feature_csv(args.train_list))
    mapping, _ = read_hidden_manifest(args.train_hidden_manifest)
    first_hidden, _ = load_hidden(next(iter(mapping.values())))
    layers = list(range(first_hidden.shape[1])) if args.layers == "all" else [int(value) for value in args.layers.split(",")]
    results = []
    cache_signature = json.dumps({
        "dataset": args.dataset, "snippets_per_video": args.snippets_per_video,
        "tail_fraction": args.tail_fraction, "specificity_weight": args.specificity_weight,
        "max_k": args.k_grid[-1], "train_list": str(Path(args.train_list).resolve()),
        "manifest": str(Path(args.train_hidden_manifest).resolve()),
    }, sort_keys=True)
    for layer in layers:
        cache_path = output / f"layer_{layer:02d}_discovery.npz"
        if cache_path.exists() and not args.clean:
            cache = np.load(cache_path)
            if str(cache["signature"].item()) != cache_signature:
                raise RuntimeError(f"stale layer cache {cache_path}; use --clean or a new --out-dir")
            center, scale = cache["center"], cache["scale"]
            positive, negative, scores = cache["positive"], cache["negative"], cache["scores"]
            selections = [row.astype(np.int64) for row in cache["selections"]]
            statistics = json.loads(str(cache["statistics"].item()))
            print(f"reused layer {layer} discovery cache", flush=True)
        else:
            center, scale, positive, negative, _, statistics = collect_statistics(
                groups, mapping, args.dataset, layer, args.snippets_per_video,
                args.tail_fraction, class_names,
            )
            gradients = concept_gradients(model, normal_text, abnormal_text, center, scale, device)
            try:
                _, _, _, scores, selections = select_circuits(
                    positive, negative, gradients, args.k_grid[-1], args.specificity_weight
                )
            except RuntimeError as error:
                print(f"skip layer {layer}: {error}", flush=True)
                continue
            np.savez_compressed(
                cache_path, center=center, scale=scale, positive=positive, negative=negative,
                scores=scores, selections=np.stack(selections),
                statistics=json.dumps(statistics, ensure_ascii=False), signature=cache_signature,
            )
        results.append({
            "layer": layer, "center": center, "scale": scale, "positive": positive,
            "negative": negative, "scores": scores, "selections": selections,
            "statistics": statistics,
        })
    candidates = [block_from_layer(result, args.sufficiency_ratio, args.k_grid) for result in results]
    if len(candidates) < args.selected_layers:
        raise RuntimeError(f"only {len(candidates)} valid layers; requested {args.selected_layers}")
    chosen = sorted(candidates, key=lambda item: item["quality"], reverse=True)[:args.selected_layers]
    chosen.sort(key=lambda item: item["layer"])
    offset, blocks = 0, []
    for block in chosen:
        blocks.append(serializable_block(block, offset)); offset += block["union_width"]
    atlas = {
        "method": "definition_sensitive_sparse_circuits_v1", "dataset": args.dataset,
        "class_names": class_names, "normal_concepts": list(NORMAL_CONCEPTS),
        "selection_source": "video labels + normal statistics + frozen CLIP text gradients; no baseline scores",
        "sufficiency_ratio": args.sufficiency_ratio, "k_grid": args.k_grid,
        "compact_width": offset, "blocks": blocks,
        "layer_ranking": sorted([
            {"layer_zero_based": item["layer"], "quality": item["quality"],
             "union_width": item["union_width"], "k_per_class": item["k_per_class"],
             "full_mass": item["full_mass"], "direction_stability": item["direction_stability"]}
            for item in candidates
        ], key=lambda item: item["quality"], reverse=True),
    }
    save_json(atlas_path, atlas)
    report = {
        "atlas": str(atlas_path),
        "train": build_split("train", args.train_list, args.train_hidden_manifest, output, chosen, args.clean, args.skip_missing_hidden),
        "test": build_split("test", args.test_list, args.test_hidden_manifest, output, chosen, args.clean, args.skip_missing_hidden),
    }
    save_json(output / "build_report.json", report)
    print(json.dumps({"atlas": atlas, "build": report}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
