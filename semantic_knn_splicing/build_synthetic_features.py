#!/usr/bin/env python3
"""Materialize LaGoVAD-style normal/anomaly temporal concatenations."""

from __future__ import annotations

import argparse
import json
import random
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

from semantic_knn_splicing.common import clean_output, save_json, seed_everything, write_csv


def random_crop(feature: np.ndarray, maximum: int, rng: random.Random) -> np.ndarray:
    if len(feature) <= maximum:
        return feature
    start = rng.randint(0, len(feature) - maximum)
    return feature[start:start + maximum]


def main() -> None:
    parser = argparse.ArgumentParser(description="Build offline KNN temporal synthesis features.")
    parser.add_argument("--pseudo-csv", required=True)
    parser.add_argument("--retrieval-cache", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--copies-per-segment", type=int, default=2)
    parser.add_argument("--max-num-clips", type=int, default=5)
    parser.add_argument("--max-normal-length", type=int, default=96)
    parser.add_argument("--retrieval-probability", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=234)
    parser.add_argument("--clean", action="store_true")
    args = parser.parse_args()
    if min(args.copies_per_segment, args.max_num_clips, args.max_normal_length) <= 0:
        parser.error("counts must be positive")
    if not 0 <= args.retrieval_probability <= 1:
        parser.error("retrieval probability must be in [0,1]")
    output = clean_output(args.out_dir, args.clean)
    feature_dir = output / "features"
    feature_dir.mkdir(exist_ok=True)
    seed_everything(args.seed)
    pseudo = pd.read_csv(args.pseudo_csv)
    cache = json.loads(Path(args.retrieval_cache).read_text(encoding="utf-8"))
    all_normal_paths = sorted({path for paths in cache.values() for path in paths})
    if not all_normal_paths:
        raise RuntimeError("retrieval cache is empty")
    rows, reused = [], 0
    total = len(pseudo) * args.copies_per_segment
    progress = tqdm(total=total, desc="build temporal synthesis", unit="sample")
    for _, row in pseudo.iterrows():
        abnormal_source = np.load(str(row["clip_path"])).astype(np.float32)
        abnormal = abnormal_source[int(row["start"]):int(row["end"])]
        if not len(abnormal):
            progress.update(args.copies_per_segment)
            continue
        neighbors = cache[str(row["pseudo_id"])]
        for copy_index in range(args.copies_per_segment):
            sample_id = f"{row['pseudo_id']}_{copy_index:02d}"
            rng = random.Random(f"{args.seed}:{sample_id}")
            target = feature_dir / f"{sample_id}.npz"
            valid_cache = False
            if target.exists() and not args.clean:
                try:
                    with np.load(target) as stored:
                        length = len(stored["feature"])
                        valid_cache = len(stored["frame_label"]) == length
                except (OSError, ValueError, EOFError, zipfile.BadZipFile):
                    valid_cache = False
            if valid_cache:
                reused += 1
            else:
                num_clips = rng.randint(1, args.max_num_clips)
                insert = rng.randint(0, num_clips - 1)
                clips, labels = [], []
                for position in range(num_clips):
                    if position == insert:
                        clips.append(abnormal)
                        labels.append(np.ones(len(abnormal), dtype=np.float32))
                    else:
                        pool = neighbors if rng.random() < args.retrieval_probability else all_normal_paths
                        normal = np.load(rng.choice(pool)).astype(np.float32)
                        normal = random_crop(normal, args.max_normal_length, rng)
                        clips.append(normal)
                        labels.append(np.zeros(len(normal), dtype=np.float32))
                feature = np.concatenate(clips).astype(np.float32)
                frame_label = np.concatenate(labels).astype(np.float32)
                length = len(feature)
                # CLIP floats compress poorly and compressed archives make this
                # offline stage roughly an order of magnitude slower. Write an
                # uncompressed archive atomically so an interruption never
                # leaves a corrupt cache that looks complete.
                temporary = target.with_suffix(".tmp.npz")
                np.savez(temporary, feature=feature, frame_label=frame_label)
                temporary.replace(target)
            rows.append([sample_id, str(target), str(row["label"]), length, str(row["pseudo_id"])])
            progress.update(1)
    progress.close()
    csv_path = output / "synthetic_train.csv"
    write_csv(csv_path, ["sample_id", "feature_path", "label", "length", "pseudo_id"], rows)
    report = {
        "method": "lagovad_knn_temporal_synthesis_from_text_topk_v1",
        "samples": len(rows), "reused": reused, "max_num_clips": args.max_num_clips,
        "retrieval_probability": args.retrieval_probability, "csv": str(csv_path),
        "storage": "uncompressed npz with atomic replacement",
    }
    save_json(output / "synthesis_report.json", report)
    print(json.dumps(report, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
