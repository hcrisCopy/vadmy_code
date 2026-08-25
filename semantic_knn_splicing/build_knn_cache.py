#!/usr/bin/env python3
"""Build LaGoVAD-style KNN retrieval over fully normal CLIP videos."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

from semantic_knn_splicing.common import clean_output, is_normal, read_source_csv, save_json


def normalized_mean(path: str, start: int | None = None, end: int | None = None) -> np.ndarray:
    feature = np.load(path, mmap_mode="r")
    if start is not None and end is not None:
        feature = feature[start:end]
    value = np.asarray(feature, dtype=np.float32).mean(axis=0)
    return value / max(float(np.linalg.norm(value)), 1e-8)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build normal-video KNN cache for pseudo segments.")
    parser.add_argument("--dataset", choices=["ucf", "xd"], required=True)
    parser.add_argument("--train-list", required=True)
    parser.add_argument("--pseudo-csv", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--neighbors", type=int, default=20)
    parser.add_argument("--clean", action="store_true")
    args = parser.parse_args()
    if args.neighbors <= 0:
        parser.error("--neighbors must be positive")
    output = clean_output(args.out_dir, args.clean)
    result_path = output / "retrieval_cache.json"
    if result_path.exists() and not args.clean:
        print(result_path.read_text(encoding="utf-8"), flush=True)
        return
    source = read_source_csv(args.train_list)
    normal = source[source["label"].map(lambda value: is_normal(args.dataset, str(value)))].reset_index(drop=True)
    if normal.empty:
        raise RuntimeError("no normal training feature found")
    normal_features = np.stack([
        normalized_mean(str(path)) for path in tqdm(normal["path"], desc="normal KNN index", unit="crop")
    ])
    pseudo = pd.read_csv(args.pseudo_csv)
    cache: dict[str, list[str]] = {}
    for _, row in tqdm(pseudo.iterrows(), total=len(pseudo), desc="query normal neighbors", unit="segment"):
        query = normalized_mean(str(row["clip_path"]), int(row["start"]), int(row["end"]))
        similarity = normal_features @ query
        count = min(args.neighbors, len(similarity))
        indices = np.argpartition(similarity, len(similarity) - count)[-count:]
        indices = indices[np.argsort(-similarity[indices])]
        cache[str(row["pseudo_id"])] = [str(normal.iloc[index]["path"]) for index in indices]
    save_json(result_path, cache)
    report = {"method": "lagovad_cosine_knn_normal_retrieval_v1", "normal_items": len(normal), "queries": len(pseudo), "neighbors": args.neighbors, "cache": str(result_path)}
    save_json(output / "knn_report.json", report)
    print(json.dumps(report, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
