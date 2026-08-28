#!/usr/bin/env python3
"""Select class-conditioned candidate spans with a frozen whole-layer lens."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from semantic_knn_splicing.common import clean_output, is_normal, write_csv
from semantic_knn_splicing.prompts import label_targets
from semantic_knn_splicing.semantic_lens import FrozenWholeLayerSemanticLens


def segment_ranges(length: int, count: int) -> list[tuple[int, int]]:
    edges = np.linspace(0, length, min(count, length) + 1, dtype=np.int64)
    return [(int(edges[index]), int(edges[index + 1])) for index in range(len(edges) - 1)]


def merge_selected(
    indices: list[int], ranges: list[tuple[int, int]], scores: np.ndarray
) -> list[tuple[int, int, float]]:
    ordered = sorted(set(indices))
    merged: list[tuple[int, int, list[float]]] = []
    for index in ordered:
        start, end = ranges[index]
        if merged and merged[-1][1] == start:
            previous = merged[-1]
            merged[-1] = (previous[0], end, previous[2] + [float(scores[index])])
        else:
            merged.append((start, end, [float(scores[index])]))
    return [(start, end, float(np.mean(values))) for start, end, values in merged]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create baseline-score-free pseudo spans with LAP dynamic thresholding."
    )
    parser.add_argument("--dataset", choices=["ucf", "xd"], required=True)
    parser.add_argument("--train-csv", required=True)
    parser.add_argument("--layer-atlas", required=True)
    parser.add_argument("--clip-weight", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--num-segments", type=int, default=32)
    parser.add_argument("--threshold-tau", type=float, default=1.0)
    parser.add_argument("--max-spans-per-crop", type=int, default=3)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--clean", action="store_true")
    args = parser.parse_args()
    if args.num_segments <= 0 or args.threshold_tau < 0 or args.max_spans_per_crop <= 0:
        parser.error("invalid segment, threshold, or span setting")
    output = clean_output(args.out_dir, args.clean)
    cache_dir = output / "score_cache"
    cache_dir.mkdir(exist_ok=True)
    device = torch.device(
        args.device if torch.cuda.is_available() and args.device.startswith("cuda") else "cpu"
    )
    lens = FrozenWholeLayerSemanticLens.from_files(
        args.layer_atlas, args.clip_weight, args.dataset, device
    )
    frame = pd.read_csv(args.train_csv)
    missing = {"clip_path", "hidden_path", "label", "key"} - set(frame.columns)
    if missing:
        raise ValueError(f"{args.train_csv} is missing columns: {sorted(missing)}")
    rows = []
    skipped_no_target = 0
    with torch.no_grad():
        iterator = tqdm(
            frame.iterrows(), total=len(frame), desc="semantic span selection", unit="crop"
        )
        for row_index, row in iterator:
            label = str(row["label"])
            if is_normal(args.dataset, label):
                continue
            targets = label_targets(args.dataset, label)
            if not targets:
                skipped_no_target += 1
                continue
            cache = cache_dir / f"{row_index:07d}_{Path(str(row['clip_path'])).stem}.npz"
            if cache.exists() and not args.clean:
                stored = np.load(cache)
                snippet_score = stored["snippet_score"]
                layer_score = stored["layer_score"]
                ranges_array = stored["ranges"]
                segment_scores = stored["segment_score"]
                threshold = float(stored["threshold"])
                ranges = [(int(value[0]), int(value[1])) for value in ranges_array]
            else:
                hidden = np.load(str(row["hidden_path"]))["hidden"].astype(np.float32)
                result = lens(torch.from_numpy(hidden).unsqueeze(0).to(device))
                target_index = torch.tensor(targets, device=device)
                target_margin = result["class_margin"][0].index_select(1, target_index)
                snippet_score = target_margin.max(dim=1).values.cpu().numpy().astype(np.float32)
                target_layer = result["layer_margin"][0].index_select(2, target_index)
                layer_score = target_layer.max(dim=2).values.cpu().numpy().astype(np.float32)
                ranges = segment_ranges(len(snippet_score), args.num_segments)
                segment_scores = np.asarray(
                    [snippet_score[start:end].mean() for start, end in ranges], dtype=np.float32
                )
                # LAP Eq. 13. Applying it to the within-video segment distribution
                # avoids a dataset-specific hand-tuned absolute score threshold.
                threshold = float(segment_scores.mean() + args.threshold_tau * segment_scores.std())
                np.savez_compressed(
                    cache,
                    snippet_score=snippet_score,
                    layer_score=layer_score,
                    segment_score=segment_scores,
                    ranges=np.asarray(ranges, dtype=np.int64),
                    threshold=np.float32(threshold),
                )
            chosen = np.where(segment_scores > threshold)[0].tolist()
            spans = merge_selected(chosen, ranges, segment_scores)
            spans = sorted(spans, key=lambda value: value[2], reverse=True)[: args.max_spans_per_crop]
            for segment_index, (start, end, score) in enumerate(spans):
                pseudo_id = f"{row_index:07d}_{segment_index:02d}"
                rows.append([
                    pseudo_id, str(row["key"]), str(row["clip_path"]), label,
                    start, end, score, threshold, str(cache),
                ])
    output_csv = output / "pseudo_segments.csv"
    write_csv(
        output_csv,
        [
            "pseudo_id", "key", "clip_path", "label", "start", "end",
            "score", "threshold", "score_cache",
        ],
        rows,
    )
    report = {
        **lens.config(),
        "selection": "LAP mean + tau * std over 32 temporal segments; adjacent segments merged",
        "threshold_tau": args.threshold_tau,
        "max_spans_per_crop": args.max_spans_per_crop,
        "segments": len(rows),
        "skipped_without_known_target": skipped_no_target,
        "csv": str(output_csv),
    }
    (output / "selection_report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(report, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
