#!/usr/bin/env python3
"""Select class-conditioned top-k anomaly segments without baseline scores."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from semantic_knn_splicing.common import clean_output, is_normal, write_csv
from semantic_knn_splicing.model import WholeLayerSemanticLocalizer, load_frozen_clip
from semantic_knn_splicing.prompts import label_targets


def segment_ranges(length: int, count: int) -> list[tuple[int, int]]:
    edges = np.linspace(0, length, min(count, length) + 1, dtype=np.int64)
    return [(int(edges[index]), int(edges[index + 1])) for index in range(len(edges) - 1)]


def merge_selected(indices: list[int], ranges: list[tuple[int, int]], scores: np.ndarray) -> list[tuple[int, int, float]]:
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


def load_model(args, device: torch.device):
    checkpoint = torch.load(args.localizer_model, map_location="cpu")
    config = checkpoint["run_config"]
    if config["dataset"] != args.dataset:
        raise RuntimeError("localizer checkpoint dataset mismatch")
    clip_model, tokenize = load_frozen_clip(args.clip_weight, device)
    model = WholeLayerSemanticLocalizer(
        config["layers"], clip_model, tokenize, args.dataset, config["context_length"]
    ).to(device)
    missing, unexpected = model.load_state_dict(checkpoint["model_state_dict"], strict=False)
    non_clip_missing = [key for key in missing if not key.startswith("clip_model.")]
    if non_clip_missing or unexpected:
        raise RuntimeError(f"invalid localizer checkpoint: missing={non_clip_missing}, unexpected={unexpected}")
    return model.eval()


def main() -> None:
    parser = argparse.ArgumentParser(description="Create baseline-score-free pseudo anomaly segments.")
    parser.add_argument("--dataset", choices=["ucf", "xd"], required=True)
    parser.add_argument("--train-csv", required=True)
    parser.add_argument("--localizer-model", required=True)
    parser.add_argument("--clip-weight", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--num-segments", type=int, default=32)
    parser.add_argument("--topk-segments", type=int, default=3)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--clean", action="store_true")
    args = parser.parse_args()
    if min(args.num_segments, args.topk_segments) <= 0 or args.topk_segments > args.num_segments:
        parser.error("invalid segment count")
    output = clean_output(args.out_dir, args.clean)
    cache_dir = output / "score_cache"
    cache_dir.mkdir(exist_ok=True)
    device = torch.device(args.device if torch.cuda.is_available() and args.device.startswith("cuda") else "cpu")
    model = load_model(args, device)
    frame = pd.read_csv(args.train_csv)
    missing = {"clip_path", "hidden_path", "label", "key"} - set(frame.columns)
    if missing:
        raise ValueError(f"{args.train_csv} is missing columns: {sorted(missing)}")
    rows = []
    with torch.no_grad():
        for row_index, row in tqdm(frame.iterrows(), total=len(frame), desc="select pseudo segments", unit="crop"):
            label = str(row["label"])
            if is_normal(args.dataset, label):
                continue
            targets = label_targets(args.dataset, label)
            if not targets:
                continue
            cache = cache_dir / f"{row_index:07d}_{Path(str(row['clip_path'])).stem}.npz"
            if cache.exists() and not args.clean:
                stored = np.load(cache)
                snippet_score = stored["score"]
            else:
                clip = np.load(str(row["clip_path"])).astype(np.float32)
                hidden = np.load(str(row["hidden_path"]))["hidden"].astype(np.float32)
                if len(clip) != len(hidden):
                    raise ValueError(f"{row['key']}: clip/hidden length mismatch")
                record = model(
                    torch.from_numpy(clip).unsqueeze(0).to(device),
                    torch.from_numpy(hidden).unsqueeze(0).to(device),
                )
                target_index = torch.tensor(targets, device=device)
                semantic = record["class_logits"][0].index_select(1, target_index).max(dim=1).values
                snippet_score = semantic.cpu().numpy().astype(np.float32)
                np.savez_compressed(cache, score=snippet_score)
            ranges = segment_ranges(len(snippet_score), args.num_segments)
            segment_scores = np.asarray([snippet_score[start:end].mean() for start, end in ranges])
            k = min(args.topk_segments, len(segment_scores))
            chosen = np.argpartition(segment_scores, len(segment_scores) - k)[-k:].tolist()
            for segment_index, (start, end, score) in enumerate(merge_selected(chosen, ranges, segment_scores)):
                pseudo_id = f"{row_index:07d}_{segment_index:02d}"
                rows.append([pseudo_id, str(row["key"]), str(row["clip_path"]), label, start, end, score])
    output_csv = output / "pseudo_segments.csv"
    write_csv(output_csv, ["pseudo_id", "key", "clip_path", "label", "start", "end", "score"], rows)
    report = {
        "method": "anomalyclip_text_topk_contiguous_segments_v1",
        "score_dependency": "frozen CLIP text directions and whole-layer localizer; no baseline anomaly score",
        "segments": len(rows), "csv": str(output_csv),
    }
    (output / "selection_report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
