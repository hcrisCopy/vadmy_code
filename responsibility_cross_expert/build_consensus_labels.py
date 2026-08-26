#!/usr/bin/env python3
"""CPL-VAD-style multi-scale refinement followed by conservative agreement."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from tqdm import tqdm

from .common import clean_output, is_normal, write_csv


THRESHOLDS = (0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95)
SCALES = (1, 2, 4, 8, 16)


def multi_scale_curves(score: np.ndarray) -> list[np.ndarray]:
    value = torch.from_numpy(score.astype(np.float32))[None, None]
    curves = []
    for scale in SCALES:
        pooled = F.avg_pool1d(value, kernel_size=scale, stride=scale, ceil_mode=True)
        restored = F.interpolate(pooled, size=len(score), mode="linear", align_corners=False)
        curves.append(restored[0, 0].numpy())
    return curves


def consecutive_segments(indices: np.ndarray, score: np.ndarray) -> list[tuple[int, int, float]]:
    if not len(indices):
        return []
    segments, start, previous = [], int(indices[0]), int(indices[0])
    values = [float(score[previous])]
    for raw in indices[1:]:
        current = int(raw)
        if current == previous + 1:
            values.append(float(score[current]))
            previous = current
            continue
        segments.append((start, previous + 1, float(np.mean(values))))
        start, previous, values = current, current, [float(score[current])]
    segments.append((start, previous + 1, float(np.mean(values))))
    return segments


def merge_segments(
    segments: list[tuple[int, int, float]], gap: int
) -> list[tuple[int, int, float]]:
    if not segments:
        return []
    merged = []
    start, end, score = segments[0]
    weight = max(1, end - start)
    for next_start, next_end, next_score in segments[1:]:
        if next_start - end <= gap:
            next_weight = max(1, next_end - next_start)
            score = (score * weight + next_score * next_weight) / (weight + next_weight)
            end, weight = next_end, weight + next_weight
        else:
            merged.append((start, end, score))
            start, end, score = next_start, next_end, next_score
            weight = max(1, end - start)
    merged.append((start, end, score))
    return merged


def flat_gaussian(length: int, flat_ratio: float) -> np.ndarray:
    result = np.ones(length, dtype=np.float32)
    flat_length = int(length * flat_ratio)
    tail = (length - flat_length) // 2
    if tail > 0:
        sigma = max(tail / 3.0, 1e-6)
        x = np.arange(tail, dtype=np.float32)
        decay = np.exp(-0.5 * ((x - (tail - 1)) / sigma) ** 2)
        result[:tail] = decay
        result[-tail:] = decay[::-1]
    return result


def refine_curve(
    score: np.ndarray,
    grouping: int,
    minimum_duration: int,
    cumulative_threshold: float,
    flat_ratio: float,
) -> np.ndarray:
    """Adapt CPL-VAD's public Generate_gt grouping/filter/soft-boundary code."""
    evidence = np.zeros(len(score), dtype=np.float32)
    for curve in multi_scale_curves(score):
        for threshold in THRESHOLDS:
            segments = consecutive_segments(np.where(curve >= threshold)[0], curve)
            segments = merge_segments(segments, grouping)
            for start, end, confidence in segments:
                if end - start > minimum_duration:
                    evidence[start:end] += confidence
    binary = evidence >= cumulative_threshold
    refined = np.zeros(len(score), dtype=np.float32)
    indices = np.where(binary)[0]
    for start, end, _ in consecutive_segments(indices, binary.astype(np.float32)):
        refined[start:end] = flat_gaussian(end - start, flat_ratio)
    return refined


def main() -> None:
    parser = argparse.ArgumentParser(description="Build sparse cross-expert consensus labels.")
    parser.add_argument("--dataset", choices=["ucf", "xd"], required=True)
    parser.add_argument("--expert-score-csv", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--grouping", type=int, default=6)
    parser.add_argument("--minimum-duration", type=int, default=4)
    parser.add_argument("--cumulative-threshold", type=float, default=22.0)
    parser.add_argument("--flat-ratio", type=float, default=0.55)
    parser.add_argument("--clean", action="store_true")
    args = parser.parse_args()
    if args.grouping < 0 or args.minimum_duration < 0:
        parser.error("grouping and minimum duration must be non-negative")
    if not 0.0 <= args.flat_ratio <= 1.0:
        parser.error("flat ratio must be in [0, 1]")
    output_dir = clean_output(args.out_dir, args.clean)
    label_dir = output_dir / "labels"
    label_dir.mkdir(exist_ok=True)
    frame = pd.read_csv(args.expert_score_csv)
    required = {"row_id", "key", "clip_path", "label", "length", "score_path"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"{args.expert_score_csv}: missing columns {sorted(missing)}")
    rows, statistics = [], {
        "valid": 0,
        "consensus_positive": 0,
        "semantic_only_raw": 0,
        "baseline_only_raw": 0,
        "both_high_raw": 0,
        "both_low_raw": 0,
        "ignored": 0,
    }
    abnormal_without_positive = 0
    for _, row in tqdm(frame.iterrows(), total=len(frame), desc="consensus labels", unit="crop"):
        target_path = label_dir / f"{int(row['row_id']):07d}.npz"
        if target_path.exists() and not args.clean:
            stored_target = np.load(target_path)
            target = stored_target["target"]
            semantic_refined = stored_target["semantic_refined"]
            baseline_refined = stored_target["baseline_refined"]
            stored_score = np.load(str(row["score_path"]))
            semantic_score = stored_score["semantic_score"]
            baseline_score = stored_score["baseline_score"]
        else:
            stored = np.load(str(row["score_path"]))
            semantic_score = stored["semantic_score"].astype(np.float32)
            baseline_score = stored["baseline_score"].astype(np.float32)
            length = int(row["length"])
            semantic_refined = refine_curve(
                semantic_score[:length], args.grouping, args.minimum_duration,
                args.cumulative_threshold, args.flat_ratio,
            )
            baseline_refined = refine_curve(
                baseline_score[:length], args.grouping, args.minimum_duration,
                args.cumulative_threshold, args.flat_ratio,
            )
            target = np.full(len(semantic_score), -1.0, dtype=np.float32)
            if is_normal(args.dataset, str(row["label"])):
                target[:length] = 0.0
            else:
                agreement = (semantic_refined > 0) & (baseline_refined > 0)
                valid_target = target[:length].copy()
                valid_target[agreement] = np.minimum(
                    semantic_refined[agreement], baseline_refined[agreement]
                )
                target[:length] = valid_target
                if not agreement.any():
                    abnormal_without_positive += 1
            padded_semantic = np.zeros_like(target)
            padded_baseline = np.zeros_like(target)
            padded_semantic[:length] = semantic_refined
            padded_baseline[:length] = baseline_refined
            semantic_refined, baseline_refined = padded_semantic, padded_baseline
            np.savez_compressed(
                target_path,
                target=target,
                semantic_refined=semantic_refined,
                baseline_refined=baseline_refined,
                length=np.int64(length),
            )
        length = int(row["length"])
        semantic_high = semantic_score[:length] >= 0.5
        baseline_high = baseline_score[:length] >= 0.5
        statistics["valid"] += length
        statistics["both_high_raw"] += int((semantic_high & baseline_high).sum())
        statistics["both_low_raw"] += int((~semantic_high & ~baseline_high).sum())
        statistics["semantic_only_raw"] += int((semantic_high & ~baseline_high).sum())
        statistics["baseline_only_raw"] += int((~semantic_high & baseline_high).sum())
        statistics["consensus_positive"] += int((target[:length] > 0).sum())
        statistics["ignored"] += int((target[:length] < 0).sum())
        rows.append([
            int(row["row_id"]), row["key"], row["clip_path"], row["label"],
            length, row["score_path"], str(target_path),
        ])
    output_csv = output_dir / "consensus_labels.csv"
    write_csv(
        output_csv,
        ["row_id", "key", "clip_path", "label", "length", "score_path", "consensus_path"],
        rows,
    )
    valid = max(1, statistics["valid"])
    report = {
        "method": "cpl_multiscale_refinement_then_cross_expert_intersection_v1",
        "dataset": args.dataset,
        "settings": {
            "scales": SCALES,
            "thresholds": THRESHOLDS,
            "grouping": args.grouping,
            "minimum_duration": args.minimum_duration,
            "cumulative_threshold": args.cumulative_threshold,
            "flat_ratio": args.flat_ratio,
        },
        "label_policy": "known normal=0; abnormal agreement=soft positive; disagreement/unselected=-1 ignore",
        **statistics,
        "consensus_positive_coverage": statistics["consensus_positive"] / valid,
        "ignored_fraction": statistics["ignored"] / valid,
        "abnormal_crops_without_positive": abnormal_without_positive,
        "csv": str(output_csv),
    }
    (output_dir / "consensus_report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(report, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
