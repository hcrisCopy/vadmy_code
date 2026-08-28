#!/usr/bin/env python3
"""Build disjoint snippet-pair splits with the established shift-global768 rule."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from tqdm import tqdm

from .common import (
    clean_output,
    is_normal,
    load_hidden,
    paired_indices,
    read_hidden_manifest,
    read_pseudo_scores,
    read_source_labels,
    resample,
    save_json,
    video_fold,
    write_csv,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build video-disjoint shift-global768 positive/negative snippet pairs.")
    parser.add_argument("--dataset", choices=["ucf", "xd"], required=True)
    parser.add_argument("--source-train-csv", required=True)
    parser.add_argument("--hidden-manifest", required=True)
    parser.add_argument("--pseudo-csv", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--top-p", type=float, default=0.10)
    parser.add_argument("--discovery-fraction", type=float, default=0.40)
    parser.add_argument("--validation-fraction", type=float, default=0.20)
    parser.add_argument("--seed", type=int, default=234)
    parser.add_argument("--clean", action="store_true")
    args = parser.parse_args()
    if not 0.0 < args.top_p <= 0.5:
        parser.error("--top-p must be in (0, 0.5]")

    output = clean_output(args.out_dir, args.clean)
    manifest_path = output / "pairs.csv"
    if manifest_path.exists() and not args.clean:
        print(f"reuse completed pair manifest: {manifest_path}", flush=True)
        return
    labels = read_source_labels(args.source_train_csv)
    hidden_paths = read_hidden_manifest(args.hidden_manifest)
    pseudo = read_pseudo_scores(args.pseudo_csv)
    pair_dir = output / "per_video"
    pair_dir.mkdir(parents=True, exist_ok=True)
    rows, skipped = [], []
    candidates = [(key, label) for key, label in sorted(labels.items()) if not is_normal(args.dataset, label)]
    for key, label in tqdm(candidates, desc="build shift pairs", unit="video"):
        if key not in hidden_paths:
            skipped.append([key, label, "missing_hidden"])
            continue
        if key not in pseudo:
            skipped.append([key, label, "missing_pseudo_score"])
            continue
        pseudo_label, score_path = pseudo[key]
        if pseudo_label != label:
            raise ValueError(f"{key}: pseudo label {pseudo_label!r} differs from source label {label!r}")
        pair_path = pair_dir / f"{key}.npz"
        if pair_path.exists() and not args.clean:
            with np.load(pair_path, allow_pickle=False) as cached:
                count = int(cached["positive"].shape[0])
                hidden_length = int(cached["hidden_length"].item())
                raw_score_length = int(cached["raw_score_length"].item())
                positive_mean = float(cached["positive_score_mean"].item())
                negative_mean = float(cached["negative_score_mean"].item())
        else:
            hidden = load_hidden(hidden_paths[key])
            raw_scores = np.asarray(np.load(score_path, allow_pickle=False), dtype=np.float32).reshape(-1)
            scores = resample(raw_scores[:, None], len(hidden))[:, 0]
            positive_idx, negative_idx = paired_indices(scores, args.top_p)
            positive, negative = hidden[positive_idx], hidden[negative_idx]
            count, hidden_length, raw_score_length = len(positive), len(hidden), len(raw_scores)
            positive_mean, negative_mean = float(scores[positive_idx].mean()), float(scores[negative_idx].mean())
            np.savez_compressed(
                pair_path,
                positive=positive,
                negative=negative,
                key=np.asarray(key),
                label=np.asarray(label),
                hidden_length=np.asarray(hidden_length, dtype=np.int64),
                raw_score_length=np.asarray(raw_score_length, dtype=np.int64),
                positive_score_mean=np.asarray(positive_mean, dtype=np.float32),
                negative_score_mean=np.asarray(negative_mean, dtype=np.float32),
            )
        fold = video_fold(key, args.seed, args.discovery_fraction, args.validation_fraction)
        rows.append([key, label, fold, str(pair_path), count, hidden_length, raw_score_length, positive_mean, negative_mean])

    write_csv(
        manifest_path,
        ["key", "label", "fold", "pair_path", "pair_count", "hidden_length", "raw_score_length", "positive_score_mean", "negative_score_mean"],
        rows,
    )
    write_csv(output / "skipped_videos.csv", ["key", "label", "reason"], skipped)
    counts = {fold: sum(row[2] == fold for row in rows) for fold in ("discovery", "train", "validation")}
    if min(counts.values()) == 0:
        raise RuntimeError(f"empty video split: {counts}")
    save_json(output / "pair_contract.json", {
        "method": "shift_global768_video_internal_top_bottom",
        "dataset": args.dataset,
        "positive_definition": "highest baseline pseudo-score top-p snippets within each abnormal video",
        "negative_definition": "lowest baseline pseudo-score top-p snippets within the same abnormal video",
        "baseline_score_dependency": True,
        "frame_ground_truth_used": False,
        "video_disjoint_folds": counts,
        "top_p": args.top_p,
        "seed": args.seed,
        "skipped_training_videos": len(skipped),
    })
    print(f"wrote {manifest_path} | folds={counts} | skipped={len(skipped)}", flush=True)


if __name__ == "__main__":
    main()
