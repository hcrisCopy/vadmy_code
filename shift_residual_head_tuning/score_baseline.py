#!/usr/bin/env python3
"""Cache frozen-baseline snippet scores used by Shift-Global768 selection."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm


PACKAGE_ROOT = Path(__file__).resolve().parent.parent
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))

from neuron_responsibility.baselines import build_baseline
from neuron_responsibility.common import clean_output, grouped_rows, is_normal_label, read_feature_csv, write_csv
from neuron_responsibility.evaluate import pad_chunks


def add_baseline_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--baseline", choices=["dsanet", "desc", "lagovad"], required=True)
    parser.add_argument("--baseline-root", required=True)
    parser.add_argument("--baseline-weight", default="")
    parser.add_argument("--sensitivity-weight", default="")
    parser.add_argument("--consistency-weight", default="")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate frozen-baseline scores for abnormal training videos.")
    add_baseline_arguments(parser)
    parser.add_argument("--dataset", choices=["ucf", "xd"], required=True)
    parser.add_argument("--source-train-csv", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--clean", action="store_true")
    args = parser.parse_args()

    out_dir = clean_output(args.out_dir, args.clean)
    score_dir = out_dir / "scores"
    score_dir.mkdir(exist_ok=True)
    groups = grouped_rows(read_feature_csv(args.source_train_csv))
    device = torch.device(args.device if torch.cuda.is_available() and args.device.startswith("cuda") else "cpu")
    adapter = build_baseline(args, str(device)).to(device)
    adapter.requires_grad_(False)
    adapter.eval()
    rows = []
    abnormal_groups = [
        (key, group) for key, group in groups.items()
        if not is_normal_label(args.dataset, str(group.iloc[0]["label"]))
    ]
    with torch.no_grad():
        for key, group in tqdm(abnormal_groups, desc=f"score {args.baseline}", unit="video"):
            labels = set(group["label"].astype(str))
            if len(labels) != 1:
                raise ValueError(f"{key}: inconsistent labels {sorted(labels)}")
            label = next(iter(labels))
            output_path = score_dir / f"{key}.npy"
            status = "reused"
            if not output_path.exists() or args.clean:
                clip = np.concatenate([
                    np.load(str(path)).astype(np.float32) for path in group["path"]
                ])
                clip_chunks, lengths = pad_chunks(clip, adapter.visual_length)
                output = adapter.forward_baseline(clip_chunks.to(device), lengths.to(device))
                values = []
                for index, length in enumerate(lengths.tolist()):
                    values.append(torch.sigmoid(output.binary_logits[index, :length]).cpu())
                scores = torch.cat(values).numpy().astype(np.float32)
                if not len(scores) or not np.isfinite(scores).all():
                    raise RuntimeError(f"{key}: baseline produced invalid scores")
                np.save(output_path, scores)
                status = "computed"
            score_count = int(np.load(output_path, mmap_mode="r").shape[0])
            rows.append([key, label, str(output_path), score_count, status])
    write_csv(out_dir / "group_scores.csv", ["key", "label", "score_path", "score_count", "status"], rows)
    print(f"wrote {out_dir / 'group_scores.csv'} with {len(rows)} abnormal videos", flush=True)


if __name__ == "__main__":
    main()
