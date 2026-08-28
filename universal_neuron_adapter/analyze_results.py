from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import average_precision_score, roc_auc_score
from tqdm import tqdm


def metric(dataset: str, truth: np.ndarray, score: np.ndarray) -> float:
    return float(roc_auc_score(truth, score) if dataset == "ucf" else average_precision_score(truth, score))


def load_videos(evaluation: Path, gt_path: Path, frames_per_snippet: int):
    frame = pd.read_csv(evaluation / "per_video.csv")
    truth = np.load(gt_path).astype(np.int8).reshape(-1)
    offset, videos = 0, []
    for row in frame.itertuples(index=False):
        curves = np.load(str(row.curve_path))
        length = int(row.snippets) * frames_per_snippet
        video_truth = truth[offset:offset + length]
        if len(video_truth) != length:
            raise RuntimeError(f"ground-truth exhausted at video {row.key}")
        videos.append((str(row.key), video_truth, np.repeat(curves["baseline"], frames_per_snippet), np.repeat(curves["corrected"], frames_per_snippet)))
        offset += length
    if offset != len(truth):
        raise RuntimeError(f"strict alignment failed: consumed={offset}, gt={len(truth)}")
    return videos


def paired_bootstrap(dataset: str, videos, repeats: int, seed: int):
    rng = np.random.default_rng(seed)
    observed_truth = np.concatenate([video[1] for video in videos])
    observed_base = np.concatenate([video[2] for video in videos])
    observed_method = np.concatenate([video[3] for video in videos])
    observed = metric(dataset, observed_truth, observed_method) - metric(dataset, observed_truth, observed_base)
    differences = []
    for _ in tqdm(range(repeats), desc=f"{dataset} paired video bootstrap", leave=False):
        sample = rng.integers(0, len(videos), size=len(videos))
        truth = np.concatenate([videos[index][1] for index in sample])
        base = np.concatenate([videos[index][2] for index in sample])
        method = np.concatenate([videos[index][3] for index in sample])
        if len(np.unique(truth)) < 2:
            continue
        differences.append(metric(dataset, truth, method) - metric(dataset, truth, base))
    values = np.asarray(differences)
    return {"observed_gain": observed, "ci95": np.quantile(values, [0.025, 0.975]).tolist(), "probability_gain_positive": float(np.mean(values > 0)), "valid_repeats": len(values), "unit": "video"}


def selection_stability(paths: list[str]):
    sets, layers = [], []
    for path in paths:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        neurons = {(int(row["layer"]), int(row["dimension"])) for row in payload["neurons"]}
        sets.append(neurons)
        layers.append(np.bincount([layer for layer, _ in neurons], minlength=13)[1:])
    jaccard = []
    for left in range(len(sets)):
        for right in range(left + 1, len(sets)):
            jaccard.append(len(sets[left] & sets[right]) / max(1, len(sets[left] | sets[right])))
    return {"pairwise_jaccard": jaccard, "mean_jaccard": float(np.mean(jaccard)) if jaccard else 1.0, "layer_counts": np.asarray(layers).tolist()}


def main() -> None:
    parser = argparse.ArgumentParser(description="Bootstrap results and generate paper-focused visualizations.")
    parser.add_argument("--results-root", required=True)
    parser.add_argument("--ucf-gt", required=True)
    parser.add_argument("--xd-gt", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--bootstrap-repeats", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument("--frames-per-snippet", type=int, default=16)
    parser.add_argument("--selection", action="append", default=[])
    args = parser.parse_args()
    root, output = Path(args.results_root), Path(args.out_dir)
    output.mkdir(parents=True, exist_ok=True)
    rows, bootstrap = [], {}
    for dataset, gt in (("ucf", Path(args.ucf_gt)), ("xd", Path(args.xd_gt))):
        for baseline in ("lagovad", "desc", "dsanet"):
            evaluation = root / dataset / baseline
            if not (evaluation / "per_video.csv").exists():
                continue
            videos = load_videos(evaluation, gt, args.frames_per_snippet)
            report = paired_bootstrap(dataset, videos, args.bootstrap_repeats, args.seed)
            bootstrap[f"{dataset}/{baseline}"] = report
            rows.append({"dataset": dataset.upper(), "baseline": baseline, "gain": 100 * report["observed_gain"], "lower": 100 * report["ci95"][0], "upper": 100 * report["ci95"][1]})
    (output / "paired_bootstrap.json").write_text(json.dumps(bootstrap, indent=2), encoding="utf-8")
    if args.selection:
        (output / "selection_stability.json").write_text(json.dumps(selection_stability(args.selection), indent=2), encoding="utf-8")
    if rows:
        frame = pd.DataFrame(rows)
        sns.set_theme(style="whitegrid", context="paper")
        figure, axes = plt.subplots(1, 2, figsize=(8.2, 3.3), sharex=False)
        for axis, dataset in zip(axes, ("UCF", "XD")):
            subset = frame[frame.dataset == dataset].reset_index(drop=True)
            positions = np.arange(len(subset))
            axis.errorbar(subset.gain, positions, xerr=[subset.gain - subset.lower, subset.upper - subset.gain], fmt="o", capsize=4, color="#176B87")
            axis.axvline(0, color="#B33A3A", linewidth=1, linestyle="--")
            axis.set_yticks(positions, subset.baseline)
            axis.set_xlabel("AUC gain (points)" if dataset == "UCF" else "AP gain (points)")
            axis.set_title(f"{dataset}: paired video bootstrap")
        figure.tight_layout()
        figure.savefig(output / "paired_gains_ci.png", dpi=300, bbox_inches="tight")
        figure.savefig(output / "paired_gains_ci.pdf", bbox_inches="tight")
        plt.close(figure)
        frame.to_csv(output / "paired_gains_ci.csv", index=False)


if __name__ == "__main__":
    main()
