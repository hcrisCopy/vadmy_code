from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.special import expit
from tqdm import tqdm

from innovation_evidence.common import (
    BASELINES, DATASETS, DATASET_NAMES, expert_curves, frame_metric,
    merge_expert_manifests, prepare_output, require_relative,
)
from universal_neuron_adapter.data import resample_curve
from universal_neuron_adapter.evaluate import logit, standardize


CONFLICT_WEIGHT = 1.2


def evaluate_baseline(dataset: str, baseline: str, source: Path, normality: Path, context: Path, truth_path: Path, frames_per_snippet: int) -> list[dict[str, object]]:
    experts = merge_expert_manifests(
        source / dataset / "expert" / "test" / "expert_scores.csv",
        context / dataset / "top32_multiscale_seed234" / "test" / "student_scores.csv",
        normality / dataset / "top32_signed_v1" / "test" / "expert3_scores.csv",
    )
    baseline_frame = pd.read_csv(source / dataset / baseline / "baseline_test" / "baseline_scores.csv")[["key", "baseline_score_path"]]
    frame = baseline_frame.merge(experts, on="key", validate="one_to_one")
    raw_parts, positive_parts, asymmetric_parts = [], [], []
    for row in tqdm(frame.itertuples(index=False), total=len(frame), desc=f"{dataset}/{baseline}: asymmetric residual"):
        base = np.load(str(row.baseline_score_path)).astype(np.float32)
        curves = expert_curves(row)
        sparse = resample_curve(curves[0], len(base))
        normality_curve = resample_curve(curves[2], len(base))
        base_z = standardize(base)
        agreement = np.minimum(np.maximum(sparse, 0.0), np.maximum(normality_curve, 0.0))
        conflict = np.minimum(np.maximum(base_z, 0.0), np.minimum(np.maximum(-sparse, 0.0), np.maximum(-normality_curve, 0.0)))
        positive = expit(logit(base) + agreement)
        asymmetric = expit(logit(base) + agreement - CONFLICT_WEIGHT * conflict)
        raw_parts.append(base)
        positive_parts.append(positive)
        asymmetric_parts.append(asymmetric)
    truth = np.load(truth_path).reshape(-1)
    variants = {"Frozen baseline": raw_parts, "Positive agreement only": positive_parts, "Agreement + conflict suppression": asymmetric_parts}
    rows = []
    for name, parts in variants.items():
        score = np.repeat(np.concatenate(parts), frames_per_snippet)
        if len(score) != len(truth):
            raise RuntimeError(f"{dataset}/{baseline}: strict frame alignment failed")
        rows.append({"dataset": dataset, "baseline": baseline, "variant": name, "metric": frame_metric(dataset, truth, score)})
    return rows


def render(dataset: str, table: pd.DataFrame, output: Path) -> None:
    order = ("Frozen baseline", "Positive agreement only", "Agreement + conflict suppression")
    colors = ("#A7A7A7", "#56B4E9", "#0072B2")
    x = np.arange(len(BASELINES)); width = 0.32
    figure, axis = plt.subplots(figsize=(6.0, 3.2), constrained_layout=True)
    baseline_values = table[table.variant == order[0]].set_index("baseline").reindex(BASELINES).metric
    for index, (variant, color) in enumerate(zip(order[1:], colors[1:])):
        selected = table[table.variant == variant].set_index("baseline").reindex(BASELINES)
        gains = 100 * (selected.metric.to_numpy() - baseline_values.to_numpy())
        axis.bar(x + (index - 0.5) * width, gains, width, label=variant, color=color)
    axis.set_xticks(x, ("LaGoVAD", "DeSC", "DSANet"))
    axis.axhline(0.0, color="#444444", linewidth=0.8)
    axis.set_ylabel("Frame AUC gain (points)" if dataset == "ucf" else "Frame AP gain (points)")
    axis.set_title(f"{DATASET_NAMES[dataset]}: asymmetric residual diagnostic")
    axis.legend(frameon=False, fontsize=7)
    axis.grid(axis="y", alpha=0.25)
    figure.savefig(output / f"{dataset}_asymmetric_residual.png", dpi=300)
    figure.savefig(output / f"{dataset}_asymmetric_residual.pdf")
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(description="Isolate positive agreement and conflict suppression on frozen baseline scores.")
    parser.add_argument("--source-root", required=True)
    parser.add_argument("--normality-root", required=True)
    parser.add_argument("--context-root", required=True)
    parser.add_argument("--annotation-root", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--frames-per-snippet", type=int, default=16)
    parser.add_argument("--clean", action="store_true")
    args = parser.parse_args()
    roots = {name: Path(value) for name, value in vars(args).items() if name.endswith("root") or name == "out_dir"}
    for name, path in roots.items():
        require_relative(path, f"--{name.replace('_', '-')}")
    output = roots["out_dir"]
    if not prepare_output(output, args.clean, "innovation3_metrics.csv"):
        return
    rows = []
    for dataset in DATASETS:
        for baseline in BASELINES:
            rows.extend(evaluate_baseline(
                dataset, baseline, roots["source_root"], roots["normality_root"], roots["context_root"],
                roots["annotation_root"] / dataset / "gt.npy", args.frames_per_snippet,
            ))
        render(dataset, pd.DataFrame(rows)[lambda x: x.dataset == dataset], output)
    pd.DataFrame(rows).to_csv(output / "innovation3_metrics.csv", index=False)


if __name__ == "__main__":
    main()
