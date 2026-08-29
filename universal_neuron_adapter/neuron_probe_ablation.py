from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import accuracy_score, average_precision_score, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

from universal_neuron_adapter.data import load_hidden_array
from universal_neuron_adapter.model import SparseNeuronExpert


LAYER_GROUPS = {
    "shallow_1_3": tuple(range(0, 3)),
    "middle_4_6": tuple(range(3, 6)),
    "transition_7_9": tuple(range(6, 9)),
    "deep_10_12": tuple(range(9, 12)),
}
COUNT_PER_LAYER = (4, 8, 16, 32)


def load_gate_ranking(checkpoint_path: str) -> np.ndarray:
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    config = checkpoint["config"]
    model = SparseNeuronExpert(
        int(config["active_per_layer"]), int(config["temporal_width"])
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    gate_scores = torch.sigmoid(model.gate_logits.detach()).cpu().numpy()
    return np.argsort(-gate_scores, axis=1)


def summarize_hidden(hidden: np.ndarray) -> np.ndarray:
    """Return upper/lower weak-label statistics for every CLS coordinate."""
    mean = hidden.mean(axis=-1, keepdims=True)
    variance = hidden.var(axis=-1, keepdims=True)
    hidden = (hidden - mean) / np.sqrt(variance + 1e-5)
    count = min(len(hidden), max(1, int(np.ceil(len(hidden) / 16))))
    upper = np.partition(hidden, len(hidden) - count, axis=0)[-count:].mean(axis=0)
    lower = np.partition(hidden, count - 1, axis=0)[:count].mean(axis=0)
    return np.stack((upper, lower), axis=-1).astype(np.float32)


def load_or_build_summaries(manifest_path: str, cache_path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if cache_path.exists():
        cached = np.load(cache_path, allow_pickle=False)
        return cached["features"], cached["labels"], cached["keys"]
    frame = pd.read_csv(manifest_path)
    summaries = []
    for row in tqdm(frame.itertuples(index=False), total=len(frame), desc=f"summarize {cache_path.stem}"):
        summaries.append(summarize_hidden(load_hidden_array(str(row.hidden_path))))
    features = np.stack(summaries)
    labels = frame["binary_label"].to_numpy(dtype=np.int64)
    keys = frame["key"].astype(str).to_numpy(dtype="U")
    np.savez_compressed(cache_path, features=features, labels=labels, keys=keys)
    return features, labels, keys


def coordinate_feature_indices(coordinates: np.ndarray) -> np.ndarray:
    flattened = coordinates[:, 0] * 768 + coordinates[:, 1]
    return np.stack((2 * flattened, 2 * flattened + 1), axis=1).reshape(-1)


def top_coordinates(ranking: np.ndarray, per_layer: int, layers: tuple[int, ...] | None = None) -> np.ndarray:
    selected_layers = tuple(range(12)) if layers is None else layers
    return np.asarray(
        [(layer, int(dimension)) for layer in selected_layers for dimension in ranking[layer, :per_layer]],
        dtype=np.int64,
    )


def same_layer_random(
    selected: np.ndarray, per_layer: int, seed: int
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    rows = []
    for layer in range(12):
        excluded = selected[selected[:, 0] == layer, 1]
        candidates = np.setdiff1d(np.arange(768), excluded)
        rows.extend((layer, int(value)) for value in rng.choice(candidates, per_layer, replace=False))
    return np.asarray(rows, dtype=np.int64)


def global_random(selected: np.ndarray, count: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    excluded = selected[:, 0] * 768 + selected[:, 1]
    candidates = np.setdiff1d(np.arange(12 * 768), excluded)
    flat = rng.choice(candidates, count, replace=False)
    return np.stack((flat // 768, flat % 768), axis=1).astype(np.int64)


def fit_probe(features: np.ndarray, labels: np.ndarray, seed: int):
    return make_pipeline(
        StandardScaler(),
        SGDClassifier(
            loss="log_loss",
            alpha=1e-4,
            class_weight="balanced",
            max_iter=300,
            tol=1e-4,
            random_state=seed,
            average=True,
        ),
    ).fit(features, labels)


def evaluate_probe(model, features: np.ndarray, labels: np.ndarray) -> dict[str, float]:
    scores = model.decision_function(features)
    predictions = (scores > 0.0).astype(np.int64)
    return {
        "auc": float(roc_auc_score(labels, scores)),
        "ap": float(average_precision_score(labels, scores)),
        "accuracy": float(accuracy_score(labels, predictions)),
    }


def train_and_score(
    train: np.ndarray,
    train_labels: np.ndarray,
    validation: np.ndarray,
    validation_labels: np.ndarray,
    test: np.ndarray,
    test_labels: np.ndarray,
    indices: np.ndarray,
    seed: int,
) -> dict[str, float]:
    model = fit_probe(train[:, indices], train_labels, seed)
    validation_metrics = evaluate_probe(model, validation[:, indices], validation_labels)
    test_metrics = evaluate_probe(model, test[:, indices], test_labels)
    return {
        **{f"validation_{name}": value for name, value in validation_metrics.items()},
        **{f"test_{name}": value for name, value in test_metrics.items()},
    }


def intervention_metrics(
    model,
    train: np.ndarray,
    train_labels: np.ndarray,
    test: np.ndarray,
    test_labels: np.ndarray,
    indices: np.ndarray,
) -> dict[str, float]:
    class_means = {
        label: train[train_labels == label].mean(axis=0) for label in (0, 1)
    }
    original = model.decision_function(test)
    modified = test.copy()
    for label in (0, 1):
        rows = test_labels == label
        modified[np.ix_(rows, indices)] = class_means[1 - label][indices]
    intervened = model.decision_function(modified)
    normal = test_labels == 0
    abnormal = test_labels == 1
    return {
        "normal_to_abnormal_logit_shift": float(np.mean(intervened[normal] - original[normal])),
        "normal_to_abnormal_flip_rate": float(np.mean((original[normal] <= 0.0) & (intervened[normal] > 0.0))),
        "abnormal_to_normal_logit_shift": float(np.mean(intervened[abnormal] - original[abnormal])),
        "abnormal_to_normal_flip_rate": float(np.mean((original[abnormal] > 0.0) & (intervened[abnormal] <= 0.0))),
    }


def plot_results(output: Path, controls: pd.DataFrame, layers: pd.DataFrame, counts: pd.DataFrame, interventions: pd.DataFrame) -> None:
    plt.rcParams.update({"font.size": 8, "font.family": "DejaVu Sans"})
    figure, axes = plt.subplots(2, 2, figsize=(10.6, 6.6), constrained_layout=True)
    colors = {"ours": "#0072B2", "control": "#999999", "accent": "#D55E00"}

    control_plot = controls.groupby("control", as_index=False).agg(
        mean=("test_auc", "mean"), std=("test_auc", "std")
    ).fillna(0.0)
    control_order = ["global_random", "same_layer_random", "hard_nonselected", "selected"]
    control_plot["order"] = control_plot.control.map(
        {name: index for index, name in enumerate(control_order)}
    )
    control_plot = control_plot.sort_values("order")
    positions = np.arange(len(control_plot))
    for position, row in zip(positions, control_plot.itertuples(index=False)):
        axes[0, 0].errorbar(
            100 * row.mean,
            position,
            xerr=100 * row.std,
            marker="o",
            markersize=7 if row.control == "selected" else 5,
            capsize=3,
            color=colors["ours"] if row.control == "selected" else colors["control"],
        )
    axes[0, 0].set_yticks(positions, control_plot.control)
    observed_low = (control_plot["mean"] - control_plot["std"]).min()
    observed_high = (control_plot["mean"] + control_plot["std"]).max()
    observed_span = max(observed_high - observed_low, np.finfo(float).eps)
    lower_bound = 100 * max(0.0, observed_low - observed_span)
    upper_bound = 100 * min(1.0, observed_high + observed_span)
    axes[0, 0].set_xlim(lower_bound, upper_bound)
    axes[0, 0].grid(axis="x", alpha=0.25)
    axes[0, 0].set_title("(a) Fixed-budget neuron specificity")
    axes[0, 0].set_xlabel("Video AUC (%)")

    axes[0, 1].bar(
        layers.layer_group,
        100 * layers.test_auc,
        color=["#56B4E9", "#009E73", "#E69F00", "#CC79A7"],
    )
    axes[0, 1].set_title("(b) Equal-budget layer localization")
    axes[0, 1].set_ylabel("Video AUC (%)")
    axes[0, 1].tick_params(axis="x", rotation=20)

    axes[1, 0].plot(
        counts.neurons,
        100 * counts.test_auc,
        marker="o",
        linewidth=2,
        color=colors["ours"],
    )
    axes[1, 0].set_title("(c) Neuron-count sensitivity")
    axes[1, 0].set_xlabel("Selected CLS coordinates")
    axes[1, 0].set_ylabel("Video AUC (%)")
    axes[1, 0].grid(axis="y", alpha=0.25)

    intervention_plot = interventions.groupby("control", as_index=False).agg(
        up=("normal_to_abnormal_flip_rate", "mean"),
        up_std=("normal_to_abnormal_flip_rate", "std"),
        down=("abnormal_to_normal_flip_rate", "mean"),
        down_std=("abnormal_to_normal_flip_rate", "std"),
    )
    intervention_plot = intervention_plot.fillna(0.0)
    positions = np.arange(len(intervention_plot))
    axes[1, 1].bar(
        positions - 0.18,
        100 * intervention_plot.up,
        0.36,
        yerr=100 * intervention_plot.up_std,
        capsize=3,
        label="normal -> abnormal",
        color="#D55E00",
    )
    axes[1, 1].bar(
        positions + 0.18,
        100 * intervention_plot.down,
        0.36,
        yerr=100 * intervention_plot.down_std,
        capsize=3,
        label="abnormal -> normal",
        color="#009E73",
    )
    axes[1, 1].set_xticks(positions, intervention_plot.control)
    axes[1, 1].set_title("(d) Directional activation intervention")
    axes[1, 1].set_ylabel("Directional prediction flips (%)")
    axes[1, 1].legend(frameon=False, fontsize=7)

    for axis in axes.flat:
        axis.spines[["top", "right"]].set_visible(False)
    figure.savefig(output / "interpretability_ablations.png", dpi=300)
    figure.savefig(output / "interpretability_ablations.pdf")
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run fixed-protocol CLS-neuron interpretability ablations.")
    parser.add_argument("--train-manifest", required=True)
    parser.add_argument("--val-manifest", required=True)
    parser.add_argument("--test-manifest", required=True)
    parser.add_argument("--expert-model", required=True)
    parser.add_argument("--dataset", choices=["ucf", "xd"], required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--seed", type=int, default=234)
    parser.add_argument("--random-seeds", nargs="+", type=int, default=[234, 3407, 2026, 17, 73])
    parser.add_argument("--clean", action="store_true")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    output = Path(args.out_dir)
    if args.clean and output.exists():
        shutil.rmtree(output)
    output.mkdir(parents=True, exist_ok=True)
    final_summary = output / "summary.json"
    if args.resume and final_summary.exists():
        print(f"reuse {final_summary}", flush=True)
        return

    train_summary, train_labels, _ = load_or_build_summaries(args.train_manifest, output / "train_summaries.npz")
    validation_summary, validation_labels, _ = load_or_build_summaries(args.val_manifest, output / "validation_summaries.npz")
    test_summary, test_labels, _ = load_or_build_summaries(args.test_manifest, output / "test_summaries.npz")
    train = train_summary.reshape(len(train_summary), -1)
    validation = validation_summary.reshape(len(validation_summary), -1)
    test = test_summary.reshape(len(test_summary), -1)
    ranking = load_gate_ranking(args.expert_model)
    selected = top_coordinates(ranking, 32)

    control_rows = []
    control_sets: list[tuple[str, int, np.ndarray]] = [("selected", args.seed, selected)]
    hard = np.asarray(
        [(layer, int(dimension)) for layer in range(12) for dimension in ranking[layer, 32:64]],
        dtype=np.int64,
    )
    control_sets.append(("hard_nonselected", args.seed, hard))
    for seed in args.random_seeds:
        control_sets.append(("same_layer_random", seed, same_layer_random(selected, 32, seed)))
        control_sets.append(("global_random", seed, global_random(selected, len(selected), seed)))
    for control, seed, coordinates in tqdm(control_sets, desc="fixed-budget probes"):
        metrics = train_and_score(
            train, train_labels, validation, validation_labels, test, test_labels,
            coordinate_feature_indices(coordinates), args.seed,
        )
        control_rows.append({"dataset": args.dataset, "control": control, "seed": seed, "coordinates": len(coordinates), **metrics})
    controls = pd.DataFrame(control_rows)
    controls.to_csv(output / "fixed_budget_controls.csv", index=False)

    layer_rows = []
    for name, group in tqdm(LAYER_GROUPS.items(), desc="layer probes"):
        coordinates = top_coordinates(ranking, 32, group)
        metrics = train_and_score(
            train, train_labels, validation, validation_labels, test, test_labels,
            coordinate_feature_indices(coordinates), args.seed,
        )
        layer_rows.append({"dataset": args.dataset, "layer_group": name, "coordinates": len(coordinates), **metrics})
    layers = pd.DataFrame(layer_rows)
    layers.to_csv(output / "layer_depth.csv", index=False)

    count_rows = []
    for count in tqdm(COUNT_PER_LAYER, desc="count probes"):
        coordinates = top_coordinates(ranking, count)
        metrics = train_and_score(
            train, train_labels, validation, validation_labels, test, test_labels,
            coordinate_feature_indices(coordinates), args.seed,
        )
        count_rows.append({"dataset": args.dataset, "per_layer": count, "neurons": len(coordinates), **metrics})
    counts = pd.DataFrame(count_rows)
    counts.to_csv(output / "count_sensitivity.csv", index=False)

    full_probe = fit_probe(train, train_labels, args.seed)
    intervention_rows = []
    intervention_sets = [("selected", args.seed, selected)]
    intervention_sets.extend(
        ("same_layer_random", seed, same_layer_random(selected, 32, seed))
        for seed in args.random_seeds
    )
    for control, seed, coordinates in tqdm(intervention_sets, desc="activation interventions"):
        metrics = intervention_metrics(
            full_probe, train, train_labels, test, test_labels,
            coordinate_feature_indices(coordinates),
        )
        intervention_rows.append({"dataset": args.dataset, "control": control, "seed": seed, "coordinates": len(coordinates), **metrics})
    interventions = pd.DataFrame(intervention_rows)
    interventions.to_csv(output / "directional_interventions.csv", index=False)

    plot_results(output, controls, layers, counts, interventions)
    payload = {
        "dataset": args.dataset,
        "seed": args.seed,
        "random_seeds": args.random_seeds,
        "selection_source": "primary sparse detector fitted on the official training split",
        "probe_protocol": "fixed StandardScaler + averaged SGD logistic readout; no test-set model selection",
        "feature_budget": "two weak-label statistics per selected CLS coordinate",
        "test_usage": "post-hoc interpretability reporting only",
        "selected_control": controls.loc[controls.control == "selected"].iloc[0].to_dict(),
    }
    final_summary.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2), flush=True)


if __name__ == "__main__":
    main()
