from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from tqdm import tqdm


DATASETS = ("ucf", "xd")
DATASET_LABELS = {"ucf": "UCF-Crime", "xd": "XD-Violence"}
COLORS = {"ucf": "#0072B2", "xd": "#D55E00"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Visualize which CLS neurons are detected and whether they are functionally important."
    )
    parser.add_argument("--source-root", required=True, help="Frozen main-run root (the 9d1a066 run).")
    parser.add_argument("--normality-root", required=True, help="Root of directional normality caches.")
    parser.add_argument("--controls-csv", required=True, help="Selected-vs-random neuron-removal summary.")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--normality-tag", default="top32_signed_v1")
    parser.add_argument("--clean", action="store_true")
    return parser.parse_args()


def require_relative(path: Path, name: str) -> None:
    if path.is_absolute():
        raise ValueError(f"{name} must be relative, got: {path}")


def prepare_output(output: Path, clean: bool) -> bool:
    require_relative(output, "--out-dir")
    expected = (
        output / "detected_neurons.png",
        output / "detected_neurons.pdf",
        output / "detected_neuron_heatmap.png",
        output / "detected_neuron_heatmap.pdf",
    )
    if clean and output.exists():
        shutil.rmtree(output)
    if all(path.exists() for path in expected):
        print(f"[resume] visualization already exists under {output}")
        return False
    output.mkdir(parents=True, exist_ok=True)
    return True


def load_primary(source_root: Path, dataset: str) -> tuple[pd.DataFrame, np.ndarray]:
    selected_path = source_root / dataset / "expert" / "selected_neurons.json"
    payload = json.loads(selected_path.read_text(encoding="utf-8"))
    neurons = pd.DataFrame(payload["neurons"])
    neurons["dataset"] = dataset
    return neurons, np.asarray(payload["layer_weights"], dtype=np.float64)


def load_normality(normality_root: Path, tag: str, dataset: str) -> tuple[pd.DataFrame, np.ndarray]:
    model_path = normality_root / dataset / tag / "normality_expert.npz"
    with np.load(model_path) as model:
        indices = np.asarray(model["indices"])
        directions = np.asarray(model["directions"])
        weights = np.asarray(model["weights"], dtype=np.float64)
    rows = []
    for layer in range(indices.shape[0]):
        for rank in range(indices.shape[1]):
            rows.append(
                {
                    "dataset": dataset,
                    "layer": layer + 1,
                    "dimension": int(indices[layer, rank]),
                    "direction": "high" if int(directions[layer, rank]) == 0 else "low",
                    "weight": float(weights[layer, rank]),
                }
            )
    mass = weights.sum(axis=1)
    mass /= max(float(mass.sum()), 1e-12)
    return pd.DataFrame(rows), mass


def video_response_table(source_root: Path, dataset: str) -> pd.DataFrame:
    manifest = pd.read_csv(source_root / dataset / "expert" / "train" / "expert_scores.csv")
    rows = []
    for item in tqdm(
        manifest.itertuples(index=False), total=len(manifest), desc=f"{dataset}: training responses"
    ):
        curve = np.asarray(np.load(str(item.expert_score_path)), dtype=np.float64).reshape(-1)
        count = max(1, int(np.ceil(0.1 * len(curve))))
        top_response = float(np.partition(curve, len(curve) - count)[-count:].mean())
        rows.append(
            {
                "dataset": DATASET_LABELS[dataset],
                "class": "Abnormal" if int(item.binary_label) else "Normal",
                "top10_response": top_response,
                "video": str(item.key),
            }
        )
    return pd.DataFrame(rows)


def plot_atlas(axis: plt.Axes, primary: pd.DataFrame) -> None:
    for dataset, offset, marker in (("ucf", -0.10, "o"), ("xd", 0.10, "^")):
        part = primary[primary["dataset"] == dataset]
        scale = 12 + 50 * part["absolute_weight"] / max(float(part["absolute_weight"].max()), 1e-12)
        axis.scatter(
            part["layer"] + offset,
            part["dimension"],
            s=scale,
            marker=marker,
            color=COLORS[dataset],
            alpha=0.55,
            linewidths=0,
            label=DATASET_LABELS[dataset],
        )
    axis.set(xlabel="CLIP visual layer", ylabel="CLS hidden dimension", xticks=range(1, 13), ylim=(-20, 788))
    axis.set_title("a  Detected-neuron atlas", loc="left", fontweight="bold")
    axis.legend(frameon=False, loc="upper left", ncol=2)


def plot_layer_weights(
    axis: plt.Axes, primary_weights: dict[str, np.ndarray], normality_mass: dict[str, np.ndarray]
) -> None:
    layers = np.arange(1, 13)
    for dataset, marker in (("ucf", "o"), ("xd", "^")):
        axis.plot(
            layers,
            100 * primary_weights[dataset],
            color=COLORS[dataset],
            marker=marker,
            linewidth=1.8,
            markersize=4,
            label=f"{DATASET_LABELS[dataset]}: learned expert",
        )
        axis.plot(
            layers,
            100 * normality_mass[dataset],
            color=COLORS[dataset],
            marker=marker,
            linewidth=1.4,
            linestyle="--",
            markersize=4,
            markerfacecolor="white",
            label=f"{DATASET_LABELS[dataset]}: directional expert",
        )
    axis.set(xlabel="CLIP visual layer", ylabel="Contribution mass (%)", xticks=layers)
    axis.set_title("b  Contribution across layers", loc="left", fontweight="bold")
    axis.legend(frameon=False, fontsize=8, ncol=2)


def plot_training_response(axis: plt.Axes, responses: pd.DataFrame) -> None:
    palette = {"Normal": "#999999", "Abnormal": "#E69F00"}
    sns.boxplot(
        data=responses,
        x="dataset",
        y="top10_response",
        hue="class",
        order=[DATASET_LABELS[item] for item in DATASETS],
        hue_order=["Normal", "Abnormal"],
        palette=palette,
        width=0.62,
        showfliers=False,
        linewidth=1.0,
        ax=axis,
    )
    sns.stripplot(
        data=responses,
        x="dataset",
        y="top10_response",
        hue="class",
        order=[DATASET_LABELS[item] for item in DATASETS],
        hue_order=["Normal", "Abnormal"],
        palette=palette,
        dodge=True,
        alpha=0.18,
        size=1.8,
        legend=False,
        ax=axis,
    )
    handles, labels = axis.get_legend_handles_labels()
    axis.legend(handles[:2], labels[:2], frameon=False, loc="upper left")
    axis.set(xlabel="", ylabel="Video top-10% neuron response")
    axis.set_title("c  Training-only response separation", loc="left", fontweight="bold")


def plot_causal_control(axis: plt.Axes, controls: pd.DataFrame) -> None:
    controls = controls.copy()
    controls["setting"] = controls["dataset"].str.upper() + " / " + controls["baseline"]
    order = [f"{dataset.upper()} / {baseline}" for dataset in DATASETS for baseline in ("lagovad", "desc", "dsanet")]
    positions = np.arange(len(order))
    selected = controls.set_index("setting").loc[order, "selected_removal_drop"].to_numpy()
    random_mean = controls.set_index("setting").loc[order, "random_removal_drop_mean"].to_numpy()
    random_std = controls.set_index("setting").loc[order, "random_removal_drop_std"].to_numpy()
    axis.barh(positions, selected, height=0.52, color="#CC79A7", alpha=0.80, label="remove detected neurons")
    axis.errorbar(
        random_mean,
        positions,
        xerr=random_std,
        fmt="o",
        color="#222222",
        markerfacecolor="white",
        capsize=2,
        markersize=4,
        label="remove matched random neurons",
    )
    for y, value in zip(positions, selected):
        axis.text(value, y, f" {value:.2f}", va="center", fontsize=7)
    axis.axvline(0, color="#555555", linewidth=0.7)
    axis.set_xscale("symlog", linthresh=0.1, linscale=0.8)
    axis.set(yticks=positions, yticklabels=order, xlabel="Performance drop (metric points)")
    axis.invert_yaxis()
    axis.set_title("d  Causal neuron-removal control", loc="left", fontweight="bold")
    axis.legend(
        frameon=False,
        fontsize=8,
        loc="upper right",
        bbox_to_anchor=(1.0, -0.16),
        borderaxespad=0,
    )


def primary_matrix(frame: pd.DataFrame, layer_weights: np.ndarray) -> np.ndarray:
    matrix = np.zeros((12, 768), dtype=np.float64)
    for row in frame.itertuples(index=False):
        matrix[int(row.layer) - 1, int(row.dimension)] = (
            float(row.gate) * float(row.absolute_weight) * float(layer_weights[int(row.layer) - 1])
        )
    maximum = max(float(matrix.max()), 1e-12)
    return matrix / maximum


def directional_matrix(frame: pd.DataFrame) -> np.ndarray:
    matrix = np.zeros((12, 768), dtype=np.float64)
    for row in frame.itertuples(index=False):
        sign = 1.0 if row.direction == "high" else -1.0
        matrix[int(row.layer) - 1, int(row.dimension)] = sign * float(row.weight)
    maximum = max(float(np.abs(matrix).max()), 1e-12)
    return matrix / maximum


def render_neuron_heatmap(
    output: Path,
    primary: pd.DataFrame,
    normality: pd.DataFrame,
    layer_weights: dict[str, np.ndarray],
) -> None:
    primary_cmap = plt.get_cmap("viridis").copy()
    primary_cmap.set_bad("white")
    directional_cmap = plt.get_cmap("RdBu_r").copy()
    directional_cmap.set_bad("white")
    figure, axes = plt.subplots(2, 2, figsize=(11.5, 5.8), constrained_layout=True)
    primary_arrays, directional_arrays = {}, {}
    meshes = []
    for column, dataset in enumerate(DATASETS):
        learned = primary_matrix(primary[primary["dataset"] == dataset], layer_weights[dataset])
        directed_frame = normality[normality["dataset"] == dataset]
        directed = directional_matrix(directed_frame)
        primary_arrays[dataset] = learned
        directional_arrays[dataset] = directed

        learned_mesh = axes[0, column].pcolormesh(
            np.arange(769),
            np.arange(13),
            np.ma.masked_equal(learned, 0.0),
            cmap=primary_cmap,
            vmin=0.0,
            vmax=1.0,
            shading="flat",
            rasterized=False,
        )
        directed_mesh = axes[1, column].pcolormesh(
            np.arange(769),
            np.arange(13),
            np.ma.masked_equal(directed, 0.0),
            cmap=directional_cmap,
            vmin=-1.0,
            vmax=1.0,
            shading="flat",
            rasterized=False,
        )
        meshes.append((learned_mesh, directed_mesh))
        for direction, marker in (("high", "^"), ("low", "v")):
            subset = directed_frame[directed_frame["direction"] == direction]
            axes[1, column].scatter(
                subset["dimension"] + 0.5,
                subset["layer"] - 0.5,
                marker=marker,
                s=3.2,
                facecolors="none",
                edgecolors="#202020",
                linewidths=0.25,
                rasterized=False,
            )
        axes[0, column].set_title(
            f"{'a' if column == 0 else 'b'}  Learned sparse expert / {DATASET_LABELS[dataset]}",
            loc="left",
            fontweight="bold",
        )
        axes[1, column].set_title(
            f"{'c' if column == 0 else 'd'}  Directional expert / {DATASET_LABELS[dataset]}",
            loc="left",
            fontweight="bold",
        )

    for row in range(2):
        for column in range(2):
            axis = axes[row, column]
            axis.set_xlim(0, 768)
            axis.set_ylim(12, 0)
            axis.set_xticks([0, 128, 256, 384, 512, 640, 768])
            axis.set_yticks(np.arange(12) + 0.5, labels=np.arange(1, 13))
            axis.set_xlabel("CLS hidden dimension")
            axis.set_ylabel("CLIP visual layer" if column == 0 else "")
            axis.grid(axis="y", color="#D9D9D9", linewidth=0.35)
            axis.text(
                0.995,
                0.03,
                "32 selected / layer",
                transform=axis.transAxes,
                ha="right",
                va="bottom",
                fontsize=7.5,
                color="#333333",
            )
    first_colorbar = figure.colorbar(meshes[0][0], ax=axes[0, :], pad=0.015, aspect=30)
    first_colorbar.set_label("Normalized neuron importance")
    second_colorbar = figure.colorbar(meshes[0][1], ax=axes[1, :], pad=0.015, aspect=30)
    second_colorbar.set_label("Signed effect: below normal  ←  0  →  above normal")
    figure.savefig(output / "detected_neuron_heatmap.png", dpi=400, bbox_inches="tight")
    figure.savefig(output / "detected_neuron_heatmap.pdf", bbox_inches="tight")
    plt.close(figure)

    for dataset, matrix in primary_arrays.items():
        pd.DataFrame(matrix, index=np.arange(1, 13)).rename_axis("layer").to_csv(
            output / f"{dataset}_learned_neuron_heatmap.csv"
        )
    for dataset, matrix in directional_arrays.items():
        pd.DataFrame(matrix, index=np.arange(1, 13)).rename_axis("layer").to_csv(
            output / f"{dataset}_directional_neuron_heatmap.csv"
        )
    caption = (
        "Detected CLS neurons form sparse, distributed patterns across all 12 CLIP visual layers. "
        "Panels (a-b) show normalized importance for the learned sparse expert; white cells are not selected. "
        "Panels (c-d) show the directional expert, where upward triangles/red indicate above-normal activation "
        "and downward triangles/blue indicate below-normal activation. Each panel contains 32 selected CLS "
        "dimensions per layer, learned independently from official training data."
    )
    (output / "detected_neuron_heatmap_caption.txt").write_text(caption, encoding="utf-8")


def main() -> None:
    args = parse_args()
    source_root = Path(args.source_root)
    normality_root = Path(args.normality_root)
    controls_path = Path(args.controls_csv)
    output = Path(args.out_dir)
    for path, name in (
        (source_root, "--source-root"),
        (normality_root, "--normality-root"),
        (controls_path, "--controls-csv"),
    ):
        require_relative(path, name)
    if not prepare_output(output, args.clean):
        return

    primary_parts, normality_parts, response_parts = [], [], []
    primary_weights: dict[str, np.ndarray] = {}
    normality_mass: dict[str, np.ndarray] = {}
    for dataset in tqdm(DATASETS, desc="load neuron evidence"):
        primary, primary_weights[dataset] = load_primary(source_root, dataset)
        normality, normality_mass[dataset] = load_normality(normality_root, args.normality_tag, dataset)
        primary_parts.append(primary)
        normality_parts.append(normality)
        response_parts.append(video_response_table(source_root, dataset))
    primary_table = pd.concat(primary_parts, ignore_index=True)
    normality_table = pd.concat(normality_parts, ignore_index=True)
    responses = pd.concat(response_parts, ignore_index=True)
    controls = pd.read_csv(controls_path)

    sns.set_theme(style="ticks", context="paper", font_scale=1.0)
    plt.rcParams.update({"font.size": 8.5, "axes.titlesize": 9.5, "axes.labelsize": 8.5})
    figure, axes = plt.subplots(2, 2, figsize=(11.2, 7.2), constrained_layout=True)
    plot_atlas(axes[0, 0], primary_table)
    plot_layer_weights(axes[0, 1], primary_weights, normality_mass)
    plot_training_response(axes[1, 0], responses)
    plot_causal_control(axes[1, 1], controls)
    sns.despine(fig=figure)
    figure.suptitle(
        "Detected CLIP CLS neurons are sparse, interpretable, and causally important",
        fontsize=12,
        fontweight="bold",
    )
    figure.savefig(output / "detected_neurons.png", dpi=300, bbox_inches="tight")
    figure.savefig(output / "detected_neurons.pdf", bbox_inches="tight")
    plt.close(figure)
    render_neuron_heatmap(output, primary_table, normality_table, primary_weights)

    primary_table.to_csv(output / "detected_neuron_atlas.csv", index=False)
    normality_table.to_csv(output / "directional_neuron_atlas.csv", index=False)
    responses.to_csv(output / "training_neuron_responses.csv", index=False)
    metadata = {
        "figure": "detected_neurons",
        "neuron_definition": "one CLIP ViT-B/16 CLS hidden-state coordinate at one visual layer",
        "selection_count": "32 dimensions per layer, 12 layers, for each expert and dataset",
        "seed_policy": {
            "data_split": 234,
            "primary_expert_and_correction_heads": 234,
            "diverse_expert_and_context_student": 3407,
            "directional_normality_expert": "deterministic",
        },
        "panel_c_data": "official training videos only; top 10% mean of per-snippet primary neuron score",
        "panel_d_data": "post-hoc test-set causal intervention; five size-matched random removals",
        "source_root": str(source_root),
        "normality_root": str(normality_root),
        "controls_csv": str(controls_path),
    }
    (output / "figure_metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"[done] wrote neuron visualization and source tables to {output}")


if __name__ == "__main__":
    main()
