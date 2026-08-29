from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle


COLORS = {
    "ink": "#18324A",
    "muted": "#5F7182",
    "line": "#91A4B5",
    "train_bg": "#F4F7FA",
    "infer_bg": "#F7FAFC",
    "data": "#DCEAF5",
    "primary": "#F4A261",
    "context": "#66C2A5",
    "normality": "#B39DDB",
    "consensus": "#42A5A5",
    "fusion": "#4C78A8",
    "temporal": "#E76F51",
    "output": "#2A9D8F",
    "white": "#FFFFFF",
}


def box(
    axis: plt.Axes,
    x: float,
    y: float,
    width: float,
    height: float,
    face: str,
    title: str,
    lines: tuple[str, ...] = (),
    edge: str | None = None,
    title_size: float = 10.0,
    body_size: float = 8.0,
) -> None:
    patch = FancyBboxPatch(
        (x, y),
        width,
        height,
        boxstyle="round,pad=0.008,rounding_size=0.012",
        linewidth=1.25,
        edgecolor=edge or face,
        facecolor=face,
        zorder=2,
    )
    axis.add_patch(patch)
    axis.text(
        x + width / 2,
        y + height - 0.032,
        title,
        ha="center",
        va="center",
        fontsize=title_size,
        fontweight="bold",
        color=COLORS["ink"],
        zorder=3,
    )
    if lines:
        axis.text(
            x + width / 2,
            y + height * 0.42,
            "\n".join(lines),
            ha="center",
            va="center",
            fontsize=body_size,
            color=COLORS["ink"],
            linespacing=1.35,
            zorder=3,
        )


def arrow(
    axis: plt.Axes,
    start: tuple[float, float],
    end: tuple[float, float],
    label: str = "",
    dashed: bool = False,
    color: str | None = None,
    connection: str = "arc3,rad=0",
) -> None:
    colour = color or COLORS["line"]
    patch = FancyArrowPatch(
        start,
        end,
        arrowstyle="-|>",
        mutation_scale=12,
        linewidth=1.35,
        linestyle="--" if dashed else "-",
        color=colour,
        connectionstyle=connection,
        zorder=1,
    )
    axis.add_patch(patch)
    if label:
        axis.text(
            (start[0] + end[0]) / 2,
            (start[1] + end[1]) / 2 + 0.016,
            label,
            ha="center",
            va="bottom",
            fontsize=7.2,
            color=COLORS["muted"],
            bbox={"facecolor": "white", "edgecolor": "none", "pad": 0.8},
            zorder=4,
        )


def pill(axis: plt.Axes, x: float, y: float, text: str, face: str) -> None:
    axis.text(
        x,
        y,
        text,
        ha="left",
        va="center",
        fontsize=7.4,
        fontweight="bold",
        color=COLORS["ink"],
        bbox={"boxstyle": "round,pad=0.28", "facecolor": face, "edgecolor": "none"},
        zorder=5,
    )


def matrix_icon(axis: plt.Axes, x: float, y: float, width: float, height: float) -> None:
    values = np.asarray(
        [[0.15, 0.35, 0.65, 0.45, 0.85], [0.75, 0.25, 0.50, 0.90, 0.40], [0.30, 0.80, 0.55, 0.20, 0.70]]
    )
    cmap = plt.get_cmap("Blues")
    rows, columns = values.shape
    for row in range(rows):
        for column in range(columns):
            axis.add_patch(
                Rectangle(
                    (x + column * width / columns, y + (rows - 1 - row) * height / rows),
                    width / columns - 0.001,
                    height / rows - 0.001,
                    facecolor=cmap(values[row, column]),
                    edgecolor="white",
                    linewidth=0.4,
                    zorder=4,
                )
            )


def curve_icon(axis: plt.Axes, x: float, y: float, width: float, height: float, color: str) -> None:
    points = np.asarray([0.12, 0.17, 0.15, 0.23, 0.21, 0.65, 0.90, 0.72, 0.34, 0.28, 0.20])
    positions = np.linspace(x, x + width, len(points))
    axis.plot(positions, y + points * height, color=color, linewidth=2.2, zorder=5)
    axis.fill_between(positions, y, y + points * height, color=color, alpha=0.15, zorder=4)


def detector_rows(axis: plt.Axes, x: float, y: float, width: float, height: float) -> None:
    rows = (
        ("e₁(t)", "Primary sparse", COLORS["primary"]),
        ("e₂(t)", "Multi-scale context", COLORS["context"]),
        ("e₃(t)", "Directional normality", COLORS["normality"]),
    )
    row_height = height / 3
    for index, (symbol, name, colour) in enumerate(rows):
        bottom = y + height - (index + 1) * row_height
        axis.add_patch(
            FancyBboxPatch(
                (x, bottom + 0.006),
                width,
                row_height - 0.012,
                boxstyle="round,pad=0.004,rounding_size=0.006",
                facecolor=colour,
                edgecolor="white",
                linewidth=1.0,
                zorder=3,
            )
        )
        axis.text(x + 0.015, bottom + row_height / 2, symbol, fontsize=8.5, fontweight="bold", va="center")
        axis.text(x + 0.052, bottom + row_height / 2, name, fontsize=7.7, va="center")


def render(output: Path) -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 8.5,
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
        }
    )
    figure, axis = plt.subplots(figsize=(16.0, 8.8))
    axis.set_xlim(0, 1)
    axis.set_ylim(0, 1)
    axis.axis("off")

    axis.add_patch(
        FancyBboxPatch(
            (0.015, 0.61), 0.97, 0.35, boxstyle="round,pad=0.008,rounding_size=0.016",
            facecolor=COLORS["train_bg"], edgecolor="#CAD7E1", linewidth=1.1,
        )
    )
    axis.add_patch(
        FancyBboxPatch(
            (0.015, 0.055), 0.97, 0.49, boxstyle="round,pad=0.008,rounding_size=0.016",
            facecolor=COLORS["infer_bg"], edgecolor="#CAD7E1", linewidth=1.1,
        )
    )
    axis.text(0.03, 0.935, "A. Training-only neuron discovery (once per dataset)", fontsize=13, fontweight="bold", color=COLORS["ink"])
    pill(axis, 0.79, 0.935, "TRAIN SPLIT ONLY · NO TEST LABELS", "#DDF3E4")
    axis.text(0.03, 0.515, "B. Single-baseline inference (same rule for all baselines and datasets)", fontsize=13, fontweight="bold", color=COLORS["ink"])
    pill(axis, 0.79, 0.515, "ONE FROZEN BASELINE · SEED 234", "#E2ECF8")

    box(axis, 0.035, 0.69, 0.105, 0.17, COLORS["data"], "Training videos", ("video-level labels", "UCF or XD train split"))
    matrix_icon(axis, 0.056, 0.655, 0.062, 0.035)
    box(axis, 0.165, 0.68, 0.105, 0.19, "#E8F0F6", "CLIP CLS states", ("12 layers × 768 dims", "one coordinate = neuron"))
    matrix_icon(axis, 0.187, 0.708, 0.061, 0.045)
    arrow(axis, (0.14, 0.775), (0.165, 0.775), "stride 16")

    box(axis, 0.305, 0.675, 0.15, 0.20, "#FCE4CF", "1  Primary sparse", ("Top-32 / layer", "MIL abnormal response", "learns which neurons"), edge=COLORS["primary"])
    box(axis, 0.475, 0.675, 0.15, 0.20, "#DDF2EA", "2  Multi-scale context", ("current + Gaussian 1.5 / 4", "linear student", "learns when evidence persists"), edge=COLORS["context"])
    box(axis, 0.645, 0.675, 0.15, 0.20, "#EEE8F7", "3  Directional normality", ("normal μ, σ + direction", "Top-32 / layer", "learns how neurons deviate"), edge=COLORS["normality"])
    arrow(axis, (0.27, 0.775), (0.305, 0.775))
    arrow(axis, (0.455, 0.775), (0.475, 0.775), dashed=True)
    arrow(axis, (0.27, 0.745), (0.645, 0.745), dashed=True, connection="arc3,rad=-0.08")

    box(axis, 0.83, 0.675, 0.135, 0.20, "#E6EEF7", "Training statistics", ("detector parameters", "normal-video priors", "event persistence width"), edge=COLORS["fusion"])
    arrow(axis, (0.795, 0.775), (0.83, 0.775))
    arrow(axis, (0.625, 0.715), (0.83, 0.715), dashed=True, connection="arc3,rad=0.05")

    box(axis, 0.035, 0.20, 0.09, 0.18, COLORS["data"], "Input video", ("snippet sequence",))
    curve_icon(axis, 0.052, 0.14, 0.056, 0.045, COLORS["fusion"])
    box(axis, 0.155, 0.36, 0.125, 0.11, "#DCE7F3", "Frozen baseline", ("LaGoVAD / DeSC / DSANet",), edge=COLORS["fusion"], body_size=7.4)
    box(axis, 0.155, 0.15, 0.125, 0.13, "#E8F0F6", "Pre-extracted CLS", ("[T, 12, 768]", "no flow / patch token"), body_size=7.5)
    arrow(axis, (0.125, 0.305), (0.155, 0.415), connection="arc3,rad=-0.12")
    arrow(axis, (0.125, 0.275), (0.155, 0.215), connection="arc3,rad=0.12")

    box(axis, 0.31, 0.36, 0.12, 0.11, "#E3EBF5", "Baseline score", ("sᵦ(t)",), edge=COLORS["fusion"], title_size=9.5, body_size=11)
    arrow(axis, (0.28, 0.415), (0.31, 0.415))
    axis.add_patch(
        FancyBboxPatch(
            (0.305, 0.115), 0.145, 0.19, boxstyle="round,pad=0.008,rounding_size=0.012",
            facecolor="white", edgecolor="#B9C8D4", linewidth=1.2, zorder=2,
        )
    )
    axis.text(0.3775, 0.277, "Three neuron evidences", ha="center", va="center", fontsize=9.5, fontweight="bold", color=COLORS["ink"])
    detector_rows(axis, 0.318, 0.13, 0.119, 0.125)
    arrow(axis, (0.28, 0.215), (0.305, 0.215))

    box(axis, 0.49, 0.19, 0.13, 0.21, "#DDF1F1", "Spectral consensus", ("R⁺ = max(corr(e), 0)", "a = principal eigenvector", "g(t) = σ(Σ aᵢeᵢ(t))"), edge=COLORS["consensus"], body_size=8.3)
    arrow(axis, (0.45, 0.215), (0.49, 0.27), "e₁,e₂,e₃")
    box(axis, 0.655, 0.17, 0.145, 0.25, "#DCE7F3", "Constrained score fusion", ("baseline-preserving logit", "+ positive agreement", "− conflict & normal prior", "neuron-gated peak support"), edge=COLORS["fusion"], body_size=8.0)
    arrow(axis, (0.43, 0.415), (0.655, 0.36), "single baseline score", connection="arc3,rad=-0.08")
    arrow(axis, (0.62, 0.295), (0.655, 0.295), "g(t)")
    box(axis, 0.83, 0.18, 0.105, 0.23, "#F9DED7", "Temporal recovery", ("training-width persistence", "local maximum expansion", "0.5-snippet alignment"), edge=COLORS["temporal"], body_size=7.7)
    arrow(axis, (0.80, 0.295), (0.83, 0.295))
    box(axis, 0.955, 0.20, 0.035, 0.19, "#DDF2EA", "", (), edge=COLORS["output"])
    curve_icon(axis, 0.959, 0.235, 0.027, 0.085, COLORS["output"])
    axis.text(0.9725, 0.17, "Frame-level\nanomaly score", ha="center", va="top", fontsize=7.4, fontweight="bold", color=COLORS["ink"])
    arrow(axis, (0.935, 0.295), (0.955, 0.295))

    arrow(axis, (0.38, 0.675), (0.36, 0.305), "frozen detector", dashed=True, color=COLORS["primary"], connection="arc3,rad=0.06")
    arrow(axis, (0.55, 0.675), (0.39, 0.305), "frozen detector", dashed=True, color=COLORS["context"], connection="arc3,rad=0.10")
    arrow(axis, (0.72, 0.675), (0.42, 0.305), "frozen detector", dashed=True, color=COLORS["normality"], connection="arc3,rad=0.12")
    arrow(axis, (0.895, 0.675), (0.875, 0.41), "training-only calibration", dashed=True, color=COLORS["fusion"])

    axis.text(
        0.5,
        0.025,
        "The same adapter consumes one baseline score stream and dataset-specific training-only CLS-neuron evidence; no second baseline, optical flow, patch token, or test-label fitting is used.",
        ha="center",
        va="center",
        fontsize=8.4,
        color=COLORS["muted"],
    )

    output.mkdir(parents=True, exist_ok=True)
    figure.savefig(output / "method_overview.svg", bbox_inches="tight", facecolor="white")
    figure.savefig(output / "method_overview.pdf", bbox_inches="tight", facecolor="white")
    figure.savefig(output / "method_overview.png", dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(figure)
    caption = (
        "Overview of the universal CLS-neuron adapter. The upper layer discovers three complementary neuron views "
        "from the official training split of each dataset: a primary sparse detector, a multi-scale context detector, "
        "and a directional normality detector. At inference, their standardized evidence is weighted by spectral "
        "consensus and conservatively fused with exactly one frozen baseline score stream, followed by training-scale "
        "temporal recovery to produce frame-level anomaly scores. No test label, second baseline, optical flow, or patch "
        "token is used by the method."
    )
    (output / "method_overview_caption.txt").write_text(caption, encoding="utf-8")
    print(f"[done] wrote method overview to {output}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Render the paper method-overview figure.")
    parser.add_argument(
        "--out-dir",
        default="../vadmy_data/universal_neuron_adapter/figures/method_overview",
    )
    parser.add_argument("--clean", action="store_true")
    args = parser.parse_args()
    output = Path(args.out_dir)
    if output.is_absolute():
        raise ValueError("--out-dir must be relative")
    if args.clean and output.exists():
        shutil.rmtree(output)
    render(output)


if __name__ == "__main__":
    main()
