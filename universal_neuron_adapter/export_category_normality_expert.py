from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

from universal_neuron_adapter.data import load_hidden_array
from universal_neuron_adapter.normality import layer_normalize, load_normality_model


def category_evidence(hidden: np.ndarray, model: dict[str, np.ndarray]) -> np.ndarray:
    normalized = layer_normalize(hidden)
    deviation = np.abs((normalized - model["normal_mean"]) / model["normal_scale"])
    curves = []
    for indices, weights in zip(model["indices"], model["weights"]):
        selected = np.take_along_axis(deviation, indices[None], axis=2)
        curves.append((selected * weights[None]).sum(axis=(1, 2)) / np.maximum(weights.sum(), 1e-6))
    return np.max(np.stack(curves), axis=0).astype(np.float32)


def main() -> None:
    parser = argparse.ArgumentParser(description="Export category-conditional normality evidence.")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--expert-model", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--clean", action="store_true")
    args = parser.parse_args()
    output = Path(args.out_dir)
    if args.clean and output.exists():
        shutil.rmtree(output)
    scores = output / "scores"
    scores.mkdir(parents=True, exist_ok=True)
    model = load_normality_model(args.expert_model)
    rows = []
    frame = pd.read_csv(args.manifest)
    for row in tqdm(frame.itertuples(index=False), total=len(frame), desc="export category normality evidence"):
        target = scores / f"{row.key}.npy"
        if not target.exists():
            np.save(target, category_evidence(load_hidden_array(str(row.hidden_path)), model))
        rows.append({"key": str(row.key), "expert4_score_path": str(target)})
    pd.DataFrame(rows).to_csv(output / "expert4_scores.csv", index=False)
    print(f"wrote {len(rows)} category normality curves", flush=True)


if __name__ == "__main__":
    main()
