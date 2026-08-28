from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.covariance import LedoitWolf
from tqdm import tqdm

from universal_neuron_adapter.covariance_expert import selected_directional_responses
from universal_neuron_adapter.data import load_hidden_array
from universal_neuron_adapter.normality import load_normality_model


def bounded_hidden(path: str, maximum_length: int) -> np.ndarray:
    hidden = load_hidden_array(path)
    if len(hidden) <= maximum_length:
        return hidden
    indices = np.linspace(0, len(hidden) - 1, maximum_length).round().astype(np.int64)
    return hidden[indices]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Fit a deterministic covariance-whitened CLS-neuron expert."
    )
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--normality-model", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--maximum-length", type=int, default=256)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--clean", action="store_true")
    args = parser.parse_args()

    output = Path(args.out_dir)
    model_path = output / "covariance_expert.npz"
    if args.clean and output.exists():
        shutil.rmtree(output)
    if args.resume and model_path.exists():
        print(f"reuse {model_path}", flush=True)
        return
    output.mkdir(parents=True, exist_ok=True)

    frame = pd.read_csv(args.manifest)
    normal = frame[frame["binary_label"] == 0]
    if normal.empty:
        raise ValueError("training manifest must contain normal videos")
    normality_model = load_normality_model(args.normality_model)
    response_blocks = []
    for row in tqdm(
        normal.itertuples(index=False),
        total=len(normal),
        desc="collect normal co-activations",
    ):
        hidden = bounded_hidden(str(row.hidden_path), args.maximum_length)
        response_blocks.append(selected_directional_responses(hidden, normality_model))
    responses = np.concatenate(response_blocks, axis=0)

    locations = []
    precisions = []
    shrinkages = []
    for layer in tqdm(range(responses.shape[1]), desc="fit layer covariance"):
        estimator = LedoitWolf(store_precision=True, assume_centered=False).fit(
            responses[:, layer]
        )
        locations.append(estimator.location_)
        precisions.append(estimator.precision_)
        shrinkages.append(estimator.shrinkage_)
    np.savez_compressed(
        model_path,
        location=np.asarray(locations, dtype=np.float32),
        precision=np.asarray(precisions, dtype=np.float32),
        shrinkage=np.asarray(shrinkages, dtype=np.float32),
    )
    metadata = {
        "definition": "layer-wise Ledoit-Wolf whitening over 32 signed CLS coordinates",
        "training_videos": int(len(normal)),
        "normal_snippets": int(len(responses)),
        "active_per_layer": int(responses.shape[-1]),
        "deterministic": True,
    }
    (output / "metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    print(json.dumps(metadata), flush=True)


if __name__ == "__main__":
    main()
