from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

from universal_neuron_adapter.context_student import context_features
from universal_neuron_adapter.data import resample_curve
from universal_neuron_adapter.normality import load_normality_model


def main() -> None:
    parser = argparse.ArgumentParser(description="Fit a multi-scale directional CLS-neuron student.")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--expert-manifest", required=True)
    parser.add_argument("--auxiliary-manifest", required=True)
    parser.add_argument("--normality-manifest", required=True)
    parser.add_argument("--normality-model", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--normal-samples", type=int, default=32)
    parser.add_argument("--positive-fraction", type=float, default=0.05)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--clean", action="store_true")
    args = parser.parse_args()

    output = Path(args.out_dir)
    model_path = output / "context_student.npz"
    if args.clean and output.exists():
        shutil.rmtree(output)
    if args.resume and model_path.exists():
        print(f"reuse {model_path}", flush=True)
        return
    if args.normal_samples < 1 or not 0.0 < args.positive_fraction <= 1.0 or args.epochs < 1:
        raise ValueError("normal-samples and epochs must be positive; positive-fraction must be in (0,1]")
    output.mkdir(parents=True, exist_ok=True)

    frame = pd.read_csv(args.manifest)
    expert = pd.read_csv(args.expert_manifest)[["key", "expert_score_path"]]
    auxiliary = pd.read_csv(args.auxiliary_manifest)[["key", "expert2_score_path"]]
    normality = pd.read_csv(args.normality_manifest)[["key", "expert3_score_path"]]
    frame = frame.merge(expert, on="key", validate="one_to_one").merge(
        auxiliary, on="key", validate="one_to_one"
    ).merge(normality, on="key", validate="one_to_one")
    model = load_normality_model(args.normality_model)
    samples, labels = [], []
    for row in tqdm(frame.itertuples(index=False), total=len(frame), desc="sample context student"):
        features = context_features(str(row.hidden_path), model)
        length = len(features)
        if int(row.binary_label) == 0:
            indices = np.unique(
                np.linspace(0, length - 1, min(args.normal_samples, length)).round().astype(np.int64)
            )
            samples.append(features[indices])
            labels.append(np.zeros(len(indices), dtype=np.int8))
            continue
        first = resample_curve(np.load(str(row.expert_score_path)), length)
        second = resample_curve(np.load(str(row.expert2_score_path)), length)
        third = resample_curve(np.load(str(row.expert3_score_path)), length)
        first = (first - first.mean()) / max(float(first.std()), 1e-6)
        second = (second - second.mean()) / max(float(second.std()), 1e-6)
        third = (third - third.mean()) / max(float(third.std()), 1e-6)
        agreement = np.maximum(
            np.nan_to_num(np.corrcoef(np.stack([first, second, third])), nan=0.0),
            0.0,
        )
        np.fill_diagonal(agreement, 1.0)
        _, eigenvectors = np.linalg.eigh(agreement)
        weights = np.abs(eigenvectors[:, -1])
        weights /= max(float(weights.mean()), 1e-6)
        teacher = weights[0] * first + weights[1] * second + 3.0 * weights[2] * third
        count = max(1, int(np.ceil(args.positive_fraction * length)))
        indices = np.argpartition(teacher, length - count)[-count:]
        samples.append(features[indices])
        labels.append(np.ones(len(indices), dtype=np.int8))

    features = np.concatenate(samples)
    targets = np.concatenate(labels)
    scaler = StandardScaler().fit(features)
    classifier = SGDClassifier(
        loss="log_loss",
        alpha=1e-4,
        class_weight="balanced",
        max_iter=args.epochs,
        tol=None,
        random_state=args.seed,
        average=True,
    ).fit(scaler.transform(features), targets)
    np.savez_compressed(
        model_path,
        mean=scaler.mean_.astype(np.float32),
        scale=scaler.scale_.astype(np.float32),
        coef=classifier.coef_.astype(np.float32),
        intercept=classifier.intercept_.astype(np.float32),
    )
    print(
        {"samples": len(features), "features": features.shape[1], "epochs": classifier.n_iter_},
        flush=True,
    )


if __name__ == "__main__":
    main()

