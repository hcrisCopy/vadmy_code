from __future__ import annotations

import argparse
import csv
import json
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from neuron_responsibility.common import base_key, is_normal_label, read_hidden_manifest


def _ordered_groups(frame: pd.DataFrame) -> list[tuple[str, pd.DataFrame]]:
    work = frame.copy()
    work["key"] = work["path"].map(base_key)
    return [(str(key), group) for key, group in work.groupby("key", sort=False)]


def _write_frame(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(path, index=False)


def prepare_manifests(
    dataset: str,
    train_csv: str,
    test_csv: str,
    train_hidden_manifest: str,
    test_hidden_manifest: str,
    out_dir: str,
    seed: int,
    val_fraction: float,
    skip_missing_hidden: bool,
) -> None:
    output = Path(out_dir)
    output.mkdir(parents=True, exist_ok=True)
    signature = {
        "dataset": dataset,
        "train_csv": train_csv,
        "test_csv": test_csv,
        "train_hidden_manifest": train_hidden_manifest,
        "test_hidden_manifest": test_hidden_manifest,
        "seed": seed,
        "val_fraction": val_fraction,
        "skip_missing_hidden": skip_missing_hidden,
    }
    signature_path = output / "prepare_signature.json"
    expected = [output / name for name in ("train_all.csv", "expert_train.csv", "expert_val.csv", "test.csv")]
    if signature_path.exists() and all(path.exists() for path in expected):
        if json.loads(signature_path.read_text(encoding="utf-8")) != signature:
            raise RuntimeError("prepared manifest signature differs; use a new output directory")
        print(f"reusing prepared manifests in {output}", flush=True)
        return

    train_hidden, _ = read_hidden_manifest(train_hidden_manifest)
    test_hidden, _ = read_hidden_manifest(test_hidden_manifest)
    missing: list[str] = []

    def build(source: str, hidden_map: dict[str, str]) -> list[dict]:
        frame = pd.read_csv(source)
        required = {"path", "label"}
        if not required.issubset(frame.columns):
            raise ValueError(f"{source}: missing columns {sorted(required - set(frame.columns))}")
        rows: list[dict] = []
        for key, group in _ordered_groups(frame):
            if key not in hidden_map:
                missing.append(key)
                if skip_missing_hidden:
                    continue
                raise KeyError(f"{source}: hidden state missing for {key}")
            labels = set(group["label"].astype(str))
            if len(labels) != 1:
                raise ValueError(f"{key}: inconsistent labels {sorted(labels)}")
            label = next(iter(labels))
            rows.append({
                "key": key,
                "label": label,
                "binary_label": int(not is_normal_label(dataset, label)),
                "clip_paths": "|".join(group["path"].astype(str).tolist()),
                "hidden_path": hidden_map[key],
            })
        return rows

    train_rows = build(train_csv, train_hidden)
    test_rows = build(test_csv, test_hidden)
    rng = random.Random(seed)
    val_keys: set[str] = set()
    for binary in (0, 1):
        keys = [row["key"] for row in train_rows if row["binary_label"] == binary]
        rng.shuffle(keys)
        count = max(1, int(round(len(keys) * val_fraction)))
        val_keys.update(keys[:count])
    expert_train = [row for row in train_rows if row["key"] not in val_keys]
    expert_val = [row for row in train_rows if row["key"] in val_keys]
    if not expert_train or not expert_val:
        raise RuntimeError("deterministic train/validation split is empty")
    _write_frame(output / "train_all.csv", train_rows)
    _write_frame(output / "expert_train.csv", expert_train)
    _write_frame(output / "expert_val.csv", expert_val)
    _write_frame(output / "test.csv", test_rows)
    signature_path.write_text(json.dumps(signature, indent=2), encoding="utf-8")
    report = {
        "train_videos": len(train_rows),
        "expert_train_videos": len(expert_train),
        "expert_val_videos": len(expert_val),
        "test_videos": len(test_rows),
        "missing_hidden": sorted(set(missing)),
    }
    (output / "prepare_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)


def load_hidden_array(path: str) -> np.ndarray:
    loaded = np.load(path, allow_pickle=False)
    hidden = loaded["hidden"] if hasattr(loaded, "files") else loaded
    hidden = np.asarray(hidden, dtype=np.float32)
    if hidden.ndim != 3 or hidden.shape[1:] != (12, 768):
        raise ValueError(f"{path}: expected [T,12,768], got {hidden.shape}")
    return hidden


def uniform_sample(array: np.ndarray, maximum_length: int) -> np.ndarray:
    if len(array) <= maximum_length:
        return array
    indices = np.linspace(0, len(array) - 1, maximum_length).round().astype(np.int64)
    return array[indices]


class HiddenVideoDataset(torch.utils.data.Dataset):
    def __init__(self, manifest: str, maximum_length: int = 256) -> None:
        self.frame = pd.read_csv(manifest)
        self.maximum_length = int(maximum_length)

    def __len__(self) -> int:
        return len(self.frame)

    def __getitem__(self, index: int) -> dict:
        row = self.frame.iloc[index]
        hidden = uniform_sample(load_hidden_array(str(row.hidden_path)), self.maximum_length)
        return {
            "key": str(row.key),
            "hidden": torch.from_numpy(hidden),
            "label": int(row.binary_label),
        }


def collate_hidden(items: list[dict]) -> dict:
    lengths = torch.tensor([len(item["hidden"]) for item in items], dtype=torch.long)
    maximum = int(lengths.max())
    hidden = torch.zeros(len(items), maximum, 12, 768, dtype=torch.float32)
    for index, item in enumerate(items):
        hidden[index, : len(item["hidden"])] = item["hidden"]
    return {
        "keys": [item["key"] for item in items],
        "hidden": hidden,
        "lengths": lengths,
        "labels": torch.tensor([item["label"] for item in items], dtype=torch.float32),
    }


def resample_curve(curve: np.ndarray, length: int) -> np.ndarray:
    curve = np.asarray(curve, dtype=np.float32).reshape(-1)
    if len(curve) == length:
        return curve
    if not len(curve):
        raise ValueError("cannot resample an empty score curve")
    return np.interp(
        np.linspace(0.0, 1.0, length, dtype=np.float32),
        np.linspace(0.0, 1.0, len(curve), dtype=np.float32),
        curve,
    ).astype(np.float32)


class ScorePairDataset(torch.utils.data.Dataset):
    def __init__(self, baseline_manifest: str, expert_manifest: str, key_manifest: str, maximum_length: int = 256) -> None:
        baseline = pd.read_csv(baseline_manifest)
        expert = pd.read_csv(expert_manifest)[["key", "expert_score_path"]]
        keys = set(pd.read_csv(key_manifest)["key"].astype(str))
        self.frame = baseline.merge(expert, on="key", validate="one_to_one")
        self.frame = self.frame[self.frame["key"].astype(str).isin(keys)].reset_index(drop=True)
        self.maximum_length = int(maximum_length)

    def __len__(self) -> int:
        return len(self.frame)

    def __getitem__(self, index: int) -> dict:
        row = self.frame.iloc[index]
        baseline = np.load(str(row.baseline_score_path)).astype(np.float32)
        expert = resample_curve(np.load(str(row.expert_score_path)), len(baseline))
        if len(baseline) > self.maximum_length:
            indices = np.linspace(0, len(baseline) - 1, self.maximum_length).round().astype(np.int64)
            baseline, expert = baseline[indices], expert[indices]
        return {
            "key": str(row.key),
            "baseline": torch.from_numpy(baseline),
            "expert": torch.from_numpy(expert),
            "label": int(row.binary_label),
        }


def collate_scores(items: list[dict]) -> dict:
    lengths = torch.tensor([len(item["baseline"]) for item in items], dtype=torch.long)
    maximum = int(lengths.max())
    baseline = torch.full((len(items), maximum), 0.5, dtype=torch.float32)
    expert = torch.full_like(baseline, 0.5)
    for index, item in enumerate(items):
        length = len(item["baseline"])
        baseline[index, :length] = item["baseline"]
        expert[index, :length] = item["expert"]
    return {
        "keys": [item["key"] for item in items],
        "baseline": baseline,
        "expert": expert,
        "lengths": lengths,
        "labels": torch.tensor([item["label"] for item in items], dtype=torch.float32),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare portable video/hidden manifests without copying features.")
    parser.add_argument("--dataset", choices=["ucf", "xd"], required=True)
    parser.add_argument("--train-csv", required=True)
    parser.add_argument("--test-csv", required=True)
    parser.add_argument("--train-hidden-manifest", required=True)
    parser.add_argument("--test-hidden-manifest", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--seed", type=int, default=234)
    parser.add_argument("--val-fraction", type=float, default=0.2)
    parser.add_argument("--skip-missing-hidden", action="store_true")
    args = parser.parse_args()
    prepare_manifests(**vars(args))


if __name__ == "__main__":
    main()

