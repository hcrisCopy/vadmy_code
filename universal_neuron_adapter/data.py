from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
from pathlib import Path

import numpy as np
import pandas as pd
import torch

CHUNK_SUFFIX = re.compile(r"__(\d+)$")


def base_key(path_or_key: str) -> str:
    return CHUNK_SUFFIX.sub("", Path(str(path_or_key)).stem)


def is_normal_label(label: str) -> bool:
    value = str(label).strip().lower().replace("_", "").replace("-", "")
    return value in {"normal", "normalvideos"}


def read_hidden_manifest(path: str) -> dict[str, str]:
    frame = pd.read_csv(path)
    required = {"key", "hidden_path"}
    if not required.issubset(frame.columns):
        raise ValueError(f"{path}: missing columns {sorted(required - set(frame.columns))}")
    frame["key"] = frame["key"].map(base_key)
    if frame["key"].duplicated().any():
        duplicates = frame.loc[frame["key"].duplicated(), "key"].tolist()
        raise ValueError(f"{path}: duplicate hidden keys, first={duplicates[:5]}")
    return dict(zip(frame["key"].astype(str), frame["hidden_path"].astype(str)))


def file_sha256(path: str) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


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
    """Build a stratified validation split exclusively from official training data."""
    if not 0.0 < val_fraction < 1.0:
        raise ValueError("val-fraction must be in (0, 1)")
    output = Path(out_dir)
    output.mkdir(parents=True, exist_ok=True)
    signature = {
        "dataset": dataset,
        "paths": [train_csv, test_csv, train_hidden_manifest, test_hidden_manifest],
        "sha256": [file_sha256(path) for path in (train_csv, test_csv, train_hidden_manifest, test_hidden_manifest)],
        "seed": seed,
        "val_fraction": val_fraction,
        "skip_missing_hidden": skip_missing_hidden,
    }
    signature_path = output / "prepare_signature.json"
    expected = [output / name for name in ("train_all.csv", "expert_train.csv", "expert_val.csv", "test.csv", "split_audit.json")]
    if signature_path.exists() and all(path.exists() for path in expected):
        if json.loads(signature_path.read_text(encoding="utf-8")) != signature:
            raise RuntimeError("prepared manifest signature differs; use a new output directory")
        print(f"reusing audited manifests in {output}", flush=True)
        return

    train_hidden = read_hidden_manifest(train_hidden_manifest)
    test_hidden = read_hidden_manifest(test_hidden_manifest)
    hidden_overlap = set(train_hidden) & set(test_hidden)
    if hidden_overlap:
        raise RuntimeError(f"hidden train/test overlap: {sorted(hidden_overlap)[:10]}")
    missing: list[str] = []

    def build(source: str, hidden_map: dict[str, str]) -> list[dict[str, object]]:
        frame = pd.read_csv(source)
        required = {"path", "label"}
        if not required.issubset(frame.columns):
            raise ValueError(f"{source}: missing columns {sorted(required - set(frame.columns))}")
        frame["key"] = frame["path"].map(base_key)
        rows: list[dict[str, object]] = []
        for key, group in frame.groupby("key", sort=False):
            if key not in hidden_map:
                missing.append(str(key))
                if skip_missing_hidden:
                    continue
                raise KeyError(f"{source}: hidden state missing for {key}")
            labels = set(group["label"].astype(str))
            if len(labels) != 1:
                raise ValueError(f"{key}: inconsistent labels {sorted(labels)}")
            label = next(iter(labels))
            rows.append({"key": str(key), "label": label, "binary_label": int(not is_normal_label(label)), "clip_paths": "|".join(group["path"].astype(str)), "hidden_path": hidden_map[str(key)]})
        return rows

    train_rows = build(train_csv, train_hidden)
    test_rows = build(test_csv, test_hidden)
    train_keys = {str(row["key"]) for row in train_rows}
    test_keys = {str(row["key"]) for row in test_rows}
    overlap = train_keys & test_keys
    if overlap:
        raise RuntimeError(f"video train/test overlap: {sorted(overlap)[:10]}")

    rng = random.Random(seed)
    val_keys: set[str] = set()
    for binary in (0, 1):
        keys = [str(row["key"]) for row in train_rows if row["binary_label"] == binary]
        if len(keys) < 2:
            raise RuntimeError(f"class {binary} needs at least two training videos")
        rng.shuffle(keys)
        count = min(len(keys) - 1, max(1, round(len(keys) * val_fraction)))
        val_keys.update(keys[:count])
    expert_train = [row for row in train_rows if row["key"] not in val_keys]
    expert_val = [row for row in train_rows if row["key"] in val_keys]
    if val_keys & test_keys:
        raise RuntimeError("validation/test overlap")
    for name, rows in (("train_all.csv", train_rows), ("expert_train.csv", expert_train), ("expert_val.csv", expert_val), ("test.csv", test_rows)):
        pd.DataFrame(rows).to_csv(output / name, index=False)
    signature_path.write_text(json.dumps(signature, indent=2), encoding="utf-8")
    audit = {
        "status": "pass",
        "policy": "all fitting and checkpoint selection use official training videos only",
        "train_videos": len(train_rows),
        "expert_train_videos": len(expert_train),
        "expert_val_videos": len(expert_val),
        "test_videos": len(test_rows),
        "train_test_key_overlap": 0,
        "train_test_hidden_overlap": 0,
        "validation_test_key_overlap": 0,
        "missing_hidden": sorted(set(missing)),
    }
    (output / "split_audit.json").write_text(json.dumps(audit, indent=2), encoding="utf-8")
    print(json.dumps(audit, indent=2), flush=True)


def resample_curve(curve: np.ndarray, length: int) -> np.ndarray:
    curve = np.asarray(curve, dtype=np.float32).reshape(-1)
    if not len(curve):
        raise ValueError("cannot resample an empty curve")
    if len(curve) == length:
        return curve
    return np.interp(np.linspace(0, 1, length), np.linspace(0, 1, len(curve)), curve).astype(np.float32)


def load_hidden_array(path: str) -> np.ndarray:
    loaded = np.load(path, allow_pickle=False)
    hidden = np.asarray(loaded["hidden"] if hasattr(loaded, "files") else loaded, dtype=np.float32)
    if hidden.ndim != 3 or hidden.shape[1:] != (12, 768):
        raise ValueError(f"{path}: expected [T,12,768], got {hidden.shape}")
    return hidden


class HiddenVideoDataset(torch.utils.data.Dataset):
    def __init__(self, manifest: str, maximum_length: int = 256) -> None:
        self.frame = pd.read_csv(manifest)
        self.maximum_length = int(maximum_length)

    def __len__(self) -> int:
        return len(self.frame)

    def __getitem__(self, index: int) -> dict[str, object]:
        row = self.frame.iloc[index]
        hidden = load_hidden_array(str(row.hidden_path))
        if len(hidden) > self.maximum_length:
            indices = np.linspace(0, len(hidden) - 1, self.maximum_length).round().astype(np.int64)
            hidden = hidden[indices]
        return {"key": str(row.key), "hidden": torch.from_numpy(hidden), "label": int(row.binary_label)}


def collate_hidden(items: list[dict[str, object]]) -> dict[str, object]:
    lengths = torch.tensor([len(item["hidden"]) for item in items], dtype=torch.long)
    hidden = torch.zeros(len(items), int(lengths.max()), 12, 768)
    for index, item in enumerate(items):
        hidden[index, : len(item["hidden"])] = item["hidden"]
    return {"keys": [item["key"] for item in items], "hidden": hidden, "lengths": lengths, "labels": torch.tensor([item["label"] for item in items], dtype=torch.float32)}


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

    def __getitem__(self, index: int) -> dict[str, object]:
        row = self.frame.iloc[index]
        baseline = np.load(str(row.baseline_score_path)).astype(np.float32)
        expert = resample_curve(np.load(str(row.expert_score_path)), len(baseline))
        if len(baseline) > self.maximum_length:
            indices = np.linspace(0, len(baseline) - 1, self.maximum_length).round().astype(np.int64)
            baseline, expert = baseline[indices], expert[indices]
        return {"key": str(row.key), "baseline": torch.from_numpy(baseline), "expert": torch.from_numpy(expert), "label": int(row.binary_label)}


def collate_scores(items: list[dict[str, object]]) -> dict[str, object]:
    lengths = torch.tensor([len(item["baseline"]) for item in items], dtype=torch.long)
    baseline = torch.full((len(items), int(lengths.max())), 0.5)
    expert = torch.full_like(baseline, 0.5)
    for index, item in enumerate(items):
        length = len(item["baseline"])
        baseline[index, :length] = item["baseline"]
        expert[index, :length] = item["expert"]
    return {"keys": [item["key"] for item in items], "baseline": baseline, "expert": expert, "lengths": lengths, "labels": torch.tensor([item["label"] for item in items], dtype=torch.float32)}


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare and audit train/validation/test manifests.")
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
