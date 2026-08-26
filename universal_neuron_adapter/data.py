from __future__ import annotations

import numpy as np
import pandas as pd
import torch


def resample_curve(curve: np.ndarray, length: int) -> np.ndarray:
    curve = np.asarray(curve, dtype=np.float32).reshape(-1)
    if not len(curve):
        raise ValueError("cannot resample an empty curve")
    if len(curve) == length:
        return curve
    return np.interp(
        np.linspace(0.0, 1.0, length, dtype=np.float32),
        np.linspace(0.0, 1.0, len(curve), dtype=np.float32),
        curve,
    ).astype(np.float32)


def load_hidden(path: str) -> np.ndarray:
    loaded = np.load(path, allow_pickle=False)
    hidden = loaded["hidden"] if hasattr(loaded, "files") else loaded
    hidden = np.asarray(hidden, dtype=np.float32)
    if hidden.ndim != 3 or hidden.shape[1:] != (12, 768):
        raise ValueError(f"{path}: expected [T,12,768], got {hidden.shape}")
    return hidden


class ConsensusHiddenDataset(torch.utils.data.Dataset):
    """CLS states paired with train-only DeSC/DSANet consensus targets."""

    def __init__(self, key_manifest: str, desc_manifest: str, dsanet_manifest: str, maximum_length: int = 256) -> None:
        keys = pd.read_csv(key_manifest)[["key", "hidden_path", "binary_label"]]
        desc = pd.read_csv(desc_manifest)[["key", "baseline_score_path"]].rename(
            columns={"baseline_score_path": "desc_path"}
        )
        dsanet = pd.read_csv(dsanet_manifest)[["key", "baseline_score_path"]].rename(
            columns={"baseline_score_path": "dsanet_path"}
        )
        self.frame = keys.merge(desc, on="key", validate="one_to_one").merge(
            dsanet, on="key", validate="one_to_one"
        )
        self.maximum_length = int(maximum_length)

    def __len__(self) -> int:
        return len(self.frame)

    def __getitem__(self, index: int) -> dict:
        row = self.frame.iloc[index]
        hidden = load_hidden(str(row.hidden_path))
        desc = resample_curve(np.load(str(row.desc_path)), len(hidden))
        dsanet = resample_curve(np.load(str(row.dsanet_path)), len(hidden))
        desc_logit = np.log(np.clip(desc, 1e-5, 1.0 - 1e-5) / np.clip(1.0 - desc, 1e-5, 1.0))
        dsanet_logit = np.log(np.clip(dsanet, 1e-5, 1.0 - 1e-5) / np.clip(1.0 - dsanet, 1e-5, 1.0))
        target = 1.0 / (1.0 + np.exp(-0.5 * (desc_logit + dsanet_logit)))
        label = int(row.binary_label)
        if not label:
            target = np.zeros_like(target)
        if len(hidden) > self.maximum_length:
            indices = np.linspace(0, len(hidden) - 1, self.maximum_length).round().astype(np.int64)
            hidden, target = hidden[indices], target[indices]
        return {
            "key": str(row.key),
            "hidden": torch.from_numpy(hidden),
            "target": torch.from_numpy(target.astype(np.float32)),
            "label": label,
        }


def collate_consensus(items: list[dict]) -> dict:
    lengths = torch.tensor([len(item["hidden"]) for item in items], dtype=torch.long)
    maximum = int(lengths.max())
    hidden = torch.zeros(len(items), maximum, 12, 768, dtype=torch.float32)
    target = torch.zeros(len(items), maximum, dtype=torch.float32)
    for index, item in enumerate(items):
        length = len(item["hidden"])
        hidden[index, :length] = item["hidden"]
        target[index, :length] = item["target"]
    return {
        "keys": [item["key"] for item in items],
        "hidden": hidden,
        "target": target,
        "lengths": lengths,
        "labels": torch.tensor([item["label"] for item in items], dtype=torch.float32),
    }
