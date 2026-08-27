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

