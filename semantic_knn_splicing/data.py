from __future__ import annotations

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from .common import is_normal, uniform_process


class BaselineTrainDataset(Dataset):
    """Load original MIL examples plus offline LaGoVAD-style synthesis."""

    def __init__(
        self,
        original_csv: str,
        synthetic_csv: str,
        dataset: str,
        sequence_length: int,
        baseline: str,
        split: str = "all",
    ) -> None:
        original = pd.read_csv(original_csv).rename(columns={"path": "clip_path"})
        if not {"clip_path", "label"}.issubset(original.columns):
            raise ValueError(f"{original_csv}: expected path/label columns")
        original["kind"] = "original"
        synthetic = pd.read_csv(synthetic_csv)
        if not {"feature_path", "label"}.issubset(synthetic.columns):
            raise ValueError(f"{synthetic_csv}: expected feature_path/label columns")
        synthetic = synthetic.rename(columns={"feature_path": "clip_path"})
        synthetic["kind"] = "synthetic"
        self.frame = pd.concat([original[["clip_path", "label", "kind"]], synthetic[["clip_path", "label", "kind"]]], ignore_index=True)
        if split == "normal":
            self.frame = self.frame[
                (self.frame["kind"] == "original")
                & self.frame["label"].map(lambda value: is_normal(dataset, str(value)))
            ]
        elif split == "abnormal":
            self.frame = self.frame[
                (self.frame["kind"] == "original")
                & ~self.frame["label"].map(lambda value: is_normal(dataset, str(value)))
            ]
        elif split == "synthetic":
            self.frame = self.frame[self.frame["kind"] == "synthetic"]
        elif split != "all":
            raise ValueError(f"unknown baseline training split: {split}")
        self.frame = self.frame.reset_index(drop=True)
        self.dataset = dataset
        self.sequence_length = sequence_length
        self.baseline = baseline

    def process(
        self, feature: np.ndarray, dense: np.ndarray, synthetic: bool
    ) -> tuple[np.ndarray, np.ndarray, int]:
        """Preserve each released baseline's training-time length transform."""
        if self.baseline == "lagovad" and synthetic and len(feature) > self.sequence_length:
            return (
                feature[:self.sequence_length].astype(np.float32),
                dense[:self.sequence_length].astype(np.float32),
                self.sequence_length,
            )
        if self.baseline not in {"dsanet", "desc"} or len(feature) <= self.sequence_length:
            feature, length = uniform_process(feature, self.sequence_length)
            dense, dense_length = uniform_process(dense[:, None], self.sequence_length)
            if length != dense_length:
                raise RuntimeError("feature and dense label produced different lengths")
            return feature, dense[:, 0], length
        edges = np.linspace(0, len(feature), self.sequence_length + 1, dtype=np.int64)
        feature_bins, dense_bins = [], []
        for index in range(self.sequence_length):
            start, end = int(edges[index]), int(edges[index + 1])
            if start == end:
                feature_bins.append(feature[start])
                dense_bins.append(dense[start])
            else:
                feature_bins.append(feature[start:end].mean(axis=0))
                dense_bins.append(dense[start:end].mean())
        return (
            np.stack(feature_bins).astype(np.float32),
            np.asarray(dense_bins, dtype=np.float32),
            self.sequence_length,
        )

    def __len__(self) -> int:
        return len(self.frame)

    def __getitem__(self, index: int) -> dict:
        row = self.frame.iloc[index]
        if row["kind"] == "synthetic":
            loaded = np.load(str(row["clip_path"]))
            clip = loaded["feature"].astype(np.float32)
            dense = loaded["frame_label"].astype(np.float32)
            synthetic = 1.0
        else:
            clip = np.load(str(row["clip_path"])).astype(np.float32)
            dense = np.zeros(len(clip), dtype=np.float32)
            synthetic = 0.0
        clip, dense, length = self.process(clip, dense, bool(synthetic))
        return {
            "clip": torch.from_numpy(clip),
            "length": torch.tensor(length, dtype=torch.long),
            "binary_label": torch.tensor(not is_normal(self.dataset, str(row["label"])), dtype=torch.float32),
            "dense_label": torch.from_numpy(dense),
            "synthetic": torch.tensor(synthetic, dtype=torch.float32),
            "label_text": str(row["label"]),
        }
