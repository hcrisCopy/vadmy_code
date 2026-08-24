from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from .common import is_normal_label


def uniform_process(feature: np.ndarray, target_length: int) -> tuple[np.ndarray, int]:
    """Match DSANet/DeSC process_feat for both aligned modalities."""
    feature = np.asarray(feature, dtype=np.float32)
    source_length = int(feature.shape[0])
    if source_length > target_length:
        output = np.zeros((target_length, feature.shape[1]), dtype=np.float32)
        boundaries = np.linspace(0, source_length, target_length + 1, dtype=np.int32)
        for index in range(target_length):
            if boundaries[index] != boundaries[index + 1]:
                output[index] = feature[boundaries[index]:boundaries[index + 1]].mean(axis=0)
            else:
                output[index] = feature[boundaries[index]]
        return output, target_length
    output = np.pad(feature, ((0, target_length - source_length), (0, 0)), mode="constant")
    return output.astype(np.float32), source_length


class AlignedFeatureDataset(Dataset):
    def __init__(
        self,
        aligned_csv: str,
        dataset: str,
        visual_length: int,
        split: str = "all",
    ) -> None:
        self.frame = pd.read_csv(aligned_csv)
        required = {"clip_path", "neuron_path", "label"}
        missing = required - set(self.frame.columns)
        if missing:
            raise ValueError(f"{aligned_csv} is missing columns: {sorted(missing)}")
        if split not in {"all", "normal", "abnormal"}:
            raise ValueError("split must be all, normal or abnormal")
        if split != "all":
            normal = self.frame["label"].map(lambda value: is_normal_label(dataset, str(value)))
            self.frame = self.frame[normal if split == "normal" else ~normal].reset_index(drop=True)
        self.dataset = dataset
        self.visual_length = int(visual_length)

    def __len__(self) -> int:
        return len(self.frame)

    def __getitem__(self, index: int) -> dict:
        row = self.frame.iloc[index]
        clip = np.load(str(row["clip_path"])).astype(np.float32)
        neuron = np.load(str(row["neuron_path"])).astype(np.float32)
        if clip.ndim != 2 or clip.shape[1] != 512:
            raise ValueError(f"{row['clip_path']}: expected [T,512], got {clip.shape}")
        if neuron.ndim != 2 or neuron.shape[0] != clip.shape[0]:
            raise ValueError(f"unaligned features: clip={clip.shape}, neuron={neuron.shape}")
        clip, length = uniform_process(clip, self.visual_length)
        neuron, neuron_length = uniform_process(neuron, self.visual_length)
        if length != neuron_length:
            raise RuntimeError("aligned modalities produced different valid lengths")
        label_text = str(row["label"])
        return {
            "clip": torch.from_numpy(clip),
            "neurons": torch.from_numpy(neuron),
            "length": torch.tensor(length, dtype=torch.long),
            "binary_label": torch.tensor(
                0.0 if is_normal_label(self.dataset, label_text) else 1.0,
                dtype=torch.float32,
            ),
            "label_text": label_text,
            "key": str(row.get("key", Path(str(row["clip_path"])).stem)),
        }


def load_full_row(row: pd.Series) -> tuple[np.ndarray, np.ndarray, str, str]:
    clip = np.load(str(row["clip_path"])).astype(np.float32)
    neuron = np.load(str(row["neuron_path"])).astype(np.float32)
    if clip.ndim != 2 or clip.shape[1] != 512 or neuron.ndim != 2 or clip.shape[0] != neuron.shape[0]:
        raise ValueError(f"invalid aligned test row: clip={clip.shape}, neuron={neuron.shape}")
    return clip, neuron, str(row["label"]), str(row.get("key", Path(str(row["clip_path"])).stem))
