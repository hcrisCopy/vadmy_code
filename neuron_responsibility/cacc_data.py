"""Datasets and grouped sampling for CACC without duplicating hidden states."""

from __future__ import annotations

from collections import OrderedDict
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, Sampler

from .common import is_normal_label, load_hidden, resample_feature


def uniform_process_nd(feature: np.ndarray, target_length: int) -> tuple[np.ndarray, int]:
    source_length = int(feature.shape[0])
    flattened = feature.reshape(source_length, -1)
    if source_length > target_length:
        output = np.zeros((target_length, flattened.shape[1]), dtype=np.float32)
        boundaries = np.linspace(0, source_length, target_length + 1, dtype=np.int32)
        for index in range(target_length):
            start, stop = int(boundaries[index]), int(boundaries[index + 1])
            output[index] = flattened[start:stop].mean(axis=0) if start != stop else flattened[start]
        length = target_length
    else:
        output = np.pad(flattened, ((0, target_length - source_length), (0, 0)))
        length = source_length
    return output.reshape((target_length,) + feature.shape[1:]).astype(np.float32), length


class CACCFeatureDataset(Dataset):
    def __init__(self, csv_path: str, dataset: str, visual_length: int, split: str = "all", cache_videos: int = 8) -> None:
        self.frame = pd.read_csv(csv_path)
        required = {"clip_path", "hidden_path", "label", "key"}
        missing = required - set(self.frame.columns)
        if missing:
            raise ValueError(f"{csv_path} is missing columns: {sorted(missing)}")
        if split not in {"all", "normal", "abnormal"}:
            raise ValueError("split must be all, normal or abnormal")
        if split != "all":
            normal = self.frame["label"].map(lambda value: is_normal_label(dataset, str(value)))
            self.frame = self.frame[normal if split == "normal" else ~normal].reset_index(drop=True)
        self.dataset = dataset
        self.visual_length = int(visual_length)
        self.cache_videos = max(1, int(cache_videos))
        self._cache: OrderedDict[str, np.ndarray] = OrderedDict()

    def __len__(self) -> int:
        return len(self.frame)

    def _hidden(self, key: str, path: str) -> np.ndarray:
        if key in self._cache:
            value = self._cache.pop(key)
            self._cache[key] = value
            return value
        value, _ = load_hidden(path)
        self._cache[key] = value
        while len(self._cache) > self.cache_videos:
            self._cache.popitem(last=False)
        return value

    def __getitem__(self, index: int) -> dict:
        row = self.frame.iloc[index]
        clip = np.load(str(row["clip_path"])).astype(np.float32)
        if clip.ndim != 2 or clip.shape[1] != 512:
            raise ValueError(f"{row['clip_path']}: expected [T,512], got {clip.shape}")
        key = str(row["key"])
        hidden = self._hidden(key, str(row["hidden_path"]))
        hidden = resample_feature(hidden.reshape(len(hidden), -1), len(clip)).reshape(
            len(clip), hidden.shape[1], hidden.shape[2]
        )
        clip, length = uniform_process_nd(clip, self.visual_length)
        hidden, hidden_length = uniform_process_nd(hidden, self.visual_length)
        if length != hidden_length:
            raise RuntimeError("CLIP and hidden states produced different valid lengths")
        label = str(row["label"])
        return {
            "clip": torch.from_numpy(clip), "hidden": torch.from_numpy(hidden),
            "length": torch.tensor(length, dtype=torch.long),
            "binary_label": torch.tensor(0.0 if is_normal_label(self.dataset, label) else 1.0),
            "label_text": label, "key": key, "sample_id": str(row["clip_path"]),
        }


class VideoGroupedSampler(Sampler[int]):
    """Shuffle videos while keeping their ten UCF crops adjacent for NPZ reuse."""

    def __init__(self, dataset: CACCFeatureDataset, seed: int) -> None:
        self.groups = [group.index.to_numpy() for _, group in dataset.frame.groupby("key", sort=False)]
        self.seed = int(seed)

    def __iter__(self):
        generator = np.random.default_rng(self.seed)
        order = generator.permutation(len(self.groups))
        for group_index in order:
            indices = self.groups[int(group_index)].copy()
            generator.shuffle(indices)
            yield from map(int, indices)

    def __len__(self) -> int:
        return sum(len(group) for group in self.groups)
