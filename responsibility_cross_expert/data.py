"""Datasets shared by semantic-expert and head-only stages."""

from __future__ import annotations

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from .common import is_normal, stable_validation_key, temporal_mean_process, uniform_process


def _filter_kind(frame: pd.DataFrame, dataset: str, kind: str) -> pd.DataFrame:
    normal = frame["label"].map(lambda value: is_normal(dataset, str(value)))
    if kind == "normal":
        return frame[normal]
    if kind == "abnormal":
        return frame[~normal]
    if kind == "all":
        return frame
    raise ValueError(f"unknown data kind: {kind}")


def _filter_fold(
    frame: pd.DataFrame,
    fold: str,
    seed: int,
    validation_fraction: float,
) -> pd.DataFrame:
    if fold == "all":
        return frame
    validation = frame["key"].map(
        lambda key: stable_validation_key(str(key), seed, validation_fraction)
    )
    if fold == "train":
        return frame[~validation]
    if fold == "validation":
        return frame[validation]
    raise ValueError(f"unknown fold: {fold}")


class WholeLayerDataset(Dataset):
    def __init__(
        self,
        csv_path: str,
        dataset: str,
        sequence_length: int,
        kind: str = "all",
        fold: str = "all",
        seed: int = 234,
        validation_fraction: float = 0.1,
        include_clip: bool = True,
    ) -> None:
        frame = pd.read_csv(csv_path)
        required = {"clip_path", "hidden_path", "label", "key"}
        missing = required - set(frame.columns)
        if missing:
            raise ValueError(f"{csv_path}: missing columns {sorted(missing)}")
        frame = _filter_kind(frame, dataset, kind)
        frame = _filter_fold(frame, fold, seed, validation_fraction)
        self.frame = frame.reset_index(drop=True)
        self.dataset = dataset
        self.sequence_length = int(sequence_length)
        self.include_clip = bool(include_clip)

    def __len__(self) -> int:
        return len(self.frame)

    def __getitem__(self, index: int) -> dict:
        row = self.frame.iloc[index]
        stored = np.load(str(row["hidden_path"]))
        hidden = stored["hidden"].astype(np.float32)
        hidden, hidden_length = temporal_mean_process(hidden, self.sequence_length)
        item = {
            "hidden": torch.from_numpy(hidden),
            "length": torch.tensor(hidden_length, dtype=torch.long),
            "binary_label": torch.tensor(
                not is_normal(self.dataset, str(row["label"])), dtype=torch.float32
            ),
            "label_text": str(row["label"]),
            "key": str(row["key"]),
            "clip_path": str(row["clip_path"]),
            "hidden_path": str(row["hidden_path"]),
        }
        if self.include_clip:
            clip = np.load(str(row["clip_path"])).astype(np.float32)
            if len(clip) != len(stored["hidden"]):
                raise ValueError(f"unaligned clip/hidden lengths for {row['key']}")
            clip, clip_length = temporal_mean_process(clip, self.sequence_length)
            if clip_length != hidden_length:
                raise RuntimeError("clip and hidden temporal transforms disagree")
            item["clip"] = torch.from_numpy(clip)
        return item


class HeadTrainingDataset(Dataset):
    def __init__(
        self,
        consensus_csv: str,
        dataset: str,
        sequence_length: int,
        baseline: str,
        kind: str,
    ) -> None:
        frame = pd.read_csv(consensus_csv)
        required = {"clip_path", "label", "consensus_path", "key"}
        missing = required - set(frame.columns)
        if missing:
            raise ValueError(f"{consensus_csv}: missing columns {sorted(missing)}")
        self.frame = _filter_kind(frame, dataset, kind).reset_index(drop=True)
        self.dataset = dataset
        self.sequence_length = int(sequence_length)
        self.baseline = baseline

    def __len__(self) -> int:
        return len(self.frame)

    def _process(
        self, feature: np.ndarray, target: np.ndarray, target_length: int
    ) -> tuple[np.ndarray, np.ndarray, int]:
        if self.baseline in {"dsanet", "desc"}:
            feature, length = temporal_mean_process(feature, self.sequence_length)
        else:
            feature, length = uniform_process(feature, self.sequence_length)
        if len(target) != self.sequence_length:
            raise ValueError(
                f"consensus target must already have {self.sequence_length} steps, got {len(target)}"
            )
        if length != target_length:
            raise RuntimeError("feature and consensus label temporal transforms disagree")
        return feature, target, length

    def __getitem__(self, index: int) -> dict:
        row = self.frame.iloc[index]
        feature = np.load(str(row["clip_path"])).astype(np.float32)
        consensus = np.load(str(row["consensus_path"]))
        target = consensus["target"].astype(np.float32)
        target_length = int(consensus["length"])
        feature, target, length = self._process(feature, target, target_length)
        return {
            "clip": torch.from_numpy(feature),
            "target": torch.from_numpy(target),
            "length": torch.tensor(length, dtype=torch.long),
            "binary_label": torch.tensor(
                not is_normal(self.dataset, str(row["label"])), dtype=torch.float32
            ),
            "label_text": str(row["label"]),
            "key": str(row["key"]),
        }
