from __future__ import annotations

import hashlib
import re
from pathlib import Path

import numpy as np
import pandas as pd
import torch


CHUNK_SUFFIX = re.compile(r"__(\d+)$")
KNOWN_SUFFIXES = {".npy", ".npz", ".avi", ".mp4", ".mkv", ".mov", ".webm"}


def base_key(path_or_key: str) -> str:
    """Return the video key used to join DSANet chunks and hidden states."""
    value = Path(str(path_or_key)).name
    if Path(value).suffix.lower() in KNOWN_SUFFIXES:
        value = Path(value).stem
    return CHUNK_SUFFIX.sub("", value)


def is_normal_label(dataset: str, label: str) -> bool:
    value = str(label).strip().lower().replace("_", "").replace("-", "")
    normal_values = {"normal", "normalvideos"} if dataset == "ucf" else {"a", "normal", "normalvideos"}
    return value in normal_values


def indices_digest(indices: np.ndarray) -> str:
    values = np.asarray(indices, dtype=np.int64)
    return hashlib.sha256(values.tobytes()).hexdigest()


def load_aligned_hidden(row: pd.Series) -> dict[str, object]:
    """Load one audited video and discard any tail outside DSANet's GT domain."""
    with np.load(str(row.hidden_path), allow_pickle=False) as archive:
        hidden = np.asarray(archive["hidden"], dtype=np.float32)
        frame_indices = np.asarray(archive["frame_indices"], dtype=np.int64)

    if hidden.ndim != 3 or hidden.shape[1:] != (12, 768):
        raise ValueError(f"{row.hidden_path}: expected [T,12,768], got {hidden.shape}")
    valid_snippets = int(row.valid_snippets)
    hidden = hidden[:valid_snippets]
    frame_indices = frame_indices[:valid_snippets]
    if indices_digest(frame_indices) != str(row.frame_indices_sha256):
        raise RuntimeError(f"{row.key}: frame_indices changed after the P0 audit")
    return {
        "key": str(row.key),
        "hidden": torch.from_numpy(hidden),
        "frame_indices": torch.from_numpy(frame_indices.copy()),
        "mask": torch.ones(valid_snippets, dtype=torch.bool),
        "label": int(row.binary_label),
        "evaluation_frames": int(row.evaluation_frames),
    }

class AlignedHiddenDataset(torch.utils.data.Dataset):
    """Dataset backed by a P0 audited manifest."""

    def __init__(self, manifest: str) -> None:
        self.frame = pd.read_csv(manifest)
        required = {
            "key",
            "binary_label",
            "hidden_path",
            "valid_snippets",
            "evaluation_frames",
            "frame_indices_sha256",
        }
        missing = required - set(self.frame.columns)
        if missing:
            raise ValueError(f"{manifest}: missing columns {sorted(missing)}")

    def __len__(self) -> int:
        return len(self.frame)

    def __getitem__(self, index: int) -> dict[str, object]:
        return load_aligned_hidden(self.frame.iloc[index])


def collate_aligned_hidden(items: list[dict[str, object]]) -> dict[str, object]:
    """Right-pad hidden states while keeping the validity mask authoritative."""
    lengths = torch.tensor([len(item["hidden"]) for item in items], dtype=torch.long)
    maximum = int(lengths.max())
    hidden = torch.zeros(len(items), maximum, 12, 768, dtype=torch.float32)
    frame_indices = torch.zeros(len(items), maximum, dtype=torch.long)
    mask = torch.zeros(len(items), maximum, dtype=torch.bool)
    for index, item in enumerate(items):
        length = int(lengths[index])
        hidden[index, :length] = item["hidden"]
        frame_indices[index, :length] = item["frame_indices"]
        mask[index, :length] = True
    return {
        "keys": [str(item["key"]) for item in items],
        "hidden": hidden,
        "frame_indices": frame_indices,
        "mask": mask,
        "lengths": lengths,
        "labels": torch.tensor([int(item["label"]) for item in items], dtype=torch.float32),
        "evaluation_frames": torch.tensor([int(item["evaluation_frames"]) for item in items], dtype=torch.long),
    }
