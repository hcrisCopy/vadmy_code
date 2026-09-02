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


class NormalContextWindowDataset(torch.utils.data.Dataset):
    """Normal-only, alignment-preserving windows for context prediction."""

    def __init__(
        self,
        manifest: str,
        maximum_length: int,
        overlap: int,
        training: bool = False,
        seed: int = 0,
        exhaustive: bool = True,
    ) -> None:
        frame = pd.read_csv(manifest)
        required = {"key", "binary_label", "hidden_path", "valid_snippets"}
        missing = required - set(frame.columns)
        if missing:
            raise ValueError(f"{manifest}: missing columns {sorted(missing)}")
        if maximum_length < 2 or not 0 <= overlap < maximum_length:
            raise ValueError("maximum_length must be >=2 and overlap in [0, maximum_length)")
        if (frame["binary_label"].astype(int) != 0).any():
            raise ValueError("context predictor manifests must contain normal videos only")
        self.frame = frame.reset_index(drop=True)
        self.maximum_length = int(maximum_length)
        self.overlap = int(overlap)
        self.training = bool(training)
        self.seed = int(seed)
        self.exhaustive = bool(exhaustive)
        self.epoch = 0
        self.windows: list[tuple[int, int, int]] = []
        stride = maximum_length - overlap
        for row_index, row in self.frame.iterrows():
            length = int(row.valid_snippets)
            if length < 2:
                continue
            if self.training:
                self.windows.append((row_index, 0, min(length, maximum_length)))
                continue
            if not self.exhaustive:
                start = max(0, (length - maximum_length) // 2)
                self.windows.append((row_index, start, min(length, start + maximum_length)))
                continue
            starts = list(range(0, max(1, length - overlap), stride))
            if starts and starts[-1] + maximum_length < length:
                starts.append(length - maximum_length)
            for start in starts:
                end = min(length, start + maximum_length)
                if end - start >= 2:
                    self.windows.append((row_index, start, end))

    def set_epoch(self, epoch: int) -> None:
        """Select a reproducible fresh crop for every training video."""
        self.epoch = int(epoch)

    def __len__(self) -> int:
        return len(self.windows)

    def __getitem__(self, index: int) -> dict[str, object]:
        row_index, start, end = self.windows[index]
        row = self.frame.iloc[row_index]
        if self.training:
            length = int(row.valid_snippets)
            if length > self.maximum_length:
                token = f"{self.seed}:{self.epoch}:{row.key}".encode("utf-8")
                digest = hashlib.sha256(token).digest()
                start = int.from_bytes(digest[:8], "little") % (
                    length - self.maximum_length + 1
                )
                end = start + self.maximum_length
        with np.load(str(row.hidden_path), allow_pickle=False) as archive:
            hidden = np.asarray(archive["hidden"][start:end], dtype=np.float32)
        return {
            "key": str(row.key),
            "start": start,
            "hidden": torch.from_numpy(hidden.copy()),
        }


def collate_context_windows(items: list[dict[str, object]]) -> dict[str, object]:
    lengths = torch.tensor([len(item["hidden"]) for item in items], dtype=torch.long)
    maximum = int(lengths.max())
    hidden = torch.zeros(len(items), maximum, 12, 768, dtype=torch.float32)
    mask = torch.zeros(len(items), maximum, dtype=torch.bool)
    for index, item in enumerate(items):
        length = int(lengths[index])
        hidden[index, :length] = item["hidden"]
        mask[index, :length] = True
    return {
        "keys": [str(item["key"]) for item in items],
        "starts": torch.tensor([int(item["start"]) for item in items]),
        "hidden": hidden,
        "mask": mask,
        "lengths": lengths,
    }


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


def uniform_temporal_average(features: np.ndarray, target_length: int) -> np.ndarray:
    """Match DSANet's training-time uniform bin averaging on any trailing shape."""
    features = np.asarray(features, dtype=np.float32)
    if len(features) <= target_length:
        return features
    boundaries = np.linspace(0, len(features), target_length + 1, dtype=np.int32)
    output = np.empty((target_length, *features.shape[1:]), dtype=np.float32)
    for index in range(target_length):
        left, right = int(boundaries[index]), int(boundaries[index + 1])
        output[index] = features[left:right].mean(axis=0) if left != right else features[left]
    return output


class AuditorTrainingDataset(torch.utils.data.Dataset):
    """Aligned hidden states and frozen host scores for B4 training.

    Long videos use DSANet's official training-time uniform bin averaging. The
    same bin boundaries are applied to hidden states and cached host scores, so
    the correction target never drifts out of temporal alignment.
    """

    def __init__(self, manifest: str, maximum_length: int) -> None:
        self.frame = pd.read_csv(manifest)
        self.maximum_length = int(maximum_length)
        self._cache: list[dict[str, object]] | None = None
        required = {
            "key",
            "binary_label",
            "hidden_path",
            "host_score_path",
            "valid_snippets",
        }
        missing = required - set(self.frame.columns)
        if missing:
            raise ValueError(f"{manifest}: missing columns {sorted(missing)}")
        if self.maximum_length < 2:
            raise ValueError("maximum_length must be at least two")

    def __len__(self) -> int:
        return len(self.frame)

    def _load_item(self, index: int) -> dict[str, object]:
        row = self.frame.iloc[index]
        length = int(row.valid_snippets)
        with np.load(str(row.hidden_path), allow_pickle=False) as archive:
            hidden = np.asarray(archive["hidden"][:length], dtype=np.float32)
        host_score = np.asarray(
            np.load(str(row.host_score_path), allow_pickle=False), dtype=np.float32
        ).reshape(-1)[:length]
        if hidden.shape != (length, 12, 768):
            raise ValueError(
                f"{row.hidden_path}: expected {(length, 12, 768)}, got {hidden.shape}"
            )
        if len(host_score) != length:
            raise ValueError(
                f"{row.host_score_path}: expected {length} scores, got {len(host_score)}"
            )
        if length > self.maximum_length:
            hidden = uniform_temporal_average(hidden, self.maximum_length)
            host_score = uniform_temporal_average(
                host_score[:, None], self.maximum_length
            )[:, 0]
        return {
            "key": str(row.key),
            "hidden": torch.from_numpy(hidden.copy()),
            "host_score": torch.from_numpy(host_score.copy()),
            "label": int(row.binary_label),
        }

    def preload(self, indices: object) -> None:
        """Read each compressed archive once; DataLoader fork workers share this cache."""
        if self._cache is not None:
            return
        cache: list[dict[str, object] | None] = [None] * len(self)
        for index in indices:
            position = int(index)
            cache[position] = self._load_item(position)
        if any(item is None for item in cache):
            raise ValueError("preload indices must cover the complete training dataset")
        self._cache = [item for item in cache if item is not None]

    def __getitem__(self, index: int) -> dict[str, object]:
        if self._cache is not None:
            return self._cache[index]
        return self._load_item(index)


class HostScoreTrainingDataset(torch.utils.data.Dataset):
    """Host-only W1 data path; it never opens hidden-state archives."""

    def __init__(self, manifest: str, maximum_length: int) -> None:
        self.frame = pd.read_csv(manifest)
        self.maximum_length = int(maximum_length)
        required = {"key", "binary_label", "host_score_path", "valid_snippets"}
        missing = required - set(self.frame.columns)
        if missing:
            raise ValueError(f"{manifest}: missing columns {sorted(missing)}")

    def __len__(self) -> int:
        return len(self.frame)

    def __getitem__(self, index: int) -> dict[str, object]:
        row = self.frame.iloc[index]
        length = int(row.valid_snippets)
        host_score = np.asarray(
            np.load(str(row.host_score_path), allow_pickle=False), dtype=np.float32
        ).reshape(-1)[:length]
        if len(host_score) != length:
            raise ValueError(f"{row.host_score_path}: host-score length mismatch")
        if length > self.maximum_length:
            host_score = uniform_temporal_average(
                host_score[:, None], self.maximum_length
            )[:, 0]
        return {
            "key": str(row.key),
            "host_score": torch.from_numpy(host_score.copy()),
            "label": int(row.binary_label),
        }


def collate_auditor_training(items: list[dict[str, object]]) -> dict[str, object]:
    """Right-pad a B4 batch; the boolean mask is authoritative everywhere."""
    lengths = torch.tensor([len(item["host_score"]) for item in items], dtype=torch.long)
    maximum = int(lengths.max())
    hidden = torch.zeros(len(items), maximum, 12, 768, dtype=torch.float32)
    host_score = torch.zeros(len(items), maximum, dtype=torch.float32)
    mask = torch.zeros(len(items), maximum, dtype=torch.bool)
    for index, item in enumerate(items):
        length = int(lengths[index])
        hidden[index, :length] = item["hidden"]
        host_score[index, :length] = item["host_score"]
        mask[index, :length] = True
    return {
        "keys": [str(item["key"]) for item in items],
        "hidden": hidden,
        "host_score": host_score,
        "mask": mask,
        "lengths": lengths,
        "labels": torch.tensor(
            [int(item["label"]) for item in items], dtype=torch.float32
        ),
    }


def collate_host_score_training(items: list[dict[str, object]]) -> dict[str, object]:
    lengths = torch.tensor([len(item["host_score"]) for item in items], dtype=torch.long)
    maximum = int(lengths.max())
    host_score = torch.zeros(len(items), maximum, dtype=torch.float32)
    mask = torch.zeros(len(items), maximum, dtype=torch.bool)
    for index, item in enumerate(items):
        length = int(lengths[index])
        host_score[index, :length] = item["host_score"]
        mask[index, :length] = True
    return {
        "keys": [str(item["key"]) for item in items],
        "host_score": host_score,
        "mask": mask,
        "lengths": lengths,
        "labels": torch.tensor(
            [int(item["label"]) for item in items], dtype=torch.float32
        ),
    }


class FinalLayerDataset(torch.utils.data.Dataset):
    def __init__(self, manifest: str, training: bool, maximum_length: int = 256) -> None:
        self.frame = pd.read_csv(manifest)
        self.training = bool(training)
        self.maximum_length = int(maximum_length)
        required = {"key", "binary_label", "feature_path", "frame_indices_path", "evaluation_frames"}
        missing = required - set(self.frame.columns)
        if missing:
            raise ValueError(f"{manifest}: missing columns {sorted(missing)}")

    def __len__(self) -> int:
        return len(self.frame)

    def __getitem__(self, index: int) -> dict[str, object]:
        row = self.frame.iloc[index]
        features = np.load(str(row.feature_path), mmap_mode="r", allow_pickle=False)
        if features.ndim != 2 or features.shape[1] != 768:
            raise ValueError(f"{row.feature_path}: expected [T,768], got {features.shape}")
        if self.training:
            features = uniform_temporal_average(features, self.maximum_length)
        else:
            features = np.asarray(features, dtype=np.float32)
        return {
            "key": str(row.key),
            "features": torch.from_numpy(np.asarray(features, dtype=np.float32).copy()),
            "label": int(row.binary_label),
            "frame_indices_path": str(row.frame_indices_path),
            "evaluation_frames": int(row.evaluation_frames),
        }


def collate_final_layer(items: list[dict[str, object]]) -> dict[str, object]:
    lengths = torch.tensor([len(item["features"]) for item in items], dtype=torch.long)
    maximum = int(lengths.max())
    features = torch.zeros(len(items), maximum, 768, dtype=torch.float32)
    mask = torch.zeros(len(items), maximum, dtype=torch.bool)
    for index, item in enumerate(items):
        length = int(lengths[index])
        features[index, :length] = item["features"]
        mask[index, :length] = True
    return {
        "keys": [str(item["key"]) for item in items],
        "features": features,
        "mask": mask,
        "lengths": lengths,
        "labels": torch.tensor([int(item["label"]) for item in items], dtype=torch.float32),
        "frame_indices_paths": [str(item["frame_indices_path"]) for item in items],
        "evaluation_frames": torch.tensor([int(item["evaluation_frames"]) for item in items], dtype=torch.long),
    }
