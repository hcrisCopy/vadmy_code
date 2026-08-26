from __future__ import annotations

import csv
import hashlib
import json
import random
import re
import shutil
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import torch


CHUNK_RE = re.compile(r"__(\d+)$")


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def stable_validation_key(key: str, seed: int, fraction: float) -> bool:
    """Deterministic video-level split without Python hash randomisation."""
    if not 0.0 < fraction < 1.0:
        raise ValueError("validation fraction must be in (0, 1)")
    digest = hashlib.sha256(f"{seed}:{key}".encode("utf-8")).digest()
    value = int.from_bytes(digest[:8], "big") / float(2**64)
    return value < fraction


def clean_output(path: str | Path, clean: bool) -> Path:
    output = Path(path)
    if clean and output.exists():
        resolved = output.resolve()
        if len(resolved.parts) < 3:
            raise ValueError(f"refusing to clean broad path: {resolved}")
        shutil.rmtree(resolved)
    output.mkdir(parents=True, exist_ok=True)
    return output


def base_key(value: str | Path) -> str:
    return CHUNK_RE.sub("", Path(str(value)).stem)


def is_normal(dataset: str, label: str) -> bool:
    if dataset == "ucf":
        return str(label).lower() == "normal"
    if dataset == "xd":
        return str(label).split("-")[0].upper() == "A"
    raise ValueError(f"unsupported dataset: {dataset}")


def read_source_csv(path: str | Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    missing = {"path", "label"} - set(frame.columns)
    if missing:
        raise ValueError(f"{path} is missing columns: {sorted(missing)}")
    return frame


def read_hidden_manifest(path: str | Path) -> dict[str, str]:
    frame = pd.read_csv(path)
    missing = {"key", "hidden_path"} - set(frame.columns)
    if missing:
        raise ValueError(f"{path} is missing columns: {sorted(missing)}")
    pools = set(frame.get("token_pool", pd.Series(["cls"])).astype(str))
    if pools != {"cls"}:
        raise ValueError(f"CLS hidden states are required, got {sorted(pools)}")
    return {str(row["key"]): str(row["hidden_path"]) for _, row in frame.iterrows()}


def load_hidden(path: str | Path) -> np.ndarray:
    loaded = np.load(path, allow_pickle=True)
    hidden = loaded if isinstance(loaded, np.ndarray) else loaded["hidden"]
    hidden = np.asarray(hidden, dtype=np.float32)
    if hidden.ndim != 3 or hidden.shape[-1] != 768:
        raise ValueError(f"{path}: expected [T,L,768], got {hidden.shape}")
    return hidden


def resample(feature: np.ndarray, length: int) -> np.ndarray:
    feature = np.asarray(feature, dtype=np.float32)
    if feature.shape[0] == length:
        return feature
    old_axis = np.linspace(0.0, 1.0, feature.shape[0], dtype=np.float32)
    new_axis = np.linspace(0.0, 1.0, length, dtype=np.float32)
    flat = feature.reshape(feature.shape[0], -1)
    columns = [np.interp(new_axis, old_axis, flat[:, index]) for index in range(flat.shape[1])]
    return np.stack(columns, axis=1).reshape((length,) + feature.shape[1:]).astype(np.float32)


def uniform_process(feature: np.ndarray, target_length: int) -> tuple[np.ndarray, int]:
    source_length = int(len(feature))
    if source_length >= target_length:
        indices = np.linspace(0, source_length - 1, target_length, dtype=np.int64)
        return np.asarray(feature[indices], dtype=np.float32), target_length
    pad = [(0, target_length - source_length)] + [(0, 0)] * (feature.ndim - 1)
    return np.pad(feature, pad, mode="constant").astype(np.float32), source_length


def temporal_mean_process(feature: np.ndarray, target_length: int) -> tuple[np.ndarray, int]:
    """DSANet/DeSC-style temporal bin averaging with zero padding."""
    feature = np.asarray(feature, dtype=np.float32)
    source_length = int(len(feature))
    if source_length <= target_length:
        pad = [(0, target_length - source_length)] + [(0, 0)] * (feature.ndim - 1)
        return np.pad(feature, pad, mode="constant").astype(np.float32), source_length
    edges = np.linspace(0, source_length, target_length + 1, dtype=np.int64)
    pooled = []
    for index in range(target_length):
        start, end = int(edges[index]), int(edges[index + 1])
        pooled.append(feature[start:end].mean(axis=0))
    return np.stack(pooled).astype(np.float32), target_length


def save_json(path: str | Path, value: object) -> None:
    with Path(path).open("w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, ensure_ascii=False)


def write_csv(path: str | Path, header: Iterable[str], rows: Iterable[Iterable[object]]) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(list(header))
        writer.writerows(rows)


def state_dict_from_checkpoint(path: str | Path) -> dict[str, torch.Tensor]:
    value = torch.load(path, map_location="cpu")
    if isinstance(value, dict):
        for key in ("model_state_dict", "state_dict"):
            if key in value:
                value = value[key]
                break
    if not isinstance(value, dict):
        raise TypeError(f"{path}: checkpoint does not contain a state dictionary")
    if value and all(str(key).startswith("module.") for key in value):
        value = {str(key)[7:]: tensor for key, tensor in value.items()}
    return value
