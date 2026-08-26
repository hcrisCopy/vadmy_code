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


def clean_output(path: str | Path, clean: bool) -> Path:
    output = Path(path)
    if clean and output.exists():
        resolved = output.resolve()
        if len(resolved.parts) < 3:
            raise ValueError(f"refusing to clean broad path: {resolved}")
        shutil.rmtree(resolved)
    output.mkdir(parents=True, exist_ok=True)
    return output


def save_json(path: str | Path, value: object) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8")


def write_csv(path: str | Path, header: Iterable[str], rows: Iterable[Iterable[object]]) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(list(header))
        writer.writerows(rows)


def base_key(value: str | Path) -> str:
    return CHUNK_RE.sub("", Path(str(value)).stem)


def is_normal(dataset: str, label: str) -> bool:
    if dataset == "ucf":
        return str(label).lower() == "normal"
    if dataset == "xd":
        return str(label).split("-")[0].upper() == "A"
    raise ValueError(f"unsupported dataset: {dataset}")


def read_source_labels(path: str | Path) -> dict[str, str]:
    frame = pd.read_csv(path)
    missing = {"path", "label"} - set(frame.columns)
    if missing:
        raise ValueError(f"{path}: missing columns {sorted(missing)}")
    result: dict[str, str] = {}
    for _, row in frame.iterrows():
        key, label = base_key(str(row["path"])), str(row["label"])
        if key in result and result[key] != label:
            raise ValueError(f"{path}: inconsistent labels for {key}")
        result[key] = label
    return result


def read_hidden_manifest(path: str | Path) -> dict[str, str]:
    frame = pd.read_csv(path)
    missing = {"key", "hidden_path"} - set(frame.columns)
    if missing:
        raise ValueError(f"{path}: missing columns {sorted(missing)}")
    pools = set(frame["token_pool"].astype(str)) if "token_pool" in frame else {"cls"}
    if pools != {"cls"}:
        raise ValueError(f"{path}: expected CLS hidden states, got {sorted(pools)}")
    result: dict[str, str] = {}
    for _, row in frame.iterrows():
        key, hidden_path = str(row["key"]), str(row["hidden_path"])
        if key in result and result[key] != hidden_path:
            raise ValueError(f"{path}: conflicting duplicate key {key}")
        result[key] = hidden_path
    return result


def read_pseudo_scores(path: str | Path) -> dict[str, tuple[str, str]]:
    frame = pd.read_csv(path)
    missing = {"key", "label", "score_path"} - set(frame.columns)
    if missing:
        raise ValueError(f"{path}: missing columns {sorted(missing)}")
    result: dict[str, tuple[str, str]] = {}
    for _, row in frame.iterrows():
        key = str(row["key"])
        if key in result:
            raise ValueError(f"{path}: duplicate pseudo-score key {key}")
        result[key] = str(row["label"]), str(row["score_path"])
    return result


def load_hidden(path: str | Path) -> np.ndarray:
    loaded = np.load(path, allow_pickle=False)
    hidden = loaded if isinstance(loaded, np.ndarray) else loaded["hidden"]
    hidden = np.asarray(hidden, dtype=np.float32)
    if hidden.ndim != 3 or hidden.shape[0] == 0 or hidden.shape[-1] != 768:
        raise ValueError(f"{path}: expected non-empty [T,L,768], got {hidden.shape}")
    if not np.isfinite(hidden).all():
        raise ValueError(f"{path}: hidden contains non-finite values")
    return hidden


def resample(values: np.ndarray, length: int) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    if values.shape[0] == length:
        return values
    if values.shape[0] == 0 or length <= 0:
        raise ValueError("source and target lengths must be positive")
    source_axis = np.linspace(0.0, 1.0, values.shape[0], dtype=np.float32)
    target_axis = np.linspace(0.0, 1.0, length, dtype=np.float32)
    flat = values.reshape(values.shape[0], -1)
    result = np.stack([np.interp(target_axis, source_axis, flat[:, i]) for i in range(flat.shape[1])], axis=1)
    return result.reshape((length,) + values.shape[1:]).astype(np.float32)


def paired_indices(scores: np.ndarray, top_p: float) -> tuple[np.ndarray, np.ndarray]:
    """Exact shift-global768 rule: equal, disjoint top/bottom tails."""
    scores = np.asarray(scores, dtype=np.float32).reshape(-1)
    if len(scores) < 2 or not np.isfinite(scores).all():
        raise ValueError("need at least two finite pseudo scores")
    count = min(max(1, int(np.ceil(top_p * len(scores)))), len(scores) // 2)
    order = np.argsort(scores, kind="mergesort")
    negative, positive = order[:count], order[-count:][::-1]
    if np.intersect1d(positive, negative).size:
        raise RuntimeError("positive and negative tails overlap")
    return positive.astype(np.int64), negative.astype(np.int64)


def video_fold(key: str, seed: int, discovery_fraction: float, validation_fraction: float) -> str:
    if discovery_fraction <= 0 or validation_fraction <= 0 or discovery_fraction + validation_fraction >= 1:
        raise ValueError("discovery/validation fractions must be positive and sum to less than one")
    digest = hashlib.sha256(f"{seed}:{key}".encode("utf-8")).digest()
    value = int.from_bytes(digest[:8], "big") / float(2**64)
    if value < discovery_fraction:
        return "discovery"
    if value < discovery_fraction + validation_fraction:
        return "validation"
    return "train"


def load_pair(path: str | Path) -> tuple[np.ndarray, np.ndarray]:
    with np.load(path, allow_pickle=False) as item:
        positive = np.asarray(item["positive"], dtype=np.float32)
        negative = np.asarray(item["negative"], dtype=np.float32)
    if positive.ndim != 3 or negative.shape[1:] != positive.shape[1:] or positive.shape[-1] != 768:
        raise ValueError(f"{path}: invalid pair shapes {positive.shape}, {negative.shape}")
    return positive, negative


def read_pair_manifest(path: str | Path, fold: str) -> pd.DataFrame:
    frame = pd.read_csv(path)
    missing = {"key", "fold", "pair_path", "pair_count"} - set(frame.columns)
    if missing:
        raise ValueError(f"{path}: missing columns {sorted(missing)}")
    selected = frame[frame["fold"] == fold].reset_index(drop=True)
    if selected.empty:
        raise RuntimeError(f"{path}: fold {fold!r} is empty")
    return selected


def selected_coordinates(metadata: dict, mode: str, seed: int) -> list[tuple[int, int]]:
    selected = [(int(item["layer_index"]), int(dim)) for item in metadata["selected"] for dim in item["dims"]]
    if mode == "selected":
        return selected
    rng = np.random.default_rng(seed)
    layers, width = int(metadata["num_layers"]), int(metadata["hidden_dim"])
    if mode == "same_layer_random":
        result = []
        for item in metadata["selected"]:
            layer, count = int(item["layer_index"]), len(item["dims"])
            available = np.setdiff1d(np.arange(width), np.asarray(item["dims"], dtype=np.int64))
            if count > len(available):
                raise ValueError(f"layer {layer}: not enough non-selected dimensions for a disjoint random control")
            result.extend((layer, int(dim)) for dim in rng.choice(available, size=count, replace=False))
        return result
    if mode == "global_random":
        selected_flat = np.asarray([layer * width + dim for layer, dim in selected], dtype=np.int64)
        available = np.setdiff1d(np.arange(layers * width), selected_flat)
        flat = rng.choice(available, size=len(selected), replace=False)
        return [(int(index // width), int(index % width)) for index in flat]
    raise ValueError(f"unknown feature mode: {mode}")


def project(hidden: np.ndarray, coordinates: list[tuple[int, int]]) -> np.ndarray:
    return np.stack([hidden[:, layer, dim] for layer, dim in coordinates], axis=1).astype(np.float32)
