from __future__ import annotations

import csv
import json
import re
import shutil
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


CHUNK_RE = re.compile(r"__(\d+)$")


def ensure_dir(path: str | Path) -> Path:
    result = Path(path)
    result.mkdir(parents=True, exist_ok=True)
    return result


def clean_output(path: str | Path, clean: bool) -> Path:
    result = Path(path)
    if clean and result.exists():
        resolved = result.resolve()
        if len(resolved.parts) < 3:
            raise ValueError(f"refusing to clean broad path: {resolved}")
        shutil.rmtree(resolved)
    return ensure_dir(result)


def feature_key(path_or_key: str) -> str:
    return Path(str(path_or_key)).stem


def base_key(path_or_key: str) -> str:
    return CHUNK_RE.sub("", feature_key(path_or_key))


def chunk_index(path_or_key: str) -> int:
    match = CHUNK_RE.search(feature_key(path_or_key))
    return int(match.group(1)) if match else 0


def is_normal_label(dataset: str, label: str) -> bool:
    if dataset == "ucf":
        return str(label).lower() == "normal"
    if dataset == "xd":
        return str(label).split("-")[0].upper() == "A"
    raise ValueError(f"unknown dataset: {dataset}")


def read_feature_csv(path: str | Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    missing = {"path", "label"} - set(frame.columns)
    if missing:
        raise ValueError(f"{path} is missing columns: {sorted(missing)}")
    return frame


def grouped_rows(frame: pd.DataFrame) -> dict[str, pd.DataFrame]:
    work = frame.copy()
    work["base_key"] = work["path"].map(base_key)
    work["chunk_index"] = work["path"].map(chunk_index)
    return {
        str(key): group.sort_values("chunk_index")
        for key, group in work.groupby("base_key", sort=True)
    }


def read_hidden_manifest(path: str | Path) -> tuple[dict[str, str], str]:
    frame = pd.read_csv(path)
    missing = {"key", "hidden_path"} - set(frame.columns)
    if missing:
        raise ValueError(f"{path} is missing columns: {sorted(missing)}")
    pools = set(frame.get("token_pool", pd.Series(["cls"])).astype(str))
    if pools != {"cls"}:
        raise ValueError(f"{path}: neuron method requires CLS hidden states, got {sorted(pools)}")
    mapping = {str(row["key"]): str(row["hidden_path"]) for _, row in frame.iterrows()}
    return mapping, "cls"


def load_hidden(path: str | Path) -> tuple[np.ndarray, dict]:
    loaded = np.load(path, allow_pickle=True)
    if isinstance(loaded, np.ndarray):
        hidden = loaded
        metadata: dict = {}
    else:
        if "hidden" not in loaded.files:
            raise ValueError(f"{path}: NPZ must contain 'hidden'")
        hidden = loaded["hidden"]
        metadata = {
            key: loaded[key].item() if loaded[key].shape == () else loaded[key]
            for key in loaded.files
            if key != "hidden"
        }
    hidden = np.asarray(hidden, dtype=np.float32)
    if hidden.ndim != 3 or hidden.shape[0] == 0:
        raise ValueError(f"{path}: expected non-empty [T,L,D], got {hidden.shape}")
    return hidden, metadata


def uniform_indices(length: int, count: int) -> np.ndarray:
    count = max(1, min(int(length), int(count)))
    return np.linspace(0, length - 1, count, dtype=np.int64)


def resample_feature(feature: np.ndarray, target_length: int) -> np.ndarray:
    feature = np.asarray(feature, dtype=np.float32)
    if feature.ndim == 1:
        feature = feature[:, None]
    if feature.shape[0] == target_length:
        return feature
    if feature.shape[0] == 0:
        return np.zeros((target_length, feature.shape[1]), dtype=np.float32)
    old_axis = np.linspace(0.0, 1.0, feature.shape[0], dtype=np.float32)
    new_axis = np.linspace(0.0, 1.0, target_length, dtype=np.float32)
    columns = [np.interp(new_axis, old_axis, feature[:, index]) for index in range(feature.shape[1])]
    return np.stack(columns, axis=1).astype(np.float32)


def save_json(path: str | Path, value: dict) -> None:
    with Path(path).open("w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, ensure_ascii=False)


def load_json(path: str | Path) -> dict:
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_csv(path: str | Path, header: Iterable[str], rows: Iterable[Iterable[object]]) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(list(header))
        writer.writerows(rows)


def portable_path(path: str | Path, anchor: str | Path) -> str:
    """Store a path relative to an artifact directory whenever possible."""
    path_obj = Path(path).resolve()
    anchor_obj = Path(anchor).resolve()
    try:
        return str(path_obj.relative_to(anchor_obj))
    except ValueError:
        return str(path_obj)


def resolve_artifact_path(value: str, artifact_file: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else Path(artifact_file).resolve().parent / path
