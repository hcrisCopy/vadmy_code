"""Released-baseline frame-level evaluation shared by training and testing."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.metrics import average_precision_score, roc_auc_score
from tqdm import tqdm

from .common import base_key


def pad_chunks(array: np.ndarray, chunk_length: int) -> tuple[torch.Tensor, torch.Tensor]:
    chunks, lengths = [], []
    for start in range(0, max(1, len(array)), chunk_length):
        part = array[start:start + chunk_length]
        if not len(part):
            continue
        lengths.append(len(part))
        if len(part) < chunk_length:
            part = np.pad(part, ((0, chunk_length - len(part)), (0, 0)), mode="constant")
        chunks.append(part)
    return (
        torch.from_numpy(np.stack(chunks).astype(np.float32)),
        torch.tensor(lengths, dtype=torch.long),
    )


def official_frame_metrics(
    adapter,
    test_list: str,
    gt_path: str,
    frames_per_snippet: int,
    device: torch.device,
    score_cache: Path | None = None,
) -> dict[str, float]:
    adapter.eval()
    frame = pd.read_csv(test_list)
    if not {"path", "label"}.issubset(frame.columns):
        raise ValueError(f"{test_list}: expected path/label columns")
    frame["key"] = frame["path"].map(base_key)
    scores = []
    if score_cache is not None:
        score_cache.mkdir(parents=True, exist_ok=True)
    with torch.no_grad():
        iterator = frame.groupby("key", sort=False)
        for key, group in tqdm(iterator, desc="official frame evaluation", unit="video", leave=False):
            cache_path = score_cache / f"{key}.npy" if score_cache is not None else None
            if cache_path is not None and cache_path.exists():
                scores.append(torch.from_numpy(np.load(cache_path).astype(np.float32)))
                continue
            clip = np.concatenate([np.load(str(path)).astype(np.float32) for path in group["path"]])
            if adapter.__class__.__name__ == "DeSCAdapter" and adapter.dataset == "xd":
                # DeSC's released XD evaluator resizes each full video to 256
                # snippets and linearly restores the score to the original T.
                original_length = len(clip)
                resized = F.interpolate(
                    torch.from_numpy(clip).T.unsqueeze(0),
                    size=adapter.visual_length,
                    mode="linear",
                    align_corners=True,
                ).transpose(1, 2)
                lengths = torch.tensor([adapter.visual_length], dtype=torch.long)
                output = adapter.forward_baseline(resized.to(device), lengths.to(device))
                video_scores = F.interpolate(
                    torch.sigmoid(output.binary_logits)[None],
                    size=original_length,
                    mode="linear",
                    align_corners=True,
                )[0, 0].cpu()
            else:
                # DSANet and LaGoVAD release full-length evaluators that split
                # long videos into max-length chunks. DeSC/UCF uses stride 256,
                # equal to its window length, which is the same partition.
                chunks, lengths = pad_chunks(clip, adapter.visual_length)
                output = adapter.forward_baseline(chunks.to(device), lengths.to(device))
                video_scores = torch.cat([
                    torch.sigmoid(output.binary_logits[index, :length]).cpu()
                    for index, length in enumerate(lengths.tolist())
                ])
            if cache_path is not None:
                np.save(cache_path, video_scores.numpy())
            scores.append(video_scores)
    prediction = np.repeat(torch.cat(scores).numpy(), frames_per_snippet)
    truth = np.load(gt_path).astype(np.int64).reshape(-1)
    usable = min(len(truth), len(prediction))
    return {
        "frame_auc": float(roc_auc_score(truth[:usable], prediction[:usable])),
        "frame_ap": float(average_precision_score(truth[:usable], prediction[:usable])),
        "frames": int(usable),
    }
