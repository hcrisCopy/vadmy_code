"""DeSC author-release temporal inference protocols.

The implementation is adapted from ``ucf_test_tta.py`` and ``xd_test_tta.py``
without importing or modifying the protected baseline source tree.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F

from .baseline_adapters import BaselineAdapter


def _as_sequence(value: np.ndarray | torch.Tensor, device: torch.device) -> torch.Tensor:
    tensor = torch.as_tensor(value, dtype=torch.float32, device=device)
    if tensor.ndim != 2 or tensor.shape[0] <= 0:
        raise ValueError(f"expected a non-empty [time, width] sequence, got {tuple(tensor.shape)}")
    return tensor


def _run(
    adapter: BaselineAdapter,
    clip: torch.Tensor,
    lengths: torch.Tensor,
    neurons: torch.Tensor | None,
) -> tuple[torch.Tensor, torch.Tensor]:
    if neurons is None:
        output = adapter.forward_baseline(clip, lengths)
    else:
        output, _ = adapter.forward_conditioned(clip, neurons, lengths)
    return torch.sigmoid(output.binary_logits), F.softmax(output.semantic_logits, dim=-1)


def _ucf_window_starts(total: int, window: int) -> list[int]:
    if total <= window:
        return [0]
    starts = list(range(0, total - window, window))
    if not starts or starts[-1] != total - window:
        starts.append(total - window)
    return starts


def _pad_window(sequence: torch.Tensor, start: int, window: int) -> tuple[torch.Tensor, int]:
    value = sequence[start : start + window]
    length = int(value.shape[0])
    if length < window:
        value = F.pad(value, (0, 0, 0, window - length))
    return value, length


def desc_official_probabilities(
    adapter: BaselineAdapter,
    clip_value: np.ndarray | torch.Tensor,
    device: torch.device,
    neuron_value: np.ndarray | torch.Tensor | None = None,
) -> dict[str, torch.Tensor]:
    """Return per-snippet DeSC binary and semantic ensemble probabilities.

    UCF follows the released overlapping 256-snippet TTA. XD follows the
    released whole-video interpolation to 256 snippets and interpolation back.
    Returned tensors are on CPU and have their original temporal length.
    """
    if getattr(adapter, "dataset", None) not in {"ucf", "xd"}:
        raise ValueError("DeSC inference requires adapter.dataset in {'ucf', 'xd'}")
    clip = _as_sequence(clip_value, device)
    neurons = None if neuron_value is None else _as_sequence(neuron_value, device)
    if neurons is not None and neurons.shape[0] != clip.shape[0]:
        raise ValueError(
            f"CLIP/neuron temporal mismatch: {clip.shape[0]} versus {neurons.shape[0]}"
        )
    total = int(clip.shape[0])
    window = int(adapter.visual_length)

    if adapter.dataset == "ucf":
        binary_sum = torch.zeros(total, device=device)
        semantic_sum: torch.Tensor | None = None
        counts = torch.zeros(total, device=device)
        for start in _ucf_window_starts(total, window):
            clip_window, length = _pad_window(clip, start, window)
            neuron_window = None
            if neurons is not None:
                neuron_window, neuron_length = _pad_window(neurons, start, window)
                if neuron_length != length:
                    raise RuntimeError("CLIP/neuron window lengths differ")
            binary, semantic = _run(
                adapter,
                clip_window.unsqueeze(0),
                torch.tensor([length], device=device),
                None if neuron_window is None else neuron_window.unsqueeze(0),
            )
            stop = start + length
            binary_sum[start:stop] += binary[0, :length]
            if semantic_sum is None:
                semantic_sum = torch.zeros(total, semantic.shape[-1], device=device)
            semantic_sum[start:stop] += semantic[0, :length]
            counts[start:stop] += 1
        if semantic_sum is None or torch.any(counts == 0):
            raise RuntimeError("DeSC UCF sliding inference left uncovered snippets")
        return {
            "binary": (binary_sum / counts).cpu(),
            "semantic": (semantic_sum / counts.unsqueeze(-1)).cpu(),
        }

    clip_aligned = F.interpolate(
        clip.transpose(0, 1).unsqueeze(0), size=window, mode="linear", align_corners=True
    ).transpose(1, 2)
    neuron_aligned = None
    if neurons is not None:
        neuron_aligned = F.interpolate(
            neurons.transpose(0, 1).unsqueeze(0), size=window,
            mode="linear", align_corners=True,
        ).transpose(1, 2)
    binary, semantic = _run(
        adapter,
        clip_aligned,
        torch.tensor([window], device=device),
        neuron_aligned,
    )
    binary_original = F.interpolate(
        binary.unsqueeze(1), size=total, mode="linear", align_corners=True
    ).squeeze(0).squeeze(0)
    semantic_original = F.interpolate(
        semantic.transpose(1, 2), size=total, mode="linear", align_corners=True
    ).transpose(1, 2).squeeze(0)
    return {"binary": binary_original.cpu(), "semantic": semantic_original.cpu()}


def desc_primary_anomaly_probability(
    probabilities: dict[str, torch.Tensor], dataset: str
) -> torch.Tensor:
    """Match the score reported by the released DeSC dataset evaluator."""
    if dataset == "ucf":
        return probabilities["binary"]
    if dataset == "xd":
        return 1.0 - probabilities["semantic"][:, 0]
    raise ValueError(f"unsupported DeSC dataset: {dataset}")


