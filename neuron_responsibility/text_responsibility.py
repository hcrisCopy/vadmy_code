"""CLIP text-grounded neuron responsibility without baseline anomaly scores."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from .baselines import label_map_for


TEMPLATES = (
    "a surveillance video of {}",
    "a video showing {}",
    "a scene of {}",
)
NORMAL_CONCEPTS = ("normal daily activity", "a safe ordinary event", "nothing abnormal")


def abnormal_concepts(dataset: str) -> list[str]:
    values = list(label_map_for(dataset).values())
    normal = "Normal" if dataset == "ucf" else "normal"
    return ["an abnormal dangerous event"] + [value for value in values if value != normal]


def _state_dict(path: str) -> dict[str, torch.Tensor]:
    value = torch.load(path, map_location="cpu")
    if isinstance(value, dict):
        for key in ("model_state_dict", "state_dict"):
            if key in value:
                value = value[key]
                break
    if not isinstance(value, dict):
        raise TypeError(f"{path}: expected a state dictionary")
    if value and all(str(key).startswith("module.") for key in value):
        value = {str(key)[7:]: tensor for key, tensor in value.items()}
    return value


def load_reference_clip(source_root: str, weight: str, device: torch.device):
    """Rebuild only the frozen CLIP carried by a released checkpoint."""
    source = str(Path(source_root) / "src")
    sys.path.insert(0, source)
    try:
        from clip.clip import tokenize
        from clip.model import build_model
    finally:
        sys.path.pop(0)
    state = _state_dict(weight)
    prefixes = ("clipmodel.", "base.clipmodel.", "model.clipmodel.")
    clip_state = {}
    for prefix in prefixes:
        candidate = {
            str(key)[len(prefix):]: tensor
            for key, tensor in state.items()
            if str(key).startswith(prefix) and "prompt_learner" not in str(key)
        }
        if "visual.proj" in candidate:
            clip_state = candidate
            break
    if not clip_state:
        raise KeyError("checkpoint does not contain a reusable clipmodel.* state")
    model = build_model(clip_state).float().to(device).eval()
    model.requires_grad_(False)
    return model, tokenize


def encode_concepts(model, tokenize, concepts: list[str] | tuple[str, ...], device) -> torch.Tensor:
    groups = []
    with torch.no_grad():
        for concept in concepts:
            tokens = tokenize([template.format(concept) for template in TEMPLATES]).to(device)
            embedded = model.token_embedding(tokens)
            encoded = model.encode_text(embedded, tokens)
            groups.append(F.normalize(encoded.float(), dim=-1).mean(dim=0))
    return F.normalize(torch.stack(groups), dim=-1)


class TextMargin:
    def __init__(self, model, normal_text: torch.Tensor, abnormal_text: torch.Tensor) -> None:
        self.ln_post = model.visual.ln_post
        self.projection = model.visual.proj
        self.normal_text = normal_text
        self.abnormal_text = abnormal_text

    def image(self, hidden: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.ln_post(hidden.float()) @ self.projection.float(), dim=-1)

    def margin(self, hidden: torch.Tensor) -> torch.Tensor:
        image = self.image(hidden)
        abnormal = image @ self.abnormal_text.T
        normal = image @ self.normal_text.T
        return torch.logsumexp(abnormal, dim=-1) - np.log(abnormal.shape[-1]) - normal.mean(-1)

    def responsibility(self, hidden: torch.Tensor, center: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        work = hidden.detach().float().requires_grad_(True)
        margin = self.margin(work)
        gradient = torch.autograd.grad(margin.sum(), work)[0]
        return margin.detach(), ((work - center) * gradient).detach()


def robust_location_scale(values: np.ndarray) -> tuple[float, float]:
    center = float(np.median(values))
    scale = float(np.median(np.abs(values - center)) * 1.4826)
    return center, max(scale, 1e-6)


def evidence_from_responsibility(
    responsibility: torch.Tensor,
    hidden: torch.Tensor,
    center: torch.Tensor,
    scale: torch.Tensor,
    indices: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    semantic = responsibility.index_select(1, indices).sum(dim=1)
    z = (hidden.float() - center) / scale
    structural = z.index_select(1, indices).square().mean(dim=1).sqrt()
    return semantic, structural


def calibrated_prior(
    semantic: np.ndarray,
    structural: np.ndarray,
    calibration: dict[str, float],
) -> np.ndarray:
    semantic_z = (semantic - calibration["semantic_center"]) / calibration["semantic_scale"]
    structural_z = (structural - calibration["structural_center"]) / calibration["structural_scale"]
    logit = np.clip(0.5 * semantic_z + 0.5 * structural_z, -12.0, 12.0)
    return (1.0 / (1.0 + np.exp(-logit))).astype(np.float32)
