"""TRACE: baseline-score-free neuron evidence for reliable snippet supervision."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn

from .losses import binary_topk_mil
from .model import valid_mask


@dataclass
class TraceThresholds:
    semantic_low: float
    semantic_high: float
    temporal_low: float
    temporal_high: float

    def as_dict(self) -> dict[str, float]:
        return {
            "semantic_low": float(self.semantic_low),
            "semantic_high": float(self.semantic_high),
            "temporal_low": float(self.temporal_low),
            "temporal_high": float(self.temporal_high),
        }

    @classmethod
    def from_dict(cls, value: dict[str, float]) -> "TraceThresholds":
        return cls(**{key: float(value[key]) for key in (
            "semantic_low", "semantic_high", "temporal_low", "temporal_high"
        )})


class TraceNeuronEvidence(nn.Module):
    """Semantic and normal-temporal teachers over selected raw CLS dimensions."""

    method_name = "temporally_responsible_anomaly_concept_evidence_v1"

    def __init__(
        self,
        selected_layers: torch.Tensor,
        selected_dimensions: torch.Tensor,
        center: torch.Tensor,
        scale: torch.Tensor,
        innovation_center: torch.Tensor,
        innovation_scale: torch.Tensor,
        hidden_width: int = 128,
        active_neurons: int = 96,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        layers = torch.as_tensor(selected_layers, dtype=torch.long).flatten()
        dimensions = torch.as_tensor(selected_dimensions, dtype=torch.long).flatten()
        statistics = [torch.as_tensor(value, dtype=torch.float32).flatten() for value in (
            center, scale, innovation_center, innovation_scale
        )]
        width = int(layers.numel())
        if not width or dimensions.numel() != width or any(value.numel() != width for value in statistics):
            raise ValueError("TRACE indices and statistics must have one shared positive width")
        if not 0 < active_neurons <= width:
            raise ValueError("active_neurons must be within the selected neuron width")
        self.neuron_width = width
        self.hidden_width = int(hidden_width)
        self.active_neurons = int(active_neurons)
        self.register_buffer("selected_layers", layers)
        self.register_buffer("selected_dimensions", dimensions)
        self.register_buffer("center", statistics[0])
        self.register_buffer("scale", statistics[1].clamp_min(1e-6))
        self.register_buffer("innovation_center", statistics[2])
        self.register_buffer("innovation_scale", statistics[3].clamp_min(1e-6))
        self.neuron_logits = nn.Parameter(torch.zeros(width))
        self.input_norm = nn.LayerNorm(width * 2)
        self.input_projection = nn.Linear(width * 2, self.hidden_width)
        self.temporal_paths = nn.ModuleList([
            nn.Conv1d(
                self.hidden_width, self.hidden_width, kernel_size=3,
                padding=dilation, dilation=dilation, groups=self.hidden_width, bias=False,
            )
            for dilation in (1, 2, 4)
        ])
        self.temporal_mix = nn.Linear(self.hidden_width * 4, self.hidden_width)
        self.semantic_head = nn.Linear(self.hidden_width, 1)
        self.dropout = nn.Dropout(dropout)

    @classmethod
    def from_artifact(
        cls, arrays: dict[str, torch.Tensor], hidden_width: int,
        active_neurons: int, dropout: float,
    ) -> "TraceNeuronEvidence":
        return cls(
            arrays["selected_layers"], arrays["selected_dimensions"],
            arrays["center"], arrays["scale"], arrays["innovation_center"],
            arrays["innovation_scale"], hidden_width, active_neurons, dropout,
        )

    def config(self) -> dict:
        return {
            "method": self.method_name,
            "selected_layers": self.selected_layers.cpu().tolist(),
            "selected_dimensions": self.selected_dimensions.cpu().tolist(),
            "center": self.center.cpu().tolist(), "scale": self.scale.cpu().tolist(),
            "innovation_center": self.innovation_center.cpu().tolist(),
            "innovation_scale": self.innovation_scale.cpu().tolist(),
            "hidden_width": self.hidden_width, "active_neurons": self.active_neurons,
        }

    @classmethod
    def from_config(cls, config: dict, dropout: float = 0.0) -> "TraceNeuronEvidence":
        return cls(
            torch.tensor(config["selected_layers"]),
            torch.tensor(config["selected_dimensions"]),
            torch.tensor(config["center"]), torch.tensor(config["scale"]),
            torch.tensor(config["innovation_center"]),
            torch.tensor(config["innovation_scale"]),
            int(config["hidden_width"]), int(config["active_neurons"]), dropout,
        )

    def neuron_gates(self) -> torch.Tensor:
        soft = torch.sigmoid(self.neuron_logits)
        indices = soft.topk(self.active_neurons).indices
        hard = torch.zeros_like(soft).scatter_(0, indices, 1.0)
        return hard + soft - soft.detach() if self.training else hard

    def selected_hidden(self, hidden: torch.Tensor) -> torch.Tensor:
        if hidden.ndim != 4:
            raise ValueError(f"expected hidden [B,T,L,D], got {tuple(hidden.shape)}")
        if int(self.selected_layers.max()) >= hidden.shape[2] or int(self.selected_dimensions.max()) >= hidden.shape[3]:
            raise ValueError("TRACE neuron index is outside the hidden-state tensor")
        return hidden[:, :, self.selected_layers, self.selected_dimensions].float()

    @staticmethod
    def neighbor_context(value: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        left = torch.cat([value[:, :1], value[:, :-1]], dim=1)
        right = torch.cat([value[:, 1:], value[:, -1:]], dim=1)
        left_mask = torch.cat([mask[:, :1], mask[:, :-1]], dim=1)
        right_mask = torch.cat([mask[:, 1:], mask[:, -1:]], dim=1)
        count = (left_mask + right_mask).clamp_min(1.0).unsqueeze(-1)
        return (left * left_mask.unsqueeze(-1) + right * right_mask.unsqueeze(-1)) / count

    def forward(self, hidden: torch.Tensor, lengths: torch.Tensor) -> dict[str, torch.Tensor]:
        mask = valid_mask(lengths, hidden.shape[1], torch.float32)
        selected = self.selected_hidden(hidden)
        standardized = (selected - self.center) / self.scale
        context = self.neighbor_context(selected, mask)
        innovation = (selected - context).abs()
        innovation_z = (innovation - self.innovation_center) / self.innovation_scale
        temporal_per_neuron = F.softplus(innovation_z).clamp_max(8.0)
        gate = self.neuron_gates()
        gated = gate / gate.sum().clamp_min(1.0)
        temporal_score = (temporal_per_neuron * gated).sum(dim=-1) * mask
        inputs = torch.cat([standardized * gate, standardized.abs() * gate], dim=-1)
        local = F.gelu(self.input_projection(self.input_norm(inputs))) * mask.unsqueeze(-1)
        channels = local.transpose(1, 2)
        paths = [F.gelu(layer(channels)).transpose(1, 2) * mask.unsqueeze(-1)
                 for layer in self.temporal_paths]
        embedding = F.gelu(self.temporal_mix(torch.cat([local, *paths], dim=-1)))
        embedding = self.dropout(embedding) * mask.unsqueeze(-1)
        semantic_logits = self.semantic_head(embedding).squeeze(-1) * mask
        return {
            "semantic_logits": semantic_logits,
            "temporal_score": temporal_score,
            "temporal_per_neuron": temporal_per_neuron,
            "standardized": standardized,
            "embedding": embedding,
            "mask": mask,
            "gates": gate,
        }

    def sparsity_loss(self) -> torch.Tensor:
        soft = torch.sigmoid(self.neuron_logits)
        return (soft * (1.0 - soft)).mean()


def trace_pretraining_losses(
    evidence: TraceNeuronEvidence,
    record: dict[str, torch.Tensor],
    labels: torch.Tensor,
    lengths: torch.Tensor,
) -> dict[str, torch.Tensor]:
    semantic = record["semantic_logits"]
    temporal = record["temporal_score"].detach()
    mask = record["mask"]
    mil = binary_topk_mil(semantic, labels, lengths)
    normal_rows = labels < 0.5
    if bool(normal_rows.any()):
        normal_mask = mask[normal_rows]
        normal = (F.softplus(semantic[normal_rows]) * normal_mask).sum() / normal_mask.sum().clamp_min(1.0)
    else:
        normal = semantic.sum() * 0.0
    temporal_normalized = temporal / temporal.detach().mean().clamp_min(1e-6)
    agreement = binary_topk_mil(semantic + temporal_normalized, labels, lengths)
    pair_mask = mask[:, 1:] * mask[:, :-1]
    smooth = ((semantic[:, 1:] - semantic[:, :-1]).square() * pair_mask).sum() / pair_mask.sum().clamp_min(1.0)
    return {"mil": mil, "normal": normal, "agreement": agreement,
            "smooth": smooth, "sparse": evidence.sparsity_loss()}


def grow_events(seed: torch.Tensor, support: torch.Tensor, mask: torch.Tensor, steps: int = 8) -> torch.Tensor:
    current = seed.bool() & mask.bool()
    support = support.bool() & mask.bool()
    for _ in range(max(0, int(steps))):
        left = F.pad(current[:, :-1], (1, 0), value=False)
        right = F.pad(current[:, 1:], (0, 1), value=False)
        updated = current | ((left | right) & support)
        if torch.equal(updated, current):
            break
        current = updated
    return current


def responsibility_sets(
    record: dict[str, torch.Tensor], labels: torch.Tensor,
    thresholds: TraceThresholds, grow_steps: int = 8,
) -> dict[str, torch.Tensor]:
    semantic_probability = torch.sigmoid(record["semantic_logits"])
    temporal = record["temporal_score"]
    mask = record["mask"].bool()
    abnormal = (labels >= 0.5).unsqueeze(1)
    normal = ~abnormal
    positive_seed = abnormal & (semantic_probability >= thresholds.semantic_high) & (
        temporal >= thresholds.temporal_high) & mask
    support = (semantic_probability >= thresholds.semantic_low) | (temporal >= thresholds.temporal_low)
    positive = grow_events(positive_seed, support, mask, grow_steps)
    reliable_negative = (semantic_probability <= thresholds.semantic_low) & (
        temporal <= thresholds.temporal_low) & mask
    risk = torch.maximum(
        semantic_probability / max(thresholds.semantic_high, 1e-6),
        temporal / max(thresholds.temporal_high, 1e-6),
    )
    return {
        "positive_seed": positive_seed.float(), "positive": positive.float(),
        "reliable_negative": reliable_negative.float(), "normal_rows": normal.float(),
        "risk": risk * mask, "mask": mask.float(),
        "semantic_probability": semantic_probability * mask, "temporal_score": temporal * mask,
    }


def _bounded(values: torch.Tensor, maximum: int) -> torch.Tensor:
    if values.numel() <= maximum:
        return values
    indices = torch.linspace(0, values.numel() - 1, maximum, device=values.device).long()
    return values.index_select(0, indices)


def smooth_ap_loss(positive: torch.Tensor, negative: torch.Tensor, temperature: float = 0.1) -> torch.Tensor:
    positive = _bounded(positive.flatten(), 128)
    negative = _bounded(negative.flatten(), 512)
    if not positive.numel() or not negative.numel():
        return (positive.sum() + negative.sum()) * 0.0
    all_scores = torch.cat([positive, negative])
    difference_all = (all_scores.unsqueeze(0) - positive.unsqueeze(1)) / temperature
    rank_all = 1.0 + torch.sigmoid(difference_all).sum(dim=1) - 0.5
    difference_pos = (positive.unsqueeze(0) - positive.unsqueeze(1)) / temperature
    rank_pos = 1.0 + torch.sigmoid(difference_pos).sum(dim=1) - 0.5
    return 1.0 - (rank_pos / rank_all.clamp_min(1.0)).mean()


def trace_student_losses(
    binary_logits: torch.Tensor,
    semantic_logits: torch.Tensor,
    sets: dict[str, torch.Tensor],
    labels: torch.Tensor,
    class_targets: torch.Tensor,
    lengths: torch.Tensor,
    hard_normal_fraction: float = 0.05,
    ranking_margin: float = 0.5,
) -> dict[str, torch.Tensor]:
    mask = valid_mask(lengths, binary_logits.shape[1], binary_logits.dtype)
    positive_mask = sets["positive"].bool()
    negative_mask = sets["reliable_negative"].bool()
    normal_rows = labels < 0.5
    hard_normal_mask = torch.zeros_like(positive_mask)
    for row in torch.where(normal_rows)[0].tolist():
        length = max(1, int(lengths[row].item()))
        count = max(1, int(round(length * hard_normal_fraction)))
        indices = sets["risk"][row, :length].topk(min(count, length)).indices
        hard_normal_mask[row, indices] = True
    supervised_mask = (positive_mask | negative_mask | hard_normal_mask) & mask.bool()
    targets = positive_mask.to(binary_logits.dtype)
    if bool(supervised_mask.any()):
        weights = torch.ones_like(binary_logits)
        weights[positive_mask] = 2.0
        binary = (F.binary_cross_entropy_with_logits(binary_logits, targets, reduction="none")
                  * weights * supervised_mask).sum() / (weights * supervised_mask).sum().clamp_min(1.0)
    else:
        binary = binary_logits.sum() * 0.0
    positive_scores = binary_logits[positive_mask]
    hard_scores = binary_logits[hard_normal_mask]
    if positive_scores.numel() and hard_scores.numel():
        pos = _bounded(positive_scores, 128)
        neg = _bounded(hard_scores, 256)
        ranking = F.softplus(ranking_margin - pos.unsqueeze(1) + neg.unsqueeze(0)).mean()
        ap = smooth_ap_loss(pos, neg)
    else:
        ranking = binary_logits.sum() * 0.0
        ap = binary_logits.sum() * 0.0
    semantic_masks, semantic_targets = [], []
    for row in range(binary_logits.shape[0]):
        if labels[row] >= 0.5:
            indices = torch.where(positive_mask[row])[0]
            target_index = int(class_targets[row].argmax().item())
        else:
            indices = torch.where(hard_normal_mask[row])[0]
            target_index = 0
        if indices.numel():
            indices = _bounded(indices, 32)
            semantic_masks.append(semantic_logits[row].index_select(0, indices))
            semantic_targets.append(torch.full((indices.numel(),), target_index, device=binary_logits.device, dtype=torch.long))
    if semantic_masks:
        semantic = F.cross_entropy(torch.cat(semantic_masks), torch.cat(semantic_targets))
    else:
        semantic = semantic_logits.sum() * 0.0
    pair_mask = mask[:, 1:] * mask[:, :-1]
    probability = torch.sigmoid(binary_logits)
    event = ((probability[:, 1:] - probability[:, :-1]).square() * pair_mask).sum() / pair_mask.sum().clamp_min(1.0)
    return {"pseudo_binary": binary, "ranking": ranking, "ap": ap,
            "semantic": semantic, "event": event,
            "positive_count": positive_mask.sum().detach(),
            "seed_count": sets["positive_seed"].sum().detach(),
            "hard_normal_count": hard_normal_mask.sum().detach()}

