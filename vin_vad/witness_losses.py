from __future__ import annotations

import math

import torch
from torch.nn import functional as F

from vin_vad.witness_router import masked_mean


def topk_bag_probability(score: torch.Tensor, validity: torch.Tensor) -> torch.Tensor:
    outputs = []
    for row, mask in zip(score, validity):
        valid = row[mask]
        if valid.numel() == 0:
            raise ValueError("every video needs at least one valid snippet")
        count = min(valid.numel(), int(valid.numel() / 16 + 1))
        outputs.append(torch.topk(valid, count).values.mean())
    return torch.stack(outputs)


def per_video_mil(score: torch.Tensor, validity: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    bag = topk_bag_probability(score, validity).clamp(1e-6, 1.0 - 1e-6)
    return F.binary_cross_entropy(bag, labels.to(score.dtype), reduction="none")


def ranking_loss(score: torch.Tensor, validity: torch.Tensor, labels: torch.Tensor, margin: float) -> torch.Tensor:
    bag = topk_bag_probability(score, validity)
    normal = bag[labels <= 0.5]
    abnormal = bag[labels > 0.5]
    if normal.numel() == 0 or abnormal.numel() == 0:
        return bag.sum() * 0.0
    return F.softplus(margin - abnormal[:, None] + normal[None, :]).mean()


def temporal_smoothness(score: torch.Tensor, validity: torch.Tensor) -> torch.Tensor:
    pair_mask = validity[:, 1:] & validity[:, :-1]
    difference = (score[:, 1:] - score[:, :-1]).square()
    if not pair_mask.any():
        return score.sum() * 0.0
    return difference[pair_mask].mean()


def temporal_responsibility_sparsity(
    evidence: torch.Tensor, validity: torch.Tensor, labels: torch.Tensor
) -> torch.Tensor:
    """Penalize responsibility support wider than the registered MIL top-k."""
    penalties = []
    for row, mask, label in zip(evidence, validity, labels):
        if label <= 0.5:
            continue
        valid = row[mask].clamp_min(1e-8)
        responsibility = valid / valid.sum()
        entropy = -(responsibility * responsibility.log()).sum()
        count = min(valid.numel(), int(valid.numel() / 16 + 1))
        excess = torch.relu(entropy - math.log(count))
        penalties.append(valid.numel() * excess / math.log(max(2, valid.numel())))
    if not penalties:
        return evidence.sum() * 0.0
    return torch.stack(penalties).mean()


def witness_objective(
    result: dict[str, torch.Tensor],
    host_score: torch.Tensor,
    validity: torch.Tensor,
    labels: torch.Tensor,
    sparsity: torch.Tensor,
    lambda_witness: float = 1.0,
    lambda_final: float = 1.0,
    lambda_normal: float = 0.5,
    lambda_sparse: float = 1e-3,
    rank_weight: float = 0.5,
    rank_margin: float = 0.5,
    smooth_weight: float = 0.02,
) -> dict[str, torch.Tensor]:
    evidence = result["evidence"]
    corrected = result["corrected_score"]
    video_loss = F.binary_cross_entropy(result["video_probability"], labels.to(evidence.dtype))
    residual = (labels - topk_bag_probability(host_score, validity)).abs().detach()
    neuron_per_video = per_video_mil(evidence, validity, labels)
    neuron_loss = (residual * neuron_per_video).sum() / residual.sum().clamp_min(1e-6)
    neuron_loss = neuron_loss + rank_weight * ranking_loss(evidence, validity, labels, rank_margin)
    neuron_loss = neuron_loss + smooth_weight * temporal_smoothness(evidence, validity)
    final_loss = per_video_mil(corrected, validity, labels).mean()
    normal_mask = labels <= 0.5
    if normal_mask.any():
        normal_evidence = -torch.log1p(-evidence.clamp(max=1.0 - 1e-6))
        normal_corrected = -torch.log1p(-corrected.clamp(max=1.0 - 1e-6))
        dense_normal = (
            masked_mean(normal_evidence, validity)[normal_mask]
            + masked_mean(normal_corrected, validity)[normal_mask]
        ).mean()
    else:
        dense_normal = evidence.sum() * 0.0
    temporal_sparse = temporal_responsibility_sparsity(evidence, validity, labels)
    # Sparse coordinates alone do not imply sparse temporal responsibility.
    # Reuse the registered sparsity coefficient for the standard weak-MIL event
    # prior. Entropy makes it invariant to a trivial rescaling of all scores.
    sparse_loss = sparsity + temporal_sparse
    total = (
        video_loss
        + lambda_witness * neuron_loss
        + lambda_final * final_loss
        + lambda_normal * dense_normal
        + lambda_sparse * sparse_loss
    )
    return {
        "total": total,
        "video": video_loss,
        "witness_mil": neuron_loss,
        "final_mil": final_loss,
        "dense_normal": dense_normal,
        "sparse": sparse_loss,
        "host_residual": residual.mean(),
    }


def variant_objective(
    variant: str,
    result: dict[str, torch.Tensor],
    host_score: torch.Tensor,
    validity: torch.Tensor,
    labels: torch.Tensor,
    sparsity: torch.Tensor | None,
    lambda_witness: float = 1.0,
    lambda_final: float = 1.0,
    lambda_normal: float = 0.5,
    lambda_sparse: float = 1e-3,
    rank_weight: float = 0.5,
    rank_margin: float = 0.5,
    smooth_weight: float = 0.02,
) -> dict[str, torch.Tensor]:
    """Apply only the losses belonging to one pre-registered F3 variant."""
    if variant == "w6":
        if sparsity is None:
            raise ValueError("w6 requires a sparsity surrogate")
        return witness_objective(
            result,
            host_score,
            validity,
            labels,
            sparsity,
            lambda_witness=lambda_witness,
            lambda_final=lambda_final,
            lambda_normal=lambda_normal,
            lambda_sparse=lambda_sparse,
            rank_weight=rank_weight,
            rank_margin=rank_margin,
            smooth_weight=smooth_weight,
        )

    corrected = result["corrected_score"]
    zero = corrected.sum() * 0.0
    final_loss = per_video_mil(corrected, validity, labels).mean()
    normal_mask = labels <= 0.5
    if normal_mask.any():
        corrected_normal = -torch.log1p(-corrected.clamp(max=1.0 - 1e-6))
        dense_corrected = masked_mean(corrected_normal, validity)[normal_mask].mean()
    else:
        dense_corrected = zero

    if variant == "w1":
        video_loss = F.binary_cross_entropy(
            result["video_probability"], labels.to(corrected.dtype)
        )
        total = video_loss + lambda_final * final_loss + lambda_normal * dense_corrected
        return {
            "total": total,
            "video": video_loss,
            "witness_mil": zero,
            "final_mil": final_loss,
            "dense_normal": dense_corrected,
            "sparse": zero,
            "host_residual": zero,
        }

    if variant == "w2":
        if sparsity is None:
            raise ValueError("w2 requires a sparsity surrogate")
        evidence = result["evidence"]
        residual = (labels - topk_bag_probability(host_score, validity)).abs().detach()
        neuron_loss = (
            residual * per_video_mil(evidence, validity, labels)
        ).sum() / residual.sum().clamp_min(1e-6)
        neuron_loss = neuron_loss + rank_weight * ranking_loss(
            evidence, validity, labels, rank_margin
        )
        neuron_loss = neuron_loss + smooth_weight * temporal_smoothness(
            evidence, validity
        )
        if normal_mask.any():
            evidence_normal = -torch.log1p(-evidence.clamp(max=1.0 - 1e-6))
            dense_evidence = masked_mean(evidence_normal, validity)[normal_mask].mean()
        else:
            dense_evidence = zero
        dense_normal = dense_corrected + dense_evidence
        temporal_sparse = temporal_responsibility_sparsity(
            evidence, validity, labels
        )
        sparse_loss = sparsity + temporal_sparse
        total = (
            lambda_witness * neuron_loss
            + lambda_final * final_loss
            + lambda_normal * dense_normal
            + lambda_sparse * sparse_loss
        )
        return {
            "total": total,
            "video": zero,
            "witness_mil": neuron_loss,
            "final_mil": final_loss,
            "dense_normal": dense_normal,
            "sparse": sparse_loss,
            "host_residual": residual.mean(),
        }
    raise ValueError("variant must be w1, w2 or w6")
