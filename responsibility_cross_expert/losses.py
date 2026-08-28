from __future__ import annotations

import torch
import torch.nn.functional as F


def binary_topk_mil(logits: torch.Tensor, labels: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
    probability = torch.sigmoid(logits)
    bag_values = []
    for row, length in zip(probability, lengths):
        valid_length = max(1, int(length.item()))
        count = min(valid_length, valid_length // 16 + 1)
        bag_values.append(row[:valid_length].topk(count).values.mean())
    return F.binary_cross_entropy(torch.stack(bag_values), labels.float())


def multiclass_topk_mil(
    logits: torch.Tensor,
    targets: torch.Tensor,
    lengths: torch.Tensor,
) -> torch.Tensor:
    bags = []
    for row, length in zip(logits, lengths):
        valid_length = max(1, int(length.item()))
        count = min(valid_length, valid_length // 16 + 1)
        bags.append(row[:valid_length].topk(count, dim=0).values.mean(dim=0))
    return -(targets * F.log_softmax(torch.stack(bags), dim=1)).sum(dim=1).mean()


def single_top_multiclass_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    lengths: torch.Tensor,
    normal_target: bool = False,
    epsilon: float = 0.1,
) -> torch.Tensor:
    classes = logits.shape[-1]
    if normal_target:
        target = torch.full_like(targets, 0.01)
        target[:, 0] = 1.0
    else:
        target = targets
    target = target / target.sum(dim=1, keepdim=True).clamp_min(1e-6)
    target = (1.0 - epsilon) * target + epsilon / classes
    bags = []
    for row, length in zip(logits, lengths):
        valid_length = max(1, int(length.item()))
        bags.append(row[:valid_length].topk(1, dim=0).values.mean(dim=0))
    return -(target * F.log_softmax(torch.stack(bags), dim=1)).sum(dim=1).mean()


def dsanet_consistency_loss(
    logits: torch.Tensor,
    original: torch.Tensor,
    reconstructed: torch.Tensor,
    lengths: torch.Tensor,
) -> torch.Tensor:
    reconstruction_score = (1.0 - F.cosine_similarity(original, reconstructed, dim=-1)) / 2.0
    classifier_score = torch.sigmoid(logits)
    mask = torch.arange(logits.shape[1], device=logits.device).unsqueeze(0) < lengths.unsqueeze(1)
    return F.mse_loss(classifier_score[mask], reconstruction_score[mask])


def text_separation_loss(text_features: torch.Tensor) -> torch.Tensor:
    normalized = F.normalize(text_features, dim=-1)
    if normalized.shape[0] <= 1:
        return normalized.sum() * 0.0
    return (normalized[1:] @ normalized[0]).abs().mean()
