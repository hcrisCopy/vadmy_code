from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F


NEG_INF = -torch.inf


def inverse_softplus(value: float) -> float:
    return math.log(math.expm1(value))


class EventChain(nn.Module):
    """Exact OR-constrained inference for E1, E2 and E3."""

    def __init__(self, variant: str) -> None:
        super().__init__()
        if variant not in {"e1", "e2", "e3"}:
            raise ValueError(f"event chain variant must be e1/e2/e3, got {variant}")
        self.variant = variant
        if variant == "e3":
            self.onset_raw = nn.Parameter(torch.tensor(inverse_softplus(1.0)))
        else:
            initial_onset = 1.0 / 256.0
            self.onset_raw = nn.Parameter(torch.tensor(math.log(initial_onset / (1.0 - initial_onset))))
        if variant == "e1":
            self.register_parameter("persistence_raw", None)
        else:
            self.persistence_raw = nn.Parameter(torch.tensor(math.log(0.9 / 0.1)))

    def onset_probability(self, lengths: torch.Tensor) -> torch.Tensor:
        lengths = lengths.to(self.onset_raw.dtype).clamp_min(1.0)
        if self.variant == "e3":
            rate = F.softplus(self.onset_raw)
            return -torch.expm1(-rate / lengths)
        return torch.sigmoid(self.onset_raw).expand_as(lengths)

    def persistence_probability(self, lengths: torch.Tensor) -> torch.Tensor:
        if self.variant == "e1":
            return self.onset_probability(lengths)
        return torch.sigmoid(self.persistence_raw).expand_as(lengths)

    def transition_log_probs(self, lengths: torch.Tensor) -> torch.Tensor:
        onset = self.onset_probability(lengths).clamp(1e-7, 1.0 - 1e-7)
        persistence = self.persistence_probability(lengths).clamp(1e-7, 1.0 - 1e-7)
        transitions = torch.stack(
            [
                torch.stack([1.0 - onset, onset], dim=-1),
                torch.stack([1.0 - persistence, persistence], dim=-1),
            ],
            dim=1,
        )
        return transitions.log()

    def log_partitions(self, emissions: torch.Tensor, mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if emissions.ndim != 2 or mask.shape != emissions.shape:
            raise ValueError("emissions and mask must both be [B,T]")
        lengths = mask.sum(dim=1)
        if torch.any(lengths < 1):
            raise ValueError("every video must contain at least one valid snippet")
        batch = emissions.shape[0]
        transitions = self.transition_log_probs(lengths)
        onset = self.onset_probability(lengths).clamp(1e-7, 1.0 - 1e-7)

        alpha = emissions.new_full((batch, 2, 2), NEG_INF)
        alpha[:, 0, 0] = torch.log1p(-onset)
        alpha[:, 1, 1] = onset.log() + emissions[:, 0]
        for time in range(1, emissions.shape[1]):
            updated = emissions.new_full((batch, 2, 2), NEG_INF)
            for current_state in (0, 1):
                unary = emissions[:, time] if current_state else 0.0
                for seen_anomaly in (0, 1):
                    candidates = []
                    for previous_state in (0, 1):
                        for previous_seen in (0, 1):
                            if seen_anomaly == (previous_seen or current_state):
                                candidates.append(
                                    alpha[:, previous_state, previous_seen]
                                    + transitions[:, previous_state, current_state]
                                )
                    if candidates:
                        updated[:, current_state, seen_anomaly] = torch.logsumexp(
                            torch.stack(candidates, dim=-1), dim=-1
                        ) + unary
            active = mask[:, time, None, None]
            alpha = torch.where(active, updated, alpha)
        log_z0 = torch.logsumexp(alpha[:, :, 0], dim=1)
        log_z1 = torch.logsumexp(alpha[:, :, 1], dim=1)
        return log_z0, log_z1

    def video_log_probs(self, emissions: torch.Tensor, mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        log_z0, log_z1 = self.log_partitions(emissions, mask)
        normalizer = torch.logaddexp(log_z0, log_z1)
        return log_z0 - normalizer, log_z1 - normalizer

    def snippet_marginals(self, emissions: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """Return p(z_t=1|H); invalid padded positions are exactly zero."""
        if emissions.ndim != 2 or mask.shape != emissions.shape:
            raise ValueError("emissions and mask must both be [B,T]")
        outputs = torch.zeros_like(emissions)
        lengths = mask.sum(dim=1)
        for batch_index, length_tensor in enumerate(lengths):
            length = int(length_tensor.item())
            values = emissions[batch_index, :length]
            length_arg = length_tensor.reshape(1)
            transition = self.transition_log_probs(length_arg)[0]
            onset = self.onset_probability(length_arg)[0].clamp(1e-7, 1.0 - 1e-7)

            forward = [torch.stack([torch.log1p(-onset), onset.log() + values[0]])]
            for time in range(1, length):
                current = []
                for state in (0, 1):
                    unary = values[time] if state else 0.0
                    current.append(
                        torch.logsumexp(forward[-1] + transition[:, state], dim=0) + unary
                    )
                forward.append(torch.stack(current))
            backward = [values.new_zeros(2) for _ in range(length)]
            for time in range(length - 2, -1, -1):
                current = []
                for state in (0, 1):
                    next_terms = torch.stack(
                        [
                            transition[state, 0] + backward[time + 1][0],
                            transition[state, 1] + values[time + 1] + backward[time + 1][1],
                        ]
                    )
                    current.append(torch.logsumexp(next_terms, dim=0))
                backward[time] = torch.stack(current)
            log_total = torch.logsumexp(forward[-1], dim=0)
            for time in range(length):
                outputs[batch_index, time] = torch.exp(
                    forward[time][1] + backward[time][1] - log_total
                )
        return outputs.masked_fill(~mask, 0.0)
