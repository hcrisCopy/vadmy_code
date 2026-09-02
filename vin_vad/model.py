from __future__ import annotations

import torch
from torch import nn

from vin_vad.base_tcn import BaseTCN
from vin_vad.context_predictor import MaskedContextPredictor, detached_distribution
from vin_vad.event_chain import EventChain
from vin_vad.host_auditor import NormalQCalibrator, TwoAxisHostAuditor, masked_topk_mean
from vin_vad.losses import topk_video_probability
from vin_vad.violation_field import ViolationField


class EventAblationModel(nn.Module):
    def __init__(self, variant: str, width: int = 512, dropout: float = 0.1) -> None:
        super().__init__()
        if variant not in {"e0", "e1", "e2", "e3"}:
            raise ValueError(f"variant must be e0/e1/e2/e3, got {variant}")
        self.variant = variant
        self.tcn = BaseTCN(width=width, dropout=dropout)
        self.chain = None if variant == "e0" else EventChain(variant)

    def forward(self, features: torch.Tensor, mask: torch.Tensor) -> dict[str, torch.Tensor]:
        emissions = self.tcn(features, mask)
        if self.variant == "e0":
            snippet_probability = torch.sigmoid(emissions).masked_fill(~mask, 0.0)
            video_probability = topk_video_probability(emissions, mask)
        else:
            _, log_p1 = self.chain.video_log_probs(emissions, mask)
            video_probability = log_p1.exp()
            snippet_probability = self.chain.snippet_marginals(emissions, mask)
        return {
            "emissions": emissions,
            "video_prob": video_probability,
            "snippet_prob": snippet_probability,
        }


class CVAVADCorrectionModel(nn.Module):
    """B4 joint model around a frozen cached host score.

    Only the normal-context predictor, directional field weights and the two
    projected correction strengths are trainable. Cached host scores are
    inputs, not parameters.
    """

    def __init__(
        self,
        predictor: MaskedContextPredictor,
        field: ViolationField,
        auditor: TwoAxisHostAuditor,
        q_calibrator: NormalQCalibrator,
        evidence_id: str = "c3",
    ) -> None:
        super().__init__()
        if evidence_id not in {"c0", "c1", "c2", "c3", "c4"}:
            raise ValueError("evidence_id must be c0/c1/c2/c3/c4")
        self.predictor = predictor
        self.field = field
        self.auditor = auditor
        self.q_calibrator = q_calibrator
        self.evidence_id = str(evidence_id)

    def evidence_forward(
        self,
        hidden: torch.Tensor,
        validity: torch.Tensor,
        labels: torch.Tensor | None = None,
        update_statistics: bool = False,
        require_context_loss: bool = True,
    ) -> dict[str, object]:
        """Compute one ablation's evidence without applying the host auditor."""
        attention_mode = "global" if self.evidence_id == "c4" else "masked"
        if require_context_loss or self.evidence_id in {"c2", "c3", "c4"}:
            distribution = self.predictor(
                hidden, validity, attention_mode=attention_mode
            )
            normalized = distribution["normalized_hidden"]
            mean, sigma = detached_distribution(distribution)
        else:
            normalized = self.predictor.normalize_hidden(hidden)
            mean = torch.zeros_like(normalized)
            sigma = torch.ones_like(normalized)
            distribution = None
        field = self.field(
            normalized,
            mean,
            sigma,
            validity,
            labels=labels,
            update_statistics=update_statistics,
        )
        return {"distribution": distribution, "field": field}

    def forward(
        self,
        hidden: torch.Tensor,
        host_score: torch.Tensor,
        validity: torch.Tensor,
        labels: torch.Tensor,
        update_statistics: bool,
    ) -> dict[str, torch.Tensor]:
        evidence_result = self.evidence_forward(
            hidden,
            validity,
            labels=labels,
            update_statistics=update_statistics,
            require_context_loss=True,
        )
        distribution = evidence_result["distribution"]
        field = evidence_result["field"]
        evidence_video = masked_topk_mean(field["evidence"], validity)
        if update_statistics:
            self.q_calibrator.update(evidence_video, labels)
            self.auditor.set_normal_q_statistics(
                float(self.q_calibrator.median),
                float(self.q_calibrator.mad),
                float(self.q_calibrator.tau_normal),
            )
        correction = self.auditor(host_score, field["evidence"], validity)
        return {
            "distribution": distribution,
            "field": field,
            "evidence_video": evidence_video,
            **correction,
        }
