from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path
from types import MethodType

import torch
import torch.nn.functional as F
from torch import nn

from .losses import (
    binary_topk_mil,
    dsanet_consistency_loss,
    multiclass_topk_mil,
    single_top_multiclass_loss,
    text_separation_loss,
)


@dataclass
class BaselineOutput:
    binary_logits: torch.Tensor
    semantic_logits: torch.Tensor
    features: torch.Tensor
    raw: object


def label_map_for(dataset: str) -> dict[str, str]:
    if dataset == "ucf":
        return {
            "Normal": "Normal", "Abuse": "Abuse", "Arrest": "Arrest", "Arson": "Arson",
            "Assault": "Assault", "Burglary": "Burglary", "Explosion": "Explosion",
            "Fighting": "Fighting", "RoadAccidents": "RoadAccidents", "Robbery": "Robbery",
            "Shooting": "Shooting", "Shoplifting": "Shoplifting", "Stealing": "Stealing",
            "Vandalism": "Vandalism",
        }
    return {
        "A": "normal", "B1": "fighting", "B2": "shooting", "B4": "riot",
        "B5": "abuse", "B6": "car accident", "G": "explosion",
    }


def class_targets(label_texts: list[str], label_map: dict[str, str], device: torch.device) -> torch.Tensor:
    prompt = list(label_map.values())
    targets = torch.zeros(len(label_texts), len(prompt), device=device)
    for row, label_text in enumerate(label_texts):
        labels = [label_text] if len(label_map) != 7 else label_text.split("-")
        for label in labels:
            if label in label_map:
                targets[row, prompt.index(label_map[label])] = 1.0
    return targets


def _load_state(path: str) -> dict:
    checkpoint = torch.load(path, map_location="cpu")
    if isinstance(checkpoint, dict):
        for key in ("model_state_dict", "state_dict"):
            if key in checkpoint:
                checkpoint = checkpoint[key]
                break
    if not isinstance(checkpoint, dict):
        raise TypeError(f"{path}: checkpoint does not contain a state dict")
    if checkpoint and all(str(key).startswith("module.") for key in checkpoint):
        checkpoint = {str(key)[7:]: value for key, value in checkpoint.items()}
    return checkpoint


def _apply_feature_modulation(
    owner: "BaselineAdapter",
    features: torch.Tensor,
    stream_name: str,
) -> torch.Tensor:
    if owner.feature_modulator is None or owner._current_neurons is None:
        return features
    modulated, record = owner.feature_modulator(
        features,
        owner._current_neurons,
        owner._current_lengths,
    )
    record["stream_name"] = stream_name
    owner._modulation_records.append(record)
    return modulated


def _apply_pre_temporal_conditioning(
    owner: "BaselineAdapter",
    features: torch.Tensor,
    stream_name: str,
) -> torch.Tensor:
    if owner.pre_temporal_conditioner is None or owner._current_neurons is None:
        return features
    conditioned, record = owner.pre_temporal_conditioner(
        features,
        owner._current_neurons,
        owner._current_lengths,
    )
    record["stream_name"] = stream_name
    owner._conditioning_records.append(record)
    return conditioned


def _capture_encode_video(module: nn.Module, owner: "BaselineAdapter", stream_name: str) -> None:
    original = module.encode_video

    def wrapped(self, *args, **kwargs):
        arguments = list(args)
        if arguments:
            arguments[0] = _apply_pre_temporal_conditioning(
                owner, arguments[0], stream_name
            )
        elif "images" in kwargs:
            kwargs["images"] = _apply_pre_temporal_conditioning(
                owner, kwargs["images"], stream_name
            )
        value = original(*arguments, **kwargs)
        value = _apply_feature_modulation(owner, value, stream_name)
        self._responsibility_features = value
        return value

    module.encode_video = MethodType(wrapped, module)


def _capture_lagovad_temporal(module: nn.Module, owner: "BaselineAdapter") -> None:
    original = module._temporal_encoding

    def wrapped(self, *args, **kwargs):
        arguments = list(args)
        if arguments:
            arguments[0] = _apply_pre_temporal_conditioning(owner, arguments[0], "main")
        elif "v_feat" in kwargs:
            kwargs["v_feat"] = _apply_pre_temporal_conditioning(
                owner, kwargs["v_feat"], "main"
            )
        features, _ = original(*arguments, **kwargs)
        features = _apply_feature_modulation(owner, features, "main")
        normalized = F.normalize(features, dim=-1)
        return features, normalized

    module._temporal_encoding = MethodType(wrapped, module)


def _batch_mask(lengths: torch.Tensor, steps: int) -> torch.Tensor:
    positions = torch.arange(steps, device=lengths.device).unsqueeze(0)
    return positions >= lengths.unsqueeze(1)


def _unfreeze_last(container: nn.Module) -> None:
    children = list(container.children())
    if not children:
        container.requires_grad_(True)
        return
    children[-1].requires_grad_(True)


class BaselineAdapter(nn.Module):
    visual_length: int

    def __init__(self) -> None:
        super().__init__()
        self.feature_modulator: nn.Module | None = None
        self.pre_temporal_conditioner: nn.Module | None = None
        self._current_neurons: torch.Tensor | None = None
        self._current_lengths: torch.Tensor | None = None
        self._modulation_records: list[dict[str, torch.Tensor | str]] = []
        self._conditioning_records: list[dict[str, torch.Tensor | str]] = []

    def attach_feature_modulator(self, modulator: nn.Module) -> None:
        if self.feature_modulator is not None:
            raise RuntimeError("a feature modulator is already attached")
        self.feature_modulator = modulator

    def attach_pre_temporal_conditioner(self, conditioner: nn.Module) -> None:
        if self.pre_temporal_conditioner is not None:
            raise RuntimeError("a pre-temporal conditioner is already attached")
        self.pre_temporal_conditioner = conditioner

    def forward_conditioned(
        self,
        clip: torch.Tensor,
        neurons: torch.Tensor,
        lengths: torch.Tensor,
    ) -> tuple[BaselineOutput, list[dict[str, torch.Tensor | str]]]:
        if self.pre_temporal_conditioner is None:
            raise RuntimeError(
                "attach a pre-temporal conditioner before calling forward_conditioned"
            )
        self._current_neurons = neurons
        self._current_lengths = lengths
        self._conditioning_records = []
        try:
            output = self.forward_baseline(clip, lengths)
            records = list(self._conditioning_records)
        finally:
            self._current_neurons = None
            self._current_lengths = None
            self._conditioning_records = []
        if not records:
            raise RuntimeError("baseline forward did not reach the pre-temporal feature hook")
        return output, records

    def forward_modulated(
        self,
        clip: torch.Tensor,
        neurons: torch.Tensor,
        lengths: torch.Tensor,
    ) -> tuple[BaselineOutput, list[dict[str, torch.Tensor | str]]]:
        if self.feature_modulator is None:
            raise RuntimeError("attach a feature modulator before calling forward_modulated")
        self._current_neurons = neurons
        self._current_lengths = lengths
        self._modulation_records = []
        try:
            output = self.forward_baseline(clip, lengths)
            records = list(self._modulation_records)
        finally:
            self._current_neurons = None
            self._current_lengths = None
            self._modulation_records = []
        if not records:
            raise RuntimeError("baseline forward did not reach the shared post-temporal feature hook")
        return output, records

    def forward_baseline(self, clip: torch.Tensor, lengths: torch.Tensor) -> BaselineOutput:
        raise NotImplementedError

    def original_loss(self, output: BaselineOutput, labels: torch.Tensor, texts: list[str], lengths: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def set_train_scope(self, scope: str) -> None:
        raise NotImplementedError


class DSANetAdapter(BaselineAdapter):
    def __init__(self, root: str, dataset: str, weight: str, device: str) -> None:
        super().__init__()
        source = str(Path(root) / "src")
        sys.path.insert(0, source)
        try:
            from model import DSANet
            module_name = "ucf_option" if dataset == "ucf" else "xd_option"
            options_module = __import__(module_name)
            self.options = options_module.parser.parse_args([])
        finally:
            sys.path.pop(0)
        self.dataset = dataset
        self.label_map = label_map_for(dataset)
        self.prompt = list(self.label_map.values())
        self.visual_length = int(self.options.visual_length)
        self.base = DSANet(
            self.options.classes_num, self.options.embed_dim, self.visual_length,
            self.options.visual_width, self.options.visual_head, self.options.visual_layers,
            self.options.attn_window, self.options.prompt_prefix, self.options.prompt_postfix,
            self.options, device,
        )
        self.base.load_state_dict(_load_state(weight), strict=True)
        _capture_encode_video(self.base, self, "main")

    def forward_baseline(self, clip: torch.Tensor, lengths: torch.Tensor) -> BaselineOutput:
        raw = self.base(clip, _batch_mask(lengths, clip.shape[1]), self.prompt, lengths, True)
        return BaselineOutput(raw[1].squeeze(-1), raw[2], self.base._responsibility_features, raw)

    def original_loss(self, output, labels, texts, lengths):
        raw = output.raw
        target = class_targets(texts, self.label_map, output.binary_logits.device)
        loss = binary_topk_mil(output.binary_logits, labels, lengths)
        loss = loss + float(self.options.loss2_weight) * multiclass_topk_mil(output.semantic_logits, target, lengths)
        loss = loss + text_separation_loss(raw[0])
        loss = loss + single_top_multiclass_loss(raw[3], target, lengths)
        loss = loss + single_top_multiclass_loss(raw[4], target, lengths, normal_target=True)
        dnp = raw[5]
        loss = loss + dsanet_consistency_loss(
            output.binary_logits, dnp["original_features"], dnp["reconstructed_features"], lengths
        ) + dnp["g_loss"]
        return loss

    def set_train_scope(self, scope: str) -> None:
        self.base.requires_grad_(False)
        if scope == "frozen":
            return
        if scope in {"heads", "temporal_heads", "evidence_adaptation", "all_non_clip"}:
            for module in (self.base.classifier, self.base.mlp1, self.base.mlp2):
                module.requires_grad_(True)
        if scope in {"temporal_only", "temporal_heads", "evidence_adaptation", "all_non_clip"}:
            _unfreeze_last(self.base.temporal.resblocks)
            for module in (self.base.gc2, self.base.gc4, self.base.linear):
                module.requires_grad_(True)
        if scope == "evidence_adaptation":
            # DNP is DSANet's normal-pattern reference module.  The definition
            # evidence loss supervises its reconstruction error, so it must be
            # trainable in the final stage while CLIP stays frozen.
            self.base.video_anomaly_refiner.requires_grad_(True)
        if scope == "all_non_clip":
            self.base.requires_grad_(True)
            self.base.clipmodel.requires_grad_(False)
        if scope not in {"frozen", "heads", "temporal_only", "temporal_heads", "evidence_adaptation", "all_non_clip"}:
            raise ValueError(f"unknown train scope: {scope}")


class DeSCAdapter(BaselineAdapter):
    def __init__(self, root: str, dataset: str, sensitivity_weight: str, consistency_weight: str, device: str) -> None:
        super().__init__()
        source = str(Path(root) / "src")
        sys.path.insert(0, source)
        try:
            from model_modular_corrected import CLIPVAD_Modular_Parallel
            consistency_module = __import__("model_gmp" if dataset == "ucf" else "model_multigmp")
            ConsistencyModel = consistency_module.CLIPVAD
            options_module = __import__("ucf_option" if dataset == "ucf" else "xd_option")
            self.options = options_module.parser.parse_args([])
        finally:
            sys.path.pop(0)
        self.dataset = dataset
        self.label_map = label_map_for(dataset)
        self.prompt = list(self.label_map.values())
        self.visual_length = int(self.options.visual_length)
        self.sensitivity = CLIPVAD_Modular_Parallel(
            self.options.classes_num, self.options.embed_dim, self.visual_length, self.options.visual_width,
            self.options.prompt_prefix, self.options.prompt_postfix, device,
            self.options.tcn_levels, self.options.tcn_kernel_size,
            getattr(self.options, "graph_layers", 1), getattr(self.options, "graph_head", 1),
        )
        if dataset == "ucf":
            self.consistency = ConsistencyModel(
                self.options.classes_num, self.options.embed_dim, self.visual_length, self.options.visual_width,
                self.options.visual_head, self.options.visual_layers, self.options.attn_window,
                self.options.prompt_prefix, self.options.prompt_postfix, device,
            )
        else:
            self.consistency = ConsistencyModel(
                self.options.classes_num, self.options.embed_dim, self.visual_length, self.options.visual_width,
                self.options.visual_head, self.options.visual_layers, self.options.attn_window,
                self.options.prompt_prefix, self.options.prompt_postfix, device, n_components=5,
            )
        self.sensitivity.load_state_dict(_load_state(sensitivity_weight), strict=True)
        self.consistency.load_state_dict(_load_state(consistency_weight), strict=True)
        _capture_encode_video(self.sensitivity, self, "sensitivity")
        _capture_encode_video(self.consistency, self, "consistency")

    def forward_baseline(self, clip: torch.Tensor, lengths: torch.Tensor) -> BaselineOutput:
        mask = _batch_mask(lengths, clip.shape[1])
        first = self.sensitivity(clip, mask, self.prompt, lengths)
        second = self.consistency(clip, mask, self.prompt, lengths)
        # The released DeSC evaluator averages stream probabilities, not raw
        # logits.  Convert the averaged probabilities back to log/logit form
        # so the shared adapter contract preserves that exact inference rule.
        binary_probability = 0.5 * (
            torch.sigmoid(first[1].squeeze(-1)) + torch.sigmoid(second[1].squeeze(-1))
        )
        binary = torch.logit(binary_probability.clamp(1e-6, 1.0 - 1e-6))
        semantic_probability = 0.5 * (
            F.softmax(first[2], dim=-1) + F.softmax(second[2], dim=-1)
        )
        semantic = torch.log(semantic_probability.clamp_min(1e-12))
        features = 0.5 * (self.sensitivity._responsibility_features + self.consistency._responsibility_features)
        return BaselineOutput(binary, semantic, features, (first, second))

    def original_loss(self, output, labels, texts, lengths):
        target = class_targets(texts, self.label_map, output.binary_logits.device)
        total = output.binary_logits.sum() * 0.0
        for stream in output.raw:
            total = total + binary_topk_mil(stream[1].squeeze(-1), labels, lengths)
            total = total + multiclass_topk_mil(stream[2], target, lengths)
        return total

    def set_train_scope(self, scope: str) -> None:
        if scope not in {"frozen", "heads", "temporal_only", "temporal_heads"}:
            raise ValueError(f"unknown DeSC train scope: {scope}")
        for model in (self.sensitivity, self.consistency):
            model.requires_grad_(False)
        if scope == "frozen":
            return
        if scope in {"heads", "temporal_heads"}:
            for model in (self.sensitivity, self.consistency):
                for module in (model.classifier, model.mlp1, model.mlp2):
                    module.requires_grad_(True)
        if scope in {"temporal_only", "temporal_heads"}:
            _unfreeze_last(self.sensitivity.tcn_module.layers)
            _unfreeze_last(self.sensitivity.gt_module.resblocks)
            self.sensitivity.fusion_mlp.requires_grad_(True)
            _unfreeze_last(self.consistency.temporal.resblocks)
            for module in (self.consistency.gc2, self.consistency.gc4, self.consistency.linear):
                module.requires_grad_(True)


class LaGoVADAdapter(BaselineAdapter):
    def __init__(self, root: str, dataset: str, weight: str, device: str) -> None:
        super().__init__()
        source = str(Path(root) / "src")
        sys.path.insert(0, source)
        try:
            from models.LaGoVAD import LaGoVADLightModel
            from models.LaGoVAD.losses import mil_loss, multi_class_mil_loss, multi_class_mil_loss_v2
            self.base = LaGoVADLightModel.load_from_checkpoint(weight, map_location="cpu")
        finally:
            sys.path.pop(0)
        self._mil_loss = mil_loss
        self._multi_class_mil_loss = multi_class_mil_loss
        self._multi_class_mil_loss_v2 = multi_class_mil_loss_v2
        self.dataset = dataset
        self.label_map = label_map_for(dataset)
        self.class_names = list(self.label_map.values())
        self.visual_length = 512
        _capture_lagovad_temporal(self.base, self)

    def forward_baseline(self, clip: torch.Tensor, lengths: torch.Tensor) -> BaselineOutput:
        batch = {"v_feat": clip, "v_feat_l": lengths}
        raw = self.base(batch, class_names=self.class_names)
        return BaselineOutput(raw["cls_bin_logits"], raw["cls_sim_mat"], raw["vis_feats"], raw)

    def original_loss(self, output, labels, texts, lengths):
        target = class_targets(texts, self.label_map, output.binary_logits.device)
        target_indices = target.argmax(dim=1)
        config = self.base.training_config
        binary_loss = self._mil_loss(
            output.binary_logits,
            labels.float(),
            lengths,
            topk_pct=config.mil_topk_pct,
        )
        if config.mul_mil_type == "v1":
            multi_loss_function = self._multi_class_mil_loss
        elif config.mul_mil_type == "v2":
            multi_loss_function = self._multi_class_mil_loss_v2
        else:
            raise ValueError(f"unsupported LaGoVAD mul_mil_type: {config.mul_mil_type}")
        multi_loss = multi_loss_function(
            output.semantic_logits,
            target_indices,
            lengths,
            topk_pct=config.mul_mil_topk_pct,
        )
        return binary_loss + float(config.mul_weight) * multi_loss

    def set_train_scope(self, scope: str) -> None:
        self.base.requires_grad_(False)
        if scope == "frozen":
            return
        if scope in {"heads", "temporal_heads", "all_non_clip"}:
            for module in (self.base.bin_head, self.base.sim_head):
                module.requires_grad_(True)
        if scope in {"temporal_only", "temporal_heads", "all_non_clip"}:
            temporal = self.base.temporal_encoder
            if hasattr(temporal, "temporal") and hasattr(temporal.temporal, "resblocks"):
                _unfreeze_last(temporal.temporal.resblocks)
                for name in ("gc2", "gc4", "linear"):
                    if hasattr(temporal, name):
                        getattr(temporal, name).requires_grad_(True)
            elif hasattr(temporal, "encoder") and hasattr(temporal.encoder, "layer"):
                _unfreeze_last(temporal.encoder.layer)
            elif hasattr(temporal, "layer"):
                _unfreeze_last(temporal.layer)
            else:
                _unfreeze_last(temporal)
            if self.base.fusion is not None:
                _unfreeze_last(self.base.fusion)
        if scope == "all_non_clip":
            self.base.requires_grad_(True)
            self.base.clip_text_model.model.requires_grad_(False)
        if scope not in {"frozen", "heads", "temporal_only", "temporal_heads", "all_non_clip"}:
            raise ValueError(f"unknown train scope: {scope}")


def build_baseline(args: argparse.Namespace, device: str) -> BaselineAdapter:
    if args.baseline == "dsanet":
        return DSANetAdapter(args.baseline_root, args.dataset, args.baseline_weight, device)
    if args.baseline == "desc":
        return DeSCAdapter(
            args.baseline_root, args.dataset, args.sensitivity_weight, args.consistency_weight, device
        )
    if args.baseline == "lagovad":
        return LaGoVADAdapter(args.baseline_root, args.dataset, args.baseline_weight, device)
    raise ValueError(f"unknown baseline: {args.baseline}")
