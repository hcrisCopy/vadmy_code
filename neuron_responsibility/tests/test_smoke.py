from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import torch

from neuron_responsibility.common import base_key, resample_feature
from neuron_responsibility.circuit_routing import ConceptCircuitRouter
from neuron_responsibility.boundary_localization import (
    IndependentNeuronLocalizer,
    NeuronBoundaryConditioner,
    boundary_supervision_loss,
    synthesize_boundary_batch,
)
from neuron_responsibility.data import AlignedFeatureDataset
from neuron_responsibility.feature_modulation import (
    SparseNeuronFeatureModulator,
    score_free_modulation_losses,
)
from neuron_responsibility.model import (
    NeuronResponsibilityProbe,
    ResponsibilityCorrectionHead,
    partition_responsibility_loss,
    probe_mil_loss,
    responsibility_mil_loss,
    responsibility_sets,
)


def test_key_and_resample() -> None:
    assert base_key("Abuse001_x264__7.npy") == "Abuse001_x264"
    feature = np.arange(12, dtype=np.float32).reshape(3, 4)
    assert resample_feature(feature, 5).shape == (5, 4)


def test_probe_and_losses_are_finite() -> None:
    torch.manual_seed(1)
    probe = NeuronResponsibilityProbe(12, hidden_width=8, dropout=0.0)
    neurons = torch.randn(2, 16, 12)
    lengths = torch.tensor([16, 11])
    labels = torch.tensor([1.0, 0.0])
    logits = probe(neurons, lengths)
    loss_probe = probe_mil_loss(logits, labels, lengths)
    base_logits = torch.randn(2, 16)
    loss_resp = responsibility_mil_loss(base_logits, torch.sigmoid(logits), labels, lengths)
    sets = responsibility_sets(torch.sigmoid(base_logits), torch.sigmoid(logits), lengths)
    assert logits.shape == (2, 16)
    assert torch.isfinite(loss_probe)
    assert torch.isfinite(loss_resp)
    probe.eval()
    assert int(probe.feature_gates().sum().item()) == 12
    correction = ResponsibilityCorrectionHead(hidden_width=4)
    corrected = correction(base_logits, torch.sigmoid(logits), lengths)
    assert torch.allclose(corrected, base_logits)
    partition_loss, partitions = partition_responsibility_loss(
        corrected, base_logits, torch.sigmoid(logits), labels, lengths,
        neuron_threshold=0.8, persistence=3,
    )
    assert torch.isfinite(partition_loss)
    assert set(partitions) == {
        "agreement_high", "baseline_only", "neuron_only", "agreement_low", "pure_normal"
    }
    assert torch.allclose(
        sets["positive"] + sets["normal"] + sets["uncertain"],
        sets["mask"],
        atol=1e-5,
    )


def test_aligned_dataset(tmp_path: Path) -> None:
    clip = np.random.randn(7, 512).astype(np.float32)
    neurons = np.random.randn(7, 12).astype(np.float32)
    clip_path = tmp_path / "Normal001__5.npy"
    neuron_path = tmp_path / "Normal001__5_neuron.npy"
    np.save(clip_path, clip)
    np.save(neuron_path, neurons)
    csv_path = tmp_path / "aligned.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["clip_path", "neuron_path", "label", "key", "length"])
        writer.writerow([clip_path, neuron_path, "Normal", "Normal001", 7])
    dataset = AlignedFeatureDataset(str(csv_path), "ucf", visual_length=16)
    item = dataset[0]
    assert item["clip"].shape == (16, 512)
    assert item["neurons"].shape == (16, 12)
    assert item["length"].item() == 7
    assert item["binary_label"].item() == 0.0


def test_score_free_feature_modulation_is_zero_initialized() -> None:
    torch.manual_seed(2)
    modulator = SparseNeuronFeatureModulator(
        neuron_width=12,
        active_indices=torch.tensor([1, 4, 8, 10]),
        thresholds=torch.tensor([0.5, 0.5, 0.5, 0.5]),
        feature_width=16,
        context_width=4,
        temporal_kernel=3,
    )
    features = torch.randn(2, 9, 16)
    neurons = torch.randn(2, 9, 12)
    lengths = torch.tensor([9, 6])
    labels = torch.tensor([1.0, 0.0])
    modulated, record = modulator(features, neurons, lengths)
    assert torch.equal(modulated, features)
    losses = score_free_modulation_losses([record], labels, lengths)
    assert set(losses) == {"auxiliary", "normal", "smooth", "sparse"}
    assert all(torch.isfinite(value) for value in losses.values())
    total = sum(losses.values())
    total.backward()
    assert modulator.auxiliary_head.weight.grad is not None
    assert modulator.config()["method"] == "sparse_neuron_feature_modulation_v1"


def test_boundary_localizer_synthesis_and_conditioning() -> None:
    torch.manual_seed(3)
    localizer = IndependentNeuronLocalizer(
        neuron_width=12,
        active_indices=torch.tensor([1, 3, 5, 7, 9, 11]),
        thresholds=torch.full((6,), 0.5),
        hidden_width=8,
        active_neurons=4,
        dropout=0.0,
    )
    normal_clip = torch.randn(2, 16, 512)
    abnormal_clip = torch.randn(2, 16, 512)
    normal_neurons = torch.randn(2, 16, 12)
    abnormal_neurons = torch.randn(2, 16, 12) + 1.0
    lengths = torch.tensor([16, 13])
    synthetic = synthesize_boundary_batch(
        localizer,
        normal_clip,
        normal_neurons,
        lengths,
        abnormal_clip,
        abnormal_neurons,
        lengths,
        min_segment=3,
        max_segment=6,
    )
    assert synthetic["clip"].shape == (4, 16, 512)
    assert synthetic["targets"][:2].sum() > 0
    assert synthetic["targets"][2:].sum() == 0
    logits = localizer(synthetic["neurons"], synthetic["lengths"])
    losses = boundary_supervision_loss(
        logits, synthetic["targets"], synthetic["lengths"], synthetic["confidence"]
    )
    assert set(losses) == {"bce", "dice", "boundary"}
    assert all(torch.isfinite(value) for value in losses.values())

    conditioner = NeuronBoundaryConditioner(
        localizer, feature_width=512, adapter_width=8, max_scale=0.25
    )
    conditioned, record = conditioner(normal_clip, normal_neurons, lengths)
    assert torch.equal(conditioned, normal_clip)
    sum(losses.values()).backward()
    conditioner_loss = conditioner(
        normal_clip, normal_neurons, lengths
    )[0].square().mean()
    conditioner_loss.backward()
    assert conditioner.up.weight.grad is not None
    assert conditioner.config()["method"] == "neuron_boundary_pre_temporal_conditioning_v1"


def test_concept_circuit_router_preserves_shape_and_gradients() -> None:
    torch.manual_seed(4)
    hidden_width, feature_width = 8, 5
    union = torch.tensor([1, 3, 6])
    class_mask = torch.tensor([[1, 1, 0], [0, 1, 1]], dtype=torch.float32)
    directions = torch.tensor([[1, -1, 0], [0, 1, -1]], dtype=torch.float32)
    router = ConceptCircuitRouter(
        union_indices=union,
        class_mask=class_mask,
        directions=directions,
        center=torch.zeros(3),
        scale=torch.ones(3),
        ln_weight=torch.ones(hidden_width),
        ln_bias=torch.zeros(hidden_width),
        projection=torch.randn(hidden_width, feature_width),
        normal_text=torch.randn(2, feature_width),
        abnormal_text=torch.randn(2, feature_width),
        normal_margin_threshold=-1.0,
        initial_gain=0.1,
    )
    hidden = torch.randn(2, 7, hidden_width)
    normalized = torch.nn.functional.layer_norm(hidden, (hidden_width,))
    clip = normalized @ router.projection
    compact = torch.cat(
        [
            hidden.index_select(-1, union),
            hidden.mean(-1, keepdim=True),
            hidden.var(-1, keepdim=True, unbiased=False),
        ],
        dim=-1,
    )
    targets = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    views = router(clip, compact, targets)
    assert views.enhanced.shape == clip.shape
    assert views.suppressed.shape == clip.shape
    assert views.evidence.shape == clip.shape[:2]
    assert torch.isfinite(views.enhanced).all()
    loss = views.enhanced.square().mean() + views.target_text_effect.mean()
    loss.backward()
    assert router.gain_logits.grad is not None
