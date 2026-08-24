from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import torch

from neuron_responsibility.common import base_key, resample_feature
from neuron_responsibility.data import AlignedFeatureDataset
from neuron_responsibility.model import (
    NeuronResponsibilityProbe,
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
