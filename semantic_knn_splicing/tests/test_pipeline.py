from __future__ import annotations

import json

import numpy as np
import pandas as pd
import torch

from semantic_knn_splicing.data import BaselineTrainDataset
from semantic_knn_splicing.prompts import PROMPT_BANKS, abnormal_class_names
from semantic_knn_splicing.select_pseudo_segments import merge_selected
from semantic_knn_splicing.semantic_lens import selected_layer_spec
from semantic_knn_splicing.train_baseline import pseudo_supervision_losses


def test_published_prompt_dictionary_counts_are_stable() -> None:
    expected = {"ucf": 33, "xd": 30}
    for dataset in ("ucf", "xd"):
        prompts = {
            prompt
            for name in abnormal_class_names(dataset)
            for prompt in PROMPT_BANKS[dataset][name]
        }
        assert len(prompts) == expected[dataset]


def test_selected_layer_weights_are_responsibility_normalized(tmp_path) -> None:
    path = tmp_path / "atlas.json"
    path.write_text(json.dumps({"blocks": [
        {"layer_zero_based": 3, "width": 10, "direction_stability": 0.5},
        {"layer_zero_based": 8, "width": 20, "direction_stability": 1.0},
    ]}), encoding="utf-8")
    layers, weights = selected_layer_spec(str(path))
    assert layers == [3, 8]
    assert np.allclose(weights, [0.2, 0.8])


def test_adjacent_threshold_segments_are_merged() -> None:
    ranges = [(0, 2), (2, 4), (4, 6), (6, 8)]
    result = merge_selected([1, 2], ranges, np.asarray([0.1, 0.6, 0.8, 0.2]))
    assert result == [(2, 6, 0.7)]


def test_pseudo_losses_ignore_padding_and_original_examples() -> None:
    logits = torch.zeros(2, 4)
    labels = torch.tensor([[0.0, 1.0, 1.0, 0.0], [1.0, 1.0, 1.0, 1.0]])
    dense, mil = pseudo_supervision_losses(
        logits, labels, torch.tensor([3, 4]), torch.tensor([1.0, 0.0]), 2
    )
    assert torch.isfinite(dense) and torch.isfinite(mil)
    assert dense.item() > 0 and mil.item() > 0


def test_training_splits_preserve_original_pairing(tmp_path) -> None:
    feature = tmp_path / "feature.npy"
    np.save(feature, np.zeros((4, 512), dtype=np.float32))
    synthetic = tmp_path / "synthetic.npz"
    np.savez_compressed(
        synthetic,
        feature=np.zeros((4, 512), dtype=np.float32),
        frame_label=np.asarray([0, 1, 1, 0], dtype=np.float32),
    )
    original_csv = tmp_path / "original.csv"
    pd.DataFrame({
        "path": [str(feature), str(feature)], "label": ["Normal", "Fighting"]
    }).to_csv(original_csv, index=False)
    synthetic_csv = tmp_path / "synthetic.csv"
    pd.DataFrame({
        "feature_path": [str(synthetic)], "label": ["Fighting"]
    }).to_csv(synthetic_csv, index=False)
    common = (str(original_csv), str(synthetic_csv), "ucf", 4, "dsanet")
    assert len(BaselineTrainDataset(*common, split="normal")) == 1
    assert len(BaselineTrainDataset(*common, split="abnormal")) == 1
    assert len(BaselineTrainDataset(*common, split="synthetic")) == 1
