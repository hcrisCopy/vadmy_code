from __future__ import annotations

import pandas as pd
import torch

from vin_vad.train_witness import (
    balanced_indices,
    comparable_configuration,
    merge_balanced_batches,
)


def test_balanced_indices_are_fixed_and_class_complete() -> None:
    frame = pd.DataFrame({"binary_label": [0, 1, 0, 1, 0, 1]})
    normal, abnormal = balanced_indices(frame, per_class=2)
    assert normal == [0, 2]
    assert abnormal == [1, 3]


def test_merge_balanced_batches_preserves_padding_and_labels() -> None:
    normal = {
        "hidden": torch.ones(1, 2, 12, 768),
        "host_score": torch.ones(1, 2),
        "mask": torch.ones(1, 2, dtype=torch.bool),
        "labels": torch.zeros(1),
    }
    abnormal = {
        "hidden": torch.ones(1, 3, 12, 768),
        "host_score": torch.ones(1, 3),
        "mask": torch.ones(1, 3, dtype=torch.bool),
        "labels": torch.ones(1),
    }
    merged = merge_balanced_batches(normal, abnormal)
    assert merged["hidden"].shape == (2, 3, 12, 768)
    assert merged["mask"].tolist() == [[True, True, False], [True, True, True]]
    assert merged["labels"].tolist() == [0.0, 1.0]


def test_resume_ignores_provenance_hash_but_not_hyperparameters() -> None:
    saved = {"git_commit": "old", "learning_rate": 3e-4, "seed": 42}
    current = {"git_commit": "new", "learning_rate": 3e-4, "seed": 42}
    assert comparable_configuration(saved) == comparable_configuration(current)
    current["learning_rate"] = 1e-4
    assert comparable_configuration(saved) != comparable_configuration(current)


def test_rng_checkpoint_tensors_are_cpu_compatible() -> None:
    state = torch.get_rng_state()
    torch.set_rng_state(state.cpu())
    assert state.dtype == torch.uint8
