from __future__ import annotations

import torch

from vin_vad.evaluate_correction import window_starts


def test_window_starts_cover_sequence_and_finish_at_tail() -> None:
    starts = window_starts(length=701, maximum_length=256, overlap=64)
    coverage = torch.zeros(701, dtype=torch.int64)
    for start in starts:
        coverage[start : start + 256] += 1
    assert starts[0] == 0
    assert starts[-1] == 701 - 256
    assert torch.all(coverage > 0)


def test_short_sequence_uses_one_window() -> None:
    assert window_starts(length=19, maximum_length=256, overlap=64) == [0]
