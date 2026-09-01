from __future__ import annotations

import numpy as np
import pytest
import torch

from vin_vad.evaluate import expand_snippet_scores, masked_mean
from vin_vad.p0_audit import npz_array_header


def test_npz_header_reads_shape_without_loading_payload(tmp_path) -> None:
    path = tmp_path / "hidden.npz"
    np.savez_compressed(path, hidden=np.zeros((3, 12, 768), dtype=np.float16))
    shape, dtype = npz_array_header(str(path), "hidden")
    assert shape == (3, 12, 768)
    assert dtype == np.dtype(np.float16)


def test_expand_snippet_scores_uses_audited_boundaries() -> None:
    scores = np.asarray([0.1, 0.8, 0.3], dtype=np.float32)
    expanded = expand_snippet_scores(scores, np.asarray([0, 16, 32]), frame_count=48)
    assert len(expanded) == 48
    np.testing.assert_array_equal(expanded[:16], np.full(16, 0.1, dtype=np.float32))
    np.testing.assert_array_equal(expanded[16:32], np.full(16, 0.8, dtype=np.float32))
    np.testing.assert_array_equal(expanded[32:], np.full(16, 0.3, dtype=np.float32))


def test_expand_rejects_invalid_frame_indices() -> None:
    with pytest.raises(ValueError):
        expand_snippet_scores(np.asarray([0.1, 0.2]), np.asarray([0, 0]), frame_count=16)


def test_expand_truncates_training_tail_at_real_frame_count() -> None:
    scores = np.asarray([0.1, 0.2, 0.9], dtype=np.float32)
    expanded = expand_snippet_scores(scores, np.asarray([0, 16, 32]), frame_count=37)
    assert len(expanded) == 37
    np.testing.assert_array_equal(expanded[32:], np.full(5, 0.9, dtype=np.float32))


def test_padding_content_does_not_change_masked_result() -> None:
    mask = torch.tensor([[True, True, False, False]])
    first = masked_mean(torch.tensor([[1.0, 3.0, 0.0, 0.0]]), mask)
    changed_padding = masked_mean(torch.tensor([[1.0, 3.0, -999.0, 999.0]]), mask)
    torch.testing.assert_close(first, changed_padding)
