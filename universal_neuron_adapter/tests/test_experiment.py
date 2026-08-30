from __future__ import annotations

import json

import pytest

from universal_neuron_adapter import experiment


def test_experiment_rejects_invalid_name(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(experiment, "git_revision", lambda: "abc")
    with pytest.raises(ValueError, match="experiment name"):
        experiment.initialize_experiment(tmp_path / "run", "bad/name", 234)


def test_experiment_refuses_mixed_revision(tmp_path, monkeypatch) -> None:
    output = tmp_path / "run"
    monkeypatch.setattr(experiment, "git_revision", lambda: "first")
    experiment.initialize_experiment(output, "formal_seed234", 234)
    monkeypatch.setattr(experiment, "git_revision", lambda: "second")
    with pytest.raises(RuntimeError, match="another code version"):
        experiment.initialize_experiment(output, "formal_seed234", 234)


def test_experiment_records_readable_name(tmp_path, monkeypatch) -> None:
    output = tmp_path / "run"
    monkeypatch.setattr(experiment, "git_revision", lambda: "abc")
    experiment.initialize_experiment(output, "formal_seed234", 234)
    payload = json.loads((output / "experiment.json").read_text(encoding="utf-8"))
    assert payload["experiment_name"] == "formal_seed234"
