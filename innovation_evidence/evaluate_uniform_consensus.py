from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

import universal_neuron_adapter.evaluate as evaluator


def uniform_weights(*curves: np.ndarray) -> np.ndarray:
    """Ablation: give every CLS-neuron view the same reliability."""
    return np.ones(len(curves), dtype=np.float32)


def main() -> None:
    evaluator.spectral_consensus_weights = uniform_weights
    evaluator.main()
    output = Path(sys.argv[sys.argv.index("--out-dir") + 1])
    metrics_path = output / "metrics.json"
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    metrics["configuration"]["event_gate_reliability"] = "uniform reliability ablation"
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
