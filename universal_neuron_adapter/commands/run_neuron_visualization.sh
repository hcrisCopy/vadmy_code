#!/usr/bin/env bash
set -euo pipefail

SOURCE_ROOT="../vadmy_data/universal_neuron_adapter/runs/9d1a066"
NORMALITY_ROOT="../vadmy_data/universal_neuron_adapter/normality_expert_cache"
CONTROLS_CSV="../vadmy_data/universal_neuron_adapter/neuron_controls/c8c8cf6/neuron_control_summary.csv"
OUTPUT_ROOT="../vadmy_data/universal_neuron_adapter/figures/detected_neurons"

python -m universal_neuron_adapter.visualize_detected_neurons \
  --source-root "${SOURCE_ROOT}" \
  --normality-root "${NORMALITY_ROOT}" \
  --normality-tag top32_signed_v1 \
  --controls-csv "${CONTROLS_CSV}" \
  --out-dir "${OUTPUT_ROOT}" \
  "$@"
