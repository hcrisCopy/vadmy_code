#!/usr/bin/env bash
set -euo pipefail

python -m universal_neuron_adapter.visualize_method_overview \
  --out-dir ../vadmy_data/universal_neuron_adapter/figures/method_overview \
  "$@"
