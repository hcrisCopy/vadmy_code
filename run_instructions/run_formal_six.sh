#!/usr/bin/env bash
# Run the three formal baselines on UCF-Crime and XD-Violence.
# Usage: EXPERIMENT_NAME=formal_seed234 bash run_instructions/run_formal_six.sh
set -euo pipefail

export EXPERIMENT_NAME="${EXPERIMENT_NAME:-formal_seed234}"

echo "formal experiment: $EXPERIMENT_NAME"
bash run_instructions/run_lagovad.sh
bash run_instructions/run_desc.sh
bash run_instructions/run_dsanet.sh

python -m universal_neuron_adapter.aggregate_metric \
  --results-root ../vadmy_data/universal_neuron_adapter
echo "six-result experiment completed: $EXPERIMENT_NAME"
