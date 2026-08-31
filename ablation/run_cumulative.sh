#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 2 || $# -gt 4 ]]; then
  echo "usage: bash ablation/run_cumulative.sh <baseline> <source-run-name> [ucf|xd|both] [--clean]" >&2
  exit 2
fi

baseline="$1"
source_name="$2"
dataset="${3:-both}"
clean=()
if [[ "${4:-}" == "--clean" ]]; then
  clean=(--clean)
elif [[ -n "${4:-}" ]]; then
  echo "fourth argument must be --clean" >&2
  exit 2
fi

case "$baseline" in
  lagovad|desc|dsanet|vadclip) ;;
  *) echo "unknown baseline: $baseline" >&2; exit 2 ;;
esac
case "$dataset" in
  ucf|xd|both) ;;
  *) echo "dataset must be ucf, xd, or both" >&2; exit 2 ;;
esac
if [[ ! "$source_name" =~ ^[A-Za-z0-9][A-Za-z0-9._-]{0,79}$ ]]; then
  echo "invalid source run name: $source_name" >&2
  exit 2
fi

python -m ablation.cumulative \
  --baseline "$baseline" \
  --dataset "$dataset" \
  --source-run "../vadmy_data/universal_neuron_adapter/runs/$source_name" \
  --output-dir "../vadmy_data/universal_neuron_adapter/ablations/$source_name" \
  "${clean[@]}"
