#!/usr/bin/env bash
set -euo pipefail
export PYTHONWARNINGS="ignore::RuntimeWarning"

OUT="../vadmy_data/witness_vad/dsanet/f0_universal_autopsy"
DATA_ROOT="$(realpath -m ../vadmy_data)"
OUT_ABS="$(realpath -m "$OUT")"
EXPECTED_OUT="$DATA_ROOT/witness_vad/dsanet/f0_universal_autopsy"

if [[ "$OUT_ABS" != "$EXPECTED_OUT" ]]; then
  echo "refusing unsafe F0 output path: $OUT_ABS" >&2
  exit 2
fi
if [[ $# -gt 1 || ( $# -eq 1 && "$1" != "--resume" && "$1" != "--clean" ) ]]; then
  echo "usage: bash run_instructions/run_witness_vad_f0_dsanet.sh [--resume|--clean]" >&2
  exit 2
fi
if [[ "${1:-}" == "--clean" && -d "$OUT_ABS" ]]; then
  echo "cleaning exact F0 output: $OUT_ABS"
  rm -rf -- "$OUT_ABS"
fi

mkdir -p "$OUT"
COMMAND="bash run_instructions/run_witness_vad_f0_dsanet.sh${1:+ $1}"
printf '%s\n' "$COMMAND" > "$OUT/command.txt"
exec > >(tee -a "$OUT/stdout.log") 2>&1

echo "F0 DSANet Universal autopsy"
echo "output: $OUT"
python -m pytest vin_vad/tests/test_f0.py -q

python -m vin_vad.universal_autopsy \
  --source-run ../vadmy_data/universal_neuron_adapter/runs/9d1a066 \
  --normality-cache-root ../vadmy_data/universal_neuron_adapter/normality_expert_cache \
  --normality-cache-name top32_signed_v1 \
  --context-cache-root ../vadmy_data/universal_neuron_adapter/context_student_cache \
  --context-cache-name top32_multiscale_seed234 \
  --historical-run ../vadmy_data/universal_neuron_adapter/runs/60bcdda \
  --output-dir "$OUT" \
  --target-tpr 0.95 \
  --device cuda \
  --resume

echo "F0 complete: $OUT/summary.md"
