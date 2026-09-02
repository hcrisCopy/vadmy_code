#!/usr/bin/env bash
set -euo pipefail

OUT="../vadmy_data/witness_vad/dsanet/f1_smoke"
DATA_ROOT="$(realpath -m ../vadmy_data)"
OUT_ABS="$(realpath -m "$OUT")"
EXPECTED_OUT="$DATA_ROOT/witness_vad/dsanet/f1_smoke"

if [[ "$OUT_ABS" != "$EXPECTED_OUT" ]]; then
  echo "refusing unsafe F1 output path: $OUT_ABS" >&2
  exit 2
fi
if [[ $# -gt 1 || ( $# -eq 1 && "$1" != "--resume" && "$1" != "--clean" ) ]]; then
  echo "usage: bash run_instructions/run_witness_vad_f1_dsanet.sh [--resume|--clean]" >&2
  exit 2
fi
if [[ "${1:-}" == "--clean" && -d "$OUT_ABS" ]]; then
  echo "cleaning exact F1 output: $OUT_ABS"
  rm -rf -- "$OUT_ABS"
fi

mkdir -p "$OUT"
COMMAND="bash run_instructions/run_witness_vad_f1_dsanet.sh${1:+ $1}"
printf '%s\n' "$COMMAND" > "$OUT/command.txt"
exec > >(tee -a "$OUT/stdout.log") 2>&1

echo "F1 Witness-VAD structure and gradient closure"
echo "output: $OUT_ABS"
python -m pytest \
  vin_vad/tests/test_witness_neurons.py \
  vin_vad/tests/test_witness_router.py \
  vin_vad/tests/test_witness_losses.py \
  -q | tee "$OUT/test_report.txt"

RESUME_ARGS=()
if [[ "${1:-}" == "--resume" ]]; then
  RESUME_ARGS=(--resume)
fi
python -m vin_vad.f1_smoke \
  --b0-root ../vadmy_data/vin_vad/dsanet/b0 \
  --out-dir "$OUT" \
  --seed 42 \
  --active-neurons 32 \
  --temporal-width 64 \
  --maximum-length 256 \
  --device cuda \
  "${RESUME_ARGS[@]}"

echo "F1 complete: $OUT/summary.md"
