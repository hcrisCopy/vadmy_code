#!/usr/bin/env bash
set -euo pipefail

ROOT="../vadmy_data/universal_neuron_adapter"
SOURCE="$ROOT/runs/9d1a066"
NORMALITY="$ROOT/normality_expert_cache"
CONTEXT="$ROOT/context_student_cache"
PROBES="$ROOT/interpretability_ablations/3f04e82/probes"
OUTPUT="../vadmy_data/innovation_evidence/$(git rev-parse --short HEAD)"
ANNOTATIONS="../vadmy_data/annotations"

mkdir -p "$OUTPUT"
exec > >(tee -a "$OUTPUT/run.log") 2>&1

python -m innovation_evidence.innovation1_neurons \
  --probe-root "$PROBES" --normality-root "$NORMALITY" --context-root "$CONTEXT" \
  --out-dir "$OUTPUT/innovation1" --clean
python -m innovation_evidence.innovation2_spectral \
  --source-root "$SOURCE" --normality-root "$NORMALITY" --context-root "$CONTEXT" \
  --annotation-root "$ANNOTATIONS" --out-dir "$OUTPUT/innovation2" \
  --frames-per-snippet 16 --clean
python -m innovation_evidence.innovation3_asymmetry \
  --source-root "$SOURCE" --normality-root "$NORMALITY" --context-root "$CONTEXT" \
  --annotation-root "$ANNOTATIONS" --out-dir "$OUTPUT/innovation3" \
  --frames-per-snippet 16 --clean

printf '%s\n' "$OUTPUT" > ../vadmy_data/innovation_evidence/current_run.txt
