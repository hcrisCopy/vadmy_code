#!/usr/bin/env bash
set -euo pipefail

OUT=../vadmy_data/vin_vad/dsanet/b2
export OMP_NUM_THREADS=8
CLEAN_ARGS=()
if [[ "${CLEAN:-0}" == "1" ]]; then
  CLEAN_ARGS=(--clean)
fi
mkdir -p "$OUT"
exec > >(tee -a "$OUT/run.log") 2>&1

python -m pytest vin_vad/tests/test_violation_field.py -q

for dataset in ucf xd; do
  python -m vin_vad.audit_violation_field \
    --dataset "$dataset" \
    --b1-dir "../vadmy_data/vin_vad/dsanet/b1/$dataset" \
    --out-dir "$OUT/$dataset" \
    --delta 1.0 \
    --statistics-momentum 0.05 \
    --batch-size 8 \
    --maximum-length 256 \
    --num-workers 2 \
    --device cuda \
    "${CLEAN_ARGS[@]}"
done

python - <<'PY'
import json
from pathlib import Path

root = Path("../vadmy_data/vin_vad/dsanet/b2")
for dataset in ("ucf", "xd"):
    summary = json.loads((root / dataset / "summary.json").read_text())
    print(
        f"{dataset}: status={summary['status']}; "
        f"normal snippets={summary['normal_snippets_seen']}; "
        f"median={summary['running_median']:.6f}; "
        f"MAD={summary['running_mad']:.6f}; "
        f"pi sum error={summary['probability_sum_error']:.3e}; "
        f"direction overlap={summary['direction_overlap_count']}"
    )
PY
