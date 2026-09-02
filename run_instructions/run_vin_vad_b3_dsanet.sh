#!/usr/bin/env bash
set -euo pipefail

OUT=../vadmy_data/vin_vad/dsanet/b3
CLEAN_ARGS=()
if [[ "${CLEAN:-0}" == "1" ]]; then
  CLEAN_ARGS=(--clean)
fi
mkdir -p "$OUT"
exec > >(tee -a "$OUT/run.log") 2>&1

python -m pytest vin_vad/tests/test_host_auditor.py -q

for dataset in ucf xd; do
  python -m vin_vad.audit_host_auditor \
    --dataset "$dataset" \
    --train-manifest "../vadmy_data/vin_vad/dsanet/b0/$dataset/evaluation/train_aligned.csv" \
    --b2-sample "../vadmy_data/vin_vad/dsanet/b2/$dataset/recompute_sample.npz" \
    --out-dir "$OUT/$dataset" \
    --alpha-cross 0.4 \
    --alpha-within 0.3 \
    --audit-kappa-cross 0.8 \
    --audit-kappa-within 0.7 \
    "${CLEAN_ARGS[@]}"
done

python - <<'PY'
import json
from pathlib import Path

root = Path("../vadmy_data/vin_vad/dsanet/b3")
for dataset in ("ucf", "xd"):
    summary = json.loads((root / dataset / "summary.json").read_text())
    print(
        f"{dataset}: status={summary['status']}; "
        f"identity={summary['identity_max_abs_error']:.3e}; "
        f"within mean error={summary['within_masked_mean_abs_error']:.3e}; "
        f"padding output error={summary['padding_output_max_abs_error']:.3e}; "
        f"padding budget error={summary['padding_budget_abs_error']:.3e}"
    )
PY
