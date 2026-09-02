#!/usr/bin/env bash
set -euo pipefail

OUT=../vadmy_data/vin_vad/dsanet/b4
export OMP_NUM_THREADS=8
CLEAN_ARGS=()
if [[ "${CLEAN:-0}" == "1" ]]; then
  CLEAN_ARGS=(--clean)
fi
mkdir -p "$OUT"
exec > >(tee -a "$OUT/run.log") 2>&1

python -m pytest \
  vin_vad/tests/test_context_predictor.py \
  vin_vad/tests/test_violation_field.py \
  vin_vad/tests/test_host_auditor.py \
  vin_vad/tests/test_b4.py -q

for dataset in ucf xd; do
  python -m vin_vad.train \
    --dataset "$dataset" \
    --train-manifest "../vadmy_data/vin_vad/dsanet/b0/$dataset/evaluation/train_aligned.csv" \
    --b1-dir "../vadmy_data/vin_vad/dsanet/b1/$dataset" \
    --b2-checkpoint "../vadmy_data/vin_vad/dsanet/b2/$dataset/violation_field_initial.pt" \
    --out-dir "$OUT/$dataset" \
    --epochs 10 \
    --batch-size 8 \
    --lr 0.0003 \
    --weight-decay 0.0001 \
    --maximum-length 256 \
    --delta 1.0 \
    --statistics-momentum 0.05 \
    --alpha-cross 0.5 \
    --alpha-within 0.25 \
    --correction-budget 0.1 \
    --lambda-context 1.0 \
    --lambda-budget 10.0 \
    --q-reservoir-capacity 4096 \
    --normal-quantile 0.95 \
    --gradient-clip 5.0 \
    --num-workers 2 \
    --seed 42 \
    --device cuda \
    --resume \
    "${CLEAN_ARGS[@]}"
done

python - <<'PY'
import json
from pathlib import Path

root = Path("../vadmy_data/vin_vad/dsanet/b4")
for dataset in ("ucf", "xd"):
    summary = json.loads((root / dataset / "summary.json").read_text())
    final = summary["final_epoch"]
    print(
        f"{dataset}: status={summary['status']}; "
        f"kappa={summary['kappa_cross']:.4f}/{summary['kappa_within']:.4f}; "
        f"correction={final['correction_size']:.4f}; "
        f"support={summary['field_support_size']}; "
        f"q reservoir={summary['normal_q_reservoir_count']}"
    )
PY
