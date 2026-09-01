#!/usr/bin/env bash
set -euo pipefail

OUT=../vadmy_data/vin_vad/dsanet/b1
export OMP_NUM_THREADS=8
CLEAN_ARGS=()
if [[ "${CLEAN:-0}" == "1" ]]; then
  CLEAN_ARGS=(--clean)
fi
mkdir -p "$OUT"
exec > >(tee -a "$OUT/run.log") 2>&1

python -m pytest vin_vad/tests/test_context_predictor.py -q

for dataset in ucf xd; do
  python -m vin_vad.train_context_predictor \
    --dataset "$dataset" \
    --source-manifest "../vadmy_data/vin_vad/dsanet/b0/$dataset/evaluation/train_aligned.csv" \
    --out-dir "$OUT/$dataset" \
    --epochs 10 \
    --patience 3 \
    --batch-size 2 \
    --lr 0.0003 \
    --weight-decay 0.0001 \
    --maximum-length 256 \
    --window-overlap 8 \
    --model-width 128 \
    --input-rank 16 \
    --head-rank 32 \
    --attention-heads 4 \
    --attention-layers 2 \
    --guard-radius 2 \
    --dropout 0.1 \
    --sigma-min 0.05 \
    --sigma-max 3.0 \
    --validation-fraction 0.1 \
    --num-workers 2 \
    --seed 42 \
    --device cuda \
    --resume \
    "${CLEAN_ARGS[@]}"
done

python - <<'PY'
import json
from pathlib import Path

root = Path("../vadmy_data/vin_vad/dsanet/b1")
for dataset in ("ucf", "xd"):
    summary = json.loads((root / dataset / "summary.json").read_text())
    print(
        f"{dataset}: status={summary['status']}; "
        f"conditional={summary['validation_conditional_nll']:.6f}; "
        f"global={summary['validation_global_nll']:.6f}; "
        f"relative improvement={100 * summary['relative_nll_improvement']:.2f}%"
    )
PY
