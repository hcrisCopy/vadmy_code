#!/usr/bin/env bash
# DSANet UCF formal protocol: train every adapter component from scratch.
# Usage: bash run_instructions/run_dsanet_ucf_reproduce.sh
# Resume: RUN_ID=<printed-id> bash run_instructions/run_dsanet_ucf_reproduce.sh
set -euo pipefail
export TOKENIZERS_PARALLELISM=false

SEED=234
RUN_ID="${RUN_ID:-$(git rev-parse --short HEAD)-dsanet-ucf-$(date +%Y%m%d-%H%M%S)}"
OUT="../vadmy_data/universal_neuron_adapter/reproductions/$RUN_ID"
DATA="$OUT/ucf/data"
PRIMARY="$OUT/ucf/primary_expert"
NORMALITY="$OUT/ucf/normality_expert"
CONTEXT="$OUT/ucf/context_student"
BASE="$OUT/ucf/dsanet"
mkdir -p "$OUT"
exec > >(tee -a "$OUT/run.log") 2>&1

echo "run id: $RUN_ID"
echo "output: $OUT"
echo "protocol: fixed training view + conservative logit correction, seed $SEED"

python -m universal_neuron_adapter.data \
  --dataset ucf \
  --train-csv ../vad_data/work_ucf/ucf_train_local.csv \
  --test-csv ../vad_data/work_ucf/ucf_test_local.csv \
  --train-hidden-manifest ../vad_data/work_ucf/clip_hidden_stride16_train_8gpu/manifest.csv \
  --test-hidden-manifest ../vad_data/work_ucf/clip_hidden_stride16_test_8gpu/manifest.csv \
  --out-dir "$DATA" --seed "$SEED" --val-fraction 0.2

python -m universal_neuron_adapter.fit_normality_expert \
  --manifest "$DATA/expert_train.csv" --out-dir "$NORMALITY" \
  --active-per-layer 32 --maximum-length 256 --resume
python -m universal_neuron_adapter.export_normality_expert \
  --manifest "$DATA/expert_train.csv" --expert-model "$NORMALITY/normality_expert.npz" \
  --out-dir "$NORMALITY/train"
python -m universal_neuron_adapter.export_normality_expert \
  --manifest "$DATA/test.csv" --expert-model "$NORMALITY/normality_expert.npz" \
  --out-dir "$NORMALITY/test"

python -m universal_neuron_adapter.train_expert \
  --train-manifest "$DATA/expert_train.csv" --val-manifest "$DATA/expert_val.csv" \
  --out-dir "$PRIMARY" --active-per-layer 32 --temporal-width 64 \
  --max-epoch 20 --batch-size 8 --lr 0.0003 --weight-decay 0.0001 \
  --sparsity-weight 0.001 --maximum-length 256 --num-workers 4 \
  --seed "$SEED" --device cuda --resume
python -m universal_neuron_adapter.export_expert \
  --manifest "$DATA/train_all.csv" --expert-model "$PRIMARY/expert_best.pth" \
  --out-dir "$PRIMARY/train" --device cuda
python -m universal_neuron_adapter.export_expert \
  --manifest "$DATA/test.csv" --expert-model "$PRIMARY/expert_best.pth" \
  --out-dir "$PRIMARY/test" --device cuda

python -m universal_neuron_adapter.fit_context_student \
  --manifest "$DATA/expert_train.csv" \
  --expert-manifest "$PRIMARY/train/expert_scores.csv" \
  --normality-manifest "$NORMALITY/train/expert3_scores.csv" \
  --normality-model "$NORMALITY/normality_expert.npz" \
  --out-dir "$CONTEXT" --normal-samples 32 --positive-fraction 0.05 \
  --epochs 20 --seed "$SEED" --resume
python -m universal_neuron_adapter.export_context_student \
  --manifest "$DATA/expert_train.csv" --student-model "$CONTEXT/context_student.npz" \
  --normality-model "$NORMALITY/normality_expert.npz" --out-dir "$CONTEXT/train"
python -m universal_neuron_adapter.export_context_student \
  --manifest "$DATA/test.csv" --student-model "$CONTEXT/context_student.npz" \
  --normality-model "$NORMALITY/normality_expert.npz" --out-dir "$CONTEXT/test"

python -m universal_neuron_adapter.cache_baseline \
  --baseline dsanet --baseline-root baseline/DSANet \
  --baseline-weight ../vadmy_data/model/DSANet/model_ucf.pth \
  --dataset ucf --manifest "$DATA/train_all.csv" --split train \
  --training-view-policy fixed --out-dir "$BASE/baseline_train" --device cuda
python -m universal_neuron_adapter.cache_baseline \
  --baseline dsanet --baseline-root baseline/DSANet \
  --baseline-weight ../vadmy_data/model/DSANet/model_ucf.pth \
  --dataset ucf --manifest "$DATA/test.csv" --split test \
  --training-view-policy fixed --out-dir "$BASE/baseline_test" --device cuda

python -m universal_neuron_adapter.train_correction \
  --baseline-manifest "$BASE/baseline_train/baseline_scores.csv" \
  --expert-manifest "$PRIMARY/train/expert_scores.csv" \
  --train-keys "$DATA/expert_train.csv" --val-keys "$DATA/expert_val.csv" \
  --baseline dsanet --dataset ucf --out-dir "$BASE/correction" \
  --loss-protocol conservative-logit-v1 --width 32 --max-epoch 15 \
  --batch-size 32 --lr 0.0003 --weight-decay 0.0001 --maximum-length 256 \
  --num-workers 4 --seed "$SEED" --device cuda --resume

python -m universal_neuron_adapter.evaluate \
  --baseline-train-manifest "$BASE/baseline_train/baseline_scores.csv" \
  --baseline-manifest "$BASE/baseline_test/baseline_scores.csv" \
  --expert-train-manifest "$PRIMARY/train/expert_scores.csv" \
  --expert-manifest "$PRIMARY/test/expert_scores.csv" \
  --expert3-train-manifest "$NORMALITY/train/expert3_scores.csv" \
  --expert3-manifest "$NORMALITY/test/expert3_scores.csv" \
  --student-train-manifest "$CONTEXT/train/student_scores.csv" \
  --student-manifest "$CONTEXT/test/student_scores.csv" \
  --correction-model "$BASE/correction/model_best.pth" \
  --gt-path ../vadmy_data/annotations/ucf/gt.npy \
  --baseline dsanet --dataset ucf --out-dir "$BASE/evaluation" \
  --frames-per-snippet 16 --event-width 41 --event-weight 1.0 \
  --normality-smoothing-blend 0.25 --persistence-weight 1.0 \
  --gaussian-sigma 0.5 --advance-snippets 0.5 --device cuda

echo "done: $BASE/evaluation/metrics.json"
