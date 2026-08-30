#!/usr/bin/env bash
# LaGoVAD + Universal CLS-neuron Adapter 一键运行（UCF + XD）
# 用法：在 vadmy_code/ 目录下执行  bash run_instructions/run_lagovad.sh
# 说明：可重复执行；fit_* 均带 --resume，中断后重跑即可续跑。
set -euo pipefail
export TOKENIZERS_PARALLELISM=false
export CUBLAS_WORKSPACE_CONFIG=:4096:8
# Formal runs use the CLIP weights already cached on the server.
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"

SEED=234
EXPERIMENT_NAME="${EXPERIMENT_NAME:-formal_seed234}"
ROOT="../vadmy_data/universal_neuron_adapter"
OUT="$ROOT/runs/$EXPERIMENT_NAME"
python -m universal_neuron_adapter.experiment \
  --name "$EXPERIMENT_NAME" --out-dir "$OUT" --seed "$SEED"
printf '%s\n' "$EXPERIMENT_NAME" > "$ROOT/current_run.txt"
exec > >(tee -a "$OUT/run_lagovad.log") 2>&1
echo "run output root: $OUT"

# ---------- 1. 数据清单 + 划分审计 ----------
python -m universal_neuron_adapter.data --dataset ucf \
  --train-csv ../vad_data/work_ucf/ucf_train_local.csv \
  --test-csv  ../vad_data/work_ucf/ucf_test_local.csv \
  --train-hidden-manifest ../vad_data/work_ucf/clip_hidden_stride16_train_8gpu/manifest.csv \
  --test-hidden-manifest  ../vad_data/work_ucf/clip_hidden_stride16_test_8gpu/manifest.csv \
  --out-dir "$OUT/ucf/data" --seed 234 --val-fraction 0.2
python -m universal_neuron_adapter.data --dataset xd \
  --train-csv ../vad_data/work_xd/xd_train_local.csv \
  --test-csv  ../vad_data/work_xd/xd_test_local.csv \
  --train-hidden-manifest ../vad_data/work_xd/clip_hidden_stride16_train_8gpu/manifest.csv \
  --test-hidden-manifest  ../vad_data/work_xd/clip_hidden_stride16_test_8gpu/manifest.csv \
  --out-dir "$OUT/xd/data" --seed 234 --val-fraction 0.2 --skip-missing-hidden

for dataset in ucf xd; do
  data="$OUT/$dataset/data"
  normality="$OUT/$dataset/normality_expert"
  context="$OUT/$dataset/context_student"
  primary="$OUT/$dataset/expert"
  base="$OUT/$dataset/lagovad"

  # ---------- 2. 方向性常态专家（共享） ----------
  python -m universal_neuron_adapter.fit_normality_expert \
    --manifest "$data/expert_train.csv" --out-dir "$normality" \
    --active-per-layer 32 --maximum-length 256 --resume
  python -m universal_neuron_adapter.export_normality_expert \
    --manifest "$data/test.csv" --expert-model "$normality/normality_expert.npz" --out-dir "$normality/test"
  python -m universal_neuron_adapter.export_normality_expert \
    --manifest "$data/expert_train.csv" --expert-model "$normality/normality_expert.npz" --out-dir "$normality/train"

  # ---------- 3. 主稀疏专家（共享） ----------
  python -m universal_neuron_adapter.train_expert \
    --train-manifest "$data/expert_train.csv" --val-manifest "$data/expert_val.csv" \
    --out-dir "$primary" --active-per-layer 32 --temporal-width 64 --max-epoch 20 --batch-size 8 \
    --lr 0.0003 --weight-decay 0.0001 --sparsity-weight 0.001 \
    --maximum-length 256 --num-workers 4 --seed 234 --device cuda --resume
  python -m universal_neuron_adapter.export_expert \
    --manifest "$data/train_all.csv" --expert-model "$primary/expert_best.pth" --out-dir "$primary/train" --device cuda
  python -m universal_neuron_adapter.export_expert \
    --manifest "$data/test.csv" --expert-model "$primary/expert_best.pth" --out-dir "$primary/test" --device cuda

  # ---------- 4. 多尺度上下文学生（共享） ----------
  python -m universal_neuron_adapter.fit_context_student \
    --manifest "$data/expert_train.csv" \
    --expert-manifest "$primary/train/expert_scores.csv" \
    --normality-manifest "$normality/train/expert3_scores.csv" \
    --normality-model "$normality/normality_expert.npz" \
    --out-dir "$context" --normal-samples 32 --positive-fraction 0.05 \
    --epochs 20 --seed 234 --resume
  python -m universal_neuron_adapter.export_context_student \
    --manifest "$data/test.csv" --student-model "$context/context_student.npz" \
    --normality-model "$normality/normality_expert.npz" --out-dir "$context/test"
  python -m universal_neuron_adapter.export_context_student \
    --manifest "$data/expert_train.csv" --student-model "$context/context_student.npz" \
    --normality-model "$normality/normality_expert.npz" --out-dir "$context/train"

  # ---------- 5. 缓存 LaGoVAD 冻结分数 ----------
  python -m universal_neuron_adapter.cache_baseline \
    --baseline lagovad --baseline-root baseline/LaGoVAD-PreVAD \
    --baseline-weight "../vadmy_data/model/LaGoVAD/best.ckpt" \
    --dataset "$dataset" --manifest "$data/train_all.csv" --split train \
    --out-dir "$base/baseline_train" --device cuda
  python -m universal_neuron_adapter.cache_baseline \
    --baseline lagovad --baseline-root baseline/LaGoVAD-PreVAD \
    --baseline-weight "../vadmy_data/model/LaGoVAD/best.ckpt" \
    --dataset "$dataset" --manifest "$data/test.csv" --split test \
    --out-dir "$base/baseline_test" --device cuda

  # ---------- 6. 训练校正头 ----------
  python -m universal_neuron_adapter.train_correction \
    --baseline-manifest "$base/baseline_train/baseline_scores.csv" \
    --expert-manifest "$primary/train/expert_scores.csv" \
    --train-keys "$data/expert_train.csv" --val-keys "$data/expert_val.csv" \
    --baseline lagovad --dataset "$dataset" --out-dir "$base/correction" \
    --width 32 --max-epoch 15 --batch-size 32 --lr 0.0003 --weight-decay 0.0001 \
    --maximum-length 256 --num-workers 4 --seed 234 --device cuda --resume

  # ---------- 7. 正式评测 ----------
  python -m universal_neuron_adapter.evaluate \
    --baseline-train-manifest "$base/baseline_train/baseline_scores.csv" \
    --baseline-manifest     "$base/baseline_test/baseline_scores.csv" \
    --expert-train-manifest "$primary/train/expert_scores.csv" \
    --expert-manifest       "$primary/test/expert_scores.csv" \
    --expert3-manifest      "$normality/test/expert3_scores.csv" \
    --expert3-train-manifest "$normality/train/expert3_scores.csv" \
    --student-manifest      "$context/test/student_scores.csv" \
    --student-train-manifest "$context/train/student_scores.csv" \
    --correction-model "$base/correction/model_best.pth" \
    --gt-path "../vadmy_data/annotations/$dataset/gt.npy" \
    --baseline lagovad --dataset "$dataset" --out-dir "$base/evaluation" \
    --frames-per-snippet 16 --event-width 41 --event-weight 1.0 \
    --normality-smoothing-blend 0.25 --persistence-weight 1.0 \
    --gaussian-sigma 0.5 --advance-snippets 0.5 --device cuda
done

python -m universal_neuron_adapter.aggregate_metric --results-root ../vadmy_data/universal_neuron_adapter
echo "done: $OUT"
