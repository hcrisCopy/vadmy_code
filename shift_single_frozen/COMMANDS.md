# 运行指令

所有命令从`vadmy_code`根目录运行。

## 分阶段Python指令：DSANet / UCF-Crime

下面是完整正式参数，不需要替换。

### 1. 锁定打分产物身份

```bash
python -m shift_single_frozen.provenance prepare-score \
  --baseline dsanet \
  --dataset ucf \
  --source-train-csv ../vad_data/work_ucf/ucf_train_local.csv \
  --baseline-weight ../vadmy_data/model/DSANet/model_ucf.pth \
  --out-dir ../vadmy_data/shift_single_frozen/ucf/dsanet/pseudo_scores
```

### 2. 使用DSANet自己的分数

```bash
python -m shift_residual_head_tuning.score_baseline \
  --baseline dsanet \
  --baseline-root baseline/DSANet \
  --baseline-weight ../vadmy_data/model/DSANet/model_ucf.pth \
  --dataset ucf \
  --source-train-csv ../vad_data/work_ucf/ucf_train_local.csv \
  --out-dir ../vadmy_data/shift_single_frozen/ucf/dsanet/pseudo_scores \
  --device cuda
```

### 3. 构建top/bottom并选神经元

```bash
python -m shift_residual_head_tuning.select_shift_neurons \
  --dataset ucf \
  --source-train-csv ../vad_data/work_ucf/ucf_train_local.csv \
  --hidden-manifest ../vad_data/work_ucf/clip_hidden_stride16_train_8gpu/manifest.csv \
  --pseudo-csv ../vadmy_data/shift_single_frozen/ucf/dsanet/pseudo_scores/group_scores.csv \
  --out-dir ../vadmy_data/shift_single_frozen/ucf/dsanet/selection \
  --top-p 0.10 \
  --topk-per-layer 64 \
  --normal-stat-snippets-per-video 256 \
  --sigma-min 1e-6

python -m shift_single_frozen.provenance seal-selection \
  --baseline dsanet \
  --dataset ucf \
  --score-provenance ../vadmy_data/shift_single_frozen/ucf/dsanet/pseudo_scores/score_provenance.json \
  --neuron-json ../vadmy_data/shift_single_frozen/ucf/dsanet/selection/selected_neurons.json \
  --out-path ../vadmy_data/shift_single_frozen/ucf/dsanet/selection/selection_provenance.json
```

### 4. 构建训练/测试对齐特征

```bash
python -m shift_residual_head_tuning.build_aligned_features \
  --source-csv ../vad_data/work_ucf/ucf_train_local.csv \
  --hidden-manifest ../vad_data/work_ucf/clip_hidden_stride16_train_8gpu/manifest.csv \
  --neuron-json ../vadmy_data/shift_single_frozen/ucf/dsanet/selection/selected_neurons.json \
  --out-dir ../vadmy_data/shift_single_frozen/ucf/dsanet/aligned/train \
  --out-csv ../vadmy_data/shift_single_frozen/ucf/dsanet/aligned_train.csv

python -m shift_residual_head_tuning.build_aligned_features \
  --source-csv ../vad_data/work_ucf/ucf_test_local.csv \
  --hidden-manifest ../vad_data/work_ucf/clip_hidden_stride16_test_8gpu/manifest.csv \
  --neuron-json ../vadmy_data/shift_single_frozen/ucf/dsanet/selection/selected_neurons.json \
  --out-dir ../vadmy_data/shift_single_frozen/ucf/dsanet/aligned/test \
  --out-csv ../vadmy_data/shift_single_frozen/ucf/dsanet/aligned_test.csv
```

### 5. 训练单残差

```bash
python -m shift_single_frozen.train \
  --baseline dsanet \
  --baseline-root baseline/DSANet \
  --baseline-weight ../vadmy_data/model/DSANet/model_ucf.pth \
  --dataset ucf \
  --train-list ../vadmy_data/shift_single_frozen/ucf/dsanet/aligned_train.csv \
  --val-list ../vadmy_data/shift_single_frozen/ucf/dsanet/aligned_test.csv \
  --gt-path ../vadmy_data/annotations/ucf/gt.npy \
  --neuron-json ../vadmy_data/shift_single_frozen/ucf/dsanet/selection/selected_neurons.json \
  --selection-provenance ../vadmy_data/shift_single_frozen/ucf/dsanet/selection/selection_provenance.json \
  --out-dir ../vadmy_data/shift_single_frozen/ucf/dsanet/training \
  --max-epoch 10 \
  --batch-size 64 \
  --lr 7e-5 \
  --weight-decay 0 \
  --residual-hidden-width 1024 \
  --residual-depth 3 \
  --frames-per-snippet 16 \
  --dsanet-ucf-eval-samples 1280 \
  --num-workers 4 \
  --seed 234 \
  --device cuda
```

中断后在这条训练指令末尾增加`--resume`；需要清空本次训练时增加`--clean`，二者不能同时使用。

### 6. 评估与可视化

```bash
python -m shift_single_frozen.evaluate \
  --baseline dsanet \
  --baseline-root baseline/DSANet \
  --baseline-weight ../vadmy_data/model/DSANet/model_ucf.pth \
  --dataset ucf \
  --test-list ../vadmy_data/shift_single_frozen/ucf/dsanet/aligned_test.csv \
  --model-path ../vadmy_data/shift_single_frozen/ucf/dsanet/training/model_best.pth \
  --gt-path ../vadmy_data/annotations/ucf/gt.npy \
  --gt-segment-path ../vadmy_data/annotations/ucf/gt_segment.npy \
  --gt-label-path ../vadmy_data/annotations/ucf/gt_label.npy \
  --out-dir ../vadmy_data/shift_single_frozen/ucf/dsanet/evaluation \
  --frames-per-snippet 16 \
  --temperature 0 \
  --device cuda

python -m shift_single_frozen.visualize_diagnostics \
  --selection-dir ../vadmy_data/shift_single_frozen/ucf/dsanet/selection \
  --training-dir ../vadmy_data/shift_single_frozen/ucf/dsanet/training \
  --out-dir ../vadmy_data/shift_single_frozen/ucf/dsanet/diagnostics
```

## 六个一键运行指令

每条都包含打分、探测、对齐、训练、评估和可视化；中断后重跑会复用完成产物并自动续训。

```bash
conda activate dsanet
bash shift_single_frozen/commands/run_dsanet_ucf.sh
```

```bash
conda activate dsanet
bash shift_single_frozen/commands/run_dsanet_xd.sh
```

```bash
conda activate dsanet
bash shift_single_frozen/commands/run_desc_ucf.sh
```

```bash
conda activate dsanet
bash shift_single_frozen/commands/run_desc_xd.sh
```

```bash
conda activate dsanet
bash shift_single_frozen/commands/run_lagovad_ucf.sh
```

```bash
conda activate dsanet
bash shift_single_frozen/commands/run_lagovad_xd.sh
```
