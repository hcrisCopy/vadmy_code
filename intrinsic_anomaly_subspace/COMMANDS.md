# 正式运行指令

以下命令全部从 `vadmy_code` 根目录运行，参数已写全，不需要手动替换。首次运行保留 `--clean`；中断续跑时删除对应阶段的 `--clean`，线性分类训练再添加 `--resume`。

## UCF-Crime

先确认复用产物存在：

```bash
for f in \
  ../vad_data/work_ucf/ucf_train_local.csv \
  ../vad_data/work_ucf/ucf_test_local.csv \
  ../vad_data/work_ucf/clip_hidden_stride16_train_8gpu/manifest.csv \
  ../vad_data/work_ucf/clip_hidden_stride16_test_8gpu/manifest.csv \
  ../vad_data/work_ucf/pseudo_8gpu/group_scores.csv \
  ../vadmy_data/annotations/ucf/gt.npy; do
  test -f "$f" || { echo "MISSING: $f"; exit 1; }
done
echo UCF_INPUTS_OK
```

### 1. 构建 shift 正负样本

```bash
python -m intrinsic_anomaly_subspace.build_shift_pairs \
  --dataset ucf \
  --source-train-csv ../vad_data/work_ucf/ucf_train_local.csv \
  --hidden-manifest ../vad_data/work_ucf/clip_hidden_stride16_train_8gpu/manifest.csv \
  --pseudo-csv ../vad_data/work_ucf/pseudo_8gpu/group_scores.csv \
  --out-dir ../vadmy_data/intrinsic_anomaly_subspace/ucf/pairs \
  --top-p 0.10 \
  --discovery-fraction 0.40 \
  --validation-fraction 0.20 \
  --seed 234 \
  --clean
```

输出：`../vadmy_data/intrinsic_anomaly_subspace/ucf/pairs/pairs.csv` 和逐视频正负 hidden 缓存。

### 2. V-FIND 式选层和神经元

```bash
python -m intrinsic_anomaly_subspace.discover_subspace \
  --pair-manifest ../vadmy_data/intrinsic_anomaly_subspace/ucf/pairs/pairs.csv \
  --out-dir ../vadmy_data/intrinsic_anomaly_subspace/ucf/discovery \
  --layer-rule threshold_union \
  --effect-threshold 1.5 \
  --probe-epochs 300 \
  --probe-lr 1e-2 \
  --weight-decay 1e-4 \
  --checkpoint-interval 10 \
  --epsilon 1e-8 \
  --seed 234 \
  --device cuda \
  --clean
```

输出：`discovery/selected_subspace.json`、`layer_metrics.csv`、`neuron_metrics.csv` 和可恢复的逐层 probe。

### 3. 训练 selected 和两个等宽随机对照

```bash
python -m intrinsic_anomaly_subspace.train_linear_readout \
  --pair-manifest ../vadmy_data/intrinsic_anomaly_subspace/ucf/pairs/pairs.csv \
  --subspace-json ../vadmy_data/intrinsic_anomaly_subspace/ucf/discovery/selected_subspace.json \
  --feature-mode selected \
  --out-dir ../vadmy_data/intrinsic_anomaly_subspace/ucf/readout/selected \
  --max-epoch 200 \
  --batch-size 1024 \
  --lr 1e-2 \
  --weight-decay 1e-4 \
  --selection-metric ap \
  --seed 234 \
  --device cuda \
  --clean

python -m intrinsic_anomaly_subspace.train_linear_readout \
  --pair-manifest ../vadmy_data/intrinsic_anomaly_subspace/ucf/pairs/pairs.csv \
  --subspace-json ../vadmy_data/intrinsic_anomaly_subspace/ucf/discovery/selected_subspace.json \
  --feature-mode same_layer_random \
  --out-dir ../vadmy_data/intrinsic_anomaly_subspace/ucf/readout/same_layer_random \
  --max-epoch 200 \
  --batch-size 1024 \
  --lr 1e-2 \
  --weight-decay 1e-4 \
  --selection-metric ap \
  --seed 234 \
  --device cuda \
  --clean

python -m intrinsic_anomaly_subspace.train_linear_readout \
  --pair-manifest ../vadmy_data/intrinsic_anomaly_subspace/ucf/pairs/pairs.csv \
  --subspace-json ../vadmy_data/intrinsic_anomaly_subspace/ucf/discovery/selected_subspace.json \
  --feature-mode global_random \
  --out-dir ../vadmy_data/intrinsic_anomaly_subspace/ucf/readout/global_random \
  --max-epoch 200 \
  --batch-size 1024 \
  --lr 1e-2 \
  --weight-decay 1e-4 \
  --selection-metric ap \
  --seed 234 \
  --device cuda \
  --clean
```

### 4. 严格帧级评测

```bash
python -m intrinsic_anomaly_subspace.evaluate_frame \
  --dataset ucf \
  --test-list ../vad_data/work_ucf/ucf_test_local.csv \
  --hidden-manifest ../vad_data/work_ucf/clip_hidden_stride16_test_8gpu/manifest.csv \
  --model-path ../vadmy_data/intrinsic_anomaly_subspace/ucf/readout/selected/model_best.pth \
  --gt-path ../vadmy_data/annotations/ucf/gt.npy \
  --out-dir ../vadmy_data/intrinsic_anomaly_subspace/ucf/evaluation/selected \
  --frames-per-snippet 16 \
  --batch-size 4096 \
  --device cuda \
  --clean

python -m intrinsic_anomaly_subspace.evaluate_frame \
  --dataset ucf \
  --test-list ../vad_data/work_ucf/ucf_test_local.csv \
  --hidden-manifest ../vad_data/work_ucf/clip_hidden_stride16_test_8gpu/manifest.csv \
  --model-path ../vadmy_data/intrinsic_anomaly_subspace/ucf/readout/same_layer_random/model_best.pth \
  --gt-path ../vadmy_data/annotations/ucf/gt.npy \
  --out-dir ../vadmy_data/intrinsic_anomaly_subspace/ucf/evaluation/same_layer_random \
  --frames-per-snippet 16 \
  --batch-size 4096 \
  --device cuda \
  --clean

python -m intrinsic_anomaly_subspace.evaluate_frame \
  --dataset ucf \
  --test-list ../vad_data/work_ucf/ucf_test_local.csv \
  --hidden-manifest ../vad_data/work_ucf/clip_hidden_stride16_test_8gpu/manifest.csv \
  --model-path ../vadmy_data/intrinsic_anomaly_subspace/ucf/readout/global_random/model_best.pth \
  --gt-path ../vadmy_data/annotations/ucf/gt.npy \
  --out-dir ../vadmy_data/intrinsic_anomaly_subspace/ucf/evaluation/global_random \
  --frames-per-snippet 16 \
  --batch-size 4096 \
  --device cuda \
  --clean
```

默认要求预测与 GT 长度完全一致。不要为了跑通随便添加 `--allow-length-crop`；只有检查确认是官方尾帧舍入差异后才允许使用。

### 5. 证据图

```bash
python -m intrinsic_anomaly_subspace.visualize_diagnostics \
  --dataset ucf \
  --discovery-dir ../vadmy_data/intrinsic_anomaly_subspace/ucf/discovery \
  --selected-eval-dir ../vadmy_data/intrinsic_anomaly_subspace/ucf/evaluation/selected \
  --same-layer-random-eval-dir ../vadmy_data/intrinsic_anomaly_subspace/ucf/evaluation/same_layer_random \
  --global-random-eval-dir ../vadmy_data/intrinsic_anomaly_subspace/ucf/evaluation/global_random \
  --out-dir ../vadmy_data/intrinsic_anomaly_subspace/ucf/diagnostics \
  --timeline-examples 4 \
  --clean
```

最终重点看 `evaluation/selected/metrics.json` 和 `diagnostics/diagnostic_summary.json`。

## XD-Violence

先确认复用产物存在：

```bash
for f in \
  ../vad_data/work_xd/xd_train_local.csv \
  ../vad_data/work_xd/xd_test_local.csv \
  ../vad_data/work_xd/clip_hidden_stride16_train_8gpu/manifest.csv \
  ../vad_data/work_xd/clip_hidden_stride16_test_8gpu/manifest.csv \
  ../vad_data/work_xd/pseudo_8gpu/group_scores.csv \
  ../vadmy_data/annotations/xd/gt.npy; do
  test -f "$f" || { echo "MISSING: $f"; exit 1; }
done
echo XD_INPUTS_OK
```

### 1. 构建 shift 正负样本

```bash
python -m intrinsic_anomaly_subspace.build_shift_pairs \
  --dataset xd \
  --source-train-csv ../vad_data/work_xd/xd_train_local.csv \
  --hidden-manifest ../vad_data/work_xd/clip_hidden_stride16_train_8gpu/manifest.csv \
  --pseudo-csv ../vad_data/work_xd/pseudo_8gpu/group_scores.csv \
  --out-dir ../vadmy_data/intrinsic_anomaly_subspace/xd/pairs \
  --top-p 0.10 \
  --discovery-fraction 0.40 \
  --validation-fraction 0.20 \
  --seed 234 \
  --clean
```

缺失 hidden 的 4 个训练视频会写入 `pairs/skipped_videos.csv`，其余视频继续运行。

### 2. V-FIND 式选层和神经元

```bash
python -m intrinsic_anomaly_subspace.discover_subspace \
  --pair-manifest ../vadmy_data/intrinsic_anomaly_subspace/xd/pairs/pairs.csv \
  --out-dir ../vadmy_data/intrinsic_anomaly_subspace/xd/discovery \
  --layer-rule threshold_union \
  --effect-threshold 1.5 \
  --probe-epochs 300 \
  --probe-lr 1e-2 \
  --weight-decay 1e-4 \
  --checkpoint-interval 10 \
  --epsilon 1e-8 \
  --seed 234 \
  --device cuda \
  --clean
```

### 3. 训练 selected 和两个等宽随机对照

```bash
python -m intrinsic_anomaly_subspace.train_linear_readout \
  --pair-manifest ../vadmy_data/intrinsic_anomaly_subspace/xd/pairs/pairs.csv \
  --subspace-json ../vadmy_data/intrinsic_anomaly_subspace/xd/discovery/selected_subspace.json \
  --feature-mode selected \
  --out-dir ../vadmy_data/intrinsic_anomaly_subspace/xd/readout/selected \
  --max-epoch 200 \
  --batch-size 1024 \
  --lr 1e-2 \
  --weight-decay 1e-4 \
  --selection-metric ap \
  --seed 234 \
  --device cuda \
  --clean

python -m intrinsic_anomaly_subspace.train_linear_readout \
  --pair-manifest ../vadmy_data/intrinsic_anomaly_subspace/xd/pairs/pairs.csv \
  --subspace-json ../vadmy_data/intrinsic_anomaly_subspace/xd/discovery/selected_subspace.json \
  --feature-mode same_layer_random \
  --out-dir ../vadmy_data/intrinsic_anomaly_subspace/xd/readout/same_layer_random \
  --max-epoch 200 \
  --batch-size 1024 \
  --lr 1e-2 \
  --weight-decay 1e-4 \
  --selection-metric ap \
  --seed 234 \
  --device cuda \
  --clean

python -m intrinsic_anomaly_subspace.train_linear_readout \
  --pair-manifest ../vadmy_data/intrinsic_anomaly_subspace/xd/pairs/pairs.csv \
  --subspace-json ../vadmy_data/intrinsic_anomaly_subspace/xd/discovery/selected_subspace.json \
  --feature-mode global_random \
  --out-dir ../vadmy_data/intrinsic_anomaly_subspace/xd/readout/global_random \
  --max-epoch 200 \
  --batch-size 1024 \
  --lr 1e-2 \
  --weight-decay 1e-4 \
  --selection-metric ap \
  --seed 234 \
  --device cuda \
  --clean
```

### 4. 严格帧级评测

```bash
python -m intrinsic_anomaly_subspace.evaluate_frame \
  --dataset xd \
  --test-list ../vad_data/work_xd/xd_test_local.csv \
  --hidden-manifest ../vad_data/work_xd/clip_hidden_stride16_test_8gpu/manifest.csv \
  --model-path ../vadmy_data/intrinsic_anomaly_subspace/xd/readout/selected/model_best.pth \
  --gt-path ../vadmy_data/annotations/xd/gt.npy \
  --out-dir ../vadmy_data/intrinsic_anomaly_subspace/xd/evaluation/selected \
  --frames-per-snippet 16 \
  --batch-size 4096 \
  --device cuda \
  --clean

python -m intrinsic_anomaly_subspace.evaluate_frame \
  --dataset xd \
  --test-list ../vad_data/work_xd/xd_test_local.csv \
  --hidden-manifest ../vad_data/work_xd/clip_hidden_stride16_test_8gpu/manifest.csv \
  --model-path ../vadmy_data/intrinsic_anomaly_subspace/xd/readout/same_layer_random/model_best.pth \
  --gt-path ../vadmy_data/annotations/xd/gt.npy \
  --out-dir ../vadmy_data/intrinsic_anomaly_subspace/xd/evaluation/same_layer_random \
  --frames-per-snippet 16 \
  --batch-size 4096 \
  --device cuda \
  --clean

python -m intrinsic_anomaly_subspace.evaluate_frame \
  --dataset xd \
  --test-list ../vad_data/work_xd/xd_test_local.csv \
  --hidden-manifest ../vad_data/work_xd/clip_hidden_stride16_test_8gpu/manifest.csv \
  --model-path ../vadmy_data/intrinsic_anomaly_subspace/xd/readout/global_random/model_best.pth \
  --gt-path ../vadmy_data/annotations/xd/gt.npy \
  --out-dir ../vadmy_data/intrinsic_anomaly_subspace/xd/evaluation/global_random \
  --frames-per-snippet 16 \
  --batch-size 4096 \
  --device cuda \
  --clean
```

### 5. 证据图

```bash
python -m intrinsic_anomaly_subspace.visualize_diagnostics \
  --dataset xd \
  --discovery-dir ../vadmy_data/intrinsic_anomaly_subspace/xd/discovery \
  --selected-eval-dir ../vadmy_data/intrinsic_anomaly_subspace/xd/evaluation/selected \
  --same-layer-random-eval-dir ../vadmy_data/intrinsic_anomaly_subspace/xd/evaluation/same_layer_random \
  --global-random-eval-dir ../vadmy_data/intrinsic_anomaly_subspace/xd/evaluation/global_random \
  --out-dir ../vadmy_data/intrinsic_anomaly_subspace/xd/diagnostics \
  --timeline-examples 4 \
  --clean
```

最终重点看 `evaluation/selected/metrics.json` 和 `diagnostics/diagnostic_summary.json`。

## 开销和冻结范围

- CLIP：不加载、不前向、不训练，直接读取现有 hidden。
- 三个 baseline：只在已有 pseudo score 中作为正负排序来源；本实验不加载或训练它们。
- Discovery：每个关键层一个 `768 -> 1` 线性 probe。
- Readout：`K -> 1`，其中 K 由 `d_n >= 1.5` 自动决定。
- 显存：通常远低于 1 GB；主要开销是 CPU 读取/解压 hidden NPZ 和逐视频 pair 缓存占用的磁盘空间。
