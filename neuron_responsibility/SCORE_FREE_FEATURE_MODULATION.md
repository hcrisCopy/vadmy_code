# 无 Baseline 分数的神经元特征调制

以下命令都从 `vadmy_code` 目录执行，参数和路径已经写全。该方案不使用 baseline 分数选择神经元、生成伪标签或划分责任区域，也不需要旧 Stage A probe。

输入继续只读复用 `../vad_data` 中的 CLIP 512D 特征与 hidden states；所有新产物写入 `../vadmy_data/neuron_responsibility`。

## 1. 复用神经元选择和对齐特征

先按 `neuron_responsibility/README.md` 的第 1、2 节生成或复用以下产物：

```text
../vadmy_data/neuron_responsibility/ucf/selection/selected_neurons.json
../vadmy_data/neuron_responsibility/ucf/aligned_train.csv
../vadmy_data/neuron_responsibility/ucf/aligned_test.csv
../vadmy_data/neuron_responsibility/xd/selection/selected_neurons.json
../vadmy_data/neuron_responsibility/xd/aligned_train.csv
../vadmy_data/neuron_responsibility/xd/aligned_test.csv
```

## 2. 正常视频证据校准

### UCF-Crime

```bash
python neuron_responsibility/calibrate_evidence.py \
  --dataset ucf \
  --train-list ../vadmy_data/neuron_responsibility/ucf/aligned_train.csv \
  --neuron-json ../vadmy_data/neuron_responsibility/ucf/selection/selected_neurons.json \
  --out-dir ../vadmy_data/neuron_responsibility/ucf/evidence_v1 \
  --active-neurons 128 \
  --normal-quantile 0.99 \
  --snippets-per-video 64
```

### XD-Violence

```bash
python neuron_responsibility/calibrate_evidence.py \
  --dataset xd \
  --train-list ../vadmy_data/neuron_responsibility/xd/aligned_train.csv \
  --neuron-json ../vadmy_data/neuron_responsibility/xd/selection/selected_neurons.json \
  --out-dir ../vadmy_data/neuron_responsibility/xd/evidence_v1 \
  --active-neurons 128 \
  --normal-quantile 0.99 \
  --snippets-per-video 64
```

输出为 `evidence_config.json`、`evidence_config.npz` 和 `calibration_signature.json`。重复执行会校验并复用；输入变化时使用同一命令并添加 `--clean`。

## 3. 训练

训练分三段：前 2 epoch 只训练零初始化神经元调制器，第 3-4 epoch 加入预测头，第 5 epoch 起再加入最后时序精炼块。CLIP 始终冻结。UCF 按帧级 AUC 选模，XD 按帧级 AP 选模。

### DSANet / UCF-Crime

```bash
python neuron_responsibility/train_feature_modulation.py \
  --baseline dsanet \
  --dataset ucf \
  --baseline-root baseline/DSANet \
  --baseline-weight ../vadmy_data/model/DSANet/model_ucf.pth \
  --train-list ../vadmy_data/neuron_responsibility/ucf/aligned_train.csv \
  --val-list ../vadmy_data/neuron_responsibility/ucf/aligned_test.csv \
  --gt-path ../vadmy_data/annotations/ucf/gt.npy \
  --evidence-config ../vadmy_data/neuron_responsibility/ucf/evidence_v1/evidence_config.npz \
  --out-dir ../vadmy_data/neuron_responsibility/ucf/dsanet_feature_modulation_v1 \
  --max-epoch 10 \
  --head-start-epoch 2 \
  --temporal-start-epoch 4 \
  --batch-size 64 \
  --adapter-lr 5e-5 \
  --head-lr 1e-5 \
  --temporal-lr 1e-6 \
  --weight-decay 0 \
  --auxiliary-weight 0.1 \
  --normal-weight 0.1 \
  --smooth-weight 0.01 \
  --sparse-weight 0.001 \
  --anchor-weight 0.01 \
  --context-width 32 \
  --temporal-kernel 5 \
  --evidence-cap 6.0 \
  --frames-per-snippet 16 \
  --dsanet-ucf-eval-samples 1280 \
  --num-workers 4 \
  --seed 234 \
  --device cuda
```

DSANet/UCF 严格使用作者 normal/abnormal 成对 batch；这里 `--batch-size 64` 表示每类 64，拼接后每步 128 个样本，并保持每 1280 个训练样本验证一次。

### DeSC / UCF-Crime

```bash
python neuron_responsibility/train_feature_modulation.py \
  --baseline desc \
  --dataset ucf \
  --baseline-root baseline/DeSC \
  --sensitivity-weight ../vadmy_data/model/DeSC/ucf_sensitivity_stream.pth \
  --consistency-weight ../vadmy_data/model/DeSC/ucf_consistency_stream.pth \
  --train-list ../vadmy_data/neuron_responsibility/ucf/aligned_train.csv \
  --val-list ../vadmy_data/neuron_responsibility/ucf/aligned_test.csv \
  --gt-path ../vadmy_data/annotations/ucf/gt.npy \
  --evidence-config ../vadmy_data/neuron_responsibility/ucf/evidence_v1/evidence_config.npz \
  --out-dir ../vadmy_data/neuron_responsibility/ucf/desc_feature_modulation_v1 \
  --max-epoch 10 --head-start-epoch 2 --temporal-start-epoch 4 \
  --batch-size 64 --adapter-lr 5e-5 --head-lr 1e-5 --temporal-lr 1e-6 \
  --weight-decay 0 --auxiliary-weight 0.1 --normal-weight 0.1 \
  --smooth-weight 0.01 --sparse-weight 0.001 --anchor-weight 0.01 \
  --context-width 32 --temporal-kernel 5 --evidence-cap 6.0 \
  --frames-per-snippet 16 --dsanet-ucf-eval-samples 1280 \
  --num-workers 4 --seed 234 --device cuda
```

### LaGoVAD / UCF-Crime

```bash
python neuron_responsibility/train_feature_modulation.py \
  --baseline lagovad \
  --dataset ucf \
  --baseline-root baseline/LaGoVAD-PreVAD \
  --baseline-weight ../vadmy_data/model/LaGoVAD/best.ckpt \
  --train-list ../vadmy_data/neuron_responsibility/ucf/aligned_train.csv \
  --val-list ../vadmy_data/neuron_responsibility/ucf/aligned_test.csv \
  --gt-path ../vadmy_data/annotations/ucf/gt.npy \
  --evidence-config ../vadmy_data/neuron_responsibility/ucf/evidence_v1/evidence_config.npz \
  --out-dir ../vadmy_data/neuron_responsibility/ucf/lagovad_feature_modulation_v1 \
  --max-epoch 20 --head-start-epoch 2 --temporal-start-epoch 4 \
  --batch-size 64 --adapter-lr 5e-5 --head-lr 1e-5 --temporal-lr 1e-6 \
  --weight-decay 0 --auxiliary-weight 0.1 --normal-weight 0.1 \
  --smooth-weight 0.01 --sparse-weight 0.001 --anchor-weight 0.01 \
  --context-width 32 --temporal-kernel 5 --evidence-cap 6.0 \
  --frames-per-snippet 16 --dsanet-ucf-eval-samples 1280 \
  --num-workers 4 --seed 234 --device cuda
```

### DSANet / XD-Violence

```bash
python neuron_responsibility/train_feature_modulation.py \
  --baseline dsanet \
  --dataset xd \
  --baseline-root baseline/DSANet \
  --baseline-weight ../vadmy_data/model/DSANet/model_xd.pth \
  --train-list ../vadmy_data/neuron_responsibility/xd/aligned_train.csv \
  --val-list ../vadmy_data/neuron_responsibility/xd/aligned_test.csv \
  --gt-path ../vadmy_data/annotations/xd/gt.npy \
  --evidence-config ../vadmy_data/neuron_responsibility/xd/evidence_v1/evidence_config.npz \
  --out-dir ../vadmy_data/neuron_responsibility/xd/dsanet_feature_modulation_v1 \
  --max-epoch 10 --head-start-epoch 2 --temporal-start-epoch 4 \
  --batch-size 96 --adapter-lr 5e-5 --head-lr 1e-5 --temporal-lr 1e-6 \
  --weight-decay 0 --auxiliary-weight 0.1 --normal-weight 0.1 \
  --smooth-weight 0.01 --sparse-weight 0.001 --anchor-weight 0.01 \
  --context-width 32 --temporal-kernel 5 --evidence-cap 6.0 \
  --frames-per-snippet 16 --dsanet-ucf-eval-samples 1280 \
  --num-workers 4 --seed 234 --device cuda
```

### DeSC / XD-Violence

```bash
python neuron_responsibility/train_feature_modulation.py \
  --baseline desc \
  --dataset xd \
  --baseline-root baseline/DeSC \
  --sensitivity-weight ../vadmy_data/model/DeSC/xd_sensitivity_stream.pth \
  --consistency-weight ../vadmy_data/model/DeSC/xd_consistency_stream.pth \
  --train-list ../vadmy_data/neuron_responsibility/xd/aligned_train.csv \
  --val-list ../vadmy_data/neuron_responsibility/xd/aligned_test.csv \
  --gt-path ../vadmy_data/annotations/xd/gt.npy \
  --evidence-config ../vadmy_data/neuron_responsibility/xd/evidence_v1/evidence_config.npz \
  --out-dir ../vadmy_data/neuron_responsibility/xd/desc_feature_modulation_v1 \
  --max-epoch 10 --head-start-epoch 2 --temporal-start-epoch 4 \
  --batch-size 96 --adapter-lr 5e-5 --head-lr 1e-5 --temporal-lr 1e-6 \
  --weight-decay 0 --auxiliary-weight 0.1 --normal-weight 0.1 \
  --smooth-weight 0.01 --sparse-weight 0.001 --anchor-weight 0.01 \
  --context-width 32 --temporal-kernel 5 --evidence-cap 6.0 \
  --frames-per-snippet 16 --dsanet-ucf-eval-samples 1280 \
  --num-workers 4 --seed 234 --device cuda
```

### LaGoVAD / XD-Violence

```bash
python neuron_responsibility/train_feature_modulation.py \
  --baseline lagovad \
  --dataset xd \
  --baseline-root baseline/LaGoVAD-PreVAD \
  --baseline-weight ../vadmy_data/model/LaGoVAD/best.ckpt \
  --train-list ../vadmy_data/neuron_responsibility/xd/aligned_train.csv \
  --val-list ../vadmy_data/neuron_responsibility/xd/aligned_test.csv \
  --gt-path ../vadmy_data/annotations/xd/gt.npy \
  --evidence-config ../vadmy_data/neuron_responsibility/xd/evidence_v1/evidence_config.npz \
  --out-dir ../vadmy_data/neuron_responsibility/xd/lagovad_feature_modulation_v1 \
  --max-epoch 20 --head-start-epoch 2 --temporal-start-epoch 4 \
  --batch-size 64 --adapter-lr 5e-5 --head-lr 1e-5 --temporal-lr 1e-6 \
  --weight-decay 0 --auxiliary-weight 0.1 --normal-weight 0.1 \
  --smooth-weight 0.01 --sparse-weight 0.001 --anchor-weight 0.01 \
  --context-width 32 --temporal-kernel 5 --evidence-cap 6.0 \
  --frames-per-snippet 16 --dsanet-ucf-eval-samples 1280 \
  --num-workers 4 --seed 234 --device cuda
```

训练输出为 `parameter_report.json`、`history.jsonl`、`checkpoint_last.pth` 和 `model_best.pth`。中断后使用完全相同的命令并添加 `--resume`；需要删除旧实验时添加 `--clean`。

## 4. DSANet / UCF-Crime 正式评测

```bash
python neuron_responsibility/evaluate_feature_modulation.py \
  --baseline dsanet \
  --dataset ucf \
  --baseline-root baseline/DSANet \
  --baseline-weight ../vadmy_data/model/DSANet/model_ucf.pth \
  --joint-model ../vadmy_data/neuron_responsibility/ucf/dsanet_feature_modulation_v1/model_best.pth \
  --test-list ../vadmy_data/neuron_responsibility/ucf/aligned_test.csv \
  --gt-path ../vadmy_data/annotations/ucf/gt.npy \
  --gt-segment-path ../vadmy_data/annotations/ucf/gt_segment.npy \
  --gt-label-path ../vadmy_data/annotations/ucf/gt_label.npy \
  --out-dir ../vadmy_data/neuron_responsibility/ucf/dsanet_feature_modulation_v1/evaluation \
  --frames-per-snippet 16 \
  --temperature 5.0 \
  --device cuda
```

正式输出在 `evaluation/metrics.json`；`binary` 是主结果，`neuron_auxiliary` 和 `fused_diagnostic` 只用于分析，不参与选模。
