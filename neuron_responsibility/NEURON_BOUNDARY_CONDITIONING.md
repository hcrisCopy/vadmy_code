# 神经元边界定位与时序前注入

以下命令均从 `vadmy_code` 目录执行。方法不使用 baseline 分数选择神经元或生成边界标签；CLIP 始终冻结。已有 `selection`、`aligned_train.csv`、`aligned_test.csv` 和 `evidence_v1` 产物直接复用。

## 1. 独立神经元定位器

定位器只读取 hidden-state 神经元、视频标签和动态拼接边界。每个数据集只训练一次，随后供三个 baseline 复用。

### UCF-Crime

```bash
python neuron_responsibility/train_boundary_localizer.py \
  --dataset ucf \
  --train-list ../vadmy_data/neuron_responsibility/ucf/aligned_train.csv \
  --val-list ../vadmy_data/neuron_responsibility/ucf/aligned_test.csv \
  --gt-path ../vadmy_data/annotations/ucf/gt.npy \
  --evidence-config ../vadmy_data/neuron_responsibility/ucf/evidence_v1/evidence_config.npz \
  --out-dir ../vadmy_data/neuron_responsibility/ucf/boundary_localizer_v1 \
  --visual-length 256 \
  --hidden-width 64 \
  --active-neurons 64 \
  --evidence-cap 6.0 \
  --dropout 0.1 \
  --max-epoch 8 \
  --batch-size 64 \
  --lr 7e-5 \
  --weight-decay 1e-4 \
  --real-mil-weight 0.2 \
  --synthetic-bce-weight 1.0 \
  --synthetic-dice-weight 1.0 \
  --boundary-weight 0.1 \
  --sparsity-weight 0.001 \
  --min-segment 4 \
  --max-segment 32 \
  --frames-per-snippet 16 \
  --num-workers 4 \
  --seed 234 \
  --device cuda
```

### XD-Violence

```bash
python neuron_responsibility/train_boundary_localizer.py \
  --dataset xd \
  --train-list ../vadmy_data/neuron_responsibility/xd/aligned_train.csv \
  --val-list ../vadmy_data/neuron_responsibility/xd/aligned_test.csv \
  --gt-path ../vadmy_data/annotations/xd/gt.npy \
  --evidence-config ../vadmy_data/neuron_responsibility/xd/evidence_v1/evidence_config.npz \
  --out-dir ../vadmy_data/neuron_responsibility/xd/boundary_localizer_v1 \
  --visual-length 256 \
  --hidden-width 64 \
  --active-neurons 64 \
  --evidence-cap 6.0 \
  --dropout 0.1 \
  --max-epoch 8 \
  --batch-size 96 \
  --lr 7e-5 \
  --weight-decay 1e-4 \
  --real-mil-weight 0.2 \
  --synthetic-bce-weight 1.0 \
  --synthetic-dice-weight 1.0 \
  --boundary-weight 0.1 \
  --sparsity-weight 0.001 \
  --min-segment 4 \
  --max-segment 32 \
  --frames-per-snippet 16 \
  --num-workers 4 \
  --seed 234 \
  --device cuda
```

输出为 `localizer_best.pth`、`checkpoint_last.pth`、`history.jsonl` 和 `run_config.json`。中断后在原命令末尾添加 `--resume`；清理重跑添加 `--clean`。

## 2. 三个 baseline 的受控适配

训练顺序固定为：前 2 epoch 只训练时序前 adapter，第 3 epoch 起解冻最后时序块，第 7 epoch 起才以极低学习率解冻预测头。边界梯度与原始 baseline 目标冲突时自动执行 PCGrad。

### DSANet / UCF-Crime

```bash
python neuron_responsibility/train_boundary_conditioning.py \
  --baseline dsanet --dataset ucf \
  --baseline-root baseline/DSANet \
  --baseline-weight ../vadmy_data/model/DSANet/model_ucf.pth \
  --train-list ../vadmy_data/neuron_responsibility/ucf/aligned_train.csv \
  --val-list ../vadmy_data/neuron_responsibility/ucf/aligned_test.csv \
  --gt-path ../vadmy_data/annotations/ucf/gt.npy \
  --localizer-model ../vadmy_data/neuron_responsibility/ucf/boundary_localizer_v1/localizer_best.pth \
  --out-dir ../vadmy_data/neuron_responsibility/ucf/dsanet_boundary_conditioning_v1 \
  --max-epoch 8 --temporal-start-epoch 2 --head-start-epoch 6 \
  --batch-size 64 --adapter-lr 5e-5 --temporal-lr 1e-6 --head-lr 2e-7 \
  --weight-decay 0 --boundary-objective-weight 1.0 \
  --synthetic-bce-weight 1.0 --synthetic-dice-weight 1.0 --boundary-shape-weight 0.1 \
  --preservation-weight 0.5 --normal-delta-weight 0.1 --anchor-weight 0.01 \
  --adapter-width 32 --max-adapter-scale 0.25 \
  --min-segment 4 --max-segment 32 --frames-per-snippet 16 \
  --dsanet-ucf-eval-samples 1280 --num-workers 4 --seed 234 --device cuda
```

### DeSC / UCF-Crime

```bash
python neuron_responsibility/train_boundary_conditioning.py \
  --baseline desc --dataset ucf \
  --baseline-root baseline/DeSC \
  --sensitivity-weight ../vadmy_data/model/DeSC/ucf_sensitivity_stream.pth \
  --consistency-weight ../vadmy_data/model/DeSC/ucf_consistency_stream.pth \
  --train-list ../vadmy_data/neuron_responsibility/ucf/aligned_train.csv \
  --val-list ../vadmy_data/neuron_responsibility/ucf/aligned_test.csv \
  --gt-path ../vadmy_data/annotations/ucf/gt.npy \
  --localizer-model ../vadmy_data/neuron_responsibility/ucf/boundary_localizer_v1/localizer_best.pth \
  --out-dir ../vadmy_data/neuron_responsibility/ucf/desc_boundary_conditioning_v1 \
  --max-epoch 8 --temporal-start-epoch 2 --head-start-epoch 6 \
  --batch-size 64 --adapter-lr 5e-5 --temporal-lr 1e-6 --head-lr 2e-7 \
  --weight-decay 0 --boundary-objective-weight 1.0 \
  --synthetic-bce-weight 1.0 --synthetic-dice-weight 1.0 --boundary-shape-weight 0.1 \
  --preservation-weight 0.5 --normal-delta-weight 0.1 --anchor-weight 0.01 \
  --adapter-width 32 --max-adapter-scale 0.25 \
  --min-segment 4 --max-segment 32 --frames-per-snippet 16 \
  --dsanet-ucf-eval-samples 1280 --num-workers 4 --seed 234 --device cuda
```

### LaGoVAD / UCF-Crime

```bash
python neuron_responsibility/train_boundary_conditioning.py \
  --baseline lagovad --dataset ucf \
  --baseline-root baseline/LaGoVAD-PreVAD \
  --baseline-weight ../vadmy_data/model/LaGoVAD/best.ckpt \
  --train-list ../vadmy_data/neuron_responsibility/ucf/aligned_train.csv \
  --val-list ../vadmy_data/neuron_responsibility/ucf/aligned_test.csv \
  --gt-path ../vadmy_data/annotations/ucf/gt.npy \
  --localizer-model ../vadmy_data/neuron_responsibility/ucf/boundary_localizer_v1/localizer_best.pth \
  --out-dir ../vadmy_data/neuron_responsibility/ucf/lagovad_boundary_conditioning_v1 \
  --max-epoch 8 --temporal-start-epoch 2 --head-start-epoch 6 \
  --batch-size 64 --adapter-lr 5e-5 --temporal-lr 1e-6 --head-lr 2e-7 \
  --weight-decay 0 --boundary-objective-weight 1.0 \
  --synthetic-bce-weight 1.0 --synthetic-dice-weight 1.0 --boundary-shape-weight 0.1 \
  --preservation-weight 0.5 --normal-delta-weight 0.1 --anchor-weight 0.01 \
  --adapter-width 32 --max-adapter-scale 0.25 \
  --min-segment 4 --max-segment 32 --frames-per-snippet 16 \
  --dsanet-ucf-eval-samples 1280 --num-workers 4 --seed 234 --device cuda
```

### DSANet / XD-Violence

```bash
python neuron_responsibility/train_boundary_conditioning.py \
  --baseline dsanet --dataset xd \
  --baseline-root baseline/DSANet \
  --baseline-weight ../vadmy_data/model/DSANet/model_xd.pth \
  --train-list ../vadmy_data/neuron_responsibility/xd/aligned_train.csv \
  --val-list ../vadmy_data/neuron_responsibility/xd/aligned_test.csv \
  --gt-path ../vadmy_data/annotations/xd/gt.npy \
  --localizer-model ../vadmy_data/neuron_responsibility/xd/boundary_localizer_v1/localizer_best.pth \
  --out-dir ../vadmy_data/neuron_responsibility/xd/dsanet_boundary_conditioning_v1 \
  --max-epoch 8 --temporal-start-epoch 2 --head-start-epoch 6 \
  --batch-size 96 --adapter-lr 5e-5 --temporal-lr 1e-6 --head-lr 2e-7 \
  --weight-decay 0 --boundary-objective-weight 1.0 \
  --synthetic-bce-weight 1.0 --synthetic-dice-weight 1.0 --boundary-shape-weight 0.1 \
  --preservation-weight 0.5 --normal-delta-weight 0.1 --anchor-weight 0.01 \
  --adapter-width 32 --max-adapter-scale 0.25 \
  --min-segment 4 --max-segment 32 --frames-per-snippet 16 \
  --dsanet-ucf-eval-samples 1280 --num-workers 4 --seed 234 --device cuda
```

### DeSC / XD-Violence

```bash
python neuron_responsibility/train_boundary_conditioning.py \
  --baseline desc --dataset xd \
  --baseline-root baseline/DeSC \
  --sensitivity-weight ../vadmy_data/model/DeSC/xd_sensitivity_stream.pth \
  --consistency-weight ../vadmy_data/model/DeSC/xd_consistency_stream.pth \
  --train-list ../vadmy_data/neuron_responsibility/xd/aligned_train.csv \
  --val-list ../vadmy_data/neuron_responsibility/xd/aligned_test.csv \
  --gt-path ../vadmy_data/annotations/xd/gt.npy \
  --localizer-model ../vadmy_data/neuron_responsibility/xd/boundary_localizer_v1/localizer_best.pth \
  --out-dir ../vadmy_data/neuron_responsibility/xd/desc_boundary_conditioning_v1 \
  --max-epoch 8 --temporal-start-epoch 2 --head-start-epoch 6 \
  --batch-size 96 --adapter-lr 5e-5 --temporal-lr 1e-6 --head-lr 2e-7 \
  --weight-decay 0 --boundary-objective-weight 1.0 \
  --synthetic-bce-weight 1.0 --synthetic-dice-weight 1.0 --boundary-shape-weight 0.1 \
  --preservation-weight 0.5 --normal-delta-weight 0.1 --anchor-weight 0.01 \
  --adapter-width 32 --max-adapter-scale 0.25 \
  --min-segment 4 --max-segment 32 --frames-per-snippet 16 \
  --dsanet-ucf-eval-samples 1280 --num-workers 4 --seed 234 --device cuda
```

### LaGoVAD / XD-Violence

```bash
python neuron_responsibility/train_boundary_conditioning.py \
  --baseline lagovad --dataset xd \
  --baseline-root baseline/LaGoVAD-PreVAD \
  --baseline-weight ../vadmy_data/model/LaGoVAD/best.ckpt \
  --train-list ../vadmy_data/neuron_responsibility/xd/aligned_train.csv \
  --val-list ../vadmy_data/neuron_responsibility/xd/aligned_test.csv \
  --gt-path ../vadmy_data/annotations/xd/gt.npy \
  --localizer-model ../vadmy_data/neuron_responsibility/xd/boundary_localizer_v1/localizer_best.pth \
  --out-dir ../vadmy_data/neuron_responsibility/xd/lagovad_boundary_conditioning_v1 \
  --max-epoch 8 --temporal-start-epoch 2 --head-start-epoch 6 \
  --batch-size 64 --adapter-lr 5e-5 --temporal-lr 1e-6 --head-lr 2e-7 \
  --weight-decay 0 --boundary-objective-weight 1.0 \
  --synthetic-bce-weight 1.0 --synthetic-dice-weight 1.0 --boundary-shape-weight 0.1 \
  --preservation-weight 0.5 --normal-delta-weight 0.1 --anchor-weight 0.01 \
  --adapter-width 32 --max-adapter-scale 0.25 \
  --min-segment 4 --max-segment 32 --frames-per-snippet 16 \
  --dsanet-ucf-eval-samples 1280 --num-workers 4 --seed 234 --device cuda
```

各实验输出为 `parameter_report.json`、`history.jsonl`、`checkpoint_last.pth` 和 `model_best.pth`。恢复或清理方式与定位器相同。

## 3. DSANet / UCF-Crime 正式评测

```bash
python neuron_responsibility/evaluate_boundary_conditioning.py \
  --baseline dsanet --dataset ucf \
  --baseline-root baseline/DSANet \
  --baseline-weight ../vadmy_data/model/DSANet/model_ucf.pth \
  --joint-model ../vadmy_data/neuron_responsibility/ucf/dsanet_boundary_conditioning_v1/model_best.pth \
  --test-list ../vadmy_data/neuron_responsibility/ucf/aligned_test.csv \
  --gt-path ../vadmy_data/annotations/ucf/gt.npy \
  --gt-segment-path ../vadmy_data/annotations/ucf/gt_segment.npy \
  --gt-label-path ../vadmy_data/annotations/ucf/gt_label.npy \
  --out-dir ../vadmy_data/neuron_responsibility/ucf/dsanet_boundary_conditioning_v1/evaluation \
  --frames-per-snippet 16 --temperature 5.0 --device cuda
```

主结果写入 `evaluation/metrics.json`；`binary` 为正式检测结果，`independent_neuron` 只用于诊断。

