# Head-only 公平对照组完整命令

以下命令均从 `vadmy_code` 执行。方法组的完整命令位于 `README.md` 的 Stage B；本文件给出六个 `baseline_only` 对照训练及其六个评测命令。对照组和方法组使用相同初始化、数据、heads、epoch、batch size、学习率和 seed，唯一差异是方法组加入 responsibility loss。

## 1. 训练 baseline-only 对照组

### DSANet / UCF-Crime

```bash
python neuron_responsibility/train_joint.py \
  --baseline dsanet \
  --dataset ucf \
  --baseline-root baseline/DSANet \
  --baseline-weight ../vadmy_data/model/DSANet/model_ucf.pth \
  --train-list ../vadmy_data/neuron_responsibility/ucf/aligned_train.csv \
  --val-list ../vadmy_data/neuron_responsibility/ucf/aligned_test.csv \
  --probe-model ../vadmy_data/neuron_responsibility/ucf/probe/probe_best.pth \
  --out-dir ../vadmy_data/neuron_responsibility/ucf/dsanet_head_baseline_only \
  --train-scope heads \
  --training-mode baseline_only \
  --max-epoch 10 \
  --batch-size 64 \
  --lr 7e-5 \
  --weight-decay 0 \
  --num-workers 4 \
  --seed 234 \
  --device cuda
```

### DeSC / UCF-Crime

```bash
python neuron_responsibility/train_joint.py \
  --baseline desc \
  --dataset ucf \
  --baseline-root baseline/DeSC \
  --sensitivity-weight ../vadmy_data/model/DeSC/ucf_sensitivity_stream.pth \
  --consistency-weight ../vadmy_data/model/DeSC/ucf_consistency_stream.pth \
  --train-list ../vadmy_data/neuron_responsibility/ucf/aligned_train.csv \
  --val-list ../vadmy_data/neuron_responsibility/ucf/aligned_test.csv \
  --probe-model ../vadmy_data/neuron_responsibility/ucf/probe/probe_best.pth \
  --out-dir ../vadmy_data/neuron_responsibility/ucf/desc_head_baseline_only \
  --train-scope heads \
  --training-mode baseline_only \
  --max-epoch 10 \
  --batch-size 64 \
  --sensitivity-lr 1e-3 \
  --consistency-lr 5e-5 \
  --weight-decay 1e-5 \
  --num-workers 4 \
  --seed 234 \
  --device cuda
```

### LaGoVAD / UCF-Crime

```bash
python neuron_responsibility/train_joint.py \
  --baseline lagovad \
  --dataset ucf \
  --baseline-root baseline/LaGoVAD-PreVAD \
  --baseline-weight ../vadmy_data/model/LaGoVAD/best.ckpt \
  --train-list ../vadmy_data/neuron_responsibility/ucf/aligned_train.csv \
  --val-list ../vadmy_data/neuron_responsibility/ucf/aligned_test.csv \
  --probe-model ../vadmy_data/neuron_responsibility/ucf/probe/probe_best.pth \
  --out-dir ../vadmy_data/neuron_responsibility/ucf/lagovad_head_baseline_only \
  --train-scope heads \
  --training-mode baseline_only \
  --max-epoch 20 \
  --batch-size 64 \
  --lr 1e-5 \
  --weight-decay 0.01 \
  --num-workers 4 \
  --seed 234 \
  --device cuda
```

### DSANet / XD-Violence

```bash
python neuron_responsibility/train_joint.py \
  --baseline dsanet \
  --dataset xd \
  --baseline-root baseline/DSANet \
  --baseline-weight ../vadmy_data/model/DSANet/model_xd.pth \
  --train-list ../vadmy_data/neuron_responsibility/xd/aligned_train.csv \
  --val-list ../vadmy_data/neuron_responsibility/xd/aligned_test.csv \
  --probe-model ../vadmy_data/neuron_responsibility/xd/probe/probe_best.pth \
  --out-dir ../vadmy_data/neuron_responsibility/xd/dsanet_head_baseline_only \
  --train-scope heads \
  --training-mode baseline_only \
  --max-epoch 10 \
  --batch-size 96 \
  --lr 1e-5 \
  --weight-decay 0 \
  --num-workers 4 \
  --seed 234 \
  --device cuda
```

### DeSC / XD-Violence

```bash
python neuron_responsibility/train_joint.py \
  --baseline desc \
  --dataset xd \
  --baseline-root baseline/DeSC \
  --sensitivity-weight ../vadmy_data/model/DeSC/xd_sensitivity_stream.pth \
  --consistency-weight ../vadmy_data/model/DeSC/xd_consistency_stream.pth \
  --train-list ../vadmy_data/neuron_responsibility/xd/aligned_train.csv \
  --val-list ../vadmy_data/neuron_responsibility/xd/aligned_test.csv \
  --probe-model ../vadmy_data/neuron_responsibility/xd/probe/probe_best.pth \
  --out-dir ../vadmy_data/neuron_responsibility/xd/desc_head_baseline_only \
  --train-scope heads \
  --training-mode baseline_only \
  --max-epoch 10 \
  --batch-size 96 \
  --sensitivity-lr 1e-3 \
  --consistency-lr 1e-5 \
  --weight-decay 1e-3 \
  --num-workers 4 \
  --seed 234 \
  --device cuda
```

### LaGoVAD / XD-Violence

```bash
python neuron_responsibility/train_joint.py \
  --baseline lagovad \
  --dataset xd \
  --baseline-root baseline/LaGoVAD-PreVAD \
  --baseline-weight ../vadmy_data/model/LaGoVAD/best.ckpt \
  --train-list ../vadmy_data/neuron_responsibility/xd/aligned_train.csv \
  --val-list ../vadmy_data/neuron_responsibility/xd/aligned_test.csv \
  --probe-model ../vadmy_data/neuron_responsibility/xd/probe/probe_best.pth \
  --out-dir ../vadmy_data/neuron_responsibility/xd/lagovad_head_baseline_only \
  --train-scope heads \
  --training-mode baseline_only \
  --max-epoch 20 \
  --batch-size 64 \
  --lr 1e-5 \
  --weight-decay 0.01 \
  --num-workers 4 \
  --seed 234 \
  --device cuda
```

## 2. 评测 baseline-only 对照组

### DSANet / UCF-Crime

```bash
python neuron_responsibility/evaluate.py \
  --baseline dsanet \
  --dataset ucf \
  --baseline-root baseline/DSANet \
  --baseline-weight ../vadmy_data/model/DSANet/model_ucf.pth \
  --joint-model ../vadmy_data/neuron_responsibility/ucf/dsanet_head_baseline_only/model_best.pth \
  --probe-model ../vadmy_data/neuron_responsibility/ucf/probe/probe_best.pth \
  --test-list ../vadmy_data/neuron_responsibility/ucf/aligned_test.csv \
  --gt-path ../vadmy_data/annotations/ucf/gt.npy \
  --gt-segment-path ../vadmy_data/annotations/ucf/gt_segment.npy \
  --gt-label-path ../vadmy_data/annotations/ucf/gt_label.npy \
  --out-dir ../vadmy_data/neuron_responsibility/ucf/dsanet_head_baseline_only/evaluation \
  --frames-per-snippet 16 \
  --temperature 5.0 \
  --device cuda
```

### DeSC / UCF-Crime

```bash
python neuron_responsibility/evaluate.py \
  --baseline desc \
  --dataset ucf \
  --baseline-root baseline/DeSC \
  --sensitivity-weight ../vadmy_data/model/DeSC/ucf_sensitivity_stream.pth \
  --consistency-weight ../vadmy_data/model/DeSC/ucf_consistency_stream.pth \
  --joint-model ../vadmy_data/neuron_responsibility/ucf/desc_head_baseline_only/model_best.pth \
  --probe-model ../vadmy_data/neuron_responsibility/ucf/probe/probe_best.pth \
  --test-list ../vadmy_data/neuron_responsibility/ucf/aligned_test.csv \
  --gt-path ../vadmy_data/annotations/ucf/gt.npy \
  --out-dir ../vadmy_data/neuron_responsibility/ucf/desc_head_baseline_only/evaluation \
  --frames-per-snippet 16 \
  --temperature 1.0 \
  --device cuda
```

### LaGoVAD / UCF-Crime

```bash
python neuron_responsibility/evaluate.py \
  --baseline lagovad \
  --dataset ucf \
  --baseline-root baseline/LaGoVAD-PreVAD \
  --baseline-weight ../vadmy_data/model/LaGoVAD/best.ckpt \
  --joint-model ../vadmy_data/neuron_responsibility/ucf/lagovad_head_baseline_only/model_best.pth \
  --probe-model ../vadmy_data/neuron_responsibility/ucf/probe/probe_best.pth \
  --test-list ../vadmy_data/neuron_responsibility/ucf/aligned_test.csv \
  --gt-path ../vadmy_data/annotations/ucf/gt.npy \
  --out-dir ../vadmy_data/neuron_responsibility/ucf/lagovad_head_baseline_only/evaluation \
  --frames-per-snippet 16 \
  --temperature 1.0 \
  --device cuda
```

### DSANet / XD-Violence

```bash
python neuron_responsibility/evaluate.py \
  --baseline dsanet \
  --dataset xd \
  --baseline-root baseline/DSANet \
  --baseline-weight ../vadmy_data/model/DSANet/model_xd.pth \
  --joint-model ../vadmy_data/neuron_responsibility/xd/dsanet_head_baseline_only/model_best.pth \
  --probe-model ../vadmy_data/neuron_responsibility/xd/probe/probe_best.pth \
  --test-list ../vadmy_data/neuron_responsibility/xd/aligned_test.csv \
  --gt-path ../vadmy_data/annotations/xd/gt.npy \
  --gt-segment-path ../vadmy_data/annotations/xd/gt_segment.npy \
  --gt-label-path ../vadmy_data/annotations/xd/gt_label.npy \
  --out-dir ../vadmy_data/neuron_responsibility/xd/dsanet_head_baseline_only/evaluation \
  --frames-per-snippet 16 \
  --temperature 1.0 \
  --device cuda
```

### DeSC / XD-Violence

```bash
python neuron_responsibility/evaluate.py \
  --baseline desc \
  --dataset xd \
  --baseline-root baseline/DeSC \
  --sensitivity-weight ../vadmy_data/model/DeSC/xd_sensitivity_stream.pth \
  --consistency-weight ../vadmy_data/model/DeSC/xd_consistency_stream.pth \
  --joint-model ../vadmy_data/neuron_responsibility/xd/desc_head_baseline_only/model_best.pth \
  --probe-model ../vadmy_data/neuron_responsibility/xd/probe/probe_best.pth \
  --test-list ../vadmy_data/neuron_responsibility/xd/aligned_test.csv \
  --gt-path ../vadmy_data/annotations/xd/gt.npy \
  --out-dir ../vadmy_data/neuron_responsibility/xd/desc_head_baseline_only/evaluation \
  --frames-per-snippet 16 \
  --temperature 1.0 \
  --device cuda
```

### LaGoVAD / XD-Violence

```bash
python neuron_responsibility/evaluate.py \
  --baseline lagovad \
  --dataset xd \
  --baseline-root baseline/LaGoVAD-PreVAD \
  --baseline-weight ../vadmy_data/model/LaGoVAD/best.ckpt \
  --joint-model ../vadmy_data/neuron_responsibility/xd/lagovad_head_baseline_only/model_best.pth \
  --probe-model ../vadmy_data/neuron_responsibility/xd/probe/probe_best.pth \
  --test-list ../vadmy_data/neuron_responsibility/xd/aligned_test.csv \
  --gt-path ../vadmy_data/annotations/xd/gt.npy \
  --out-dir ../vadmy_data/neuron_responsibility/xd/lagovad_head_baseline_only/evaluation \
  --frames-per-snippet 16 \
  --temperature 1.0 \
  --device cuda
```

## 3. 结果比较

对每个 baseline 和数据集，比较以下两个文件中的 `binary` AUC/AP：

```text
.../{baseline}_head_baseline_only/evaluation/metrics.json
.../{baseline}_head_responsibility/evaluation/metrics.json
```

两者的差值才是责任学习的公平增益。`parameter_report.json` 中的 `baseline_trainable_parameters` 和 `trainable_tensors` 应当在同一 baseline 的两个实验中完全一致。
