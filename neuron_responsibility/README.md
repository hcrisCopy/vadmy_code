# Neuron-Decoupled Responsibility Learning

本目录实现一套共享方法，适配 LaGoVAD、DeSC、DSANet 和 UCF-Crime、XD-Violence。baseline 源码不修改；CLIP 始终冻结；已有 512D CLIP 特征和 CLS hidden states 直接复用。

## 输入约定

先把可复用产物放到 `../vadmy_data/features/{ucf,xd}/source/`。新代码不引用其他项目目录。

```text
../vadmy_data/features/ucf/source/ucf_train_local.csv
../vadmy_data/features/ucf/source/ucf_test_local.csv
../vadmy_data/features/ucf/source/hidden_train/manifest.csv
../vadmy_data/features/ucf/source/hidden_test/manifest.csv
```

特征 CSV 使用 `path,label`；manifest 使用 `key,hidden_path,token_pool`，其中 `token_pool=cls`。CSV 中 `path` 指向原始 `[T,512]` CLIP 文件，hidden 文件保存 `[T,L,768]`。XD 目录结构相同。

## 1. 每个数据集只做一次 neuron selection

UCF-Crime：

```bash
python neuron_responsibility/select_neurons.py \
  --dataset ucf \
  --source-train-csv ../vadmy_data/features/ucf/source/ucf_train_local.csv \
  --hidden-manifest ../vadmy_data/features/ucf/source/hidden_train/manifest.csv \
  --out-dir ../vadmy_data/neuron_responsibility/ucf/selection \
  --top-p 0.10 \
  --topk-global 768 \
  --normal-stat-snippets-per-video 256 \
  --sigma-min 1e-6
```

XD-Violence：

```bash
python neuron_responsibility/select_neurons.py \
  --dataset xd \
  --source-train-csv ../vadmy_data/features/xd/source/xd_train_local.csv \
  --hidden-manifest ../vadmy_data/features/xd/source/hidden_train/manifest.csv \
  --out-dir ../vadmy_data/neuron_responsibility/xd/selection \
  --top-p 0.10 \
  --topk-global 768 \
  --normal-stat-snippets-per-video 256 \
  --sigma-min 1e-6
```

输出为 `selection/selected_neurons.json`、normal mean/std、tail statistics 和可恢复的 `bag_cache/`。需要重算时显式添加 `--clean`。

## 2. 构建轻量对齐清单

下面以 UCF 为例；XD 只需将 `ucf` 改成 `xd`。原始 512D CLIP 文件不会复制，输出仅包含选中 neuron 特征。

```bash
python neuron_responsibility/build_aligned_features.py \
  --source-csv ../vadmy_data/features/ucf/source/ucf_train_local.csv \
  --hidden-manifest ../vadmy_data/features/ucf/source/hidden_train/manifest.csv \
  --neuron-json ../vadmy_data/neuron_responsibility/ucf/selection/selected_neurons.json \
  --out-dir ../vadmy_data/neuron_responsibility/ucf/aligned/train \
  --out-csv ../vadmy_data/neuron_responsibility/ucf/aligned_train.csv

python neuron_responsibility/build_aligned_features.py \
  --source-csv ../vadmy_data/features/ucf/source/ucf_test_local.csv \
  --hidden-manifest ../vadmy_data/features/ucf/source/hidden_test/manifest.csv \
  --neuron-json ../vadmy_data/neuron_responsibility/ucf/selection/selected_neurons.json \
  --out-dir ../vadmy_data/neuron_responsibility/ucf/aligned/test \
  --out-csv ../vadmy_data/neuron_responsibility/ucf/aligned_test.csv
```

## 3. Stage A：冻结所有 baseline，训练共享 probe

UCF：

```bash
python neuron_responsibility/train_probe.py \
  --dataset ucf \
  --train-list ../vadmy_data/neuron_responsibility/ucf/aligned_train.csv \
  --val-list ../vadmy_data/neuron_responsibility/ucf/aligned_test.csv \
  --out-dir ../vadmy_data/neuron_responsibility/ucf/probe \
  --visual-length 256 \
  --hidden-width 128 \
  --max-epoch 10 \
  --batch-size 64 \
  --lr 7e-5 \
  --weight-decay 1e-4 \
  --sparsity-weight 1e-3 \
  --normal-instance-weight 0.25 \
  --num-workers 4 \
  --seed 234 \
  --device cuda
```

XD：使用相同结构，将路径改为 `xd`，并显式使用 `--batch-size 96 --lr 1e-5`。输出 `probe_best.pth`、`checkpoint_last.pth` 和 `history.jsonl`；中断后添加 `--resume`。

## 4. Stage B：正式 joint training

DSANet 和 LaGoVAD 默认使用 `--train-scope temporal_heads`：CLIP 冻结，probe 冻结，只训练 baseline 时序模块、融合模块和 heads。DeSC 的本地作者发布目录没有训练脚本和 GMP 损失实现，因此统一适配器有意限制为 `--train-scope heads`：两个预训练时序流固定，只更新其分类/投影 heads，避免臆造作者训练目标。

### DSANet / UCF

```bash
python neuron_responsibility/train_joint.py \
  --baseline dsanet \
  --dataset ucf \
  --baseline-root baseline/DSANet \
  --baseline-weight ../vadmy_data/model/DSANet/model_ucf.pth \
  --train-list ../vadmy_data/neuron_responsibility/ucf/aligned_train.csv \
  --val-list ../vadmy_data/neuron_responsibility/ucf/aligned_test.csv \
  --probe-model ../vadmy_data/neuron_responsibility/ucf/probe/probe_best.pth \
  --out-dir ../vadmy_data/neuron_responsibility/ucf/dsanet \
  --train-scope temporal_heads \
  --max-epoch 10 \
  --batch-size 64 \
  --lr 7e-5 \
  --weight-decay 0 \
  --responsibility-weight 1.0 \
  --num-workers 4 \
  --seed 234 \
  --device cuda
```

### DeSC / UCF

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
  --out-dir ../vadmy_data/neuron_responsibility/ucf/desc \
  --train-scope heads \
  --max-epoch 10 \
  --batch-size 64 \
  --sensitivity-lr 1e-3 \
  --consistency-lr 5e-5 \
  --weight-decay 1e-5 \
  --responsibility-weight 1.0 \
  --num-workers 4 \
  --seed 234 \
  --device cuda
```

### LaGoVAD / UCF

```bash
python neuron_responsibility/train_joint.py \
  --baseline lagovad \
  --dataset ucf \
  --baseline-root baseline/LaGoVAD-PreVAD \
  --baseline-weight ../vadmy_data/model/LaGoVAD/best.ckpt \
  --train-list ../vadmy_data/neuron_responsibility/ucf/aligned_train.csv \
  --val-list ../vadmy_data/neuron_responsibility/ucf/aligned_test.csv \
  --probe-model ../vadmy_data/neuron_responsibility/ucf/probe/probe_best.pth \
  --out-dir ../vadmy_data/neuron_responsibility/ucf/lagovad \
  --train-scope temporal_heads \
  --max-epoch 20 \
  --batch-size 64 \
  --lr 1e-5 \
  --weight-decay 0.01 \
  --responsibility-weight 1.0 \
  --num-workers 4 \
  --seed 234 \
  --device cuda
```

XD 使用相同命令结构：路径改为 `xd`；DSANet 使用 `model_xd.pth --batch-size 96 --lr 1e-5`；DeSC 使用 `xd_*_stream.pth --batch-size 96 --sensitivity-lr 1e-3 --consistency-lr 1e-5 --weight-decay 1e-3`；LaGoVAD 使用同一 `best.ckpt --batch-size 64 --lr 1e-5 --weight-decay 0.01`。

训练输出为 `checkpoint_last.pth`、`model_best.pth`、`history.jsonl` 和 `parameter_report.json`。后者逐项记录实际可训练张量，并在启动时强制断言 CLIP 可训练参数为 0。中断后在原命令末尾添加 `--resume`。

## 训练与存储开销

默认 `K=768, H=128` 的 probe 约含 0.151M 参数（FP32 权重约 0.6 MB），每个 snippet 约 0.148M MAC；长度 256 的单视频约 38M MAC。Stage B 仍以 baseline 前向/反向为主要开销，probe 只做冻结前向。选中 neuron 文件额外占 `T × 768 × 4` bytes，即约 3 KB/snippet；原 512D CLIP 文件不复制，完整 hidden states 也不复制。DeSC 本身需要双流前向，因此三者中训练开销最高。

## 5. 正式评测与互补性诊断

DSANet/UCF 示例：

```bash
python neuron_responsibility/evaluate.py \
  --baseline dsanet \
  --dataset ucf \
  --baseline-root baseline/DSANet \
  --baseline-weight ../vadmy_data/model/DSANet/model_ucf.pth \
  --joint-model ../vadmy_data/neuron_responsibility/ucf/dsanet/model_best.pth \
  --probe-model ../vadmy_data/neuron_responsibility/ucf/probe/probe_best.pth \
  --test-list ../vadmy_data/neuron_responsibility/ucf/aligned_test.csv \
  --gt-path baseline/DSANet/list/gt_ucf.npy \
  --gt-segment-path baseline/DSANet/list/gt_segment_ucf.npy \
  --gt-label-path baseline/DSANet/list/gt_label_ucf.npy \
  --out-dir ../vadmy_data/neuron_responsibility/ucf/dsanet/evaluation \
  --frames-per-snippet 16 \
  --temperature 5.0 \
  --device cuda
```

输出为 `evaluation/metrics.json`、`per_video.csv` 和可恢复的逐视频 `scores/`。删除 `--joint-model` 即可评测冻结作者权重。DeSC 改用两个 stream 权重；LaGoVAD 使用 `best.ckpt`。XD 的 ground truth 改成 `gt.npy`、`gt_segment.npy`、`gt_label.npy`，temperature 使用 `1.0`。检测 mAP 仅在对应 baseline 发布目录存在 `{dataset}_detectionMAP.py` 时计算；否则明确记录 `detection_map_skipped`，不会跨 baseline 借用评测代码。

`fused_diagnostic` 只用于判断 neuron 与 baseline 是否互补，不作为正式主结果或选模指标。
