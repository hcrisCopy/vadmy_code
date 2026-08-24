# Neuron-Decoupled Responsibility Learning

本目录实现一套共享方法，适配 DSANet、DeSC、LaGoVAD，以及 UCF-Crime、XD-Violence。baseline 源码不修改，CLIP 始终冻结。

当前正式主方案是独立神经元边界定位、动态边界合成和时序编码前注入，完整指令见
`neuron_responsibility/NEURON_BOUNDARY_CONDITIONING.md`。旧的 post-temporal 特征调制、
probe、分区责任和 score correction 只用于复现实验对照，不再作为主方案。

以下所有命令均从 `vadmy_code` 目录执行。命令已经分别写全，不需要手动替换数据集、权重或输出路径。

## 0. 产物复用与目录边界

旧项目已经生成的 512D CLIP 特征清单和 CLS hidden states 直接只读复用：

```text
../vad_data/work_ucf/ucf_train_local.csv
../vad_data/work_ucf/ucf_test_local.csv
../vad_data/work_ucf/clip_hidden_stride16_train_8gpu/manifest.csv
../vad_data/work_ucf/clip_hidden_stride16_test_8gpu/manifest.csv

../vad_data/work_xd/xd_train_local.csv
../vad_data/work_xd/xd_test_local.csv
../vad_data/work_xd/clip_hidden_stride16_train_8gpu/manifest.csv
../vad_data/work_xd/clip_hidden_stride16_test_8gpu/manifest.csv
```

不要把 hidden states 复制到 `vadmy_data`。它们体积大且已经是三个 baseline 共用的同一套 CLIP ViT-B/16 产物。新代码不导入 `vad_code` 的任何源码，只读取 `vad_data` 中的数据产物；所有新产物均写入 `../vadmy_data/neuron_responsibility`。

先检查输入和作者权重是否齐全：

```bash
for f in \
  ../vad_data/work_ucf/ucf_train_local.csv \
  ../vad_data/work_ucf/ucf_test_local.csv \
  ../vad_data/work_ucf/clip_hidden_stride16_train_8gpu/manifest.csv \
  ../vad_data/work_ucf/clip_hidden_stride16_test_8gpu/manifest.csv \
  ../vad_data/work_xd/xd_train_local.csv \
  ../vad_data/work_xd/xd_test_local.csv \
  ../vad_data/work_xd/clip_hidden_stride16_train_8gpu/manifest.csv \
  ../vad_data/work_xd/clip_hidden_stride16_test_8gpu/manifest.csv \
  ../vadmy_data/model/DSANet/model_ucf.pth \
  ../vadmy_data/model/DSANet/model_xd.pth \
  ../vadmy_data/model/DeSC/ucf_sensitivity_stream.pth \
  ../vadmy_data/model/DeSC/ucf_consistency_stream.pth \
  ../vadmy_data/model/DeSC/xd_sensitivity_stream.pth \
  ../vadmy_data/model/DeSC/xd_consistency_stream.pth \
  ../vadmy_data/model/LaGoVAD/best.ckpt; do
  test -f "$f" || { echo "MISSING: $f"; exit 1; }
done
echo "all reusable inputs and weights exist"
```

将小体积 benchmark ground truth 复制为本项目共享评测输入：

```bash
mkdir -p ../vadmy_data/annotations/ucf ../vadmy_data/annotations/xd
cp baseline/DSANet/list/gt_ucf.npy ../vadmy_data/annotations/ucf/gt.npy
cp baseline/DSANet/list/gt_segment_ucf.npy ../vadmy_data/annotations/ucf/gt_segment.npy
cp baseline/DSANet/list/gt_label_ucf.npy ../vadmy_data/annotations/ucf/gt_label.npy
cp baseline/DSANet/list/gt.npy ../vadmy_data/annotations/xd/gt.npy
cp baseline/DSANet/list/gt_segment.npy ../vadmy_data/annotations/xd/gt_segment.npy
cp baseline/DSANet/list/gt_label.npy ../vadmy_data/annotations/xd/gt_label.npy
```

## 1. 神经元选择：每个数据集只运行一次

### UCF-Crime

```bash
python neuron_responsibility/select_neurons.py \
  --dataset ucf \
  --source-train-csv ../vad_data/work_ucf/ucf_train_local.csv \
  --hidden-manifest ../vad_data/work_ucf/clip_hidden_stride16_train_8gpu/manifest.csv \
  --out-dir ../vadmy_data/neuron_responsibility/ucf/selection \
  --top-p 0.10 \
  --topk-global 768 \
  --normal-coverage-quantile 0.95 \
  --max-per-layer 96 \
  --normal-stat-snippets-per-video 256 \
  --sigma-min 1e-6
```

### XD-Violence

```bash
python neuron_responsibility/select_neurons.py \
  --dataset xd \
  --source-train-csv ../vad_data/work_xd/xd_train_local.csv \
  --hidden-manifest ../vad_data/work_xd/clip_hidden_stride16_train_8gpu/manifest.csv \
  --out-dir ../vadmy_data/neuron_responsibility/xd/selection \
  --top-p 0.10 \
  --topk-global 768 \
  --normal-coverage-quantile 0.95 \
  --max-per-layer 96 \
  --normal-stat-snippets-per-video 256 \
  --sigma-min 1e-6
```

输出包括 `selected_neurons.json`、normal mean/std、局部对比/覆盖率统计和可恢复的逐视频 `local_contrast_v2_*` 缓存。只有需要强制重算时才在原命令后添加 `--clean`。

## 2. 构建对齐 neuron 特征

原始 512D CLIP 文件不会复制。输出只保存选中的 768 个 neuron 值。

### UCF-Crime train

```bash
python neuron_responsibility/build_aligned_features.py \
  --source-csv ../vad_data/work_ucf/ucf_train_local.csv \
  --hidden-manifest ../vad_data/work_ucf/clip_hidden_stride16_train_8gpu/manifest.csv \
  --neuron-json ../vadmy_data/neuron_responsibility/ucf/selection/selected_neurons.json \
  --out-dir ../vadmy_data/neuron_responsibility/ucf/aligned/train \
  --out-csv ../vadmy_data/neuron_responsibility/ucf/aligned_train.csv
```

### UCF-Crime test

```bash
python neuron_responsibility/build_aligned_features.py \
  --source-csv ../vad_data/work_ucf/ucf_test_local.csv \
  --hidden-manifest ../vad_data/work_ucf/clip_hidden_stride16_test_8gpu/manifest.csv \
  --neuron-json ../vadmy_data/neuron_responsibility/ucf/selection/selected_neurons.json \
  --out-dir ../vadmy_data/neuron_responsibility/ucf/aligned/test \
  --out-csv ../vadmy_data/neuron_responsibility/ucf/aligned_test.csv
```

### XD-Violence train

```bash
python neuron_responsibility/build_aligned_features.py \
  --source-csv ../vad_data/work_xd/xd_train_local.csv \
  --hidden-manifest ../vad_data/work_xd/clip_hidden_stride16_train_8gpu/manifest.csv \
  --neuron-json ../vadmy_data/neuron_responsibility/xd/selection/selected_neurons.json \
  --out-dir ../vadmy_data/neuron_responsibility/xd/aligned/train \
  --out-csv ../vadmy_data/neuron_responsibility/xd/aligned_train.csv \
  --skip-missing-hidden
```

当前 XD 训练集有 4 个视频未提取 hidden states。上面的命令会跳过这些视频对应的全部 CSV 行，并写入 `aligned/train/skipped_rows.csv`；已生成的对齐文件会直接校验并复用。

### XD-Violence test

```bash
python neuron_responsibility/build_aligned_features.py \
  --source-csv ../vad_data/work_xd/xd_test_local.csv \
  --hidden-manifest ../vad_data/work_xd/clip_hidden_stride16_test_8gpu/manifest.csv \
  --neuron-json ../vadmy_data/neuron_responsibility/xd/selection/selected_neurons.json \
  --out-dir ../vadmy_data/neuron_responsibility/xd/aligned/test \
  --out-csv ../vadmy_data/neuron_responsibility/xd/aligned_test.csv
```

## 3. Stage A：训练共享 probe

Stage A 不加载任何 baseline，只训练 neuron probe。

### UCF-Crime

```bash
python neuron_responsibility/train_probe.py \
  --dataset ucf \
  --train-list ../vadmy_data/neuron_responsibility/ucf/aligned_train.csv \
  --val-list ../vadmy_data/neuron_responsibility/ucf/aligned_test.csv \
  --out-dir ../vadmy_data/neuron_responsibility/ucf/probe_v2 \
  --visual-length 256 \
  --hidden-width 128 \
  --active-neurons 128 \
  --max-epoch 10 \
  --batch-size 64 \
  --lr 7e-5 \
  --weight-decay 1e-4 \
  --sparsity-weight 1e-3 \
  --normal-instance-weight 0.25 \
  --ranking-weight 0.2 \
  --smoothness-weight 0.05 \
  --anomaly-sparsity-weight 0.01 \
  --normal-quantile 0.99 \
  --num-workers 4 \
  --seed 234 \
  --device cuda
```

### XD-Violence

```bash
python neuron_responsibility/train_probe.py \
  --dataset xd \
  --train-list ../vadmy_data/neuron_responsibility/xd/aligned_train.csv \
  --val-list ../vadmy_data/neuron_responsibility/xd/aligned_test.csv \
  --out-dir ../vadmy_data/neuron_responsibility/xd/probe_v2 \
  --visual-length 256 \
  --hidden-width 128 \
  --active-neurons 128 \
  --max-epoch 10 \
  --batch-size 96 \
  --lr 1e-5 \
  --weight-decay 1e-4 \
  --sparsity-weight 1e-3 \
  --normal-instance-weight 0.25 \
  --ranking-weight 0.2 \
  --smoothness-weight 0.05 \
  --anomaly-sparsity-weight 0.01 \
  --normal-quantile 0.99 \
  --num-workers 4 \
  --seed 234 \
  --device cuda
```

Stage A 输出 `probe_best.pth`、`checkpoint_last.pth` 和 `history.jsonl`。中断后使用完全相同的命令并在末尾添加 `--resume`。

## 4. Stage B：责任引导的 baseline 训练

前 2 个 epoch 只训练零初始化的校正头；第 3 个 epoch 起解冻每个 baseline 功能对应的分类头和最后时序精炼块。CLIP 与 probe 始终冻结。校正头、heads、最后时序块分别使用 `5e-5 / 1e-5 / 1e-6`。

### DSANet / UCF-Crime

```bash
python neuron_responsibility/train_joint.py \
  --baseline dsanet \
  --dataset ucf \
  --baseline-root baseline/DSANet \
  --baseline-weight ../vadmy_data/model/DSANet/model_ucf.pth \
  --train-list ../vadmy_data/neuron_responsibility/ucf/aligned_train.csv \
  --val-list ../vadmy_data/neuron_responsibility/ucf/aligned_test.csv \
  --gt-path ../vadmy_data/annotations/ucf/gt.npy \
  --probe-model ../vadmy_data/neuron_responsibility/ucf/probe_v2/probe_best.pth \
  --out-dir ../vadmy_data/neuron_responsibility/ucf/dsanet_gradual \
  --max-epoch 10 \
  --warmup-epochs 2 \
  --batch-size 64 \
  --correction-lr 5e-5 \
  --head-lr 1e-5 \
  --temporal-lr 1e-6 \
  --weight-decay 0 \
  --responsibility-weight 1.0 \
  --anchor-weight 1e-4 \
  --persistence 3 \
  --frames-per-snippet 16 \
  --dsanet-ucf-eval-samples 1280 \
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
  --gt-path ../vadmy_data/annotations/ucf/gt.npy \
  --probe-model ../vadmy_data/neuron_responsibility/ucf/probe_v2/probe_best.pth \
  --out-dir ../vadmy_data/neuron_responsibility/ucf/desc_gradual \
  --max-epoch 10 \
  --warmup-epochs 2 \
  --batch-size 64 \
  --correction-lr 5e-5 \
  --head-lr 1e-5 \
  --temporal-lr 1e-6 \
  --weight-decay 0 \
  --responsibility-weight 1.0 \
  --anchor-weight 1e-4 \
  --persistence 3 \
  --frames-per-snippet 16 \
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
  --gt-path ../vadmy_data/annotations/ucf/gt.npy \
  --probe-model ../vadmy_data/neuron_responsibility/ucf/probe_v2/probe_best.pth \
  --out-dir ../vadmy_data/neuron_responsibility/ucf/lagovad_gradual \
  --max-epoch 20 \
  --warmup-epochs 2 \
  --batch-size 64 \
  --correction-lr 5e-5 \
  --head-lr 1e-5 \
  --temporal-lr 1e-6 \
  --weight-decay 0 \
  --responsibility-weight 1.0 \
  --anchor-weight 1e-4 \
  --persistence 3 \
  --frames-per-snippet 16 \
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
  --gt-path ../vadmy_data/annotations/xd/gt.npy \
  --probe-model ../vadmy_data/neuron_responsibility/xd/probe_v2/probe_best.pth \
  --out-dir ../vadmy_data/neuron_responsibility/xd/dsanet_gradual \
  --max-epoch 10 \
  --warmup-epochs 2 \
  --batch-size 96 \
  --correction-lr 5e-5 \
  --head-lr 1e-5 \
  --temporal-lr 1e-6 \
  --weight-decay 0 \
  --responsibility-weight 1.0 \
  --anchor-weight 1e-4 \
  --persistence 3 \
  --frames-per-snippet 16 \
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
  --gt-path ../vadmy_data/annotations/xd/gt.npy \
  --probe-model ../vadmy_data/neuron_responsibility/xd/probe_v2/probe_best.pth \
  --out-dir ../vadmy_data/neuron_responsibility/xd/desc_gradual \
  --max-epoch 10 \
  --warmup-epochs 2 \
  --batch-size 96 \
  --correction-lr 5e-5 \
  --head-lr 1e-5 \
  --temporal-lr 1e-6 \
  --weight-decay 0 \
  --responsibility-weight 1.0 \
  --anchor-weight 1e-4 \
  --persistence 3 \
  --frames-per-snippet 16 \
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
  --gt-path ../vadmy_data/annotations/xd/gt.npy \
  --probe-model ../vadmy_data/neuron_responsibility/xd/probe_v2/probe_best.pth \
  --out-dir ../vadmy_data/neuron_responsibility/xd/lagovad_gradual \
  --max-epoch 20 \
  --warmup-epochs 2 \
  --batch-size 64 \
  --correction-lr 5e-5 \
  --head-lr 1e-5 \
  --temporal-lr 1e-6 \
  --weight-decay 0 \
  --responsibility-weight 1.0 \
  --anchor-weight 1e-4 \
  --persistence 3 \
  --frames-per-snippet 16 \
  --num-workers 4 \
  --seed 234 \
  --device cuda
```

Stage B 输出 `checkpoint_last.pth`、`model_best.pth`、`history.jsonl` 和 `parameter_report.json`。UCF 按官方帧级 AUC 选优，XD 按官方帧级 AP 选优；DSANet/UCF 还保持作者的每 1280 个训练样本验证一次。中断后使用完全相同的命令并在末尾添加 `--resume`。

模型选择依据来自各自发布代码：DSANet/UCF 固定训练间隔按帧级 AUC，DSANet/XD 每 epoch 按帧级 AP；LaGoVAD 每 epoch 保存 UCF AUC / XD AP 最优 checkpoint。DeSC 发布包没有训练脚本，只有官方 UCF AUC / XD AP 测试代码，因此采用相同主指标并每 epoch 验证，不虚构它未发布的 step 频率。

## 5. 正式评测

三个 baseline 的正式帧级异常结果均使用 `metrics.json` 中的 `binary` AUC/AP。`semantic` 是附加诊断；DSANet 额外计算 detection mAP。`fused_diagnostic` 只分析 neuron 与 baseline 的互补性，不作为正式主结果或选模指标。

### DSANet / UCF-Crime

```bash
python neuron_responsibility/evaluate.py \
  --baseline dsanet \
  --dataset ucf \
  --baseline-root baseline/DSANet \
  --baseline-weight ../vadmy_data/model/DSANet/model_ucf.pth \
  --joint-model ../vadmy_data/neuron_responsibility/ucf/dsanet_gradual/model_best.pth \
  --probe-model ../vadmy_data/neuron_responsibility/ucf/probe_v2/probe_best.pth \
  --test-list ../vadmy_data/neuron_responsibility/ucf/aligned_test.csv \
  --gt-path ../vadmy_data/annotations/ucf/gt.npy \
  --gt-segment-path ../vadmy_data/annotations/ucf/gt_segment.npy \
  --gt-label-path ../vadmy_data/annotations/ucf/gt_label.npy \
  --out-dir ../vadmy_data/neuron_responsibility/ucf/dsanet_gradual/evaluation \
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
  --joint-model ../vadmy_data/neuron_responsibility/ucf/desc_gradual/model_best.pth \
  --probe-model ../vadmy_data/neuron_responsibility/ucf/probe_v2/probe_best.pth \
  --test-list ../vadmy_data/neuron_responsibility/ucf/aligned_test.csv \
  --gt-path ../vadmy_data/annotations/ucf/gt.npy \
  --out-dir ../vadmy_data/neuron_responsibility/ucf/desc_gradual/evaluation \
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
  --joint-model ../vadmy_data/neuron_responsibility/ucf/lagovad_gradual/model_best.pth \
  --probe-model ../vadmy_data/neuron_responsibility/ucf/probe_v2/probe_best.pth \
  --test-list ../vadmy_data/neuron_responsibility/ucf/aligned_test.csv \
  --gt-path ../vadmy_data/annotations/ucf/gt.npy \
  --out-dir ../vadmy_data/neuron_responsibility/ucf/lagovad_gradual/evaluation \
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
  --joint-model ../vadmy_data/neuron_responsibility/xd/dsanet_gradual/model_best.pth \
  --probe-model ../vadmy_data/neuron_responsibility/xd/probe_v2/probe_best.pth \
  --test-list ../vadmy_data/neuron_responsibility/xd/aligned_test.csv \
  --gt-path ../vadmy_data/annotations/xd/gt.npy \
  --gt-segment-path ../vadmy_data/annotations/xd/gt_segment.npy \
  --gt-label-path ../vadmy_data/annotations/xd/gt_label.npy \
  --out-dir ../vadmy_data/neuron_responsibility/xd/dsanet_gradual/evaluation \
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
  --joint-model ../vadmy_data/neuron_responsibility/xd/desc_gradual/model_best.pth \
  --probe-model ../vadmy_data/neuron_responsibility/xd/probe_v2/probe_best.pth \
  --test-list ../vadmy_data/neuron_responsibility/xd/aligned_test.csv \
  --gt-path ../vadmy_data/annotations/xd/gt.npy \
  --out-dir ../vadmy_data/neuron_responsibility/xd/desc_gradual/evaluation \
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
  --joint-model ../vadmy_data/neuron_responsibility/xd/lagovad_gradual/model_best.pth \
  --probe-model ../vadmy_data/neuron_responsibility/xd/probe_v2/probe_best.pth \
  --test-list ../vadmy_data/neuron_responsibility/xd/aligned_test.csv \
  --gt-path ../vadmy_data/annotations/xd/gt.npy \
  --out-dir ../vadmy_data/neuron_responsibility/xd/lagovad_gradual/evaluation \
  --frames-per-snippet 16 \
  --temperature 1.0 \
  --device cuda
```

评测输出为 `metrics.json`、`per_video.csv` 和可恢复的逐视频 `scores`。如果模型、probe、测试清单或评测参数发生变化，程序会拒绝复用旧缓存；此时添加 `--clean` 或换一个 `--out-dir`。

## 6. 开销与冻结范围

默认 `K=768, H=128` 的 probe 约含 0.151M 参数，FP32 权重约 0.6 MB；每个 snippet 约 0.148M MAC，长度 256 的单视频约 38M MAC。选中 neuron 文件额外占 `T × 768 × 4` bytes，即约 3 KB/snippet。

| 阶段 | CLIP | Probe | Baseline |
|---|---|---|---|
| Stage A | 不执行 | 训练 | 不加载 |
| DSANet Stage B | 冻结 | 冻结 | heads + 最后一层 temporal + gc2/gc4/linear |
| DeSC Stage B | 冻结 | 冻结 | 双流 heads + 各自最后时序块 |
| LaGoVAD Stage B | 冻结 | 冻结 | bin/sim heads + temporal encoder 最后一层 |

DeSC 是双流前向，因此三者中训练/推理开销最高。其余新增开销主要来自很小的冻结 probe 前向。
