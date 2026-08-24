# Neuron-Decoupled Responsibility Learning

本目录实现一套共享方法，适配 DSANet、DeSC、LaGoVAD，以及 UCF-Crime、XD-Violence。baseline 源码不修改，CLIP 始终冻结。

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
  --normal-stat-snippets-per-video 256 \
  --sigma-min 1e-6
```

输出包括 `selected_neurons.json`、normal mean/std、tail statistics 和可恢复的逐视频 `bag_cache`。只有需要强制重算时才在原命令后添加 `--clean`。

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
  --out-csv ../vadmy_data/neuron_responsibility/xd/aligned_train.csv
```

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

### XD-Violence

```bash
python neuron_responsibility/train_probe.py \
  --dataset xd \
  --train-list ../vadmy_data/neuron_responsibility/xd/aligned_train.csv \
  --val-list ../vadmy_data/neuron_responsibility/xd/aligned_test.csv \
  --out-dir ../vadmy_data/neuron_responsibility/xd/probe \
  --visual-length 256 \
  --hidden-width 128 \
  --max-epoch 10 \
  --batch-size 96 \
  --lr 1e-5 \
  --weight-decay 1e-4 \
  --sparsity-weight 1e-3 \
  --normal-instance-weight 0.25 \
  --num-workers 4 \
  --seed 234 \
  --device cuda
```

Stage A 输出 `probe_best.pth`、`checkpoint_last.pth` 和 `history.jsonl`。中断后使用完全相同的命令并在末尾添加 `--resume`。

## 4. Stage B：责任引导的 baseline 训练

Stage B 中 CLIP 和 probe 固定。DSANet、LaGoVAD 训练时序/融合模块及 heads。DeSC 作者发布目录没有训练脚本和可核验的 GMP 训练损失，因此限制为 head-only。

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
  --out-dir ../vadmy_data/neuron_responsibility/xd/dsanet \
  --train-scope temporal_heads \
  --max-epoch 10 \
  --batch-size 96 \
  --lr 1e-5 \
  --weight-decay 0 \
  --responsibility-weight 1.0 \
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
  --out-dir ../vadmy_data/neuron_responsibility/xd/desc \
  --train-scope heads \
  --max-epoch 10 \
  --batch-size 96 \
  --sensitivity-lr 1e-3 \
  --consistency-lr 1e-5 \
  --weight-decay 1e-3 \
  --responsibility-weight 1.0 \
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
  --out-dir ../vadmy_data/neuron_responsibility/xd/lagovad \
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

Stage B 输出 `checkpoint_last.pth`、`model_best.pth`、`history.jsonl` 和 `parameter_report.json`。后者记录实际可训练张量，程序强制断言 CLIP 可训练参数为 0。中断后使用完全相同的命令并在末尾添加 `--resume`。

## 5. 正式评测

三个 baseline 的正式帧级异常结果均使用 `metrics.json` 中的 `binary` AUC/AP。`semantic` 是附加诊断；DSANet 额外计算 detection mAP。`fused_diagnostic` 只分析 neuron 与 baseline 的互补性，不作为正式主结果或选模指标。

### DSANet / UCF-Crime

```bash
python neuron_responsibility/evaluate.py \
  --baseline dsanet \
  --dataset ucf \
  --baseline-root baseline/DSANet \
  --baseline-weight ../vadmy_data/model/DSANet/model_ucf.pth \
  --joint-model ../vadmy_data/neuron_responsibility/ucf/dsanet/model_best.pth \
  --probe-model ../vadmy_data/neuron_responsibility/ucf/probe/probe_best.pth \
  --test-list ../vadmy_data/neuron_responsibility/ucf/aligned_test.csv \
  --gt-path ../vadmy_data/annotations/ucf/gt.npy \
  --gt-segment-path ../vadmy_data/annotations/ucf/gt_segment.npy \
  --gt-label-path ../vadmy_data/annotations/ucf/gt_label.npy \
  --out-dir ../vadmy_data/neuron_responsibility/ucf/dsanet/evaluation \
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
  --joint-model ../vadmy_data/neuron_responsibility/ucf/desc/model_best.pth \
  --probe-model ../vadmy_data/neuron_responsibility/ucf/probe/probe_best.pth \
  --test-list ../vadmy_data/neuron_responsibility/ucf/aligned_test.csv \
  --gt-path ../vadmy_data/annotations/ucf/gt.npy \
  --out-dir ../vadmy_data/neuron_responsibility/ucf/desc/evaluation \
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
  --joint-model ../vadmy_data/neuron_responsibility/ucf/lagovad/model_best.pth \
  --probe-model ../vadmy_data/neuron_responsibility/ucf/probe/probe_best.pth \
  --test-list ../vadmy_data/neuron_responsibility/ucf/aligned_test.csv \
  --gt-path ../vadmy_data/annotations/ucf/gt.npy \
  --out-dir ../vadmy_data/neuron_responsibility/ucf/lagovad/evaluation \
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
  --joint-model ../vadmy_data/neuron_responsibility/xd/dsanet/model_best.pth \
  --probe-model ../vadmy_data/neuron_responsibility/xd/probe/probe_best.pth \
  --test-list ../vadmy_data/neuron_responsibility/xd/aligned_test.csv \
  --gt-path ../vadmy_data/annotations/xd/gt.npy \
  --gt-segment-path ../vadmy_data/annotations/xd/gt_segment.npy \
  --gt-label-path ../vadmy_data/annotations/xd/gt_label.npy \
  --out-dir ../vadmy_data/neuron_responsibility/xd/dsanet/evaluation \
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
  --joint-model ../vadmy_data/neuron_responsibility/xd/desc/model_best.pth \
  --probe-model ../vadmy_data/neuron_responsibility/xd/probe/probe_best.pth \
  --test-list ../vadmy_data/neuron_responsibility/xd/aligned_test.csv \
  --gt-path ../vadmy_data/annotations/xd/gt.npy \
  --out-dir ../vadmy_data/neuron_responsibility/xd/desc/evaluation \
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
  --joint-model ../vadmy_data/neuron_responsibility/xd/lagovad/model_best.pth \
  --probe-model ../vadmy_data/neuron_responsibility/xd/probe/probe_best.pth \
  --test-list ../vadmy_data/neuron_responsibility/xd/aligned_test.csv \
  --gt-path ../vadmy_data/annotations/xd/gt.npy \
  --out-dir ../vadmy_data/neuron_responsibility/xd/lagovad/evaluation \
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
| DSANet Stage B | 冻结 | 冻结 | temporal、GCN、linear、heads |
| DeSC Stage B | 冻结 | 冻结 | 两个 stream 的 heads |
| LaGoVAD Stage B | 冻结 | 冻结 | temporal encoder、fusion、heads |

DeSC 是双流前向，因此三者中训练/推理开销最高。其余新增开销主要来自很小的冻结 probe 前向。
