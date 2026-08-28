# TRACE：DSANet / UCF-Crime

TRACE 用视频级标签和正常视频统计探测 CLIP CLS hidden states 中的原始维度，不读取 baseline 异常分数。训练分三段：先学习神经元证据，再解冻 DSANet 分类头，最后额外解冻最后一个时序块。CLIP 始终冻结。

以下命令均在 `vadmy_code` 目录运行，现有 hidden states、对齐 CSV、正常统计和作者 logits 都直接复用。

## 1. 准备神经元

```bash
python neuron_responsibility/prepare_trace.py \
  --dataset ucf \
  --train-csv ../vadmy_data/neuron_responsibility/ucf/cacc_v1/prepared/train.csv \
  --normal-artifact ../vadmy_data/neuron_responsibility/ucf/cacc_v1/prepared/cacc_artifact.npz \
  --out-dir ../vadmy_data/neuron_responsibility/ucf/trace_v1/prepared \
  --snippets-per-video 256 \
  --top-fraction 0.10 \
  --neurons-per-layer 64 \
  --layer-coverage 0.80 \
  --max-layers 3 \
  --clean
```

产物是 `../vadmy_data/neuron_responsibility/ucf/trace_v1/prepared/trace_artifact.npz` 和选择报告。选层由神经元证据量决定，不固定第几层。

## 2. 训练

```bash
python neuron_responsibility/train_trace.py \
  --baseline dsanet \
  --baseline-root baseline/DSANet \
  --baseline-weight ../vadmy_data/model/DSANet/model_ucf.pth \
  --dataset ucf \
  --train-list ../vadmy_data/neuron_responsibility/ucf/cacc_v1/prepared/train.csv \
  --val-list ../vadmy_data/neuron_responsibility/ucf/cacc_v1/prepared/test.csv \
  --artifact ../vadmy_data/neuron_responsibility/ucf/trace_v1/prepared/trace_artifact.npz \
  --gt-path baseline/DSANet/list/gt_ucf.npy \
  --teacher-cache ../vadmy_data/neuron_responsibility/ucf/dsanet_boundary_conditioning_v1/author_train_logits.pth \
  --out-dir ../vadmy_data/neuron_responsibility/ucf/trace_v1/train \
  --max-epoch 6 \
  --evidence-epochs 2 \
  --temporal-start-epoch 4 \
  --batch-size 64 \
  --hidden-width 128 \
  --active-neurons 96 \
  --dropout 0.10 \
  --evidence-lr 5e-5 \
  --head-lr 2e-6 \
  --temporal-lr 5e-7 \
  --weight-decay 0.0 \
  --evidence-normal-weight 0.10 \
  --evidence-agreement-weight 0.20 \
  --evidence-smooth-weight 0.001 \
  --evidence-sparse-weight 0.01 \
  --pseudo-weight 0.50 \
  --ranking-weight 0.20 \
  --ap-weight 0.10 \
  --semantic-weight 0.10 \
  --event-weight 0.001 \
  --preservation-weight 0.20 \
  --baseline-anchor-weight 0.005 \
  --normal-low-quantile 0.50 \
  --normal-high-quantile 0.95 \
  --hard-normal-fraction 0.05 \
  --grow-steps 8 \
  --frames-per-snippet 16 \
  --eval-samples 1280 \
  --num-workers 4 \
  --cache-videos 8 \
  --seed 234 \
  --device cuda \
  --clean
```

训练按 DSANet 的 UCF 规则以 frame AUC 选择 `../vadmy_data/neuron_responsibility/ucf/trace_v1/train/model_best.pth`，并每适配 1280 个样本评测一次。中断后把末尾的 `--clean` 改为 `--resume`，其余参数保持不变即可续训。

## 3. 正式评测

```bash
python neuron_responsibility/evaluate_trace.py \
  --baseline dsanet \
  --baseline-root baseline/DSANet \
  --baseline-weight ../vadmy_data/model/DSANet/model_ucf.pth \
  --dataset ucf \
  --test-list ../vadmy_data/neuron_responsibility/ucf/cacc_v1/prepared/test.csv \
  --gt-path baseline/DSANet/list/gt_ucf.npy \
  --model-path ../vadmy_data/neuron_responsibility/ucf/trace_v1/train/model_best.pth \
  --out-dir ../vadmy_data/neuron_responsibility/ucf/trace_v1/evaluation \
  --frames-per-snippet 16 \
  --device cuda \
  --clean
```

结果写入 `../vadmy_data/neuron_responsibility/ucf/trace_v1/evaluation/metrics.json`。同时报告作者模型、TRACE 适配模型以及两类证据的独立指标；逐视频分数会缓存，评测中断后去掉 `--clean` 重跑即可续接。
