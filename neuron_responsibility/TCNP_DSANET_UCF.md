# TCNP：DSANet / UCF-Crime 完整运行指令

TCNP（Text-Conditioned Normal-Prototype Neuron Probe）只使用训练视频级标签、冻结 CLIP 文本梯度选出的原始 CLS hidden-state 维度和正常训练视频原型，不读取 DSANet 异常分数。神经元定义始终是“某一层 CLS hidden state 的某一个原始维度”。

本轮直接复用已经生成的 DANCE 紧凑回路产物：

- `../vadmy_data/neuron_responsibility/ucf/dance_v1/circuits/definition_circuits.json`
- `../vadmy_data/neuron_responsibility/ucf/dance_v1/circuits/train.csv`
- `../vadmy_data/neuron_responsibility/ucf/dance_v1/circuits/test.csv`

这些是 `vadmy_data` 内本项目自己的产物；CSV 中记录的 `../vad_data` 特征路径只是复用数据文件，不跨项目 import 任何代码。

## 1. 训练并审计神经元探针

```bash
python neuron_responsibility/train_tcnp_probe.py \
  --dataset ucf \
  --train-list ../vadmy_data/neuron_responsibility/ucf/dance_v1/circuits/train.csv \
  --test-list ../vadmy_data/neuron_responsibility/ucf/dance_v1/circuits/test.csv \
  --atlas ../vadmy_data/neuron_responsibility/ucf/dance_v1/circuits/definition_circuits.json \
  --gt-path ../vadmy_data/annotations/ucf/gt.npy \
  --out-dir ../vadmy_data/neuron_responsibility/ucf/tcnp_v1/probe \
  --epochs 8 \
  --batch-size 64 \
  --lr 3e-3 \
  --weight-decay 1e-4 \
  --top-fraction 0.10 \
  --consistency-weight 0.20 \
  --validation-fraction 0.20 \
  --normal-quantile 0.98 \
  --prototype-count 32 \
  --prototype-samples 50000 \
  --candidate-threshold 0.50 \
  --frames-per-snippet 16 \
  --num-workers 4 \
  --seed 234 \
  --device cuda
```

质量门槛写入 `gate_report.json`：正常训练视频片段误报率不高于 2%，异常训练视频候选覆盖率不低于 50%，且已覆盖视频的异常类别准确率不低于 30%。文本回路语义是主定位信号；正常原型距离和跨层一致性作为独立审计量，不强制取三者交集。划分按视频完成，十裁剪只取一个代表，杜绝同视频泄漏。测试帧标注只产生一次最终诊断，不参与门槛、训练或阈值校准。

中断后原命令末尾添加 `--resume`；确认废弃旧输出时添加 `--clean`，二者不能同时使用。

## 2. 质量门槛通过后适配 DSANet

下列程序会自行读取 `gate_report.json`。门槛失败时会主动终止，不允许低质量伪定位再次污染 baseline。

```bash
python neuron_responsibility/train_definition_evidence.py \
  --baseline dsanet \
  --baseline-root baseline/DSANet \
  --baseline-weight ../vadmy_data/model/DSANet/model_ucf.pth \
  --dataset ucf \
  --train-list ../vadmy_data/neuron_responsibility/ucf/dance_v1/circuits/train.csv \
  --val-list ../vadmy_data/neuron_responsibility/ucf/dance_v1/circuits/test.csv \
  --atlas ../vadmy_data/neuron_responsibility/ucf/dance_v1/circuits/definition_circuits.json \
  --probe-model ../vadmy_data/neuron_responsibility/ucf/tcnp_v1/probe/probe_best.pth \
  --gate-report ../vadmy_data/neuron_responsibility/ucf/tcnp_v1/probe/gate_report.json \
  --gt-path ../vadmy_data/annotations/ucf/gt.npy \
  --teacher-cache ../vadmy_data/neuron_responsibility/ucf/dsanet_boundary_conditioning_v1/author_train_logits.pth \
  --out-dir ../vadmy_data/neuron_responsibility/ucf/tcnp_v1/dsanet \
  --max-epoch 6 \
  --temporal-start-epoch 2 \
  --reference-start-epoch 4 \
  --batch-size 64 \
  --head-lr 2e-5 \
  --temporal-lr 2e-6 \
  --reference-lr 1e-6 \
  --binary-weight 0.20 \
  --semantic-weight 0.20 \
  --normal-weight 0.10 \
  --dnp-weight 0.10 \
  --preservation-weight 0.50 \
  --anchor-weight 0.01 \
  --dsanet-ucf-eval-samples 1280 \
  --pilot-samples 5120 \
  --pilot-min-gain 0.003 \
  --num-workers 4 \
  --seed 234 \
  --device cuda
```

解冻遵循 DSANet 结构逐步进行：epoch 1–2 仅 heads，epoch 3–4 为 temporal + heads，epoch 5–6 才加入 DSANet 的正常参考/DNP 相关小模块；CLIP 始终冻结。选模完全沿用作者 DSANet/UCF 的每 1280 个样本按帧级 AUC 保存最优模型。默认 pilot 在 5120 个样本时检查；若尚未比作者 AUC 高至少 0.003（0.3 个百分点），程序自动停止，不进入更昂贵的解冻阶段。

## 3. 正式评测

```bash
python neuron_responsibility/evaluate_definition_evidence.py \
  --baseline dsanet \
  --baseline-root baseline/DSANet \
  --baseline-weight ../vadmy_data/model/DSANet/model_ucf.pth \
  --dataset ucf \
  --test-list ../vadmy_data/neuron_responsibility/ucf/dance_v1/circuits/test.csv \
  --gt-path ../vadmy_data/annotations/ucf/gt.npy \
  --model-path ../vadmy_data/neuron_responsibility/ucf/tcnp_v1/dsanet/model_best.pth \
  --out-dir ../vadmy_data/neuron_responsibility/ucf/tcnp_v1/evaluation \
  --frames-per-snippet 16 \
  --device cuda
```

## 4. 预计开销

探针只训练每个视频的一个代表 crop，UCF 共约 1610 个样本，而非 16100 个十裁剪样本。当前复用回路宽度为 1254、两层、13 个异常类别，探针只有约 2 万个可训练标量；32 个正常原型约占 0.16 MB。探针阶段通常明显低于 4 GB 显存。DSANet 适配阶段与之前 head/temporal 部分解冻相近，单卡 4090 可运行；CLIP 不执行也不反向传播。
