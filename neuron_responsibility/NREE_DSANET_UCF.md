# NREE：DSANet / UCF-Crime

NREE 保留作者 DSANet 作为主路径。DANCE 已探测的类别神经元只负责选择低秩类别专家；快、慢专家学习突发和持续事件，当前模型峰值只用于生成停止梯度的完整事件软目标。神经元探测和路由不读取 DSANet 异常分数。

直接复用 `dance_v1/circuits` 中的两层紧凑神经元及 CSV，不重新提取 CLIP hidden states。初始专家输出严格为零，作者权重作为第一个候选参与选模；之后仍按 DSANet 的 UCF 帧级 AUC、每 1280 个训练样本选模。

## 1. 正式训练

```bash
python neuron_responsibility/train_event_experts.py \
  --baseline dsanet \
  --dataset ucf \
  --baseline-root baseline/DSANet \
  --baseline-weight ../vadmy_data/model/DSANet/model_ucf.pth \
  --train-list ../vadmy_data/neuron_responsibility/ucf/dance_v1/circuits/train.csv \
  --val-list ../vadmy_data/neuron_responsibility/ucf/dance_v1/circuits/test.csv \
  --gt-path ../vadmy_data/annotations/ucf/gt.npy \
  --atlas ../vadmy_data/neuron_responsibility/ucf/dance_v1/circuits/definition_circuits.json \
  --out-dir ../vadmy_data/neuron_responsibility/ucf/nree_v1/dsanet \
  --max-epoch 10 \
  --head-start-epoch 1 \
  --temporal-start-epoch 2 \
  --full-start-epoch 4 \
  --batch-size 64 \
  --expert-lr 7e-5 \
  --baseline-lr 7e-6 \
  --weight-decay 0 \
  --route-weight 0.10 \
  --event-weight 0.05 \
  --normal-weight 0.10 \
  --smooth-weight 0.01 \
  --anchor-weight 0.01 \
  --expert-rank 32 \
  --slow-dilation 4 \
  --route-top-fraction 0.10 \
  --frames-per-snippet 16 \
  --dsanet-ucf-eval-samples 1280 \
  --checkpoint-steps 10 \
  --num-workers 4 \
  --seed 234 \
  --device cuda \
  --clean
```

输出位于 `../vadmy_data/neuron_responsibility/ucf/nree_v1/dsanet/`：`model_best.pth` 是作者初始权重与全部训练检查点中 AUC 最高者，`checkpoint_last.pth` 用于中断恢复，`history.jsonl` 和 `parameter_report.json` 记录训练与开销。

中断后使用以下完整命令恢复：

```bash
python neuron_responsibility/train_event_experts.py \
  --baseline dsanet \
  --dataset ucf \
  --baseline-root baseline/DSANet \
  --baseline-weight ../vadmy_data/model/DSANet/model_ucf.pth \
  --train-list ../vadmy_data/neuron_responsibility/ucf/dance_v1/circuits/train.csv \
  --val-list ../vadmy_data/neuron_responsibility/ucf/dance_v1/circuits/test.csv \
  --gt-path ../vadmy_data/annotations/ucf/gt.npy \
  --atlas ../vadmy_data/neuron_responsibility/ucf/dance_v1/circuits/definition_circuits.json \
  --out-dir ../vadmy_data/neuron_responsibility/ucf/nree_v1/dsanet \
  --max-epoch 10 \
  --head-start-epoch 1 \
  --temporal-start-epoch 2 \
  --full-start-epoch 4 \
  --batch-size 64 \
  --expert-lr 7e-5 \
  --baseline-lr 7e-6 \
  --weight-decay 0 \
  --route-weight 0.10 \
  --event-weight 0.05 \
  --normal-weight 0.10 \
  --smooth-weight 0.01 \
  --anchor-weight 0.01 \
  --expert-rank 32 \
  --slow-dilation 4 \
  --route-top-fraction 0.10 \
  --frames-per-snippet 16 \
  --dsanet-ucf-eval-samples 1280 \
  --checkpoint-steps 10 \
  --num-workers 4 \
  --seed 234 \
  --device cuda \
  --resume
```

## 2. 正式评测

```bash
python neuron_responsibility/evaluate_event_experts.py \
  --baseline dsanet \
  --dataset ucf \
  --baseline-root baseline/DSANet \
  --baseline-weight ../vadmy_data/model/DSANet/model_ucf.pth \
  --test-list ../vadmy_data/neuron_responsibility/ucf/dance_v1/circuits/test.csv \
  --gt-path ../vadmy_data/annotations/ucf/gt.npy \
  --atlas ../vadmy_data/neuron_responsibility/ucf/dance_v1/circuits/definition_circuits.json \
  --model-path ../vadmy_data/neuron_responsibility/ucf/nree_v1/dsanet/model_best.pth \
  --out-dir ../vadmy_data/neuron_responsibility/ucf/nree_v1/evaluation \
  --frames-per-snippet 16 \
  --device cuda \
  --clean
```

输出 `metrics.json`、`per_video.csv` 和可复用的 `frame_scores.npz`，同一次运行同时报告发布权重和 NREE 的帧级 AUC/AP。若 UCF AUC 绝对提升不足 0.3 个百分点，不调整测试集参数；先停止并检查事件目标和路由诊断。
