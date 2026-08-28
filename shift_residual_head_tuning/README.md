# Shift残差＋打分头微调

## 一句话说明

完全复现旧 Shift-Global768 的正负样本、神经元选择和残差注入，只把 baseline 的二分类打分头从冻结改为可训练。

## 方法流程

```text
冻结的作者baseline
  -> 异常训练视频内 top 10% / bottom 10% 分数片段
  -> 每层按跨视频 ShiftScore 选 Top-64
  -> 12层 × 64维 = 768D可解释神经元特征
  -> 768→1024→1024→512零初始化残差
  -> 原512D CLIP + 门控残差
  -> 冻结的baseline时序/文本模块
  -> 可训练的原二分类打分头
```

正样本和负样本来自同一个异常视频，避免场景差异主导神经元选择。正常训练视频只估计 hidden z-score 的均值和标准差，不充当负样本。

## 唯一解冻范围

| baseline | 可训练的baseline参数 | 仍冻结 |
|---|---|---|
| DSANet | `classifier` | CLIP、时序、GCN、DNP、文本交互及其他头 |
| DeSC | sensitivity/consistency 两流的 `classifier` | 两流CLIP、时序、GMP、文本交互及其他头 |
| LaGoVAD | `bin_head` | CLIP文本编码、时序、融合、相似度头 |

残差分支始终可训练。训练开始时最后一层为零，所以模型初始输出严格等于作者 checkpoint；门控初值 `sigmoid(-4)≈0.018`。

## 运行

完整命令见 [COMMANDS.md](COMMANDS.md)。六个脚本都已写死数据集、权重、学习率和输出路径，不需要手动替换参数。中断后重跑同一命令会复用打分、选择和特征产物，并自动 `--resume` 训练。

## 关键产物

```text
../vadmy_data/shift_residual_head_tuning/<dataset>/<baseline>/
  pseudo_scores/group_scores.csv
  selection/selected_neurons.json
  selection/video_pairs.csv
  aligned_train.csv / aligned_test.csv
  training/parameter_report.json
  training/checkpoint_last.pth / model_best.pth
  training/history.jsonl
  evaluation/metrics.json
  diagnostics/selection_evidence.png
  diagnostics/training_diagnostics.png
```

`parameter_report.json` 是解冻范围审计；出现任何 CLIP、时序或文本可训练参数都应停止实验。
