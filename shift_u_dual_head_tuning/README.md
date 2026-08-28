# U形双注入＋打分头微调

## 控制变量

这条实验把前两条方案组合起来：

- 正负样本仍是同一异常视频内 frozen baseline 分数 top 10% / bottom 10%；
- 仍然每层Top-64，共768个CLIP hidden神经元；
- 仍使用共享主干的early/late U形双注入；
- 相对“冻结baseline的U形双注入”，唯一变化是解冻原二分类打分头。

## 训练结构

```text
                         ┌→ early residual → 原CLIP输入
768D神经元 → 共享主干 ───┤
                         └→ late residual  → 时序输出

原CLIP + early → 冻结时序模块 → + late → 冻结的后续预测路径 → 可训练打分头
```

可训练参数：

| 部分 | DSANet | DeSC | LaGoVAD |
|---|---|---|---|
| U形双注入 | 共享主干、双出口、双门控 | 相同模块供两流共享 | 共享主干、双出口、双门控 |
| baseline打分头 | `classifier` | 两流各自的`classifier` | `bin_head` |

冻结参数：CLIP、文本编码/Prompt、时序模块、视觉文本交互、DSANet DNP、DeSC GMP、语义分类头。这里的“打分头”只指最终生成 snippet 二分类分数的线性层，不把其前面的融合或时序模块算入打分头。

训练开始时两个注入出口均为零，初始前向等于作者checkpoint；打分头从作者权重开始，不重新初始化。

## 共享输入产物

三条Shift对照实验共同使用：

```text
../vadmy_data/shift_residual_head_tuning/<dataset>/<baseline>/
  pseudo_scores/
  selection/
  aligned_train.csv
  aligned_test.csv
```

因此差异不会来自重新打分、重新选神经元或重新对齐特征。

## 输出

```text
../vadmy_data/shift_u_dual_head_tuning/<dataset>/<baseline>/
  training/parameter_report.json
  training/checkpoint_last.pth
  training/model_best.pth
  training/history.jsonl
  evaluation/metrics.json
  diagnostics/reused_selection_evidence.png
  diagnostics/u_dual_head_diagnostics.png
```

checkpoint只保存双注入参数、打分头参数和优化器状态，不复制整个冻结baseline。

运行命令见 [COMMANDS.md](COMMANDS.md)，方案质量见 [IDEA_REVIEW.md](IDEA_REVIEW.md)。
