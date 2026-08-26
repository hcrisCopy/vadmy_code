# 冻结Baseline的U形双注入

## 唯一变化

本实验与 Shift-Global768 冻结baseline残差实验保持一致：

- 正样本仍是异常训练视频内 frozen baseline 分数最高的10% snippets；
- 负样本仍是同一视频内分数最低的10% snippets；
- 正常视频仍只估计hidden z-score；
- 仍然每层Top-64，共768维；
- baseline所有参数仍然冻结。

唯一变化是把一次早期残差注入改成早期＋后期双注入。

## 结构

```text
                         ┌─ zero-init early projection ─→ 原CLIP输入
768D selected neurons ─→ 共享神经元主干
                         └─ zero-init late projection  ─→ 时序编码后视觉瓶颈

原CLIP + early residual → 冻结时序模块 → + late residual
                                           ↓
                              冻结视觉文本交互与打分头
```

它不是带解码器的U-Net，而是U形跨深度跳连：同一神经元证据一条路径参与时序建模，另一条路径绕过冻结时序模块，防止细粒度异常证据被平滑掉。

三个baseline的后期共同位置：

| baseline | early | late |
|---|---|---|
| DSANet | Transformer/GCN之前 | `encode_video`之后、classifier/文本对齐/DNP之前 |
| DeSC | 两流时序模块之前 | 两流时序输出之后、classifier/文本对齐/GMP之前 |
| LaGoVAD | temporal encoder之前 | temporal encoder之后、视觉文本fusion/bin head之前 |

DeSC两条流复用同一个双注入模块，不增加baseline专属分支。

## 复用产物

为了保证控制变量，脚本直接复用：

```text
../vadmy_data/shift_residual_head_tuning/<dataset>/<baseline>/
  pseudo_scores/
  selection/
  aligned_train.csv
  aligned_test.csv
```

缺少时脚本会调用上一方案已经审查过的准备程序补齐，但不会运行“解冻打分头”的训练。

## 输出

```text
../vadmy_data/shift_u_dual_injection/<dataset>/<baseline>/
  training/parameter_report.json
  training/checkpoint_last.pth
  training/model_best.pth
  training/history.jsonl
  evaluation/metrics.json
  diagnostics/reused_selection_evidence.png
  diagnostics/u_branch_diagnostics.png
  diagnostics/late_early_energy_ratio.png
```

正式命令见 [COMMANDS.md](COMMANDS.md)，方案边界与风险见 [IDEA_REVIEW.md](IDEA_REVIEW.md)。
