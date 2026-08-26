# 责任层双专家（暂不使用 KNN）

## 方法在做什么

原方法失败在于：冻结 CLIP 文本相似度会从每个异常视频强制选峰值，然后让 baseline 学这些有噪标签。

新方法分成两个互不依赖的专家：

```text
责任探测选中的完整 CLIP 层
    -> 小型 Adapter + 正常视觉原型
    -> 语义专家曲线

作者 baseline 512-D 特征
    -> 作者时间主干
    -> baseline 时间专家曲线

两条曲线分别做公开的 CPL 多尺度片段精炼
    -> 两者都支持的异常片段保留
    -> 分歧片段设为 ignore
    -> 纯正常视频保持确定的 0 标签
    -> 只训练 baseline 最终二分类头
```

责任层探测不读取 baseline 分数。baseline 分数只在第二阶段作为独立时间证据参与伪标签校验。

## 为什么比旧方案干净

- 没有 KNN、视频拼接或光流。
- 不反传 CLIP，不提取 patch token。
- hidden states 不再直接充当真值，而是先训练成可独立验证的语义专家。
- 三个 baseline 统一在 anomaly curve 和 binary head 接口，不假装它们具有相同的视觉文本融合结构。
- DSANet 只训练 `classifier`；DeSC 只训练两个独立 `classifier`；LaGoVAD 只训练 `bin_head`。
- 每项损失的实际梯度路径可生成表格和柱状图。

## 主要产物

| 产物 | 用途 |
|---|---|
| `normal_prototype.npz` | 纯正常视频的分层视觉原型 |
| `semantic_expert_best.pth` | 不使用 baseline 分数训练的责任层语义专家 |
| `expert_scores.csv` | 每个 crop 的语义/时间专家曲线索引 |
| `consensus_labels.csv` | 正常=0、共识异常=soft positive、其余=-1 ignore |
| `gradient_path_audit.png` | 证明新增损失到底更新哪些模块 |
| `expert_agreement_matrix.png` | 查看两专家一致、分歧比例 |
| `class_layer_evidence_heatmap.png` | 查看每类异常由哪些完整层提供证据 |
| `layer_gate_weights.png` | 比较责任探测先验与训练后的整层权重 |
| `temporal_consensus_examples.png` | 查看保留的时间段是否合理 |
| `model_best.pth` | 按 baseline 作者指标选出的 head-only 模型 |

正式命令见 `responsibility_cross_expert/COMMANDS.md`。所有输出都在同级 `../vadmy_data/responsibility_cross_expert/`。
