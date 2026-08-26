# Baseline专属探测＋单残差＋冻结baseline

## 一句话说明

每个baseline先用自己的冻结异常分数，在同一异常视频内选择top 10%和bottom 10%片段；由这些片段探测768个CLIP神经元，再通过一个零初始化残差模块注入时序编码器之前。整个baseline保持冻结。

## 训练结构

```text
当前baseline自己的冻结分数
          ↓
异常视频内 top 10% / bottom 10%
          ↓
每层Top-64神经元，共12×64=768维
          ↓
LayerNorm → 3层MLP → 512维残差 → 小门控
                                  ↓
原512维CLIP特征 ──────────────── 加法
                                  ↓
                 冻结baseline的时序与预测路径
                                  ↓
                            snippet异常分数
```

只有外接残差模块训练。CLIP、文本编码器、时序模块、视觉文本交互模块和打分头全部冻结，且运行时进行可训练参数审计。

### DeSC不是“两条流一起注入”

DeSC论文将Sensitivity与Consistency作为两个独立优化的模型。当前实验只把残差送入Sensitivity流，并只用该流的`MIL + visual-text alignment`训练；Consistency流从输入、权重到输出均保持作者模型不变，只在推理时参与概率集成。这避免了此前把同一残差强加给两个目标不同的流。

评估严格沿用作者发布协议：UCF采用256 snippet滑窗并对重叠预测平均，主指标使用binary score；XD先将整段视频线性缩放到256 snippet，预测后再还原时间长度，主指标使用`1 - normal semantic probability`。

## 为什么selection不能跨baseline复用

DSANet、DeSC和LaGoVAD对同一snippet产生的分数不同，所以top/bottom片段和最终选中的神经元也可能不同。本方案输出严格隔离为：

```text
../vadmy_data/shift_single_frozen/<dataset>/<baseline>/
```

`score_provenance.json`记录baseline、数据集、训练CSV和作者权重签名；`selection_provenance.json`再绑定`selected_neurons.json`的SHA256。训练发现错配会直接停止。

修正后的DeSC产物使用`desc_sensitivity_v2`目录，避免覆盖旧版“两流同时注入”的结果。

CLIP hidden states不依赖baseline，因此继续读取`../vad_data/`中的已有产物，不重复提取。

## 主要产物

```text
pseudo_scores/score_provenance.json
pseudo_scores/group_scores.csv
selection/selected_neurons.json
selection/selection_provenance.json
aligned_train.csv
aligned_test.csv
training/parameter_report.json
training/checkpoint_last.pth
training/model_best.pth
training/history.jsonl
evaluation/metrics.json
diagnostics/baseline_specific_selection.png
diagnostics/single_residual_training.png
```

命令见[COMMANDS.md](COMMANDS.md)，设计审查见[IDEA_REVIEW.md](IDEA_REVIEW.md)。
