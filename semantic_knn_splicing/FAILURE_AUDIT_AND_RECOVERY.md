# 失败审计与下一步方案

## 结论先行

当前方法失败不是单一超参数问题，而是三类问题叠加：

1. **根本设计问题**：把未经目标域训练的 CLIP 中间层文本相似度直接当成片段真值。UCF 实验中异常文本相对正常文本的平均边际为 `-0.0127`，但视频内动态阈值仍会从每个异常视频强制选出“相对最高”的片段。因此候选段覆盖率很高，不代表正确率高。
2. **关键迁移细节不等价**：借用了 LaGoVAD 的 KNN 拼接和伪监督形式，却没有复现它成立的前提、训练阶段和完整损失。尤其是 LaGoVAD 的动态合成用于 PreVAD 预训练，在 UCF/XD 域微调配置中其伪监督和负样本对比权重均为 0；它还包含我们遗漏的负样本对比损失。
3. **解冻路径错误且过多**：新增伪帧损失只监督 DSANet 的二分类 `logits1`，实际只更新 `mlp2 + classifier`；同时解冻的 `mlp1 + text_adapter` 收不到这项新监督，只会被作者原损失继续改写。5,248,513 个可训练参数中，约 3,148,288 个（60%）与新监督没有梯度连接。

所以，继续微调阈值、权重或学习率不能解决本质问题。下一步应从“单一语义曲线自我标注”改为“独立语义专家与原 baseline 时间专家交叉校验”，并把 baseline 的可训练范围缩到与新监督直接相连的二分类头。

---

## 1. 实验事实

| 模型 | UCF frame AUC | frame AP | 相对作者权重 |
|---|---:|---:|---:|
| DSANet 作者权重 | 89.4446 | 37.4196 | — |
| 当前方法最佳，sample 1,280 | 89.4983 | 34.8382 | AUC +0.0537，AP -2.5814 |
| 当前方法第 10 epoch | 85.9718 | 29.9767 | AUC -3.4729，AP -7.4429 |

训练损失持续下降，而 AUC/AP 下降，说明模型成功拟合了合成标签，但合成标签与真实帧级排序不一致。最佳点出现在第一次验证，随后持续退化，也说明从已经收敛的作者权重继续进行高自由度联合训练产生了遗忘。

## 2. 失败原因分级

### 2.1 决定性原因：语义分数没有经过目标域适配

我们当前流程是：选出第 8、12 层完整 CLS 表示，直接与冻结 CLIP 文本嵌入计算异常—正常边际，再用视频内阈值取连续峰值。这里存在三个跳跃：

- CLIP 的图文预训练相似度不等于监控视频中的异常性；
- 图像级可见语义不等于长视频中的时间边界；
- 视频内相对最大值不等于绝对正样本。

TPWNG 并没有直接用原始 CLIP 相似度产生伪标签，而是先用排序损失和分布不一致损失做域适配，再引入正常视觉提示，最后以正常性引导生成伪标签 [4]。这正好反证了我们省略的步骤不是装饰，而是语义分数可用的前提。

### 2.2 决定性原因：LaGoVAD 机制被拆开后不再等价

| LaGoVAD 原实现 | 当前实现 | 后果 |
|---|---|---|
| 在带有类别/描述的 PreVAD 片段上动态合成 | 在弱标签 UCF 长视频中先用原始文本分数猜异常段 | anchor 的可信度完全不同 |
| 每次取样在线动态合成 | 预生成 39,700 个固定样本 | 多样性下降，错误标签被重复训练 |
| 同时合成正常与异常样本 | 主要构造异常候选及其正常上下文 | 缺少合成正常的负 MIL 约束 |
| `L_dvs + L_neg + language-guided detection` | 只迁移 dense BCE 与正段 MIL | 丢失关键的困难负样本对齐 |
| DVS 用于 PreVAD 预训练 | UCF/XD finetune 配置把三项额外权重设为 0 | 把预训练增强误当成目标域微调方法 |

LaGoVAD 完整消融中，去掉负样本对比后 UCF 从 81.12 降到 77.66，约 -3.46；去掉动态合成后降到 79.18，约 -1.94 [3]。在 UCF 上，被遗漏的负样本对比贡献甚至大于我们重点迁移的拼接本身。

### 2.3 决定性实现问题：按模块名称解冻，而不是按梯度路径解冻

DSANet 二分类路径为：

```text
CLIP 512-D snippet
  -> temporal encoder
  -> visual_features
  -> visual_features + mlp2(visual_features)
  -> classifier
  -> logits1
```

文本路径为：

```text
text_adapter -> text_features -> mlp1 -> semantic logits
```

当前 dense/MIL 伪监督只读取 `logits1`。因此：

| 解冻模块 | 参数量 | 收到伪帧损失梯度？ | 判断 |
|---|---:|---:|---|
| `classifier` | 513 | 是 | 必要 |
| `mlp2` | 2,099,712 | 是 | 可选，但自由度偏大 |
| `mlp1` | 2,099,712 | 否 | 不应由二分类伪标签阶段解冻 |
| `text_adapter` | 1,048,576 | 否 | 需要语义/对比损失，不能靠二分类伪标签训练 |

这也解释了“解冻视觉文本交互模块”为什么没有把 hidden states 的语义证据传给二分类器：新增损失根本没有走到这些参数。

### 2.4 次要但真实的优化细节偏差

DSANet 作者代码：

- 主干和 normality refiner 使用两个优化器；
- refiner 使用 StableAdamW 和 100 iteration warm-up；
- 每个 epoch 结束重新载入历史最佳 checkpoint，再开始下一 epoch。

当前代码从作者最佳权重出发，使用一个新 AdamW、作者训练学习率和 cosine scheduler，并持续从当前状态训练。遗漏“每轮回到最佳点”会放大后期崩塌；从已收敛权重以训练初始学习率继续更新也过于激进。但由于第一次验证已经没有实际增益，这些细节只能解释持续退化，不能挽救错误监督。

### 2.5 可解释性没有真正进入最终模型

当前 selected layers 只用于离线生成伪段；最终 DSANet 输入仍是作者的 512-D CLIP 特征，不读取第 8/12 层表示。于是 hidden states 的作用完全被压缩成一批有噪标签。层选择本身可解释，但它不是“神经元级信息增强模型表示”；一旦伪标签不准，最终模型没有第二条路径利用这些层的信息。

---

## 3. 三个 baseline 的真正共性

### 3.1 共同计算接口

三个模型的最小公共结构不是“视觉文本融合”，而是：

```text
冻结的 512-D CLIP snippet 特征
  -> baseline 自己的时间建模
  -> snippet 级二分类打分器
  -> frame anomaly score
```

它们都有语义分支，但语义分支与最终二分类分数的连接方式不同。

| Baseline | 主要时间能力 | 文本/语义如何参与 | 最终二分类路径 | 论文中主要增益来源 |
|---|---|---|---|---|
| DSANet | Transformer/GCN + 正常模式重构 | 语义对齐主要是辅助分支；`logits1` 本身不依赖文本 | `visual -> mlp2 -> classifier` | Adapter、SG-NM、DCSA 协同；完整模型相对 VadCLIP 在 XD 为 +2.44 AP [1] |
| DeSC | 独立的敏感流 TCN+GT 与稳定流 GCN+GMP | 语义主要服务稳定性/类别一致性 | 两个独立二分类器，推理时平均 | 独立训练最关键：联合 86.18，独立流 88.46/88.35，融合 89.37 UCF [2] |
| LaGoVAD | 2 层 RoFormer | 文本经 co-attention/fusion 直接参与 language-guided detection | binary head + similarity head | 语言引导、DVS、困难负样本对比共同作用 [3] |

### 3.2 哪个模块最关键

答案分两层：

1. **对三个 baseline 都成立的最关键可训练接口：最终二分类打分头。** 新的片段级监督必须首先只更新真正输出 anomaly score 的 head。视觉文本融合不是三个模型共有的功能模块：LaGoVAD 的融合直接影响二分类，DSANet/DeSC 的二分类分支则可以不经过文本。
2. **最关键的性能能力不是同一个内部模块，而是“时间专家 + 语义专家的互补”。** DSANet 用正常模式补足 MIL 峰值，DeSC 用敏感/稳定双流互补，LaGoVAD 用语言定义和困难正常负样本补足视觉打分。共同规律是引入独立证据并控制冲突，而不是一起解冻更多参数。

因此不应再规定“三个 baseline 都解冻名为 mlp1/mlp2/fusion 的模块”。通用性应该定义在输入输出接口和训练原则上，而不是强行让内部参数名相同。

---

## 4. 有依据的恢复方向

### 4.1 将整层 hidden states 训练成独立语义专家

保留责任探测选出的少量完整层，但不再直接用冻结余弦相似度作为标签。对每个选中层只训练一个低开销投影/Adapter，CLIP 图像和文本 backbone 均冻结：

```text
selected CLS layers -> small layer projections -> semantic anomaly/category curve
```

训练依据：

- 视频级正常/异常排序损失；
- 纯正常视频构造 normality visual prompt/prototype；
- 异常文本与正常文本的对照边际；
- 层权重保持稀疏并输出类别—层贡献，保留可解释性。

TPWNG 证明正常性引导和目标域排序适配对 CLIP 伪标签是必要步骤 [4]；DSANet 的正常模式距离也显示正常建模能改善时间分离 [1]。这里不微调 CLIP，只训练预计算 hidden states 上的小投影，符合单卡开销限制。

### 4.2 不再自我标注：二分类时间专家与语义专家交叉校验

已有 baseline 的原始二分类曲线作为**时间专家**；selected-layer 语义分支作为**语义专家**。二者先独立训练，随后交换或共同过滤伪标签：

- 两者都高：高置信异常；
- 两者都低：高置信正常；
- 二者分歧：设为 ignore，不反向传播；
- 连续区间做多尺度一致性、短孤立片段过滤和边界软化。

CPL-VAD 的消融显示，仅双向交换而不做一致性精炼在 UCF 反而从 86.60 降到 86.42；加入 CAR 后达到 88.24 [5]。这说明“两个分数直接平均/互教”仍不够，必须先清理伪标签。需要注意：CPL-VAD 当前公开仓库有伪标签生成和测试模型，但 README 明确说明训练代码尚未发布，因此现阶段只能把它作为研究依据，不能声称完整复用了其训练实现。

### 4.3 训练改为分阶段/交替，而不是所有损失联合更新

DeSC 的最强直接证据是联合优化 86.18，两个流独立训练为 88.46/88.35，协同推理为 89.37 [2]。因此下一版应采用：

1. 冻结 baseline，独立训练 selected-layer 语义专家；
2. 固定两个 teacher，离线生成经过一致性过滤的 sparse pseudo labels；
3. 只训练 baseline 的 binary head；
4. 每轮重新生成标签时使用旧 teacher，不让同一个模型即时学习自己的输出。

MIST 的开源训练代码提供了稀疏连续采样和两阶段 self-training 实现 [6]；CVPR 2023 的伪标签完整性/不确定性工作表明，不确定性感知精炼可单独带来约 +1.83 UCF AUC [7]。它们共同支持“少而准、分阶段”而不是“每个异常视频都强制给正标签”。

### 4.4 困难正常负样本优先于继续扩张 KNN 拼接

如果保留 LaGoVAD 机制，优先迁移其已开源的 in-video negative mining / contrastive loss，而不是先扩大合成数据。它直接约束：语义前景应接近异常文本，而同视频的低分正常片段应远离异常文本。只有在两专家一致的候选段上使用该损失，避免让错误 anchor 进一步放大。

KNN 拼接降级为后续可选增强：只有当独立语义专家在测试集仅用于诊断的 frame AUC/AP 和两专家 agreement precision 达标后，才动态生成 normal–abnormal–normal 序列；否则不进入完整训练。

---

## 5. 解冻策略

### 第一优先级：只解冻二分类头

| Baseline | 第一阶段可训练 | 冻结 |
|---|---|---|
| DSANet | `classifier` | `mlp1`、`mlp2`、text adapter、时序主干、CLIP |
| DeSC | 两个 stream 各自的 `classifier`，保持分开优化 | `mlp1/mlp2`、两个时序主干、text prompt、CLIP |
| LaGoVAD | `bin_head`；若确认该版本的融合 gate 是标量，可单独训练 gate | fusion 主体、`sim_head`、soft prompt、temporal encoder、CLIP |

DSANet 的 `classifier` 只有 513 个参数，head-only 是最安全的因果试验：如果可靠伪标签连 head-only 都不能提升，继续解冻更深模块只会增加拟合噪声的能力。

### 第二优先级：信号验证后再加入最后的视觉校准层

只有 head-only 在多个随机种子均超过作者权重，且 AP 不下降，才加入与二分类路径直接相连的最后视觉投影：DSANet/DeSC 的 `mlp2`，LaGoVAD 的 binary fusion gate 或 bin head 前最后一层。仍不解冻文本模块。

### 文本模块的独立规则

text adapter、soft prompt、semantic head 只能由文本对比、类别 MIL、正常/异常对照损失更新，不能用二分类 dense 伪标签顺带解冻。语义专家和 binary head 使用独立优化器、独立阶段和独立 checkpoint。

---

## 6. 下一轮实验的止损顺序

1. **E0 梯度审计**：对每项损失分别 backward，记录每个模块的梯度范数与两损失梯度余弦；不训练。目的是证明监督实际到达哪里。
2. **E1 语义专家单测**：冻结 baseline，只训练 selected-layer 小投影；输出逐层/逐类贡献、正常/异常边际分布、test frame 曲线。若独立语义曲线明显弱于 baseline，不生成伪标签。
3. **E2 标签质量闸门**：实现双专家 agreement/ignore mask；报告正标签覆盖率、分歧率、边界长度分布。在 UCF test 上仅作诊断，不据此选超参。
4. **E3 head-only**：从作者最佳权重初始化，只训练 binary classifier，学习率显著低于从头训练；保持作者选模指标和最佳权重回载方式。至少 3 个种子。
5. **E4 视觉最后层**：仅当 E3 稳定提升后加入 `mlp2`/对应视觉校准层。
6. **E5 KNN/DVS**：仅当伪标签质量已被验证，再做在线动态拼接、补齐正常合成和困难负样本对比。

不满足前一阶段门槛就停止，不再直接跑完整 10 epoch。这能避免再次用算力验证一个在前置诊断中已经不成立的假设。

## 7. 最终判断

当前方法的主要失败责任排序为：

```text
错误的伪标签前提
  > LaGoVAD 迁移不等价
  > 解冻模块与梯度路径不匹配
  > 联合优化造成冲突
  > 学习率/调度/回载等训练细节
```

“解冻太多”确实成立，但不是简单地把 5.25M 改成 2M 就够了。正确做法是先把 hidden states 变成经目标域训练、可独立验证的语义专家，再用它与 baseline 时间专家交叉清理伪标签；baseline 端首先只动最终二分类头。这个方向同时保留：

- **可解释性**：明确展示哪些完整层、哪些类别语义、哪些时间段提供证据；
- **通用性**：三个 baseline 只需暴露原 anomaly curve 和 binary head，不要求内部结构相同；
- **低开销**：复用预计算 CLS hidden states，不提取 patch token/光流，不反传 CLIP；
- **文献依据**：正常性引导、两阶段伪标签、独立专家、困难负样本和一致性精炼均有论文或开源实现支撑。

但必须诚实说明：这些依据提高的是方案可信度，不构成“必然超过强 baseline 2–3 点”的保证。下一步价值在于用 E0–E3 低成本实验先证明监督质量和因果路径，再决定是否值得完整训练。

## 参考文献

[1] Yin et al. *Learning to Tell Apart: Weakly Supervised Video Anomaly Detection via Disentangled Semantic Alignment*. AAAI, 2026. Official code: `lessiYin/DSANet`.

[2] Zheng et al. *Decoupled Sensitivity-Consistency Learning for Weakly Supervised Video Anomaly Detection*. arXiv:2603.19780, 2026. Official code: `imzht/DeSC`.

[3] Liu et al. *Language-guided Open-world Video Anomaly Detection under Weak Supervision*. ICLR, 2026. Official code: `Kamino666/LaGoVAD-PreVAD`.

[4] Yang, Liu, and Wu. *Text Prompt with Normality Guidance for Weakly Supervised Video Anomaly Detection*. CVPR, 2024.

[5] Lee et al. *Cross Pseudo Labeling for Weakly Supervised Video Anomaly Detection*. ICASSP/arXiv:2602.17077, 2026. Public repository currently lacks training scripts.

[6] Feng, Hong, and Zheng. *MIST: Multiple Instance Self-Training Framework for Video Anomaly Detection*. CVPR, 2021. Official code: `fjchange/MIST_VAD`.

[7] Zhang et al. *Exploiting Completeness and Uncertainty of Pseudo Labels for Weakly Supervised Video Anomaly Detection*. CVPR, 2023.
