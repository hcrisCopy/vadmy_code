# P1 失败根因与 ViN-VAD 模块收缩建议

> 日期：2026-09-01
> 结论先行：P1 的 NO-GO 判断正确，但它只否定了“用 exact-OR/Markov chain 从视频标签直接学定位”这条路线，没有否定“上下文违背神经元”动机。下一版应删除 OR-chain 核心贡献，保留正式 baseline 的检测分数，把 contextual violation 做成有界、可审计的残差证据，而不是重新训练一个替代 DSANet 的检测器。

## 摘要

本文结合 P1 正式结果、当前实现、旧方案代码和 2021--2026 年相关一手文献，回答三个问题：P1 为什么不能提点；旧方案中真正有效的是什么；下一步哪个最小模块值得验证。证据显示，P1 同时改变了输入表示、检测头和 MIL 目标：它没有复用 DSANet 的时序 Transformer、双通道 GCN、正常性重建和语义对齐检测分数，而是用原始冻结 CLIP 最后一层 CLS 重新训练三层 TCN。其 E0 在 UCF-Crime 只有 83.44 AUC，已经比项目中 DSANet executable baseline 89.445 低约 6 点，因此 E1--E3 测到的是一个较弱新检测器上的 pooling/chain 行为，不是“DSANet + 新模块”的增益能力。更根本地，exact OR 等价于长序列 noisy-OR：正常 bag 对每个局部假阳性重复惩罚，异常 bag 又可由大量很小的局部分数轻易满足，导致定位梯度饱和。P1 的下降训练准确率、几乎未学习的 transition 参数和跨数据集不稳定与这一机制吻合。建议将论文收缩为一个主张：冻结 host detector，通过正常上下文预测得到稀疏、定向的神经元违背证据，再以有界 residual 修正 host logit，并用 context swap、activation intervention 和跨 host 迁移证明该证据确实提供了 host 没有的上下文信息。

## 1. 研究问题与范围

本轮只回答三个问题：

1. P1 失败主要来自实现错误、训练目标，还是事件先验本身？
2. 旧方案中与“提点”稳定相关的是哪类信息，而不是哪段后处理代码？
3. 在保留“正常上下文 + 定向违背 + 内部机制验证”动机下，哪个最小模块最值得先跑？

证据优先级为：本项目正式运行结果与代码 > 同任务一手论文 > 相邻任务的一手论文。未经正式实验验证的模块建议均标记为“假说”，不是实验结论。

## 2. P1 到底测试了什么

### 2.1 P1 不是 DSANet 上的增量模块

P1 的输入来自 `universal_neuron_adapter/extract_hidden_states.py`：脚本通过 `baseline/DSANet/src/clip` 加载冻结 CLIP ViT-B/16，保存 12 层 CLS hidden states。DSANet 与 VadCLIP 仓库中的 `clip/model.py` 和 `clip/clip.py` 哈希相同，所以“这些特征属于 VadCLIP/DSANet 共用的 CLIP 特征体系”是对的；但它们只是**原始冻结 CLIP 内部状态**，不是 VadCLIP 或 DSANet 训练后的任务检测特征。

正式 DSANet 的检测路径见 `baseline/DSANet/src/model.py`：

- 冻结 CLIP 输出 512 维视觉特征；
- local Transformer 建模短期时间关系；
- similarity/temporal 双通道 GCN 建模长期关系；
- binary classifier 输出 `S_det`；
- 训练时还有正常原型重建、检测/重建一致性和视觉-文本语义损失。

DSANet 论文也明确说明，正式 coarse-grained inference 直接使用该检测分支的 `S_det`，而不是 CLIP 最后一层 hidden state 上另训一个 TCN：[DSANet, AAAI 2026](https://arxiv.org/html/2511.10334v1)。

P1 的 `vin_vad/base_tcn.py` 则只读取 `[T,768]` 的最后层 CLS，重新训练三层 residual TCN。因而：

> P1 是“raw CLIP CLS + 新 TCN + 新 bag objective”的独立检测器消融，不是“正式 DSANet + event module”的增量消融。

最直接的证据是 UCF：P1 E0 为 83.44 AUC，而项目中干净 DSANet executable baseline 为 89.445。事件模块加入之前已经丢了约 6 AUC，说明“提点锚点”被换掉了。

### 2.2 P1 同时改了三件事，不能只把失败归因于 Markov smoothing

E0--E3 表面只改 pooling/event chain，但相对旧方案还发生了三项根本变化：

1. **表示替换**：全 12 层 CLS 稀疏证据变成最后一层 CLS 全维输入。
2. **host 替换**：正式 DSANet 检测曲线变成从零训练的 TCN 曲线。
3. **目标替换**：top-k MIL 变成 exact-OR likelihood，E2/E3 再加入状态持续性。

因此 P1 能回答“exact OR 是否适合作为这个新检测器的主训练目标”，不能回答“上下文违背证据能否修正 DSANet”。

## 3. exact OR 为什么结构性地不利于定位

### 3.1 E1 就是带先验的 noisy-OR

E1 中各状态独立。令

\[
q_t=\sigma(\operatorname{logit}h+\eta_t),\qquad
P(y=1)=1-\prod_t(1-q_t).
\]

记 \(Q=\prod_t(1-q_t)\)。对 emission 的梯度为

\[
\frac{\partial\mathcal L_{y=0}}{\partial\eta_t}=q_t,
\qquad
\frac{\partial\mathcal L_{y=1}}{\partial\eta_t}
=-\frac{Q}{1-Q}q_t.
\]

这产生两个相反但同时存在的问题：

- **正常视频逐点受罚**：负 bag 的损失是所有位置负证据的累积，连续假阳性会被按帧/片段重复惩罚。
- **异常视频过早满足**：只要许多位置都有很小的 \(q_t\)，bag probability 就会接近 1，正 bag 梯度随 \(Q\to0\) 消失，不再要求模型制造清晰局部峰值。

Wang 等人在弱标注语音/声音事件长序列上已经给出同样结论：noisy-OR 对假阳性过严、对漏检过松，可获得合理 bag prediction，却产生过小且不适合定位的 frame prediction；其根因包括相邻帧相关性违反独立假设，以及乘积在长序列中快速饱和：[Comparing Max and Noisy-Or Pooling, Interspeech 2018](https://maigoakisame.github.io/papers/interspeech18a.pdf)。PUMA 的理论分析也证明，标准 noisy-OR 的正 bag probability 会随 bag size 指数趋近 1：[PUMA, KDD 2023](https://lorenzo-perini.github.io/files/KDD_Paper.pdf)。

这不是说 top-k 没有偏差。UMIL、D²MIL 和 The Road Less Seen 都指出 top-k/高分选择会带来上下文偏差、噪声监督或漏掉低分异常：[UMIL, CVPR 2023](https://openaccess.thecvf.com/content/CVPR2023/html/Lv_Unbiased_Multiple_Instance_Learning_for_Weakly_Supervised_Video_Anomaly_Detection_CVPR_2023_paper.html)、[D²MIL, CVPR 2026](https://openaccess.thecvf.com/content/CVPR2026/html/Zhao_Learning_from_Noisy_Supervision_A_Denoising-Debiasing_Framework_for_Weakly_Supervised_CVPR_2026_paper.html)、[The Road Less Seen, CVPR 2026](https://openaccess.thecvf.com/content/CVPR2026/html/Acharya_The_Road_Less_Seen_Segment_Exploration_for_Weakly_Supervised_Video_CVPR_2026_paper.html)。结论只是：**把 top-k 换成 exact OR 并不会自动得到更正确的 snippet credit assignment。**

### 3.2 E3 只校准“零 emission 的事件先验”，没有校准学习问题

E3 使用

\[
h_T=1-\exp(-\rho/T),
\]

确实把 zero-emission 下的先验事件概率固定在约 \(1-e^{-1}=0.632\)。P1 的 `fixed_emission.json` 验证：E1/E2 从 T=16 的 0.061 增长到 T=1024 的 0.982，E3 在各长度均约 0.632。

但这只解决一个非常窄的问题：没有视觉证据时，视频更长不应自动更异常。它没有解决：

- 正 bag 梯度饱和；
- 负 bag 对相关连续片段重复惩罚；
- emission 强度、onset 和 persistence 之间不可辨识；
- 训练和测试的长度分布不一致。

特别是 `vin_vad/data.py` 在训练时把所有长视频平均压到 256 snippets，而测试使用完整长度。E3 训练时几乎只见到 \(h_{256}\)，测试时却按真实 T 改变 hazard。换言之，长度校准公式在测试时改变了模型的 transition regime，但 TCN/chain 没在这些 regime 下联合训练。

### 3.3 persistence 在弱 bag 标签下不可辨识

E2/E3 只用视频级标签。对一个异常 bag，“一个强而短的事件”和“一个弱而长的事件”都可提高 \(P(y=1)\)。没有边界监督或额外可识别约束时，TCN emission、onset 和 persistence 可以互相补偿。

正式运行后，参数几乎停在初始化：

| 数据集 | 变体 | onset@256 | persistence |
|---|---:|---:|---:|
| UCF | E2 | 0.003919 | 0.899785 |
| UCF | E3 | 0.003906 | 0.899769 |
| XD | E2 | 0.003909 | 0.899972 |
| XD | E3 | 0.003900 | 0.899972 |

这说明“可学习事件持续性”在当前监督下实际上没有学到数据集中的事件结构，只是在使用手工初始化的 0.9 先验。

固定 emission 机制实验还显示，T=64 时 E3 的单个孤立峰会产生约 19.43 的 posterior mass，连续中等证据约 20.67；它确实形成了宽事件，但“宽”不等于“边界正确”。LEC-VAD、PE-MIL 和 LAS-VAD 都把事件完整性/边界/语义分组作为额外学习问题，而不是认为持续性先验本身足够：[LEC-VAD, ICML 2025](https://proceedings.mlr.press/v267/wang25l.html)、[PE-MIL, CVPR 2024](https://openaccess.thecvf.com/content/CVPR2024/html/Chen_Prompt-Enhanced_Multiple_Instance_Learning_for_Weakly_Supervised_Video_Anomaly_Detection_CVPR_2024_paper.html)、[LAS-VAD, CVPR 2026](https://arxiv.org/abs/2603.00550)。

## 4. 远程重算：掉点发生在哪里

本轮在同一正式 checkpoint 上额外只读重算了两种帧分数：

- raw：\(\sigma(\eta_t)\)，即 TCN emission；
- posterior：正式 evaluator 使用的 \(p(z_t=1\mid H)\)。

| 数据集 | 变体 | raw AUC/AP | posterior AUC/AP | 判断 |
|---|---:|---:|---:|---|
| UCF | E0 | AUC 83.444 | AUC 83.444 | top-k 基线 |
| UCF | E1 | AUC 81.264 | AUC 81.300 | 主要是训练目标把 emission 学差 |
| UCF | E2 | AUC 79.907 | AUC 81.344 | chain 后验部分修复 raw，但仍低于 E0 |
| UCF | E3 | AUC 80.091 | AUC 79.943 | 长度校准后验再次伤害排序 |
| XD | E0 | AP 73.836 | AP 73.836 | top-k 基线 |
| XD | E1 | AP 74.706 | AP 74.707 | E1 在 XD 有小幅正效应 |
| XD | E2 | AP 58.348 | AP 64.501 | persistence 后验修复一部分，但训练已严重退化 |
| XD | E3 | AP 55.975 | AP 62.378 | 同上，且远低于 E0/E1 |

训练集表现也按 E0→E1→E2/E3 下降：

| 数据集 | E0 acc | E1 acc | E2 acc | E3 acc |
|---|---:|---:|---:|---:|
| UCF | 95.44 | 93.36 | 88.87 | 87.57 |
| XD | 93.97 | 89.31 | 80.59 | 82.92 |

因此不能简单归因于“推理时平滑过度”。更准确的分解是：

1. E1 在 UCF 的主要损失发生在 exact-OR 训练；
2. E2/E3 在 XD 的主要损失也先发生在训练，后验只救回部分排序；
3. persistence/length calibration 对不同数据集的后验影响不稳定，是第二层问题；
4. 最早的能力缺口是 P1 用弱新检测器替代了正式 DSANet。

## 5. 旧方案为什么偶尔能提点

旧代码 `universal_neuron_adapter/evaluate.py` 很复杂，而且包含多次观察公开测试集后形成的权重、视频级 LR、滤波和融合规则，不能作为论文方法继续使用。但它提供了有价值的失败诊断。

### 5.1 真正有效的不是“事件链”，而是三类信息

1. **保留强 host 排序**。旧方案始终以冻结 baseline score 为主，只加 residual；P1 则完全替换 host detector。对 DeSC/DSANet 这类强 host，已有历史记录显示直接 neuron evidence 的单步收益通常只有 0.02--0.05 pp，说明强排序只能小改，不能重学。
2. **全层、定向内部证据**。`universal_neuron_adapter/model.py` 使用 12 层稀疏 neuron expert；`normality.py` 证明 directional deviation 比 absolute deviation 更有效。P1 只保留最后层且没有正常上下文残差。
3. **正常负证据**。旧消融中最大增益来自视频级正常/异常抑制，event propagation 只是第二来源，并且在 DSANet/XD 单独为负。这说明旧系统真正补到的更像“host 对正常视频的置信校准”，不是“从零恢复事件边界”。

### 5.2 旧方案不能直接复活

旧方案的问题同样明确：

- 测试集参与反复开发，增益不是 blind estimate；
- 四个独立拟合/后处理阶段，难以归因；
- DSANet/UCF 干净复现的 paired gain 约 +0.972 pp，但 bootstrap 区间跨 0；
- UCF detection mAP 从 13.014 降到 9.592，说明 frame AUC 提升可能主要来自正常视频压分，而非边界更准。

所以应保留它揭示的**信息类型**，删除其 LR、经验门控、max filter、median filter 和数据集相关插值。

## 6. 文献地图与新颖性边界

| 方向 | 代表一手工作 | 已经解决什么 | 我们仍可回答什么 |
|---|---|---|---|
| top-k/MIL 偏差 | [RTFM](https://openaccess.thecvf.com/content/ICCV2021/papers/Tian_Weakly-Supervised_Video_Anomaly_Detection_With_Robust_Temporal_Feature_Magnitude_Learning_ICCV_2021_paper.pdf), [UMIL](https://openaccess.thecvf.com/content/CVPR2023/html/Lv_Unbiased_Multiple_Instance_Learning_for_Weakly_Supervised_Video_Anomaly_Detection_CVPR_2023_paper.html), [D²MIL](https://openaccess.thecvf.com/content/CVPR2026/html/Zhao_Learning_from_Noisy_Supervision_A_Denoising-Debiasing_Framework_for_Weakly_Supervised_Video_CVPR_2026_paper.html) | 特征幅值、上下文去偏、噪声样本处理 | 不再把“更好的 MIL 选择”写成主创新 |
| 上下文异常 | [CMRL](https://openaccess.thecvf.com/content/CVPR2023/html/Cho_Look_Around_for_Anomalies_Weakly-Supervised_Anomaly_Detection_via_Context-Motion_Relational_CVPR_2023_paper.html), [PE-MIL](https://openaccess.thecvf.com/content/CVPR2024/html/Chen_Prompt-Enhanced_Multiple_Instance_Learning_for_Weakly_Supervised_Video_Anomaly_Detection_CVPR_2024_paper.html), [LAS-VAD](https://arxiv.org/abs/2603.00550), [SESAD](https://arxiv.org/abs/2607.10298) | context-motion、normal prompt、意图、结构化证据选择 | 条件预测“这个内部单元在正常上下文下本应怎样响应” |
| 事件完整性 | [Exploiting Completeness and Uncertainty](https://arxiv.org/abs/2212.04090), [LEC-VAD](https://proceedings.mlr.press/v267/wang25l.html), [The Road Less Seen](https://openaccess.thecvf.com/content/CVPR2026/html/Acharya_The_Road_Less_Seen_Segment_Exploration_for_Weakly_Supervised_Video_CVPR_2026_paper.html) | 伪标签完整性、边界、低分高不确定片段 | event 本身已拥挤，不宜再作为第一贡献 |
| 正常性建模 | [DSANet](https://arxiv.org/html/2511.10334v1) | 视频内正常原型重建并约束检测分数 | 正常上下文的 neuron-wise conditional distribution，而非 feature reconstruction |
| 内部单元定位 | [LAKE](https://arxiv.org/abs/2604.07802), [DNA](https://arxiv.org/abs/2601.22515), [V-FIND](https://arxiv.org/abs/2608.03008) | 稀疏敏感神经元、层/单元定位、功能干预 | WSVAD 中的条件违背、时间定位、跨 host transfer |
| 上下文内部干预 | [SteerVAD](https://arxiv.org/html/2602.24021v1) | MLLM attention-head RSA、global-context controller、各向异性 steering | 最强近邻；必须证明 masked normal predictor 优于 global controller/class separability |
| OR/noisy-OR 反证 | [Wang et al.](https://maigoakisame.github.io/papers/interspeech18a.pdf), [PUMA](https://lorenzo-perini.github.io/files/KDD_Paper.pdf) | 长序列饱和与定位失败 | 直接解释 P1，不再把 exact OR 当默认正确目标 |

最危险的近邻不是一般 WSVAD，而是 SteerVAD。它已经用全局上下文控制内部 attention heads 并做几何 steering。因此论文不能声称“首次 context-aware internal VAD”或“首次干预 VAD 神经元”。可防守的差异必须精确为：

> SteerVAD 用类别可分性选择 MLLM heads，再由全局语义向量生成 steering；ViN-VAD 学习冻结 CLIP 坐标在**遮蔽目标后的正常时间上下文条件分布**，把实际响应相对该条件分布的**定向预测残差**作为证据，并验证该证据能否在视频级弱标签下跨不同 host detector 修正时间定位。

如果 masked conditional residual 不能稳定优于 SteerVAD-style global controller、global z-score 和 raw activation，这个新颖性边界就不成立。

## 7. 建议的唯一主模块

### 7.1 名称与目的

建议暂称 **Host-Preserving Contextual Violation Adapter（HP-CVA）**。

它只做一件事：在不替换 host detector 的前提下，用正常上下文预测残差对 host logit 做小幅、有界修正。

### 7.2 最小结构

**输入 A：正式 host score**

\[
b_t=\operatorname{logit}(s_t^{host}).
\]

`s_host` 必须来自 baseline 原 evaluator 使用的正式检测分支。例如 DSANet 使用 `S_det`，不能再用最后层 CLS TCN 代替。

**输入 B：正常上下文条件违背**

用正常训练视频学习 masked predictor，目标保护区间不可见：

\[
(\mu_{t,u},\sigma_{t,u})=g_u(H_{\setminus G_t}),
\qquad
r_{t,u}=\frac{h_{t,u}-\operatorname{sg}(\mu_{t,u})}
{\operatorname{sg}(\sigma_{t,u})+\epsilon}.
\]

构造定向违背并用稀疏非负权重聚合：

\[
v^+_{t,u}=\operatorname{ReLU}(r_{t,u}-\delta),\quad
v^-_{t,u}=\operatorname{ReLU}(-r_{t,u}-\delta),\quad
e_t=\sum_u\pi_uv_{t,u}.
\]

**有界 residual，而非替代检测器**

用正常训练集的 median/MAD 标准化 \(e_t\)，得到 \(\bar e_t\)，再写成：

\[
s_t=\sigma\left(
b_t+\alpha_{max}\tanh(\gamma)\tanh(\bar e_t)
\right).
\]

其中 \(\alpha_{max}\) 是统一协议中的固定 logit 改动上限，\(\gamma\) 是全局可学习标量。这个结构有三个好处：

1. 初始化 \(\gamma=0\) 时严格等于 host；
2. 最坏改动被结构性限制，不会让新证据重写强 baseline；
3. \(\bar e_t<0\) 时提供旧方案中有效的正常负证据，\(\bar e_t>0\) 时提供局部违背正证据，不需要额外视频级 LR。

这是待验证假说，不是已证明最终公式。作者需要决定是否接受“有界 residual”这一技术路线。

### 7.3 为什么先不加事件模块

- P1 已证明 current OR-chain 不适合作为主训练目标；
- event completeness、connected components、temporal exploration 已有强近邻；
- 旧方案的 event propagation 在 DSANet/XD 单独为负；
- contextual violation 本身尚未被干净验证，再叠 event module 会失去归因。

时间建模先只存在于两处：host 原有时序网络，以及 masked normal-context predictor。若 HP-CVA 已经稳定提点但事件 mAP 仍差，再单独讨论边界模块；不要现在预埋。

## 8. 最小正式实验：只做能决定论文生死的部分

### 阶段 R0：host identity

目标：证明输入确实是正式 DSANet，而不是另训替代检测器。

- 缓存 DSANet 官方 `S_det`；
- 同 evaluator 复现 UCF/XD executable baseline；
- 检查 hidden states 与 host score 的视频、snippet、frame boundary 一一对齐；
- `gamma=0` 时输出逐点严格等于 host。

不过关就停止。

### 阶段 R1：context predictor 机制验证

只训练正常上下文预测器，不接 detector。核心对照只有四个：

| ID | 证据 | 回答什么 |
|---|---|---|
| C0 | raw activation | 高激活是否足够 |
| C1 | global normal z-score | 正常锚是否足够 |
| C2 | masked contextual absolute residual | 上下文预测是否有效 |
| C3 | masked contextual directional residual | 方向是否必要 |

只看：normal conditional NLL、context-matched swap 后的 \(\Delta\mu/\Delta r\)、same-action/different-context 排序。C3 不优于 C1/C2，就停止“contextual violation”主张。

### 阶段 R2：DSANet 上的 host-preserving adapter

只跑：Host、Host+C0、Host+C1、Host+C2、Host+C3。dense softmax/entmax 放到 C3 成立后再补，不要现在跑。

报告三类指标即可：

1. 官方 frame AUC/AP；
2. abnormal-video-only AUC/AP 或 detection mAP，防止只靠正常视频压分；
3. paired per-video bootstrap，仅对 Host vs Full 和 Full vs 最强直接对照。

判定：UCF/XD 都不低于 host，至少一个数据集有清楚增益；同时 abnormal-only/event metric 不下降。若只涨总体 AUC、事件指标仍降，不能讲“定位更准”。

### 阶段 R3：最强近邻与迁移

只补两个实验：

- SteerVAD-style global-context controller vs masked normal predictor；
- 在 DeSC 或 VadCLIP 上零结构修改迁移。先选一个第二 host，不必立即跑四 baseline × 两数据集。

这两项分别守住新颖性和通用性。

### 阶段 R4：机制干预

只有 R2/R3 成立才做：

- context-matched donor swap；
- selected-unit erase；
- normal-donor activation patch；
- 同层随机、幅值匹配未选中单元、随机 donor 三个控制。

论文只能写“这些单元对模型预测有功能贡献”，不能写现实世界因果或“偷窃神经元”。

## 9. 审稿人视角的最终判断

### 可以保留

- “异常是相对正常时间上下文的内部定向违背”这一动机；
- 正常视频是可靠锚；
- 多层 CLS 坐标、稀疏方向单元、context-matched intervention；
- 跨 host 的 frozen adapter 设定。

### 应删除或降级

- **删除核心贡献二：Exact weak-label event inference。** 目前理论和实验均不支持。
- event chain 最多作为失败分析/附录，不再作为 Full 模型底座。
- 不再用 final-layer TCN 代表 DSANet。
- 不复活旧方案的视频级 LR、经验滤波和多系数融合。
- “event completeness”“context-aware”“internal intervention”均不能单独 claim first。

### 论文故事收缩为一句话

> 现有 WSVAD host 能识别显著异常，却无法判断同一内部响应在当前时间上下文中是否反常；我们从正常视频学习被遮蔽目标的 neuron-wise conditional expectation，把定向预测残差作为稀疏违背证据，并以有界 residual 修正冻结 host，再通过上下文替换、内部干预和跨 host 迁移验证该证据。

这条故事比“contextual neurons + exact OR chain + compositional tags”窄，但更容易被实验讲实。tag 不是当前生死点；只有检测与干预成立后再做。

## 10. 当前不确定性

1. 本轮 raw/posterior 重算使用正式 checkpoint 和正式 test manifest，但只是诊断，没有形成新的模型或调参依据。
2. HP-CVA 公式尚未运行，不能声称一定提点。
3. SteerVAD 与本方案重叠较大，masked conditional prediction 必须通过直接实验建立差异。
4. 旧方案的 normal suppression 提示了有效信号，但其历史增益受测试集开发污染，只能用于提出假说。
5. 如果 R1 机制成立而 R2 不提点，论文可转向机制分析，但不应包装成强 detection method；是否接受该定位由作者决定。

## 11. 现有产物与复核入口

- P1 总表：`../vadmy_data/vin_vad/dsanet/p1/summary.csv`
- 固定 emission：`../vadmy_data/vin_vad/dsanet/p1/fixed_emission.json`
- 单模型训练：`../vadmy_data/vin_vad/dsanet/p1/<dataset>/<variant>/train_summary.json`
- 正式评测：`../vadmy_data/vin_vad/dsanet/p1/<dataset>/<variant>/evaluation/`
- P1 代码：`vin_vad/`
- 旧方案诊断代码：`universal_neuron_adapter/`
- 最终旧设计：`docs/ws-vad-ultimate-v6-2026-08-31.md`
- 当前搭建指南：`docs/vin-vad-build-ablation-guide-2026-09-01.md`

本轮没有修改训练代码、baseline、远程权重或 `vad_data`；没有启动新正式实验。
