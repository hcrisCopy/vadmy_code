# 从消融证据到显著增益：三类视觉-文本 WSVAD Baseline 与神经元探测的升级路线

## 摘要

本文围绕三个问题展开：三篇目标 Baseline 中哪些设计真正带来了显著的 AUC/AP 增益；这些设计是否存在跨模型共性；如何把这些共性与不依赖 Baseline 打分的 CLIP 神经元探测结合，形成有依据的通用增强方法。逐表核对表明，大增益并不来自滑窗、平滑或普通残差注入，而集中在四种机制：把冲突目标拆成互补专家；以正常性和负证据压制假阳性；把视频级弱标签转化为可信的 snippet 级候选；在不破坏视觉空间的前提下进行任务特定的视觉-文本对齐。DeSC 的解耦优化相对联合训练在 UCF-Crime/XD-Violence 上分别提高 3.19 AUC/6.96 AP；VadCLIP 的时序建模提高 12.29 AP；TPWNG 的正常性引导提高 1.96 AUC/2.36 AP；MGFN 的 magnitude contrastive loss 提高 2.85 AUC/3.69 AP。这些独立结果共同指向同一结论：要获得 2--3 点增益，神经元不能继续作为附加特征被放大，而应成为独立的细粒度证据源，用来改变弱监督下的实例选择、负样本抑制和互补专家训练。本文据此提出 Causal Neuron Evidence Decoupling（CNED）：由正常流形偏离、异常概念因果响应和双时间尺度证据构成独立神经元专家，通过可信候选筛选、分阶段优化和置信度融合增强任意 Baseline。该路线有论文消融依据，但现有文献不能保证在强 Baseline 上稳定提升 2--3 点；因此给出分阶段 go/no-go 门槛，先验证候选质量和独立专家上限，再投入三模型六组正式训练。

## 1. 研究问题

- RQ1：LaGoVAD、DeSC 和 DSANet 中，哪些模块或训练策略对 AUC/AP 产生了最大、可归因的增益？
- RQ2：这些显著模块与相近 WSVAD、CLIP 图像异常检测工作的有效机制有什么共性？
- RQ3：怎样把这些共性和 CLIP 神经元因果探测结合，形成通用于三个 Baseline、两个数据集且单卡 RTX 4090 可运行的方法？

本文的核心判断不是“把多个有效 trick 堆在一起”，而是：**大增益方法都在改善弱标签到细粒度证据的转换；神经元探测只有进入这一转换过程，而不是进入特征残差，才有机会产生显著指标增益。**

## 2. 调研方法

研究首先逐页抽取本地三篇 Baseline 的主结果与消融表，随后沿四条检索路线核对相关工作：（1）正常性、负证据和 memory；（2）MIL 实例选择、伪标签和不确定性；（3）局部/全局及瞬时/持续时序建模；（4）CLIP prompt、视觉-文本对齐与神经元解释。纳入条件是论文正文或官方论文页能够验证具体机制；具体增益只从同一论文、同一特征和同一评测设置的消融中计算，不把跨论文榜单差值当成模块因果效应。

三篇 Baseline 的本地来源为 `paper/LANGUAGE-GUIDED OPEN-WORLD VIDEO ANOMALY.pdf`、`paper/Decoupled Sensitivity-Consistency Learning for.pdf` 和 `paper/DSANET.pdf`。LaGoVAD 使用开放世界/零样本协议，其数字不能直接与另外两篇标准同域 UCF AUC/XD AP 相加比较；DSANet 的模块消融主要在 XD-Violence 上报告 AP 和 fine-grained AVG；DeSC 同时报告标准 UCF AUC 和 XD AP。

## 3. 证据分类

调研结果可分为四类互补机制：

1. **冲突解耦与互补专家**：瞬时/持续、分类/定位、通用/类别特定目标分别优化，最后融合。
2. **正常性锚定与假阳性抑制**：显式正常原型、正常 prompt、负样本挖掘或正常流形距离。
3. **弱标签细化与证据选择**：动态 top-k、伪标签、置信度/不确定性筛选、前景或关键片段选择。
4. **受控的视觉-文本适配**：调整 text-side 或小型 adapter，保留预训练视觉空间，利用更具体的事件语义。

这四类并非四个可随意叠加的 trick。第一类解决优化冲突，第二类控制 false positive，第三类控制 weak-label noise，第四类提供异常含义；它们分别对应不同误差来源。

## 4. 三篇 Baseline 的显著消融证据

### 4.1 DeSC：最大增益来自解耦，不是 TTA

DeSC 的联合模型仅有 86.18 UCF AUC/80.22 XD AP；独立训练 temporal sensitivity stream 后达到 88.46/85.04，独立 semantic consistency stream 达到 88.35/85.48，最终协同融合达到 89.37/87.18 [2]。因此，相对联合训练的完整增益是 **+3.19 AUC/+6.96 AP**。这一结果比任何内部小模块都大，说明主要瓶颈是瞬时敏感性和持续稳定性之间的梯度冲突。

| DeSC 消融 | UCF AUC | XD AP | 可归因结论 |
|---|---:|---:|---|
| Unified joint training | 86.18 | 80.22 | 冲突目标的折中点 |
| Temporal stream | 88.46 | 85.04 | 相对 unified：+2.28/+4.82 |
| Semantic stream | 88.35 | 85.48 | 相对 unified：+2.17/+5.26 |
| Collaborative DeSC | 89.37 | 87.18 | 相对 unified：+3.19/+6.96 |
| Basic ensemble | 89.13 | 86.47 | 已获得大部分融合收益 |
| Sliding-window TTA | 89.37 | 87.18 | 仅 +0.24/+0.71 |

内部结构进一步支持这一解释：parallel TCN+GT 相比 TCN-only 提高 2.66 AUC/5.26 AP，相比 GT-only 提高 2.20 AUC/0.49 AP；GMP 只提高 0.65 AUC/0.81 AP。**可以借鉴的是独立专家和不同时间尺度，不是照搬滑窗 TTA。**

### 4.2 DSANet：显著增益来自 text adapter 与正常-异常语义分离

DSANet 以 VadCLIP 为基线，在 XD-Violence 上从 84.51 AP/24.70 AVG 提升到 86.95/28.87 [1]。Adapter 单独带来的 AP 增益不大，但 fine-grained AVG 从 24.70 增至 28.15，即 +3.45；在完整结构中，SG-NM 和 DCSA 进一步形成协同，最终获得 **+2.44 AP/+4.17 AVG**。

| DSANet 组件 | XD AP | fine-grained AVG | 相对上一可比项 |
|---|---:|---:|---|
| VadCLIP baseline | 84.51 | 24.70 | - |
| Adapter | 85.00 | 28.15 | +0.49/+3.45 |
| Adapter + SG-NM | 85.94 | 28.39 | +0.94/+0.24 |
| Adapter + DCSA | 85.67 | 28.25 | +0.67/+0.10 |
| Full | 86.95 | 28.87 | 相对 baseline：+2.44/+4.17 |

Text tuning 消融更有启发性：no tuning、manual prompt、learned prompt 和 adapter 分别为 81.57/27.60、81.05/28.05、82.88/28.26、86.95/28.87。Adapter 相比 learned prompt 提高 **4.07 AP**，说明简单 prompt token 不足以完成任务域对齐，内部 text representation adaptation 才能显著改变细粒度类别分离。SG-NM 用正常模式指导判断，DCSA 将 detection 与 semantic alignment 解耦；二者共同避免仅靠异常视频 top-k 形成的语义偏差。

### 4.3 LaGoVAD：显著增益来自条件化定义、动态合成和负样本

LaGoVAD 的消融属于零样本跨域协议 [3]。完整模型的七数据集 detection average 为 69.98；移除 dynamic video synthesis loss（Ldvs）后为 65.73，下降 4.25；移除 negative loss（Lneg）后为 68.92，下降 1.06。按单数据集看，移除 Ldvs 使 UCF/XD 分别下降 1.94/4.01，移除 Lneg 使其下降 3.46/3.59。关闭 language-guided detection 时，平均 detection 只下降 0.11，但 UCF/XD 分别下降 **4.48/2.86**，同时分类平均下降 6.34，说明语言条件在特定域和分类上非常关键，平均数掩盖了它的作用。

| LaGoVAD 设计 | 完整值 | 对照值 | UCF/XD 变化 |
|---|---:|---:|---:|
| Dynamic video synthesis loss | Det. Avg. 69.98 | w/o：65.73 | -1.94/-4.01 |
| In-sample negative loss | Det. Avg. 69.98 | w/o：68.92 | -3.46/-3.59 |
| Language-guided detection | Det. Avg. 69.98 | w/o：69.87 | -4.48/-2.86 |
| KNN retrieval in synthesis | - | w/o KNN | UCF -1.14，XD -5.30 |
| Video-specific description | UCF 83.03 | manual 81.12 | +1.91 |

LaGoVAD 的可借鉴重点有三个：异常定义必须进入 detection，而不是只在末端分类；构造片段时要保持局部语义相似性，随机拼接会制造噪声；负样本挖掘对抑制跨域 false positive 的作用大于普通正样本增强。

## 5. 相近工作中可复用的显著 trick

### 5.1 时序和局部-全局结构：大增益来自互补建模

VadCLIP 从无时序建模的 72.22 AP 提升到 84.51 AP，LGT-Adapter 贡献 **+12.29 AP**；local Transformer+GCN 在 fine-grained AVG 上也明显优于单一全局 Transformer [4]。STPrompt 中，空间选择注意力与 temporal adapter 从约 84.54 的单分支基线推进到 88.08 UCF AUC，且 distance Transformer 优于 vanilla Transformer 1.76 AUC [7]。RTFM 的 PDC+TSA 从 77.39 提升到 82.12 UCF AUC，说明长短时依赖互补 [8]。MGFN 中，Glance-Focus 比简单 focus-focus 提高 3.14 AUC/6.36 AP [9]。

共同点不是“Transformer 有效”，而是**异常时间尺度未知时，单一感受野会系统性漏掉一类异常**。DeSC 进一步表明，若不同尺度目标存在冲突，串联或联合优化仍不够，独立训练后融合更有效。

### 5.2 正常性与负证据：对 AUC/AP 都有稳定贡献

UR-DMU 的 dual memory 相比主干提高 4.08 UCF AUC/5.99 XD AP；仅 normal memory 已显著优于仅 abnormal memory，而 dual memory 再提高 1.78 AUC/1.26 AP [10]。TPWNG 的 normality guidance 提高 1.96 AUC/2.36 AP，normal visual prompt 提高 2.54 AUC/2.10 AP [6]。MGFN 的 magnitude contrastive loss 提高 2.85 AUC/3.69 AP [9]。这些结果与 LaGoVAD 的 Lneg、DSANet 的 SG-NM、DeSC 的 GMP 形成独立收敛：**异常不应只由“像不像异常”决定，还必须由“是否离开正常模式”验证。**

### 5.3 可靠候选、伪标签和证据聚焦：最直接修复弱标签

LAP 从 visual-only baseline 的 87.0 UCF AUC/81.3 XD AP 提升到 88.9/86.5；其中 visual-text feature synthesis 先贡献 +1.2/+2.8，multi-prompt learning 再贡献 +0.3/+0.9，pseudo anomaly labeling 再贡献 +0.4/+1.5 [5]。动态阈值优于静态阈值 0.5 AUC/2.1 AP，完整句子 prompt 优于短语 0.7 AUC/3.1 AP。

D2MIL 是更严格的 plug-and-play 对照：通过丢弃高损失疑似噪声，再用 VLM 找回被误删的 hard anomaly，在 UCF-Crime 上给不同 MIL Baseline 带来 +0.22 到 +1.61 AUC [13]。这说明“清理候选”可靠，但单独作为训练 trick 在强模型上通常不足 2 点。The Road Less Seen 不再只利用 top-k，而以 temporal clustering 覆盖多样片段、以 uncertainty sampling 探索低分难例；在 UCF-Crime 上将 UR-DMU 的 AP 从 35.48 提至 38.33，其中 VLM 融合进一步把 36.42 推至 38.33 [14]。

MuST-VAD 提供了一个重要反例和启示：互学习使 UCF AUROC 仅提高 0.48，但 AP 提高 5.21；若把 key clips 换成随机窗口，AUROC 降到 86.54，甚至低于 88.15 的初始模型 [15]。因此，**关键片段选择决定高分区域的 precision，往往更显著影响 AP；若候选错了，增加大模型或更多训练反而有害。**

### 5.4 神经元异常检测：有效做法是正常流形与稀疏子空间，不是激活放大

LAKE 从少量正常样本中按正常 patch token 的通道方差选择 Top-100 神经元，在该子空间建立 normal gallery，以最近邻距离度量结构偏离，再用 normal/anomalous text 相似度做辅助验证 [12]。Top-100 相比 random-100 将 MVTec-AD image AUROC 从 81.6 提至 94.7，pixel PRO 从 45.8 提至 88.9；作为 WinCLIP 插件时 image AUROC 从 90.4 提至 92.8，即 **+2.4 点**。但这项证据来自工业图像 patch token，而不是视频 CLS snippet，不能直接宣称同样增益会出现在 UCF/XD。

LAKE 对我们的关键纠正是：神经元的价值不是“把异常神经元数值放大”，而是**选择一个低冗余的坐标系，在该坐标系内度量与正常流形的偏离**。我们上一版 CNCR 已证明选中维度对异常文本概念的干预效应是随机维度的 38.6 倍，却没有建立稳定的 normal manifold，也没有改变 MIL 的候选选择，因此出现“解释成立、检测不涨”的结果是符合上述文献的。

## 6. 跨论文共同机制

| 共同机制 | 独立证据 | 为什么提升指标 | 对当前工作的含义 |
|---|---|---|---|
| 冲突目标解耦 | DeSC、VadCLIP、STPrompt、LEC-VAD | 避免瞬时/持续、分类/定位梯度互相折中 | 神经元证据应独立成专家，不注入 Baseline feature |
| 正常性验证 | UR-DMU、TPWNG、MGFN、LaGoVAD、LAKE | 抑制背景变化和正常动作造成的高分假阳性 | 同时发现 normal-manifold neurons 与 anomaly-concept neurons |
| 候选聚焦与去噪 | LAP、D2MIL、Road Less Seen、MuST-VAD | 把视频级标签转成更可信的 snippet 监督 | 神经元首先用于选片段和置信度，不用于残差增强 |
| 受控语义适配 | DSANet、VadCLIP、TPWNG、LaGoVAD | 让异常定义与目标域事件对齐，同时保留视觉空间 | 优先 text adapter；视觉 CLIP 和已有 hidden states 保持冻结 |
| 多尺度证据 | DeSC、RTFM、MGFN、PEL4VAD | 同时覆盖短促和持续异常并减少边界遗漏 | 从同一神经元证据产生 high-pass/low-pass 两个 temporal expert |

最大的共同点可以表述为：**有效模块都在改变“哪些 snippet 应该被相信，以及应依据哪一类证据相信”，而不是简单增加 feature dimension。**

## 7. 建议主方法：Causal Neuron Evidence Decoupling

### 7.1 方法定位

CNED 是一个训练期细粒度监督增强器和推理期轻量证据专家。它不依赖 Baseline score 探测神经元，不修改三个 Baseline 的输入维数，不把 hidden state 拼接进视觉特征。三个 Baseline 仍保留原始 CLIP 512-D feature 和各自作者结构；CNED 只通过统一的 snippet index、soft target 和标量 evidence score 对接。

### 7.2 三类独立证据

1. **Normal-manifold evidence**：只使用正常训练视频，在 CLIP hidden dimensions 中选择能稳定描述正常流形、同时对随机扰动具有高下游影响的稀疏维度。在选中子空间建立分层 normal prototypes 或近邻 gallery。对 snippet 计算稳健 Mahalanobis/KNN deviation。该分支对应 LAKE、UR-DMU 和 SG-NM。
2. **Concept-causal evidence**：使用类别文本和 normal text，通过 activation patching/ablation 选择对目标概念具有正向、特异、可复现因果效应的维度。分数使用 target effect 减去 off-target effect 和 normal effect，而不是直接取 activation 大小。该分支继承当前 CNCR 已通过的因果门控，但不进行特征增强。
3. **Temporal evidence**：对前两类证据分别生成 transient high-pass expert 与 sustained low-pass expert。二者独立训练或固定变换，禁止共享冲突损失；最后由小型 reliability gate 融合。这直接对应 DeSC 的最大消融收益。

### 7.3 神经元证据如何改变弱监督训练

对异常视频，不生成“一刀切”的硬伪标签，而生成三个集合：

- reliable positive：normal deviation 高、concept effect 高，并被至少一个 temporal expert 支持；
- reliable negative：两个证据都低，包括异常视频内部的大量正常 snippet；
- uncertain pool：证据冲突或靠近阈值的 snippet，前期不监督，后期只以低权重参与。

阈值采用每视频动态分位数并设置全局上限，避免 UCF 和 XD 使用不同算法；数据集差异只由训练分布估计。候选要求跨相邻 snippet 连续或来自多个 temporal clusters，防止只覆盖最显著的一小段。训练损失由作者原始 MIL、候选 ranking/contrastive loss、正常不变性和 soft distillation 组成。异常类别文本只监督 concept-causal branch，normal gallery 只监督 normal branch，Baseline 的最后时序块只接收蒸馏后的 soft evidence，避免端到端冲突。

### 7.4 分阶段优化，而不是渐进解冻的另一种写法

- Stage A：冻结全部 Baseline，离线发现 normal/concept circuits，建立 gallery，并验证 selected-vs-random、跨类别重合和 normal FPR。
- Stage B：冻结 Baseline，训练独立 evidence head 和两个 temporal experts。此阶段必须先证明 evidence-only score 有检测能力。
- Stage C：关闭 feature routing，只解冻每个 Baseline 作者本来训练的最后语义/时序适配部分，以低学习率蒸馏 CNED soft target；CNED 与 Baseline 分开 optimizer、交替更新。
- Stage D：固定两个专家，只训练不超过两层的 score fusion/calibration head。推理输出是 Baseline score 与 CNED evidence score 的置信度加权融合，而不是修改 512-D feature。

这种分阶段方式借鉴 DeSC 的冲突解耦、DSANet 的 text-side adaptation 和 D2MIL 的去噪-找回逻辑，同时保留通用接口。

## 8. 哪些 trick 可以直接借鉴，哪些不应作为主创新

### 8.1 直接借鉴

- **Normal vs anomaly 双锚点**：来自 TPWNG、UR-DMU、LAKE；应用到 neuron subspace，而不是再建全维 memory。
- **动态分位数与不确定区间**：来自 LAP、D2MIL；用于候选选择，不能用固定 0.5 阈值跨数据集。
- **多样片段覆盖**：来自 Road Less Seen；每个 temporal cluster 最多选择少量候选，避免 top-k 全挤在同一事件峰值。
- **transient/sustained 分支独立优化**：来自 DeSC；这是最有力的结构级依据。
- **text adapter 而非 visual encoder fine-tuning**：DSANet 与 TPWNG 都支持这一选择；TPWNG 中 image-side fine-tuning 反而下降。
- **hard normal mining**：来自 LaGoVAD Lneg、UR-DMU 和 MuST-VAD；重点处理正常视频中的最高 evidence/最高 Baseline score 片段。
- **轻量 score smoothing/sliding overlap**：仅作为最终边界修正，预期 0.2--0.7 点，不作为论文核心。

### 8.2 不应继续投入的方向

- 把 768-D hidden states 或选中神经元再次拼接进 512-D feature。
- 对选中神经元做正残差放大、门控增强或固定 suppression。
- 只依据 Baseline score 定义神经元或异常候选；这会把 Baseline 错误反馈为监督上限。
- 仅增加平滑、TTA、温度或 top-k 网格搜索并期待 2--3 点。
- 同时解冻 CLIP image encoder 与 Baseline；TPWNG 的消融显示小数据下容易破坏视觉空间。
- 把各论文模块的增益简单相加；消融增益高度非加性，尤其在强 Baseline 上会收缩。

## 9. 2--3 点目标的可行性与硬门槛

文献支持“该机制可能产生 2--3 点”，但**不支持“在三个强 Baseline、两个数据集上都必然提高 2--3 点”**。D2MIL 在较强 UCF Baseline 上通常只有 +0.22 到 +1.61 AUC，LAKE 的 +2.4 来自图像 patch setting，MuST-VAD 的 +5.21 主要体现在 AP 而 AUROC 仅 +0.48。DSANet 已达到 89.44 UCF AUC；若只使用同一批 frozen CLIP CLS feature，要到 91.44--92.44，需要新的 snippet supervision 明显修正 Baseline 排序，而不是普通插件校准。

为避免再进行无依据的大训练，建议设定以下 go/no-go：

1. **独立证据门槛**：在不使用 Baseline score 时，CNED evidence-only 的 UCF AUC/XD AP 必须至少超过 raw CLIP text similarity 2 点；selected circuits 必须显著优于 random/全部维度。
2. **候选质量门槛**：仅作离线诊断时，以 test GT 计算 top-q snippet precision、event coverage 和 normal FPR。相对 Baseline top-k，event coverage 至少提高 10 个百分点，normal FPR 不得上升。
3. **DSANet 单模型门槛**：training-only candidate supervision 必须先带来至少 +1.0 AUC；若不足，不进入三 Baseline 扩展。score fusion 再贡献至少 +0.5，才继续完整方案。
4. **因果对照门槛**：random neurons、activation-only neurons、concept-only neurons、normal-only neurons、full CNED 五组必须齐全。若 random 与 selected 差异小，不能把提升归因于解释性神经元。
5. **指标双约束**：UCF 同时报告 AUC 和 AP，XD 报告 AP；不接受 AUC 微升、AP 明显下降的 checkpoint。模型选择仍遵循各 Baseline 作者主指标。

若前两项未通过，说明 CLS snippet neuron subspace 的信息不足。此时唯一有强依据的结构升级是补提 CLIP patch-token 或低成本 motion foreground evidence，因为 STPrompt、LAKE 和 TLMA 的显著增益都依赖局部/前景信息；继续围绕 CLS 做 loss 微调不合理。

## 10. 结论

RQ1：三篇 Baseline 中，最大增益分别来自 DSANet 的 text adapter 与正常-异常语义分离、DeSC 的冲突目标解耦和互补专家、LaGoVAD 的语言条件化检测、动态合成与负样本挖掘。小型 TTA、平滑和单正则通常不足 1 点。

RQ2：跨论文共性是正常性锚定、可信 snippet 选择、冲突证据解耦、受控视觉-文本适配和多尺度时序覆盖。它们共同改善视频级弱标签到片段级判别依据的转换。显著增益的来源不是更多维 feature，而是更可靠的 evidence assignment。

RQ3：最有依据的结合方式是 CNED：神经元探测提供 normal-manifold、concept-causal 和 dual-temporal 三类独立证据，用于候选筛选、去噪、蒸馏和 score-level 专家融合；不再进行 feature injection。该方法保持 Baseline-score-free neuron discovery、三个 Baseline 的统一接口和单卡 4090 可行性。能否达到 +2--3 点必须由 evidence-only 与候选质量门槛先验证；若门槛失败，应补充 patch/motion 局部证据，而不是继续小幅调参。

## References

[1] Wenti Yin, Huaxin Zhang, et al., "Learning to Tell Apart: Weakly Supervised Video Anomaly Detection via Disentangled Semantic Alignment," arXiv:2511.10334, 2025.

[2] Hantao Zheng, Ning Han, Yawen Zeng, Hao Chen, "Decoupled Sensitivity-Consistency Learning for Weakly Supervised Video Anomaly Detection," arXiv:2603.19780, 2026.

[3] Zihao Liu, Xiaoyu Wu, Jianqin Wu, Xuxu Wang, Linlin Yang, "Language-Guided Open-World Video Anomaly Detection under Weak Supervision," ICLR, 2026.

[4] Peng Wu, Xuerong Zhou, et al., "VadCLIP: Adapting Vision-Language Models for Weakly Supervised Video Anomaly Detection," AAAI, 2024.

[5] Chenchen Tao, Xiaohao Peng, et al., "Learning Suspected Anomalies from Event Prompts for Video Anomaly Detection," arXiv:2403.01169, 2024.

[6] Zhiwei Yang, Jing Liu, Peng Wu, "Text Prompt with Normality Guidance for Weakly Supervised Video Anomaly Detection," CVPR, 2024.

[7] Peng Wu, Xuerong Zhou, et al., "Weakly Supervised Video Anomaly Detection and Localization with Spatio-Temporal Prompts," ACM Multimedia, 2024.

[8] Yu Tian, Guansong Pang, et al., "Weakly-Supervised Video Anomaly Detection with Robust Temporal Feature Magnitude Learning," ICCV, 2021.

[9] Yingxian Chen, Zhengzhe Liu, et al., "MGFN: Magnitude-Contrastive Glance-and-Focus Network for Weakly-Supervised Video Anomaly Detection," AAAI, 2023.

[10] Hang Zhou, Junqing Yu, Wei Yang, "Dual Memory Units with Uncertainty Regulation for Weakly Supervised Video Anomaly Detection," AAAI, 2023.

[11] Yujiang Pu, Xiaoyu Wu, Lulu Yang, Shengjin Wang, "Learning Prompt-Enhanced Context Features for Weakly-Supervised Video Anomaly Detection," arXiv:2306.14451, 2024.

[12] Shaotian Li, Shangze Li, et al., "Latent Anomaly Knowledge Excavation: Unveiling Sparse Sensitive Neurons in Vision-Language Models," arXiv:2604.07802, 2026.

[13] Yaxin Zhao, Yang Wang, et al., "Learning from Noisy Supervision: A Denoising-Debiasing Framework for Weakly Supervised Video Anomaly Detection," CVPR, 2026.

[14] Anusha Acharya, Hitesh Sapkota, Qi Yu, Xumin Liu, "The Road Less Seen: Segment Exploration for Weakly Supervised Video Anomaly Detection," CVPR, 2026.

[15] Satoshi Hashimoto, Hitoshi Nishimura, Mori Kurokawa, "MuST-VAD: Mutual Structured Learning for Video Anomaly Detection," arXiv:2608.06913, 2026.

[16] Yu Wang, Shiwei Chen, "Learning Event Completeness for Weakly Supervised Video Anomaly Detection," arXiv:2506.13095, 2025.
