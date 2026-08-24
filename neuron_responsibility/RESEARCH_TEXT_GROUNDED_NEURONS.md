# 从图像异常解释到视频异常检测：视觉-文本对齐应当成为独立的神经元责任监督

## 摘要

本报告研究三个问题：图像异常检测中的可解释方法哪些能够迁移到弱监督视频异常检测；NUS Fei Shen 团队的稀疏敏感神经元工作能提供哪些可复用机制；以及如何在单卡 RTX 4090、共享 CLIP ViT-B/16 特征和 hidden states 的条件下，设计同时适配 DSANet、DeSC、LaGoVAD 与 UCF-Crime、XD-Violence 的方法。

调研结果表明，图像异常检测中真正产生性能收益的“解释”通常不是事后热图，而是参与预测的结构化证据：正常流形偏离、文本语义验证、稀疏概念瓶颈或可干预的内部单元。LAKE 使用正常样本的高方差通道构造稀疏正常流形，并以 normal/anomalous 文本相似度作为次级验证；TextSpan、SpLiCE 和 CLIP-Dissect 则说明 CLIP 的联合空间可以把内部贡献转换为文本概念。但是，单个 hidden dimension 可能是 polysemantic 的，高方差也不等同于对输出有因果作用，因此仅按异常/正常激活差、baseline 分数相关性或通道方差挑选神经元，都不足以构成可信解释。

三个视频 baseline 的共同瓶颈并非缺少文本，而是文本监督依赖模型自身先产生的 snippet 权重：模型先猜测异常片段，再用这些片段进行文本对齐，形成 confirmation bias。报告据此提出 **Neuron-Text Responsibility Prior（NTRP）**：利用缓存的多层 CLS hidden states、CLIP 最终视觉投影和文本概念库，离线建立稀疏、文本可命名、具有下游影响并跨数据子集稳定的神经元责任图；再将其产生的 snippet 证据仅作为训练期的独立实例责任监督，约束三个 baseline 的原始 anomaly logits。该方案不使用 baseline 分数探测神经元，主版本推理时没有新增开销，并保留作者原损失和选模规则。

## 1. 研究问题

- **RQ1：** 图像异常检测的可解释性方法中，哪些机制能迁移到只有视频级标签的 VAD，而哪些只是不可迁移的空间热图？
- **RQ2：** 稀疏敏感神经元、CLIP 文本分解和因果干预各自提供什么证据；如何避免把相关性误写成神经元责任？
- **RQ3：** 如何利用三个 baseline 共有的视觉-文本对齐，在单卡 4090 上构造 baseline-independent、dataset-independent 且可能带来实质增益的方法？

## 2. 调研方法

检索覆盖六个视角：解释型图像/视频异常检测综述、CLIP 异常检测、Fei Shen 的异常神经元工作、CLIP 内部文本分解、神经元因果解释的反例、公开实现与计算成本。共执行 23 个检索式，审阅 248 个候选结果；正文只采用能够从论文原文、官方会议页面、作者主页或官方代码核验的方法。另对本地 DSANet、DeSC、LaGoVAD 论文和仓库代码逐项核对。

需要校准的一点是：LAKE 是 2026 年预印本，检索时尚无独立引用与公开官方代码，其实验结果可作为强假设和设计参考，但不能当作已被重复验证的定论。

## 3. 方法谱系：有性能作用的解释与事后解释应当分开

### 3.1 正常流形和样例检索：解释是“离哪个正常模式最远”

PatchCore 以局部特征到正常 memory bank 的最近邻距离定位异常；WinCLIP+ 同时利用正常参考图像与语言定义；DictAS 把异常判断改写为稀疏字典查询失败；LAKE 则把 memory bank 限制到由正常样本高方差通道构成的 100 维子空间。四者虽然实现不同，但解释对象一致：测试特征无法由正常参考模式解释。

这一机制适合视频迁移，因为 image patch 可以对应为 video snippet。不过工业图像通常拥有同类产品的正常 support set，而 UCF/XD 的正常视频跨场景、视角和行为差异很大。直接构造一个全局正常 gallery 会把 scene shift 当异常；视频版本必须采用“全局场景簇 + 视频内稳健正常中心”的层级正常流形，而不是照搬 LAKE 的单类别 memory bank。

### 3.2 视觉-文本语义：文本更适合验证异常，而不是单独决定异常

WinCLIP 用 normal/anomalous 两类状态词和模板集合显著改善原始 CLIP；AnomalyCLIP 学习 object-agnostic normality/abnormality prompts，并同时优化全局与局部语义；VCP-CLIP 用图像全局与局部上下文更新文本提示；LAKE 将文本分支设为几何偏离后的语义验证器，并报告文本权重超过 0.5 时性能明显下降。

这些结果共同指向一个边界：CLIP 文本空间能够排除“视觉变化但语义正常”的假阳性，却不适合替代结构异常证据。视频中的 generic prompt，如 “an anomalous event”，比工业缺陷描述更抽象，因此文本证据必须与神经元正常流形偏离相互同意，不能单独作为伪标签。

### 3.3 神经元和概念解释：可命名、可预测、可干预是三个不同层级

CLIP-Dissect 用数据集中激活神经元的样例和 CLIP 文本相似度自动命名视觉网络神经元；SpLiCE 把最终 CLIP embedding 分解为少量非负文本概念；TextSpan 进一步利用 ViT residual 与 attention 的可加结构，把最终视觉表示分解到 layer、head 和 patch，并用文本方向命名贡献。它们说明语言可以成为内部表示的解释接口，但不自动证明某个原始维度是单义、稳定和因果的。

Fei Shen 团队的 LAKE 以正常样本通道方差选择 Top-K 神经元，并报告 Top-100 明显优于随机 100；其优势是训练免费、稀疏且 memory 开销低。该团队的 Precise Shield 则采用更严格的“激活强度 × 下游影响”定位安全神经元，再只更新对应 LoRA 行。后者对本项目更有启发：神经元必须既对输入发生响应，又能影响目标输出。

Makelov 等人的反例说明，即使 activation patching 产生了预期输出，也可能通过 OOD 干预激活 dormant pathway，从而制造“找到有意义子空间”的假象。因此，本项目不应把单一选择分数称为因果解释。可信的神经元责任至少需要四项证据：

1. **Sensitivity：** 对目标概念有足够激活变化；
2. **Downstream influence：** 能改变 CLIP anomaly-vs-normal 文本 margin；
3. **Specificity：** 对少数可命名概念集中，而不是对所有概念普遍响应；
4. **Stability：** 在视频子集、UCF/XD 和随机种子间重复出现。

## 4. 三个 baseline 的共同结构与共同缺陷

| 模型 | 共享视觉输入 | 文本如何进入 | snippet 责任如何产生 | 共同风险 |
|---|---|---|---|---|
| DSANet | 冻结 CLIP ViT-B/16 的 512-D snippet 特征 | learnable prompt，event/background prototype 与类别文本对比 | 初始 anomaly logits 将视频拆成 event-centric 与 background-centric | 模型自己的早期分数决定后续语义监督对象 |
| DeSC | 冻结 CLIP ViT-B/16 特征 | sensitivity/consistency stream 都包含视觉-文本 alignment | MIL logits 与 alignment score 参与时序敏感分支和 Gaussian mixture prior | 语义先验仍由当前模型预测加权，易强化错误片段 |
| LaGoVAD | CLIP 图像特征 + CLIP 文本定义 | temporal encoder 后与 anomaly definitions 做 Transformer fusion | binary score 对视觉片段加权，形成正/负特征用于 hard-negative contrast | binary head 先选片段，文本对齐再确认该选择 |

三者的公共计算图可以写成：

```text
CLIP snippet feature -> temporal encoder -> provisional score
                                      -> score-weighted visual pooling
text prompt --------------------------> visual-text alignment
```

因此它们并不是“没有使用视觉-文本对齐”，而是 **文本对齐发生得太晚，而且依赖 provisional score**。在只有视频级标签时，一旦 provisional score 把背景、镜头切换或高运动片段选为异常，后续 alignment loss 会继续拟合该片段。这是三个结构不同的 baseline 共享的 confirmation bias，也是比“在哪一层注入残差”更本质的突破口。

## 5. 提议：Neuron-Text Responsibility Prior

### 5.1 核心原则

NTRP 不把神经元特征拼接进 baseline，也不以 baseline score 为 teacher。它建立一条独立证据链：

```text
cached CLIP hidden states
        +
CLIP text concept bank
        |
text-grounded neuron responsibility atlas
        |
structural deviation x semantic agreement
        |
training-only snippet responsibility prior
        |
original baseline loss + confidence-masked auxiliary MIL loss
```

这条证据链直接利用三个 baseline 的共同来源 CLIP，因此 probe 产物可以在三个 baseline 间复用；UCF/XD 只更换从训练标签自动生成的概念词，不改变网络或算法。

### 5.2 阶段 A：建立文本概念库

文本库分为三个互斥组，并使用固定模板 ensemble：

- normal/benign state：normal activity、ordinary event、safe interaction、routine motion；
- generic anomaly state：abnormal event、dangerous activity、violent event、accident；
- dataset label concepts：从训练 CSV 的类别字段或视频名自动解析，不手工为 UCF/XD 分别写模块。

文本编码只运行一次并缓存。解释时输出具体类别概念和 generic state，避免只有一个无法说明内容的 anomaly scalar。

### 5.3 阶段 B：从缓存 hidden states 计算责任，而不是相关性

对于最后一层 CLS hidden state `h`，CLIP 的 post-LayerNorm 和 visual projection 可以直接把 `h` 映射到共享 512-D 空间。首先必须验证该映射重建现有 512-D 特征，误差不超过预设阈值；不通过则停止，而不是训练一个任意 bridge。

定义文本 margin：

```text
m(h) = logsumexp(sim(project(h), T_anomaly))
     - logsumexp(sim(project(h), T_normal))
```

神经元的局部下游贡献采用 `Grad x centered activation`：

```text
r_d(h) = (h_d - median_normal_d) * partial m(h) / partial h_d
```

这比“异常均值减正常均值”多了 downstream influence，也不需要重新跑图像编码器。对多层 hidden states，利用 residual delta `Delta h_l = h_l - h_(l-1)` 分配 layer-wise contribution；它是对最终残差流的局部归因，不应宣称为完整因果路径。最终 Top-K 依据以下乘积/几何平均排序：

```text
importance = sensitivity * downstream_influence
           * concept_specificity * split_stability
```

只保留跨 5 个数据划分稳定、概念熵较低的神经元。随后对 Top-K 做 mean-ablation 与随机 K 对照；只有真实文本 margin 下降显著高于随机选择，才能称为责任神经元。

### 5.4 阶段 C：构造 baseline-independent snippet evidence

每个 snippet 产生两类证据：

1. **结构偏离 `d_t`：** 在 selected-neuron space 中，到层级正常流形的距离。层级流形由 normal training snippets 的场景簇和当前视频最大稳健簇共同组成，以减小跨场景误报。
2. **语义 margin `m_t`：** selected neurons 对 anomaly concepts 与 normal concepts 的责任差。

二者使用 agreement gate，而不是自由 MLP：

```text
e_t = ranknorm(d_t) * sigmoid(calibrated(m_t))
```

只有结构异常且文本语义也异常的片段得到高置信证据。镜头切换可能有高 `d_t`，但通常没有异常文本 margin；语义相似但视觉正常的片段则没有高结构偏离。

### 5.5 阶段 D：训练期责任监督，推理期零新增开销

保留每个 baseline 的作者损失 `L_author`。对最终 binary anomaly logits `s_t` 增加同一个外部损失：

```text
L = L_author
  + lambda_conf * sum_t w_t * BCE(sigmoid(s_t), stopgrad(e_t))
  + lambda_rank * Rank(s_positive, s_reliable_normal)
```

其中 `w_t` 只在 `e_t` 的高、低分位取非零值，中间不确定片段不监督。异常视频的高证据片段用于 positive instance；正常视频和异常视频中的低证据稳定片段用于 reliable negative。该损失只要求 baseline 暴露 binary logits，因此三个模型使用完全相同的 wrapper。

主实验应采用 teacher-only-at-training：测试时仍只跑原 baseline，额外推理参数和 FLOPs 为零。只有诊断实验才融合 `e_t`，用于判断 neuron prior 是否具有互补信息，不作为主结果。

“零新增推理开销”特指主检测路径。需要生成解释时，可读取已经提取的 test hidden states，输出 top concepts、top neurons 和结构偏离来源；若在线系统本来只保存 512-D feature，则需要在同一次 CLIP feature extraction 中额外缓存所选 64-128 个 hidden dimensions。解释模式的存储开销不应混入检测模型 FLOPs。由于 teacher 只在训练期出现，解释是否忠实于最终 baseline 还必须额外验证：对 selected neurons 做 mean ablation、重新投影为 512-D feature 并运行 baseline，确认对应 snippet score 的下降显著大于 random-K。否则输出只能称为“训练责任依据”，不能称为最终预测的忠实解释。

### 5.6 解冻策略

根据已完成的 DSANet 实验，head 解冻的梯度冲突最高且导致性能下降。NTRP 的默认策略应为：

- neuron atlas 和 evidence 全程冻结；
- baseline head 固定；
- 只解冻最靠近输入的 feature normalization/adapter 以及最后一个 temporal block；
- 固定 sample-count 评估，按作者规则选择 AUC/AP 最优检查点；
- 不设置“最后阶段自动解冻 head”。

这不是因为所有 baseline 的同名模块相同，而是因为统一原则相同：允许 temporal representation 接受新的 instance responsibility，保留作者已训练好的 decision boundary。

## 6. 可解释性如何被定量验证

仅展示 neuron index 或曲线不够。NTRP 至少报告：

| 维度 | 指标 | 通过标准的建议起点 |
|---|---|---:|
| Faithfulness | selected-neuron mean ablation 导致的文本 margin 下降 / random-K | 大于 3 倍 |
| Sparsity | 达到 90% 累积责任所需神经元数 | 不超过 128 |
| Concept specificity | 每个神经元责任概念分布的归一化熵 | 明显低于 random-K |
| Stability | 5-fold neuron set Jaccard | 大于 0.4 |
| Cross-dataset generality | UCF 与 XD 的 generic-neuron Jaccard | 大于 0.25 |
| Localization quality | prior 的 top-5% snippet precision | 至少 25%-30% |
| Complementarity | baseline 与 prior 错误不一致性、oracle fusion AUC | held-out validation 至少 +0.5 AUC 点再进入完整训练 |

这些阈值不是论文结论，而是节省实验成本的工程 gate。若 prior 自身仍只有约 73 AUC、top-5% precision 约 13%，应停止联合训练，先修复 probe；不能期待 baseline 解冻弥补低质量责任监督。

## 7. 单卡 4090 开销评估

| 阶段 | 主要操作 | 预计显存/存储特征 | 是否每个 baseline 重跑 |
|---|---|---|---|
| 文本库 | 数十到数百条 prompt 的 CLIP text encode | 可忽略，512-D 缓存 | 否 |
| 最后一层责任 | cached 768-D CLS 经 LN + projection；自动微分 margin | 小批量矩阵运算，远低于完整 CLIP 反传 | 否 |
| 多层 residual attribution | 相邻 hidden states 作差并乘最终局部归因 | 纯缓存计算 | 否 |
| 正常流形 | 64-128 维 FAISS/torch kNN + 场景聚类 | 可离线分块；远小于全 768/512 memory bank | 否 |
| baseline 训练 | 原模型 + 一个 confidence-masked loss | 主版本几乎不增加模型显存 | 是 |
| baseline 推理 | 原作者模型 | **零新增参数、零新增 CLIP forward** | 是 |

与重新运行 CLIP 全层 gradient probing 相比，利用最后 CLS hidden state 与视觉 projection 是更适合当前产物和 4090 的路线。只有在最后层责任通过 gate、但上限仍不足时，才值得抽取少量原始帧做完整 layer intervention；不应一开始就重跑全部视频。

## 8. 风险、反证与优先实验

### 8.1 最大风险

- **CLIP 对 surveillance anomaly 文本不敏感。** 工业缺陷中的 “cracked bottle” 比 “abnormal event” 更具体；UCF/XD 的语义 gap 更大。
- **原始维度 polysemantic。** 概念 specificity 可能很低，此时应从单维神经元升级为稀疏非负 concept direction，但解释对象必须改称“概念方向”，不能继续称单神经元。
- **视频内最大簇不一定正常。** 长时间暴力或火灾会违反“异常稀疏”假设，需要训练集正常场景簇兜底。
- **teacher 可能只复制最终 CLIP feature。** 必须比较 final 512 text margin、all-neuron margin、selected-neuron margin；若 selected 版本没有增加互补性，稀疏化只是压缩而不是新知识。

### 8.2 最小可证伪实验顺序

1. 验证 final CLS hidden state 经 CLIP LN/projection 是否精确重建现有 512-D feature。
2. 只在 UCF 上建立 final-layer responsibility atlas，不训练任何 baseline。
3. 报告 selected vs random neuron ablation、概念熵、5-fold 稳定性。
4. 在从训练视频固定划出的 held-out validation 上，计算 prior 的 snippet ranking、top-1/5/10% precision，以及与作者 DSANet score 的相关性和错误互补性；正式 test GT 不参与方法 gate 或超参数选择。
5. 只有 validation oracle fusion 达到至少 +0.5 AUC 点，才训练 DSANet 的 NTRP wrapper。最终论文表格仍严格沿用各 baseline 的官方测试和选模协议，并明确披露协议差异。
6. DSANet 有稳定收益后，复用同一 UCF atlas 到 DeSC/LaGoVAD；随后再在 XD 重建 dataset evidence，但不改方法或超参数结构。

这一顺序把最便宜、最能推翻方法的测试放在前面。它比直接运行八个 epoch 更适合当前算力，也能明确失败究竟来自 CLIP 文本语义、神经元定位、正常流形还是 baseline 优化。

## 9. 对研究问题的回答

**RQ1：** 可迁移的不是图像异常热图本身，而是“正常参考无法解释的结构偏离 + 文本语义验证 + 稀疏内部责任”三件套。patch token 应改写为 temporal snippet，类别正常 gallery 应改为层级场景正常流形。

**RQ2：** Fei Shen 团队最有价值的启发不是简单 Top-K，而是 LAKE 的结构/语义双证据以及 Precise Shield 的 activation × downstream impact。单纯激活差或通道方差只能说明预测相关性；加入下游影响、概念 specificity、跨划分稳定性和真实 ablation 后，才可谨慎称为神经元责任。

**RQ3：** 最有希望的统一突破口是把神经元-文本证据作为训练期独立 instance responsibility prior，打断三个 baseline 共有的“自己打分、自己选片段、再用文本确认”的闭环。NTRP-Lite 复用现有 hidden states 与 512-D features，探测不依赖 baseline，训练只增加外部 loss，推理零新增开销，符合单卡 4090 与通用性要求。

这一结论是基于结构分析和相邻领域证据得出的优先研究假设，不是对显著增益的保证。LAKE 的强结果来自同类工业图像、patch token 和正常 support set，视频监控中的场景多样性与抽象异常文本可能削弱迁移效果；因此必须先通过低成本、可证伪的 responsibility gate，再决定是否投入三个 baseline、两个数据集的完整训练。

## 参考文献

[1] K. Roth et al., "Towards Total Recall in Industrial Anomaly Detection," CVPR, 2022.

[2] J. Jeong et al., "WinCLIP: Zero-/Few-Shot Anomaly Classification and Segmentation," CVPR, 2023.

[3] Q. Zhou et al., "AnomalyCLIP: Object-agnostic Prompt Learning for Zero-shot Anomaly Detection," ICLR, 2024.

[4] Z. Qu et al., "VCP-CLIP: A Visual Context Prompting Model for Zero-Shot Anomaly Segmentation," ECCV, 2024.

[5] Z. Qu et al., "DictAS: A Framework for Class-Generalizable Few-Shot Anomaly Segmentation via Dictionary Lookup," ICCV, 2025.

[6] S. Li et al., "Latent Anomaly Knowledge Excavation: Unveiling Sparse Sensitive Neurons in Vision-Language Models," arXiv:2604.07802, 2026.

[7] T. Oikarinen and T.-W. Weng, "CLIP-Dissect: Automatic Description of Neuron Representations in Deep Vision Networks," ICLR, 2023.

[8] U. Bhalla et al., "Interpreting CLIP with Sparse Linear Concept Embeddings," NeurIPS, 2024.

[9] Y. Gandelsman, A. A. Efros, and J. Steinhardt, "Interpreting CLIP's Image Representation via Text-Based Decomposition," ICLR, 2024.

[10] G. Goh et al., "Multimodal Neurons in Artificial Neural Networks," Distill, 2021.

[11] A. Makelov et al., "Is This the Subspace You Are Looking for? An Interpretability Illusion for Subspace Activation Patching," ICLR, 2024.

[12] E. Shi et al., "Precise Shield: Explaining and Aligning VLLM Safety via Neuron-Level Guidance," arXiv:2604.08881, 2026.

[13] Y. Wang et al., "Unveiling the Unseen: A Comprehensive Survey on Explainable Anomaly Detection in Images and Videos," arXiv:2302.06670, 2023.

[14] W. Yin et al., "Learning to Tell Apart: Weakly Supervised Video Anomaly Detection via Disentangled Semantic Alignment," AAAI, 2026.

[15] H. Zheng et al., "Decoupled Sensitivity-Consistency Learning for Weakly Supervised Video Anomaly Detection," arXiv:2603.19780, 2026.

[16] Z. Liu et al., "Language-Guided Open-World Video Anomaly Detection under Weak Supervision," ICLR, 2026.
