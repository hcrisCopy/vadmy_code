# 从失败的神经元教师到神经元路由专家：下一步优化方向的证据审计

## 摘要

本报告回答三个问题：三篇 baseline 中哪些机制真正贡献 AUC/AP；近年 WS-VAD 中哪些增益机制得到相互支持；CLIP hidden-state 神经元怎样参与模型而不再充当弱定位教师。现有实验与论文证据共同否定了“CLS 单维度探针直接生成片段伪标签”这条路线。当前最有依据的方向是：保留 baseline 作为主定位器，把类别敏感神经元降级为可解释的专家路由信号，再用类别专家、互补时间尺度和完整事件损失产生可学习的残差。它有望提升指标，但没有证据保证在三个已经很强且结构不同的 baseline 上同时提升 2–3 点。

## 1. 研究问题

- RQ1：DSANet、DeSC、LaGoVAD 的大幅增益究竟来自什么？
- RQ2：哪些机制在其他 WS-VAD 工作中重复出现，并且与我们的失败结果一致？
- RQ3：神经元怎样提供 baseline 尚未利用的信息，而不是用更弱的分数覆盖 baseline？

## 2. 方法

证据包括三篇本地 baseline 原文及其消融、CVF/arXiv 可验证原文，以及本项目 DSANet/UCF 的 TRACE、DANCE、CACC、TCNP 实验。检索分别覆盖正常性建模、时间完整性、类别专家、CLIP 适配和神经元解释五个方向。工业图像 AD 的结果只用于分析机制，不直接外推到视频。

## 3. Baseline 消融揭示的共同机制

| 工作 | 最大或关键消融 | 指标变化 | 可以支持的判断 |
|---|---|---:|---|
| DSANet | frozen text encoder → text Adapter | XD AP 81.57 → 86.95 | 任务内适配文本空间比固定 prompt 更重要 |
| DSANet | VadCLIP baseline → Adapter+SG-NM+DCSA | XD AP 84.51 → 86.95 | 正常性、类别语义和适配具有协同效应 |
| DeSC | unified joint optimization → collaborative decoupled streams | UCF AUC 86.18 → 89.37；XD AP 80.22 → 87.18 | 分离冲突目标比在单一分支堆损失有效 |
| DeSC | TCN only / GT only → parallel TCN+GT | UCF 85.80/86.26 → 88.46；XD 79.78/84.55 → 85.04 | 短时突变与全局关系互补 |
| DeSC | non-overlap ensemble → sliding-window TTA | UCF 89.13 → 89.37；XD 86.47 → 87.18 | 边界平滑有稳定但较小增益 |
| LaGoVAD | 去掉 dynamic video synthesis | Det. Avg. 69.98 → 65.73 | 训练事件组合与时间分布的多样性是主要来源 |
| LaGoVAD | 去掉 negative mining | Det. Avg. 69.98 → 68.92 | 困难负例有约 1 点量级贡献 |
| LaGoVAD | class name → video-specific description | UCF 80.44 → 83.03 | 更具体的事件语义可能帮助定位，但标准测试没有此描述 |

三者并不是靠同一种结构取胜，但共同点十分明确：**显著增益来自改变优化分工、训练分布或表示空间，而不是在固定特征旁边拼接一个弱分数。** DSANet 适配文本空间并显式建模正常性，DeSC 让冲突的快/慢目标分别优化，LaGoVAD 用动态事件合成和负例改变训练分布。我们的历次残差注入、固定 probe 和伪标签实验没有改变这三个核心因素，所以反复出现 0 增益并不意外。

## 4. 相关 WS-VAD 证据

### 4.1 类别专家与完整事件学习

GS-MoE 的表格消融从 UR-DMU 86.97 AUC 出发：加入 Temporal Gaussian Splatting 为 87.84，加入类别 experts 为 89.53，再加入 gate 为 91.58。最大单步增益来自 gate（+2.05 AUC），而完整的类别专家路径累计 +4.61。论文正文声称 TGS 单独 +1.77，但表格实际为 +0.87，存在内部数字不一致，应以表格差值为准。该方法还显示，遮蔽对应类别 expert 后类别 AUC 接近随机，支持类别专门化具有因果作用，而非仅增加参数。

不过，GS-MoE 在 XD 的常规 AP 为 82.89，低于 DSANet/DeSC 的约 87 AP；因此它支持“类别专家和完整事件损失能显著改善 UCF”，不支持“直接复制便能在两个数据集都提高 2–3 点”。DeSC 在两个数据集上的结果则支持互补时间尺度更稳定，但 DeSC 本身已经吃到了这部分收益。

LAS-VAD 使用 anomaly-connected components 把相邻且语义一致的帧组成事件，再配合 intention prototypes 和跨意图对比；TPWNG 用正常性引导生成伪标签并学习自适应时间上下文。它们与 GS-MoE 的共同点不是某个具体网络层，而是把异常从孤立 top-k 点改造成结构化事件。这与我们观察到的“独立 probe 召回/精度不够、top/bottom 排序扰动强 baseline”完全一致。

### 4.2 正常性是约束，不是充分定位器

NG-MIL、TPWNG、DSANet 和 DeSC 都利用正常模式抑制误报，但方式不同：NG-MIL 用纯正常视频建立原型，DSANet 从视频内提取动态正常原型，DeSC 用 Gaussian mixture 作为平滑先验。独立证据支持正常性有用，却没有支持“正常距离本身足以定位全部异常”。本项目 TCNP 的正常原型距离在训练留出集 2% FPR 下只覆盖 46.84% 异常视频；强制与文本分数取交集时覆盖降至 22.78%。因此正常性应作为负约束或 expert 的一支，不能继续当唯一教师。

## 5. 为什么 LAKE 在图像有效、我们的 CLS 迁移失败

LAKE 在工业图像 AD 中从正常样本的 **patch token** 方差选择 Top-100 神经元，在该子空间建立正常 patch gallery，通过最近邻距离和 max pooling 定位局部缺陷；更深层 patch 再与正常/异常文本做语义验证。Top-100 对随机 100 的 MVTec image AUROC 为 94.7 对 81.6，说明它的神经元选择确有价值。但论文消融同时表明视觉结构分数必须占主导，文本权重最优为 0.3，达到或超过 0.5 会明显退化。

这和本项目有三个本质差异：

1. LAKE 有空间 patch，可直接形成局部正常图库；我们的 CLS 已经把前景、背景和动作压成一个向量。
2. 工业 AD 的正常训练图像是干净的，视频异常袋同时包含大量正常片段，无法用同样方式建立异常侧定位依据。
3. LAKE 原文明确把 temporal video 扩展列为未来工作，不能把其图像结果当作 CLS 视频方案的有效性证明。

因此，LAKE 对我们真正有用的启发不是“继续把更多 CLS 维度融合成异常分数”，而是：神经元信号应承担它能完成的任务。CLS 类别神经元适合判断“这段更像哪类事件/应由哪个专家处理”，不适合单独判断精确开始和结束时间。

## 6. 推荐方案：Neuron-Routed Event Experts（NREE）

### 6.1 核心设计

保持三个 baseline 的作者异常分数为主路径，在其时序特征与最终 head 之间增加同一套低秩 residual experts：

- `fast expert`：小卷积核/高频差分，处理爆炸、打斗等突发事件；
- `slow expert`：膨胀卷积或轻量局部 attention，处理偷窃、虐待等持续事件；
- `class experts`：每类一个低秩 residual，而不是复制完整 Transformer；
- `normal expert`：只用正常视频的 dense negative loss 抑制误报；
- `neuron router`：用稳定的类别敏感 CLS 神经元决定各 expert 权重，不直接输出最终异常分数。

最终预测为：

`s_final = s_author + gamma * sum_c r_c(h_neuron) * delta_s_c(f_baseline)`。

其中 `gamma` 初始化为 0，保证训练开始严格复现作者 baseline；`r_c` 是神经元路由，`delta_s_c` 是低秩专家残差。即使神经元无效，模型也可以回到作者路径，不再出现弱探针强制覆盖强 baseline 的结构性错误。

### 6.2 时间监督

异常视频不用固定 hard pseudo-label，而采用 event-complete self-training：

1. 从当前模型的类别分数寻找多个局部峰，而不是单个 top-k；
2. 根据峰附近曲率/连通区估计事件宽度，生成 stop-gradient Gaussian/connected-component 软目标；
3. fast/slow experts 分别拟合窄、宽事件；
4. 正常视频全部片段提供可信 dense negative；
5. 每轮目标随模型更新，因此不是把初始 baseline 分数当不可突破的教师上限。

该部分分别对应 GS-MoE 的 TGS、LAS-VAD 的 connected components 和 DeSC 的快慢互补，但做成可插拔低秩残差以适配三种 baseline。

### 6.3 神经元解释性的可证伪设计

神经元发现不读取 baseline 分数，只用训练视频类别、冻结 CLIP 文本方向和跨 fold 稳定性。与之前不同，评价目标改为路由而非片段 AUC：

- 路由类别准确率和 expert specialization；
- 遮蔽目标类别神经元后，对应 expert 使用率与类别 MIL 性能是否显著下降；
- 随机同规模神经元、全维 router、无神经元 learned router 三个公平对照；
- 若神经元 router 不优于普通 learned router，就只能把它写成解释工具，不能声称它带来性能。

这使“可解释性”成为可检验的机制贡献，而不是用一张 neuron 排名图代替因果证据。

## 7. 实验顺序和止损线

不应直接做三 baseline × 两数据集全量实验。先在 DSANet/UCF 做四个严格递进的 pilot：

| Pilot | 目的 | 继续条件 |
|---|---|---|
| 作者 baseline + event-complete loss | 验证时间完整性是否对强 baseline 仍有效 | 5120 样本至少 +0.3 AUC 点 |
| + low-rank class experts + learned gate | 验证 GS-MoE 机制能否迁移到 CLIP baseline | 比上一项至少 +0.3 点 |
| learned gate → neuron router | 验证神经元是否提供性能或同等性能下的解释性 | 不低于 learned gate，且通过遮蔽因果测试 |
| + fast/slow experts | 验证 DeSC 式互补是否仍有增量 | 至少 +0.2 点，否则不扩展 |

只有 UCF pilot 超过作者 1 点后，才值得跑 XD；只有 DSANet 两个数据集均为正增益，才适配 DeSC 和 LaGoVAD。这样不会再为一个根本不工作的假设连续消耗多轮训练。

## 8. 开销

采用每类 `512→32→1` 的低秩 expert，UCF 13 类约 0.22M 参数；fast/slow depthwise temporal blocks、router 和 gate 合计可控制在 1M 参数以内。无需 patch token，不重新执行 CLIP，复用现有 CLS hidden states。相对 baseline 预计增加约 10–25% 训练计算，显存主要仍由 baseline 决定，单卡 4090 可行。紧凑神经元建议限制到每类 16–32 个并取 union，避免当前 1254 维“稀疏回路”名不副实。

## 9. 结论

RQ1：三个 baseline 的强增益来自适配、分工和训练分布，而不是固定外部特征注入。RQ2：类别专家、互补时间尺度、正常负约束和完整事件学习具有最强的交叉证据，但没有单一方法证明能在 UCF/XD 和所有强 baseline 上统一提升 2–3 点。RQ3：CLS 神经元最合理的角色是可解释的类别路由器，而不是时间定位教师。

因此，下一步值得实现的是 NREE 的分阶段 pilot，而不是继续优化 TCNP 阈值或再训练一个独立 probe。若 event-complete loss 与普通 learned experts 都不能先超过 DSANet，则应停止“通用插件 +2–3 点”的目标，转而把工作定位为解释性分析论文；这是当前证据允许的最诚实边界。

## References

[1] DSANet authors, “Learning to Tell Apart: Weakly Supervised Video Anomaly Detection via Normality Modeling and Disentangled Semantic Alignment,” 2025.

[2] DeSC authors, “Decoupled Sensitivity-Consistency Learning for Weakly Supervised Video Anomaly Detection,” 2025.

[3] LaGoVAD authors, “Language-Guided Open-World Video Anomaly Detection,” ICLR, 2026.

[4] G. D’Amicantonio et al., “Mixture of Experts Guided by Gaussian Splatters Matters: A New Approach to Weakly-Supervised Video Anomaly Detection,” ICCV, 2025.

[5] Y. Wang and S. Zhao, “Weakly Supervised Video Anomaly Detection with Anomaly-Connected Components and Intention Reasoning,” CVPR, 2026.

[6] Z. Yang, J. Liu, and P. Wu, “Text Prompt with Normality Guidance for Weakly Supervised Video Anomaly Detection,” CVPR, 2024.

[7] S. Park et al., “Normality Guided Multiple Instance Learning for Weakly Supervised Video Anomaly Detection,” WACV, 2023.

[8] S. Li et al., “Latent Anomaly Knowledge Excavation: Unveiling Sparse Sensitive Neurons in Vision-Language Models,” arXiv:2604.07802, 2026.

[9] B. Bau et al., “Network Dissection: Quantifying Interpretability of Deep Visual Representations,” CVPR, 2017.

[10] T. Oikarinen and T.-W. Weng, “CLIP-Dissect: Automatic Description of Neuron Representations in Deep Vision Networks,” ICLR, 2023.
