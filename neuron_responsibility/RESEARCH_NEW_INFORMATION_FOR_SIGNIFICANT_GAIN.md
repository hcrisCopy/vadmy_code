# 面向显著增益的新信息注入调研

## 摘要

已有实验表明，只在同一份 CLIP hidden states 上重新选择维度、残差注入或用 baseline 分数生成伪标签，主要是在重组已有证据，难以突破强 baseline。本文围绕三类真正新增的信息源展开：训练期运动模态、可验证的合成事件边界、带完整性与不确定性建模的时间伪标签。文献消融共同表明，新增信息只有经过跨模态对齐、蒸馏或可靠性筛选后才稳定有效；直接拼接甚至可能降低指标。结合单卡 RTX 4090、无 patch token、三个 baseline 通用的约束，优先建议采用“训练期光流教师 + RGB/CLIP 学生蒸馏”，神经元探测用于定位和解释被迁移的内部功能子空间，而不再承担伪标签教师的角色。

## 1. 研究问题

1. 哪些方法确实引入了 baseline 当前没有的信息，并由消融证明能够提高 AUC/AP？
2. 哪些信息可以在训练期使用、测试期移除，从而兼顾单卡开销与三个 baseline 的通用性？
3. 如何保留“CLIP hidden-state 神经元可解释性”，同时避免再次退化为同源特征重排？

## 2. 证据汇总

| 方向 | 代表工作与消融证据 | 真正新增的信息 | 对本项目的含义 |
|---|---|---|---|
| 多模态训练期诱导 | π-VAD：UR-DMU 在 UCF-Crime 为 86.97；单独 motion 为 87.92；五模态为 90.33。XD-Violence 从 81.66 提升到 85.37 [1] | 光流、姿态、深度、全景分割、文本；XD 还使用音频 | 2–3 点级增益来自独立模态及其蒸馏，不是继续挖同一 RGB/CLIP 向量 |
| 上下文—运动关系 | CMRL：UCF 基础模型 82.84，完整模型 86.07；在已有模块上加入 CoMo 由 85.22 到 86.07 [2] | 显式建模运动与场景上下文的条件关系 | 光流不能单独判异常；应学习“该场景下该运动是否异常” |
| 完整性与不确定性伪标签 | CUPL：UCF 基线 82.86；多头完整性 84.89；不确定性细化 84.69；二者结合 86.22 [3] | 多个相互分歧的时间假设，以及伪标签置信度 | 不能再用单个 baseline 分数平滑后自我监督；必须有多视角和拒绝不可靠片段的机制 |
| RGB/Flow 交叉教师 | UGCT 在三个弱监督时间动作定位 baseline 上，mAP@0.5 均提升超过 4 点 [4] | RGB 与 Flow 两个独立视角互相提供伪标签，EMA 教师与不确定性降低噪声 | 属于邻近任务证据，但直接支持“异源教师优于同源自训练” |
| 可验证事件合成 | LaGoVAD 的 Dynamic Video Synthesis 消融中，平均检测指标由 65.73 提升到 69.98 [5] | 通过片段组合得到精确的合成异常区间和可控异常时长 | 对 DSANet/DeSC 是低成本的边界监督；但 LaGoVAD 已包含该机制，不能作为三个 baseline 共同的新核心 |
| 自训练定位 | MIST 中伪标签自训练相对其基础模型有大幅增益，稀疏连续采样也优于均匀采样 [6] | 片段级训练信号与局部连续上下文 | 证明时间伪标签有价值，但旧弱 baseline 上的增益不能直接外推到当前强模型 |

## 3. 最重要的负面证据

π-VAD 的直接多模态 late fusion 使 UR-DMU 在 UCF-Crime 上由约 86.9 降到 83.6，而采用跨模态诱导、语义对齐和蒸馏后达到 90.33 [1]。因此，以下做法不应作为主方案：

- 直接把光流特征或更多 hidden states 拼到 baseline 输入；
- 把 CLIP 时间差分称为新运动信息。它仍由同一 RGB 表征确定，并非独立信息源；
- 用一个 baseline 的高分片段监督同一个 baseline。该过程会固化原有漏检和错误峰值；
- 让少量被选神经元直接充当门控器。现有实验中的门控准确率不足，缺少独立教师来定义其功能。

## 4. 推荐方案：运动诱导的可解释神经元蒸馏

### 4.1 核心结构

1. 对训练集按现有 snippet 划分离线提取一条光流/运动特征流；同一份缓存供三个 baseline 共用。
2. 用 RGB/CLIP 分支和运动分支形成两个训练视角。运动分支只在训练期存在，并采用 EMA 教师或先训练后冻结。
3. 不直接拼接两路特征。将运动教师的中间时序关系、异常排序和边界变化蒸馏给 baseline 可训练的时序模块。
4. 只在两个视角一致且不确定性低的片段上施加正/负片段监督；分歧片段降权，不生成硬标签。
5. hidden states 用来定义可解释的“神经元子空间”：按层选择能够稳定预测运动教师关系或边界变化的一组维度，而不是按 baseline 异常分数选择维度。训练时约束可训练适配器对该子空间的读出与运动教师一致。
6. 测试时移除光流教师，仅保留 baseline、轻量时序适配器及神经元读出，因此推理输入仍为现有 CLIP 特征和 hidden states。

### 4.2 为什么仍然属于神经元探测

探测对象仍是 CLIP ViT-B/16 各层 CLS hidden-state 的维度或低秩维度组，但功能定义改变了：旧方法寻找“与 baseline 分数相关的神经元”，新方法寻找“对独立运动关系、时间边界和上下文—运动不一致敏感的神经元”。前者解释 baseline 已经会什么，后者定位 baseline 缺失信息在 CLIP 内部能够被读取或承载的位置。

### 4.3 三个 baseline 的统一接口

统一模块只需要三个张量：snippet 表征、baseline 中间时序表征、snippet anomaly logits。对 DSANet、DeSC、LaGoVAD 分别接入各自最后一个时序建模模块之前或之后，但损失与教师完全一致。允许三个 baseline 解冻的具体层不同；通用性要求的是相同的信息源、目标函数和训练协议，而不是强行解冻同名参数。

## 5. 次优但低成本的辅助项

动态事件合成可先用于 DSANet 的快速验证：在特征级构造 normal–abnormal–normal 序列，合成标签由拼接位置直接得到，并加入边界损失。它几乎不增加显存和特征存储，但必须做长度随机化、边界过渡和困难正常片段采样，以降低模型识别拼接伪影的风险。由于 LaGoVAD 已有 DVS，它应作为训练辅助，而不是论文的统一核心创新。

CUPL 式完整性—不确定性细化适合放在运动教师之后：多头覆盖不同异常区间，RGB/Flow 一致性与 MC/头间方差共同决定片段权重。它不应先于独立运动信息单独实现，否则仍可能只是更精致地复制 baseline 偏差。

## 6. 开销与实验顺序

| 阶段 | 额外训练成本 | 推理成本 | 建议判据 |
|---|---:|---:|---|
| A. 特征级事件合成 | 很低 | 无 | 先看 DSANet-UCF 是否稳定超过 baseline，而非单次最好值 |
| B. 单运动模态离线缓存 | 一次性提取；存储约为一条同维 snippet 特征流 | 无 | motion-only 教师应提供与 RGB 不完全重合且有正增益的排序/边界信号 |
| C. 跨模态蒸馏 | 训练期约 1.5–2 倍分支开销，可预提特征降低显存 | 仅轻量适配器 | DSANet-UCF 达到稳定增益后再迁移 DeSC/LaGoVAD |
| D. 多头不确定性细化 | 多次前向或多头训练 | 无或可移除 | 只有在伪标签覆盖率提高且噪声受控时保留 |

不建议一次实现五模态 π-VAD。原论文的训练期模态骨干总开销很大，并在 24 GB Titan RTX 上实验；单卡 4090 更合理的做法是只保留证据最直接的 motion 流，离线预提取后复用。

## 7. 结论

最有依据的突破口不是“更复杂的神经元选择”，而是“用独立运动信息训练 baseline，再用神经元解释和约束信息转移”。运动教师解决新信息来源，跨模态蒸馏解决测试期成本，完整性与不确定性解决弱标签的时间定位，神经元子空间提供可解释性。四者职责清晰，且只有第一项需要新增一次性数据产物。

## 参考文献

[1] Majhi et al., “Just Dance with π! A Poly-modal Inductor for Weakly-supervised Video Anomaly Detection,” CVPR 2025. https://openaccess.thecvf.com/content/CVPR2025/html/Majhi_Just_Dance_with_pi_A_Poly-modal_Inductor_for_Weakly-supervised_Video_CVPR_2025_paper.html

[2] Cho et al., “Look Around for Anomalies: Weakly-Supervised Anomaly Detection via Context-Motion Relational Learning,” CVPR 2023. https://openaccess.thecvf.com/content/CVPR2023/html/Cho_Look_Around_for_Anomalies_Weakly-Supervised_Anomaly_Detection_via_Context-Motion_Relational_CVPR_2023_paper.html

[3] Zhang et al., “Exploiting Completeness and Uncertainty of Pseudo Labels for Weakly Supervised Video Anomaly Detection,” CVPR 2023. https://openaccess.thecvf.com/content/CVPR2023/html/Zhang_Exploiting_Completeness_and_Uncertainty_of_Pseudo_Labels_for_Weakly_Supervised_CVPR_2023_paper.html

[4] Yang et al., “Uncertainty Guided Collaborative Training for Weakly Supervised Temporal Action Detection,” CVPR 2021. https://openaccess.thecvf.com/content/CVPR2021/html/Yang_Uncertainty_Guided_Collaborative_Training_for_Weakly_Supervised_Temporal_Action_Detection_CVPR_2021_paper.html

[5] Zhu et al., “Language-guided Open-world Video Anomaly Detection under Weak Supervision,” ICLR 2026. https://arxiv.org/abs/2503.13160

[6] Feng et al., “MIST: Multiple Instance Self-Training Framework for Video Anomaly Detection,” CVPR 2021. https://openaccess.thecvf.com/content/CVPR2021/html/Feng_MIST_Multiple_Instance_Self-Training_Framework_for_Video_Anomaly_Detection_CVPR_2021_paper.html
