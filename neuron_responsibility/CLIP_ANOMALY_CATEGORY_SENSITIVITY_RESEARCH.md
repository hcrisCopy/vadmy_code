# CLIP 在视频异常类别上的敏感性：从异常无感知到定义盲视

## 摘要

本调研考察三个问题：原始或冻结 CLIP 是否能够区分视频中的正常/异常语义，是否能够进一步区分不同异常类别，以及这一局限对基于 CLIP 的弱监督视频异常检测意味着什么。检索结果不支持“CLIP 对所有异常类别完全不敏感”这一绝对说法，但多条独立证据支持一个更准确的判断：**未经异常任务适配的 CLIP 更擅长识别场景和物体语义，而对正常—异常状态、细粒度动作关系以及由文本定义指定的异常类别缺少稳定的判别边界。** 这一问题在图像异常检测中被称为 anomaly unawareness，在视频异常检测中被称为 weak abnormality awareness；最新工作进一步揭示了 definition blindness，即模型可以定位“某处有异常”，却几乎不随异常定义或目标类别改变分数。

## 1. 研究问题

- RQ1：是否存在直接证据表明 CLIP 难以区分正常与异常语义？
- RQ2：是否存在直接证据表明 CLIP 或 CLIP-based VAD 对异常类别、异常定义不敏感？
- RQ3：这种局限主要来自哪里，对当前神经元探测研究有什么影响？

## 2. 检索方法

检索覆盖四个互补方向：CLIP-based 视频异常检测、CLIP-based 图像异常检测、细粒度异常提示和动作理解、对 open-world VAD 评测的反思。纳入标准为原论文能够直接支持正常/异常语义重叠、类别可分性不足、提示粒度不足或查询定义不敏感中的至少一项。方法论文仅提出适配模块但没有分析原始 CLIP 局限的，不作为主要证据。最终核心证据包括 4 篇同行评审论文和 1 篇与 LaGoVAD 直接相关的最新预印本。

## 3. 证据分类

现有证据可以分为三层，它们描述的是同一问题由弱到强的不同表现：

1. **正常—异常边界模糊**：normal 与 anomaly prompt 在 CLIP 文本空间高度重叠，视觉特征对二者给出接近的相似度。
2. **细粒度异常类别不足**：CLIP 偏向场景、物体等粗粒度语义，对 punching、stealing、shooting 等由人—物交互、动作和上下文决定的差异不稳定。
3. **异常定义不敏感**：模型能生成通用异常分数，但在保持视频不变、只改变目标异常定义时，时间分数曲线变化很小。

## 4. 直接视频证据

### 4.1 Alert-CLIP：CLIP 存在 weak abnormality awareness

Alert-CLIP 在 UCF-Crime 上直接比较正常描述和代表性异常类别描述的文本嵌入。论文发现，原始 CLIP 中正常描述与 Abuse、Assault、Robbery、Burglary、Explosion 等异常描述存在明显语义纠缠；在视频—文本对齐中，CLIP 对 normal prompt 和正确 anomaly prompt 经常给出接近的分数，有时甚至偏向错误描述。该问题同时存在于文本空间和视频—文本空间，因此不能仅归因于某一个手工 prompt。

其控制实验进一步提供了量化证据。在 zero-shot 设置下，原始 CLIP 在 XD-Violence 上为 63.33 AUC / 34.35 AP，在 UCF-Crime 上为 61.91 AUC / 11.16 AP；用语义相反但视觉相近的 hard-negative captions 重塑边界后，分别提高到 77.54 / 55.18 和 75.75 / 17.90。这个结果不能证明所有提升都来自类别敏感性，但说明原始 CLIP 的正常—异常和细粒度语义边界存在很大的可修正空间。

### 4.2 VadCLIP：原始 CLIP 特征不能有效区分 WSVAD 类别

VadCLIP 的 t-SNE 分析明确写到：CLIP 从图文预训练获得的通用能力仍不足以有效区分 WSVAD 中的不同类别，经过面向 WSVAD 的专门优化后，视觉类别边界才变得更清晰并围绕对应文本类别分布。其案例还显示，只因画面出现枪支，模型就可能把正常片段误判为 shooting。这与后续 AnomalyCLIP、Alert-CLIP 的分析一致：CLIP 容易依赖显著物体语义，却没有稳定理解“枪是否正在被用于异常行为”这一动作与上下文条件。

VadCLIP 同时构成反例边界：CLIP 并非没有异常信息。它在加入 temporal adapter、learnable/visual prompts、MIL alignment 和专门优化后可以取得较强结果。因此准确表述应是“vanilla/frozen CLIP 的类别判别边界不足”，而不是“CLIP 无法用于异常检测”。

### 4.3 Definition Blindness：直接命中 LaGoVAD

2026 年 7 月的预印本《Rethinking Open-World Video Anomaly Detection: Diagnosing Definition Blindness》提出了更严格的干预式检验：固定同一个视频，只改变用户提供的异常定义，观察时间分数是否随目标类别改变。论文图 1 直接展示 LaGoVAD：视频同时包含 shooting 和 explosion 时，LaGoVAD 能定位异常区间，但在不同目标定义下输出的分数曲线几乎不变。

论文把这一现象称为 definition blindness。作者指出，现有 Drift@5 和常规 AUC/AP 被 target-vs-normal 分离主导；在常见数据集上，target-vs-normal 的权重是 target-vs-other-anomaly 的 7.2–26.8 倍。因此模型即使只学习一个与查询无关的“通用异常分数”，也能取得看起来不错的指标。作者在 UCF-Crime、XD-Violence 和 MSAD 上测试 VadCLIP、LaGoVAD、PLOVAD 及通用 VLM，发现多个强模型存在接近零的 definition-response margin。去除不同定义共享的异常证据后，最强基线在 DC-Disc 上提高 7.3–16.0 AUROC，在 DC-DetΔ 上提高 15.5–28.3 AUROC。

这是目前与本项目最直接的证据，因为它不只是讨论 CLIP 的一般局限，而是直接表明 LaGoVAD 可能主要在做“异常—正常分离”，没有充分实现“按哪一种异常定义判断”。但该论文目前仍是预印本，结论需要独立复现后才能作为强事实写入正式论文。

### 4.4 ASK-HINT：抽象类别名遗漏了真正的判别动作

ASK-HINT 发现，冻结 VLM 使用抽象问题“视频里是否发生异常”时，容易忽略定义复杂异常所需的人—物交互和动作语义；换成 punching、kicking、attacking、wrestling 等细粒度动作问题后，同一视频可以由错误的 normal 判断转为正确的 anomaly 判断。它进一步把 UCF-Crime 类别组织为 violence、property crimes、public safety 等组，并用细粒度动作提示提高 UCF-Crime 和 XD-Violence 的结果。

ASK-HINT 研究的是更广义冻结 VLM，而非只分析 CLIP 的 512 维空间，因此它不是 CLIP 类别不敏感的单独证明；但它与 Alert-CLIP、VadCLIP 形成机制上的独立汇合：异常类别名本身太粗，真正可区分的证据往往是动作、角色、物体状态和上下文关系。

## 5. 图像异常检测的相邻证据

AnomalyCLIP 指出，原始 VLM 的 zero-shot anomaly detection 较弱，是因为模型更关注前景物体的类别语义，而不是图像中的 normality/abnormality。AA-CLIP 随后把问题明确命名为 anomaly unawareness：原始 CLIP 的 normal 与 abnormal 文本嵌入缺少清晰边界，甚至在有明显缺陷的 carpet 和 zipper 图像上，对 normal prompt 的相似度高于正确异常描述。

两者与视频证据的共同点不是“局部缺陷等于视频犯罪”，而是 CLIP 预训练目标的偏置一致：它学习“图里是什么”，却没有被明确训练去分离“同一个对象/场景处于正常状态还是异常状态”。视频还额外叠加了动作和时间关系，使这一问题更严重。

## 6. 综合判断

| 局限层次 | 直接证据 | 对 VAD 的含义 |
|---|---|---|
| 正常/异常语义纠缠 | AA-CLIP、Alert-CLIP | normal 与 anomaly prompt 不是天然良好的二分类锚点 |
| 类别可分性不足 | VadCLIP、Alert-CLIP | scene/object cues 可能压过动作与上下文差异 |
| 提示粒度不足 | ASK-HINT | 类别名称应拆成可观察的动作—角色—物体证据 |
| 定义不敏感 | Definition Blindness | LaGoVAD 等方法可能输出查询无关的通用异常曲线 |
| 可适配而非不可用 | VadCLIP、Alert-CLIP | 问题可通过针对性对齐和难负样本训练修正 |

因此，对 RQ1 的回答是肯定的：同行评审图像和视频论文均发现原始 CLIP 的正常/异常语义边界模糊。对 RQ2，证据也偏向肯定，但必须限定为 vanilla/frozen CLIP 和现有 CLIP-based VAD：它们经常能感知通用异常，却对异常类别和查询定义缺乏足够响应；对 LaGoVAD 的最直接证据目前来自尚未同行评审的预印本。对 RQ3，问题并非单纯缺少 patch-level 空间细节，而是文本语义、视觉动作证据和评测目标三者共同造成的类别盲区。

## 7. 对当前研究方向的启示

继续使用 baseline anomaly score 选择神经元，会优先找到“所有异常共同激活”的通用异常神经元，而不一定找到能够区分 Assault、Robbery、Explosion 等类别的功能单元。这正好会复制 definition blindness。

更有依据的方向是把探测目标从 anomaly responsiveness 改为 **definition sensitivity**：固定视频特征，分别输入匹配类别、容易混淆的其他异常类别和 normal 定义，定位对 matched-vs-confounder margin 有主要贡献的神经元。训练时不只拉开 normal 与 anomaly，还应加入 anomaly-vs-anomaly hard negatives；评测时除 AUC/AP 外，增加“换定义后分数是否改变”的干预式指标。该方向不需要保存 patch token，现有 CLS hidden states 和 CLIP 文本特征即可进行第一轮验证。

## 8. 局限

Alert-CLIP 的 VAD 分析是当前最直接的同行评审证据，但其提升同时来自数据、对齐损失与训练策略，不能把全部数值增益归因于单一的“类别不敏感”。Definition Blindness 对 LaGoVAD 的诊断非常直接，但目前只是 2026 年 7 月预印本。ASK-HINT 使用的冻结 VLM 范围比 CLIP 更广。现有证据足以支持开展本项目自己的干预实验，但正式论文中最好将“CLIP 类别不敏感”写成需要由本项目实验进一步验证的动机，而不是未经限定的既定事实。

## 参考文献

[1] Y. Zhu et al., "Alert-CLIP: Abnormality-aware Latent-Enhanced Representation Tuning of CLIP for Video Anomaly Detection," CVPR, 2026.

[2] P. Wu et al., "VadCLIP: Adapting Vision-Language Models for Weakly Supervised Video Anomaly Detection," AAAI, 2024.

[3] I. Song and J. Lee, "Rethinking Open-World Video Anomaly Detection: Diagnosing Definition Blindness," arXiv:2607.20780, 2026.

[4] S. Zou et al., "Unlocking Vision-Language Models for Video Anomaly Detection via Fine-Grained Prompting," WACV, 2026.

[5] Q. Zhou et al., "AnomalyCLIP: Object-agnostic Prompt Learning for Zero-shot Anomaly Detection," ICLR, 2024.

[6] W. Ma et al., "AA-CLIP: Enhancing Zero-shot Anomaly Detection via Anomaly-Aware CLIP," CVPR, 2025.
