# CLIP 在 UCF-Crime 与 XD-Violence 上的类别判断：从类别标签到动作原语

## 摘要

本文调研 CLIP 及相邻 VLM 工作是否研究过 UCF-Crime、XD-Violence 中不同异常类别的判断依据，尤其是类别与运动动作之间的对应关系。检索结果形成三条文献分支：第一类在 CLIP 输出空间学习类别方向并报告细粒度识别；第二类用事件描述、动作或多维属性替代单个类别名；第三类通过光流、合成 event stream 或多模态训练检验不同异常类别对运动信息的需求。证据共同说明，单一类别名不足以描述异常，多个类别共享动作原语，同一动作是否异常又受对象与场景制约。现有工作尚未把这些类别—动作关系定位到 CLIP ViT 各层 CLS hidden-state 的神经元组，也没有完成神经元干预验证。因此，本项目不应寻找“一类对应一个神经元”，而应建立可跨数据集共享的动作原语库，并探测负责这些原语的层级神经元组。

## 1. 研究问题

- RQ1：哪些研究真正评估了 CLIP 在 UCF-Crime、XD-Violence 上的类别级异常判断？
- RQ2：哪些研究把类别拆成动作、对象、场景或运动属性，并提供了什么具体对应关系？
- RQ3：这些成果能否直接支持 CLIP hidden-state 神经元探测，尚存什么空白？

## 2. 调研方法

检索覆盖 2023-2026 年的 CLIP 弱监督 VAD、开放词汇 VAD、动作/属性提示、解释型 VLM VAD 以及 RGB-motion/event 融合。只纳入能够从论文正文、补充材料或正式项目页核实的方法和结果。将证据分为三类：CLIP 类别空间、外加的动作语义先验、独立运动模态。类别级 AUC 只能证明类别表现差异，LLM 生成的动作表只能作为先验，二者均不能被表述为 CLIP 神经元已经学会了相应动作。

## 3. 文献分类

| 分支 | 代表工作 | 实际研究对象 | 能否解释 CLIP 内部神经元 |
|---|---|---|---|
| 类别方向与细粒度识别 | VadCLIP、AnomalyCLIP、OVVAD、MoME | 最终 CLIP 特征空间中的类别方向、类别专家或文本相似度 | 否，只解释输出空间 |
| 动作与属性提示 | LAP、A2VAD、ASK-HINT、VADTree | 类别对应的动作、对象、场景、外观属性 | 否，主要是人工或 LLM 生成先验 |
| 独立运动证据 | π-VAD、Cross-Modal Event Encoder、CMRL | 光流、event stream、上下文—运动关系 | 能验证运动是否有增量价值，但没有定位 CLS 神经元 |

## 4. CLIP 类别空间研究

VadCLIP 使用类别名和可学习 prompt 将片段视觉特征与正常及异常类别对齐，证明冻结 CLIP 可以同时做粗粒度检测与细粒度识别 [1]。AnomalyCLIP 更直接地研究 CLIP latent space：先用正常原型重新中心化视觉空间，再为每个异常类别学习 text-driven direction，片段在某一方向上的投影用于估计类别概率 [2]。两者均使用最终 CLIP 表征，而没有检查 ViT 中间层或单个神经元。

OVVAD 把任务分成 class-agnostic detection 和 class-specific categorization，并用场景名词、动作动词和合成 novel anomalies 改善未见类别识别 [3]。MoME 则进一步指出，纯类别级多样性会忽略跨类别共享模式；其 LLM anomaly prototypes 同时路由专用专家与通用专家。将细粒度 prototype 换成粗类别名后，UCF-Crime 和 XD-Violence 分别下降 1.9 与 8.5 个点 [4]。这组结果支持“动作/属性原语比类别名更有效”，但专家路由依然不是 CLIP hidden-state 神经元定位。

类别级分析还揭示了类别难度并不均匀。MoME 相对 VadCLIP 在 UCF-Crime 的 Assault 上由 31.7 提升到 85.5，在 Robbery 上由 81.9 提升到 92.0；在 XD-Violence 的 Explosion 上由 60.0 提升到 66.1，Car Accident 由 38.6 提升到 48.4，而 Riot 接近饱和 [4]。这说明类别标签相同不意味着视觉证据同质，通用异常模式与类别特有模式需要共同建模。

## 5. 从类别名到动作、对象和场景

LAP 是最直接的 CLIP 证据之一。它使用 CLIP ViT-L/14 视觉特征，把单个标签改写成原子事件句子，例如 shooting 被描述为“持枪瞄准/开枪”，Abuse、Assault、Fighting 被分解为踢、打、推、追逐并攻击，Road Accident 被分解为车辆碰撞、撞击行人或翻车 [5]。LAP 的类别级分析显示，它在 Assault、Explosion、RoadAccidents、Robbery 上相对 RTFM 改善明显，但在 Shoplifting、Fighting 等动作细微或难以用单句描述的类别上仍较弱。完整 event prompt 使 XD-Violence AP 相对 VadCLIP 提升约 2 点，说明类别动作化有用，但文本仍可能无法解决细微运动。

A2VAD 把 prompt 扩展到 appearance、motion、context 三类属性，并通过前景—背景对比对齐减少模型依赖场景背景 [6]。ASK-HINT 虽然主要使用冻结大 VLM 而非纯 CLIP，却提供了重要的相邻证据：action-centric prompt 相比抽象问题可将部分 UCF-Crime 类别 AUC 提高最多约 30%，并发现 Arson—Explosion、Robbery—Stealing—Shoplifting 等类别共享语义簇 [7]。其跨类别实验表明，physical confrontation 等动作原语可以迁移到未显式提供 prompt 的类别。

VADTree 给出了最完整的 UCF/XD 多维先验表，将每类异常拆为 scene、character/object、action/behavior [8]。这些表由 LLM 生成并作为 VLM 推理先验，不是从 CLIP 内部实证发现；但它们可作为构建动作原语库的文献依据。

### 5.1 UCF-Crime 的动作/行为原语

| 类别 | 文献中的动作/行为线索 | 更适合的原语组 |
|---|---|---|
| Abuse | 推搡、拖拽、反复击打、压制 | physical contact；restraint |
| Arrest | 强制控制、搜身、押送、按倒 | restraint；escort |
| Arson | 投掷燃烧物、快速逃离、回看火势 | ignition；throw；flee |
| Assault | 突然扑击、挥舞武器、防御姿态 | lunge；weapon swing；defense |
| Burglary | 窥视、撬锁、翻找物品 | forced entry；rummage |
| Explosion | 闪光/火焰突然出现、人群下蹲或逃散 | abrupt expansion；crowd scatter |
| Fighting | 拳打、脚踢、拉扯、多人纠缠 | repeated strike；mutual contact |
| Road Accidents | 急刹、碰撞、翻车、撞击行人 | deceleration；impact；trajectory break |
| Robbery | 抢夺、威胁、逃跑、车辆突然停驶或加速 | snatch；threat；flee |
| Shooting | 瞄准、连续射击、寻找掩体 | aim；fire；take cover |
| Shoplifting | 藏匿物品、反复张望、快速离开货架 | conceal；look-around；leave |
| Stealing | 手伸入口袋、转移赃物 | reach；take；handover |
| Vandalism | 砸击、喷涂、踢坏设施 | damage；spray；kick |

### 5.2 XD-Violence 的动作/行为原语

| 类别 | 文献中的动作/行为线索 | 更适合的原语组 |
|---|---|---|
| Abuse | 突发攻击、抓扯、受害者退缩或逃跑、持续身体接触 | aggressive contact；recoil；flee |
| Explosion | 光/烟快速扩张、人群散开、烟火持续 | expansion；crowd scatter |
| Fighting | 反复拳打脚踢、高强度动作、旁观者反应 | repeated strike；high intensity |
| Car Accident | 快速减速/撞击、碰撞后的聚集或救援 | deceleration；impact；post-event gathering |
| Shooting | 人群下蹲/奔跑、受害者倒地、事后执法活动 | panic motion；fall；post-event response |
| Riot | 投掷物体、群体暴力、波浪式混乱散开 | throw；group aggression；chaotic dispersion |

这两个表不是一一对应关系。相同原语会服务多个类别，例如 repeated strike 同时出现在 Abuse、Assault、Fighting；flee 同时出现在 Arson、Robbery、Shooting；take/conceal 同时支撑 Burglary、Robbery、Shoplifting、Stealing。因此，“一个类别对应一组独占神经元”的假设不符合现有证据。

## 6. 哪些类别确实受益于运动信息

π-VAD 的五模态训练表明，运动、姿态、深度、分割和文本经过跨模态诱导后可以改善 RGB 学生 [9]。其 UCF-Crime 类别级结果中，Explosion 从 0.47 提升到 0.78、Fighting 从 0.79 提升到 0.91、RoadAccidents 从 0.68 提升到 0.81、Shooting 从 0.77 提升到 0.86；并非所有类别都改善，Abuse 和 Assault 略有下降。这说明外部动态信息的作用具有类别条件性。

Cross-Modal Event Encoder 的补充实验把普通视频通过 frame differencing、thresholding、stacking 转换成 synthetic event stream，再接入 VadCLIP [10]。在 UCF-Crime 上，event 输入相对 image 输入使 Assault 从 56.44 提升到 72.03、Fighting 从 58.14 提升到 79.27、Shoplifting 从 64.27 提升到 73.29；但 image 在总体上仍更强。与 π-VAD 相比，这项结果更直接说明 motion-centric signal 对局部、短暂动作有互补价值，而不是证明所有异常都应改用运动流。

CMRL 提醒了另一限制：相同 running、striking 等动作是否异常取决于场景和对象关系，因此应建模 context-motion relation，而不是把运动强度直接等同于异常 [11]。综合三项研究，异常类别至少可分为 motion-dominant、appearance/change-dominant 和 context/intent-dominant；实际类别通常同时占据两到三类。

## 7. 交叉综合：对神经元探测的直接启发

现有研究支持三个结论。

第一，CLIP 最终特征中确实存在可学习的类别方向，但现有工作尚未证明这些方向由哪些中间层 CLS 神经元实现。AnomalyCLIP 的 latent direction、VadCLIP 的文本相似度、MoME 的专家均位于 CLIP 输出之后。

第二，类别不是合适的最小解释单元。LAP、ASK-HINT、VADTree 和 MoME 均指向共享动作/属性原语；类别名过粗，动作原语又必须与对象和场景联合解释。

第三，运动表示对 Assault、Fighting、Shoplifting、RoadAccidents 等类别有直接增量价值，但对 Explosion、Arson 等类别，烟火、碎片和场景改变同样关键；对 Arrest、Stealing、Shoplifting，单纯运动更无法表达意图。

因此，本项目更合适的探测目标是“跨类别动作—属性神经元组”，而非 13/6 个类别神经元。建议建立约 8-12 个共享原语：strike/contact、restraint、reach/take/conceal、forced entry、aim/fire、ignition/smoke、impact/deceleration、damage、crowd scatter、flee/chase。每个原语同时拥有动作文本、对象文本和场景文本三个条件。

## 8. 可验证的研究方案

1. 用 LAP/VADTree 的类别描述构建统一原语词典，UCF 与 XD 共享同义原语，不使用 baseline 分数。
2. 为每个原语构造多条 action-centric prompt，并以 CLIP text embedding 定义输出空间方向。
3. 对每层 CLS hidden states 训练严格受限的线性 probe，比较类别标签 probe 与动作原语 probe；采用跨视频划分，防止场景泄漏。
4. 引入 synthetic event 或稀疏光流教师，检验某神经元组是否同时预测文本原语与独立运动信号。
5. 对候选神经元组进行置零、替换和放大干预，报告原语分数、类别级 AUC/AP 和时间边界变化。只有干预产生可重复的特异性影响，才能称为功能神经元组。
6. 在 DSANet、DeSC、LaGoVAD 中统一接入这些神经元组；模型特有模块只作为下游消费者，不参与神经元定义。

## 9. 开放问题

- 据检索结果，尚无研究建立 UCF/XD 类别—动作原语—CLIP 层级 CLS 神经元的完整映射。
- 现有动作表多由人工或 LLM 生成，缺乏真实片段级动作标注验证。
- XD-Violence 是多标签数据，单一类别级评价会混合共同出现的动作；需要多原语预测而非 softmax 单类别预测。
- 大量类别依赖意图和上下文，仅靠光流或运动强度无法区分 Shoplifting 与正常拿取、Arrest 与普通接触。
- 类别级 AUC 的提高不能自动证明模型使用了正确动作；必须增加神经元干预和背景反事实实验。

## 10. 结论

RQ1：已有多项工作研究 CLIP 在 UCF/XD 上的细粒度类别判断，代表性路线包括类别方向、开放词汇分类和类别专家，但它们集中在 CLIP 输出空间。

RQ2：已有工作系统地把异常类别拆为动作、对象、场景和运动属性，并有类别级实验支持 action-centric prompt 与独立运动表示的价值。最稳定的规律不是类别独占动作，而是多个类别共享 physical contact、conceal/take、fire/smoke、impact 等原语。

RQ3：这些研究能够为本项目提供原语词典和验证指标，却不能直接替代 hidden-state 神经元探测。尚未被研究的关键问题正是：CLIP 各层哪些神经元组承载这些原语、它们如何与场景条件结合、干预这些神经元能否稳定改变三个 baseline 的时间定位与类别表现。

## 参考文献

[1] Wu, Zhou, Pang, et al., "VadCLIP: Adapting Vision-Language Models for Weakly Supervised Video Anomaly Detection," AAAI, 2024.

[2] Zanella, Liberatori, Menapace, et al., "Delving into CLIP Latent Space for Video Anomaly Recognition," Computer Vision and Image Understanding, 2024.

[3] Wu, Zhou, Pang, et al., "Open-Vocabulary Video Anomaly Detection," CVPR, 2024.

[4] Sun, Chen, Wu, et al., "Joint Learning of General and Diverse Patterns with Mixture of Memory Experts for Weakly-Supervised Video Anomaly Detection," CVPR, 2026.

[5] Tao, Peng, Wang, et al., "Learning Suspected Anomalies from Event Prompts for Video Anomaly Detection," ACM TOMM, 2026.

[6] Xu, Wang, Xu, et al., "A2VAD: Attribute-Augmented Prompt Learning for Weakly Supervised Video Anomaly Detection," Pattern Recognition, 2026.

[7] Zou, Tian, Wesemann, et al., "Unlocking Vision-Language Models for Video Anomaly Detection via Fine-Grained Prompting," WACV, 2026.

[8] Li, Xu, Rao, et al., "VADTree: Explainable Training-Free Video Anomaly Detection via Hierarchical Granularity-Aware Tree," NeurIPS, 2025.

[9] Majhi, D'Amicantonio, Dantcheva, et al., "Just Dance with pi! A Poly-modal Inductor for Weakly-supervised Video Anomaly Detection," CVPR, 2025.

[10] Jeong, Chen, Yun, et al., "Cross-Modal Event Encoder: Bridging Image-Text Knowledge to Event Streams," WACV, 2026.

[11] Cho, Kim, Hwang, et al., "Look Around for Anomalies: Weakly-Supervised Anomaly Detection via Context-Motion Relational Learning," CVPR, 2023.
