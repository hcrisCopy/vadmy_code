# 少数完整 CLIP 层、文本异常适配与 KNN 合成：面向帧级定位的证据调研

## 摘要

本报告围绕一个受限目标展开：不提取光流、patch token、姿态、描述模型或其他新视频模态，只复用现有 CLIP ViT-B/16 的 12 层 CLS hidden states 和 512 维特征，寻找能够同时适配 DSANet、DeSC 与 LaGoVAD 的帧级定位增强方法。现有证据支持三个相互依赖的判断。第一，LaGoVAD 的 KNN 并非把邻居向量拼到帧特征，而是检索语义相近的正常视频并在时间轴上合成长视频；其增益来自可控的异常占比、已知的合成边界以及场景相近的困难正常背景。第二，CLIP 对异常语义的不敏感不能只靠改写一句类别名解决：VadCLIP、TPWNG、DSANet、LaGoVAD、AA-CLIP 和 Alert-CLIP 共同指向文本适配、正常/异常对照、困难负例和细粒度描述的组合。第三，完整中间层比散乱单维度更符合本项目已观察到的分布式回路；但中间层不能直接与 CLIP 文本特征比较，必须使用轻量层投影器进入统一文本空间。由此，本报告建议优先验证“完整层选择—对照式文本适配—正常 KNN 时间拼接”三段式方案，而不是继续让单神经元或 baseline 分数充当片段教师。

## 1. 研究问题

- RQ1：LaGoVAD 的 KNN 动态视频合成为什么能提高时间定位，它能否直接迁移到 UCF-Crime 和 XD-Violence？
- RQ2：DSANet、DeSC、LaGoVAD及相关工作如何处理 CLIP 对异常类别和动作语义不敏感的问题，哪些技巧有消融依据？
- RQ3：怎样用少数完整 CLS 层代替分散神经元，并只解冻文本—视觉交互模块和打分头，形成低开销、跨 baseline 的统一方法？

## 2. 调研方法

证据包括三篇本地 baseline 原文、发布代码，以及 2023—2026 年正式发表或可核验预印本。检索分为四条路径：VAD 中的文本 prompt 与 normality guidance；CLIP 的异常语义缺陷及文本适配；视频任务中的多层 CLIP 表征；KNN/事件合成与弱监督时间边界。只把论文正文、补充材料或正式项目页能够核实的方法和消融数字用于结论。图像异常检测论文只用于解释 CLIP 文本空间缺陷，不把其像素级结果直接外推到视频。

## 3. LaGoVAD 的 KNN 到底做了什么

LaGoVAD 的 Dynamic Video Synthesis 首先决定合成长视频包含多少段，再选择一个 anchor 视频。对于异常样本，anchor 是异常视频；其他位置放入正常视频。正常视频不是任意抽取，而是从预先计算的 KNN 缓存中检索与 anchor 视觉语义相近的正常视频。最终在时间维拼接这些片段，并根据异常 anchor 的插入位置生成稠密二值伪标签。KNN 距离离线预计算，因此训练时主要增加特征加载和拼接成本，推理时不需要检索。

这不是两种容易混淆的操作：

- 不是把 K 个正常邻居的向量拼接到每个 snippet 的通道维；
- 不是用 KNN 直接判断异常；
- 而是把相似正常序列拼到异常序列前后，改变训练视频的异常持续时间和背景组成。

LaGoVAD 补充材料的消融提供了直接依据。固定最大 5 段、合成概率 0.7 时，使用 KNN 的 UCF-Crime AUC / XD-Violence AUC 为 81.12 / 74.25；去掉 KNN 后为 79.98 / 68.95，分别下降 1.14 和 5.30 点。只保留 1 段、不做多段合成时为 79.18 / 71.41，说明“多段时间合成”和“KNN 场景匹配”都有贡献。模块级消融中，去掉整个动态合成后，七数据集平均检测指标从 76.42 降到 73.51。

但是，这些数字来自 LaGoVAD 在 PreVAD 上预训练后进行的零样本评测，不能直接当作 UCF/XD 弱监督微调的预期增益。PreVAD 的异常源视频经过面向异常事件的数据收集和描述标注，而 UCF/XD 的异常训练视频是未裁剪长视频，内部包含大量正常片段。若把整段 UCF/XD 异常视频标成 1 再与正常视频拼接，只是把视频级标签伪装成帧级标签，会制造系统性噪声。

因此，直接可迁移的是三个机制，而不是原始标签生成方式：

1. 从全正常训练视频建立场景匹配的 KNN 正常库；
2. 在时间轴构造正常—候选异常—正常序列，随机化异常相对持续时间；
3. 对检索正常段使用可信的 dense negative，对候选异常段只使用置信度加权软标签。

候选异常段必须由独立于 baseline 分数的语义模型产生，并通过跨视频留出验证；否则 KNN 只能保证正常部分可靠，不能凭空获得真实异常边界。

## 4. 三个 baseline 如何处理 CLIP 文本不敏感

三个 baseline 都使用 CLIP 视觉特征，但对文本问题的处理程度不同。它们的共同结论不是“类别名已经足够”，而是必须让文本空间、视觉空间或二者交互适应异常检测。

| 方法 | 文本/交互技巧 | 可训练部分 | 有依据的作用 | 局限 |
|---|---|---|---|---|
| DSANet | 浅层及输出端 Lightweight Text Adapter；事件/背景原型分别与异常类和 normal 对齐 | 文本 Adapter、时序模块、分类头 | 文本无适配 81.57 AP，手工 prompt 81.05，学习 prompt 82.88，Adapter 86.95；说明内部文本适配明显强于改一句模板 | 事件/背景划分仍由当前检测分数引导 |
| DeSC | 继承 VadCLIP 式 learnable text prompt 与视觉引导文本特征；两个独立时间流各自学习 | prompt、两个时间流、融合与头 | 主要增益来自 sensitivity/consistency 解耦：统一训练 86.18/80.22，协同推理 89.37/87.18 | 论文没有证明新的文本描述是主要增益来源 |
| LaGoVAD | soft prompt；训练时随机使用类别名或异常描述；跨模态 fusion；in-sample hard-negative mining | soft prompt、fusion、检测/分类头 | 类别名 80.44 UCF AUC，人工类别描述 81.12，视频特定描述 83.03；困难负例也有独立增益 | 视频特定描述在标准 UCF/XD 测试时不可用 |

DSANet 的结果最直接地否定了“只换模板即可解决语义不敏感”：手工模板没有超过无适配文本，而 Lightweight Text Adapter 相对无适配提升 5.38 AP。VadCLIP 给出一致但较早的证据：在 XD 上，hand-crafted prompt 为 81.06 AP，learnable prompt 为 84.51；平均帧视觉 prompt 为 81.34，而 anomaly-focus visual prompt 为 84.51。两项结果共同说明，可学习的任务内校准和异常片段条件化比固定句式更有效。

DeSC 的定位增益主要来自另一条轴：短时敏感流捕捉突发变化，语义一致流用 Gaussian mixture prior 抑制噪声，滑窗协同推理再平滑边界。它证明时间优化分工很重要，却没有提供“更好的类别描述”带来大幅增益的消融。因此，在 DeSC 上新增文本方法时，不能把其原有 2—3 点增益归因于文本。

## 5. 其他文本技巧给出的交叉证据

### 5.1 正常—异常对照比单独异常描述更可靠

TPWNG 用事件描述与视频帧对齐生成伪标签，但不是直接使用原始 CLIP：它设计正常/异常 ranking loss、distributional inconsistency loss、learnable text prompt 和 normality visual prompt，并只微调文本端最终投影等少量部分。论文明确报告，原始 CLIP 存在 VAD domain bias，单项损失均能改善结果，其中异常 ranking loss贡献最大。

PE-MIL 虽然使用 I3D 而非 CLIP 视觉特征，却提供了与帧级边界直接相关的独立证据：abnormal-aware prompt 用类别语义覆盖多样异常模式，normal context prompt 专门放大异常与上下文的差异，使边界更清楚。它在 UCF 上的提升较小，而在 XD 上达到 88.05 AP，提示 prompt 对类别复杂、场景多样的数据更可能有收益，但并非跨数据集等幅有效。

图像异常检测中的 AnomalyCLIP、AA-CLIP 与视频方法 Alert-CLIP 从另一侧解释了原因：原始 CLIP 主要编码前景对象/类别语义，normal 与 abnormal 文本特征常高度纠缠。Alert-CLIP在视频场景中进一步采用 video-label、region-text 和 region-semantic 多级对齐，并使用多个困难负样本拉开正常/异常语义。这些工作支持“重塑文本边界”，但其 patch/region 路线不在本项目开销范围内；可借用的是 text adapter、normal/abnormal pair 和 hard negative，不是 patch 分支。

### 5.2 描述必须是可观察事件，而不是抽象类别名

LaGoVAD 的人工描述示例是“Explosion, often resulting in fire, smoke, and scattered debris”，它把抽象类名展开为可见属性；视频特定描述进一步加入对象、地点和具体行为，因此比类别名高 2.59 AUC。LAP 用原子事件句子而非类别词典，ASK-HINT 用 action-centric guiding questions；两者都说明动作、对象和结果的组合比“abnormal”或单个类名更可判别。

但描述越多并不必然越好。ASK-HINT 的问题数量消融中，6 个精选问题优于 3、9 和 12 个问题；随机问题明显更差。这个结果与 LaGoVAD 的 hard-negative mining 一致：关键是少量有区分力的语义方向，而不是堆积大量同义句。

适合 UCF/XD 的文本单元应是对照式原子定义，例如：

- Fighting：`people repeatedly punching or kicking each other` 对 `people hugging, playing, or standing close without aggressive contact`；
- Shoplifting：`a person conceals an item and leaves without paying` 对 `a customer picks up an item and pays normally`；
- Explosion：`flames, smoke, debris, and abrupt scene disruption` 对 `ordinary fire, lighting change, or smoke without an explosive event`；
- Road Accident：`vehicles collide, overturn, or abruptly break their trajectories` 对 `vehicles brake, turn, or stop normally`。

这里的 normal counterpart 非常重要，因为单独的 `fire`、`running`、`holding an object` 都可能出现在正常视频中。

### 5.3 最有依据的轻量文本配置

综合证据，推荐的文本端不是固定手工 prompt，也不是完全解冻文本编码器，而是：

1. 每类 3—6 条人工可观察原子描述；
2. 每条异常描述配一个容易混淆的正常/近邻反例；
3. 冻结 CLIP text backbone，仅训练 shared soft prompt 或低秩 text adapter；
4. 训练时加入类别对齐、normal/abnormal margin 和 in-batch hard-negative loss；
5. 所有 prompt 在两个数据集共享动作原语，类别仅决定原语组合。

这同时吸收了 VadCLIP 的 learnable prompt、DSANet 的 text adapter、TPWNG 的 normality guidance、LaGoVAD 的 description/hard negative，而不增加新的视频特征提取。

## 6. 为什么改用少数完整 hidden-state 层

本项目 DANCE 已经给出内部证据：在 12 层中，第 8 层和第 12 层的类别回路质量最高，但每类每层都触及 128 维搜索上限，跨类并集达到 1254 维。这说明异常语义不是由少数独立维度承担，而是分布在层内子空间。继续挑散乱维度既丢失协同结构，也造成不同类别神经元大量重叠。

相邻视频研究支持使用多层完整表征。STAN 从 CLIP 多层引出轻量时空分支，同时迁移低层和高层知识；视频 action recognition 的研究也显示后几层更抽象、适合动作语义，但不同深度仍提供互补信息。图像异常检测中的 GenCLIP 等工作使用 multi-layer prompting，也支持不同层包含不同粒度线索。它们不能证明第 8/12 层一定适合 UCF/XD，却支持“先选层，再融合完整层”的方法形态。

层选择仍可保留神经元级可解释性，但神经元只用于解释和选层，不再作为输入裁剪：

1. 在每层计算跨 fold、跨 prompt 的文本责任分数；
2. 统计稳定责任神经元数量、责任总质量和类别覆盖度；
3. 用参与率或 Top-K 证据覆盖率衡量证据是否分布式；
4. 选择责任质量高、跨 fold 稳定且与其他已选层互补的 2 层；
5. 下游使用这两层完整的 768 维 CLS，而不是只取被排名的维度。

这回答了“利用 hidden states 的可解释性”与“不要分散神经元”的冲突：解释对象仍是神经元责任分布，模型输入单位改为完整功能层。

中间层不能直接乘 CLIP 的 768×512 最终投影，因为后续 Transformer 与 patch token 的交互尚未发生。因此每个被选层需要一个轻量投影器：

\[
z_t^{(l)}=\operatorname{Norm}\big(P_l(\operatorname{LN}_l(h_t^{(l)}))\big),
\quad P_l:\mathbb{R}^{768}\rightarrow\mathbb{R}^{512}.
\]

`P_l` 建议采用 LayerNorm 加低秩线性 Adapter，并用文本对齐损失训练。不能把中间层直接套用最后层 `ln_post @ visual.proj` 后声称已进入准确的 CLIP 输出空间。

## 7. 推荐方案：Layer-Selected Contrastive KNN Synthesis

### 7.1 阶段 A：完整层选择

对 12 层使用完全相同的跨视频分折程序。维度级文本责任只用于形成每层的稳定责任密度、类别覆盖和 prompt 鲁棒性。选择 2 层完整 CLS；已有 UCF 结果可把第 8、12 层作为候选，但正式方法不得硬编码这两个数字。UCF 与 XD 可以得到不同层，算法、阈值搜索空间和评价协议必须相同；同一数据集的三个 baseline 复用同一层选择产物。

### 7.2 阶段 B：对照式文本空间

建立跨数据集共享的动作—对象—结果 prompt bank。冻结 CLIP text backbone，训练 shared soft prompt / low-rank text adapter 和每层 projector。每个 snippet 得到多层文本 margin：

\[
m_{t,c}=\sum_{l\in\mathcal{L}}\alpha_l
\left[\max_{p\in P_c^+}\cos(z_t^{(l)},p)
-\max_{q\in P_c^-}\cos(z_t^{(l)},q)\right].
\]

其中正 prompt 是异常定义，负 prompt 是相似但正常的反事实定义。`alpha_l` 是稀疏层门控，只在已选的两层间学习。训练只使用视频级标签、正常视频 dense negative、MIL 类别损失和 hard-negative contrastive loss，不读取 baseline 异常分数。

### 7.3 阶段 C：KNN 正常拼接与软边界监督

用完整选层特征对所有纯正常训练视频建立离线 KNN 索引。对于异常训练视频：

1. 用跨 fold 冻结的文本 margin 找到高置信、时间连续的候选事件区域，而不是单个 top-k 点；
2. 检索与候选视频场景最接近的正常片段；
3. 随机生成 `normal prefix + candidate event + normal suffix`；
4. 检索正常区域标记为确定的 0；候选事件只使用置信度加权的 soft positive；
5. 随机改变候选事件相对长度和位置，训练边界与持续时间鲁棒性。

为避免自我确认，生成候选事件的文本模型必须来自不包含该视频的训练 fold；低置信视频不参与 dense positive，只保留原始 MIL。KNN 使用完整正常视频标签，因此 negative 边界比 positive 边界更可靠，损失权重应体现这种不对称。

### 7.4 阶段 D：接入三个 baseline

统一插件把原始 512 维 CLIP 特征与两个完整层投影做零初始化门控融合：

\[
f'_t=f_t+\gamma\sum_{l\in\mathcal{L}}\alpha_l A_l(h_t^{(l)}),
\quad \gamma(0)=0.
\]

然后只解冻功能等价的两类模块：

- 文本—视觉交互：DSANet 的 text adapter / visual-guided text interaction，DeSC 的 prompt 与视觉引导文本交互，LaGoVAD 的 soft prompt 与 fusion；
- 打分头：各 baseline 的 binary / alignment / classification head。

CLIP 图像和文本 backbone 保持冻结。第一阶段也冻结大部分时序 backbone；只有合成边界任务在独立留出集有效后，才考虑解冻最后一个时序块。这样既尊重三个 baseline 的结构差异，又用“功能范围”而非完全相同的参数名定义通用解冻策略。

训练损失由作者原损失、合成 dense boundary loss、对照文本 margin、normal false-positive penalty 和初始权重 anchor 构成。测试时不使用 KNN，不运行额外模型，只加载已缓存的两个完整 CLS 层并执行轻量投影。

## 8. 为什么该方案比之前的方法更有依据

之前的神经元方案失败在两个环节：类别相关 CLS 回路被当成了时间定位教师；baseline 或弱文本分数的局部峰又被扩展成伪事件，导致错误被放大。新方案把职责重新分开：神经元责任只负责选层和解释；文本对照负责形成异常语义方向；KNN 正常拼接负责提供场景匹配的 dense negative 和可控时间边界；baseline 交互模块和打分头负责最终适配。

这种组合分别有独立证据：LaGoVAD 支持 KNN 时间合成和细粒度描述，VadCLIP/DSANet 支持 learnable prompt / text adapter，TPWNG/PE-MIL 支持 normality-guided prompt 和边界监督，STAN 支持多层完整 CLIP 表征。不存在一篇论文直接证明四者组合一定超过强 DSANet 2—3 点，因此这仍是需要验证的研究假设，而不是已有结果的机械叠加。

## 9. 必须先做的止损实验

1. **Prompt audit**：在训练集跨视频分折上比较类别名、人工原子描述、learnable prompt、对照 prompt；指标是 held-out 视频类别 MIL、正常 FPR 和候选区域时间连续性。对照 prompt 若不优于类别名，不进入 baseline 训练。
2. **Layer audit**：比较最后层、DANCE候选第8+12完整层、自动选择两层、12层全融合；要求自动两层在同等参数下优于最后层，并报告层门控和责任密度。
3. **KNN synthesis audit**：在人工已知的合成边界上，比较随机正常拼接、KNN正常拼接和无拼接；若KNN不能降低场景相似正常段的误报，则停止该分支。
4. **DSANet/UCF pilot**：先只训练 layer adapter、文本交互和head，作者权重作为选模候选；按作者固定样本间隔计算帧级 AUC。至少超过作者 0.3 点才进入完整训练，超过 1 点才扩展 XD 和另两个 baseline。
5. **反事实检查**：把异常 prompt 换成同场景正常描述、打乱层、打乱 KNN 邻居；若性能不下降，说明模型并未使用声称的机制。

## 10. 开销与边界

两个完整层每个 snippet 增加 1536 个已缓存浮点数读取；若以 float16 存储约为 3 KB/snippet。两个 `768→r→512` 低秩投影、层门控、soft prompt 和文本 Adapter 可控制在约 1M 参数量级。KNN 索引和距离离线计算，合成仅发生在训练期；文本 prompt 编码可缓存。相对 baseline 的主要新增成本是 hidden-state I/O 和小型投影，单卡 RTX 4090 可行。

方案明确禁止：光流、patch token、姿态、深度、检测/分割模型、SwinBERT/MLLM逐snippet描述以及重新提取重型视频特征。LAP、ASK-HINT、Alert-CLIP 中涉及描述模型、MLLM或region/patch的部分不进入实现，只借用原子事件、对照语义和困难负例思想。

## 11. 结论

RQ1：LaGoVAD 的 KNN 通过时间拼接相似正常视频构造可控长视频和稠密伪边界，而不是通道拼接。其机制有明显消融收益，但 UCF/XD 异常训练视频未裁剪，不能直接把整段当成异常帧；必须采用可信正常 dense negative 与跨 fold 语义候选 soft positive。

RQ2：三个 baseline 和相邻工作共同表明，CLIP 异常语义不敏感不能靠单个类名或固定模板解决。最稳定的技巧是 learnable prompt / text adapter、正常—异常对照、视觉条件化文本、细粒度可观察描述和 hard-negative alignment。DSANet 的文本 Adapter 是三篇 baseline 中最强的直接文本消融依据，DeSC 的主要收益则来自时间流解耦。

RQ3：已有 DANCE 结果显示类别证据高度分布式，因此应以神经元责任密度选择少数完整层，再用轻量投影进入统一文本空间。最终值得验证的方案是“完整层选择＋对照式文本适配＋KNN正常时间拼接＋只解冻交互和打分头”。它有清晰的文献依据和可证伪门槛，但尚没有证据保证在三个强 baseline 上统一提升 2—3 点。

## 参考文献

[1] Z. Liu, X. Wu, J. Wu, et al., “Language-Guided Open-World Video Anomaly Detection under Weak Supervision,” ICLR, 2026.

[2] W. Yin, H. Zhang, X. Wang, et al., “Learning to Tell Apart: Weakly Supervised Video Anomaly Detection via Disentangled Semantic Alignment,” AAAI, 2026.

[3] H. Zheng, N. Han, Y. Zeng, and H. Chen, “Decoupled Sensitivity-Consistency Learning for Weakly Supervised Video Anomaly Detection,” arXiv:2603.19780, 2026.

[4] P. Wu, X. Zhou, G. Pang, et al., “VadCLIP: Adapting Vision-Language Models for Weakly Supervised Video Anomaly Detection,” AAAI, 2024.

[5] Z. Yang, J. Liu, and P. Wu, “Text Prompt with Normality Guidance for Weakly Supervised Video Anomaly Detection,” CVPR, 2024.

[6] J. Chen, L. Li, L. Su, et al., “Prompt-Enhanced Multiple Instance Learning for Weakly Supervised Video Anomaly Detection,” CVPR, 2024.

[7] C. Tao, X. Peng, C. Wang, et al., “Learning Suspected Anomalies from Event Prompts for Video Anomaly Detection,” ACM TOMM, 2026.

[8] S. Zou, X. Tian, L. Wesemann, et al., “Unlocking Vision-Language Models for Video Anomaly Detection via Fine-Grained Prompting,” WACV, 2026.

[9] W. Ma, X. Zhang, Q. Yao, et al., “AA-CLIP: Enhancing Zero-Shot Anomaly Detection via Anomaly-Aware CLIP,” CVPR, 2025.

[10] Y. Zhu, M. Zhang, H. Sun, et al., “Alert-CLIP: Abnormality-aware Latent-Enhanced Representation Tuning of CLIP for Video Anomaly Detection,” CVPR, 2026.

[11] R. Liu, J. Huang, G. Li, et al., “Revisiting Temporal Modeling for CLIP-Based Image-to-Video Knowledge Transferring,” CVPR, 2023.

[12] D. Kim, C. Park, S. Cho, et al., “GenCLIP: Generalizing CLIP Prompts for Zero-shot Anomaly Detection,” arXiv:2504.14919, 2025.
