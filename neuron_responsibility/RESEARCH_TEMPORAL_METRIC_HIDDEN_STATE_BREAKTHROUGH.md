# 从“解释隐藏状态”到“修正片段排序”：面向三类视觉-文本 WSVAD Baseline 的提分路线

## 摘要

本报告围绕三个问题展开：LaGoVAD、DeSC 和 DSANet 的结构共性中，哪一个接口最适合通用增强；哪些已发表机制确实改善了时间定位、帧级 AUC 或 AP；现有多层 CLIP ViT-B/16 CLS hidden states 应怎样参与检测，而不再重复无效的特征残差注入。调研覆盖目标三篇 Baseline、2023--2026 年弱监督视频异常检测、弱监督时间动作定位、CLIP hidden-state 解释和不平衡排序优化工作，并结合本项目在 DSANet/UCF-Crime 上的 CNCR、DANCE 和 CACC 结果。

证据形成了一个明确判断：**三个 Baseline 的共同瓶颈不是缺少另一组视觉特征，而是视频级弱标签无法可靠决定“哪些 snippet 应被当成异常，以及异常事件应延伸到哪里”。hidden-state 解释模块要提分，必须从推理期残差改为训练期的独立证据教师，用于 hard-normal 抑制、低显著异常探索、事件完整性传播和可靠片段排序。** 其中，语义教师负责回答“这是什么异常”，正常时序预测教师负责回答“它是否偏离该视频的正常动态”；只有二者一致的片段才成为正种子，其余片段通过时间连通图完成事件扩展。最终监督三个 Baseline 共有的二分类定位分支与视觉-文本语义分支，并对高置信片段直接优化 pairwise AUC 与 AP 排序代理。

这一方向比现有方案更有依据，但不能保证三个 Baseline、两个数据集都稳定提升 2--3 点。已有证据表明 XD-Violence AP 更可能获得 2--4 点提升；在当前 DSANet 的 89.44 UCF AUC 上，2--3 点属于高难目标，应先通过独立候选质量门槛再进入 Baseline 训练。

## 1. 研究问题与范围

- RQ1：三套 Baseline 的真正结构共性是什么，通用模块应该接在哪里？
- RQ2：哪些机制分别改善边界完整性、正常/异常全局排序和高分区域 precision，并能解释 AUC/AP 的变化？
- RQ3：怎样把已有 CLS hidden states 转换为可解释、与 Baseline 分数独立、且单卡 RTX 4090 可运行的片段级监督？

报告只把同一论文、同一设置中的消融差值当作模块证据，不把跨论文榜单差值直接解释为因果增益。边界 mAP、帧级 AUROC 和 AP 衡量不同误差，不能相互替代。

## 2. 三个 Baseline 的共同计算骨架

尽管三篇论文分别强调异常定义、时间行为和正常模式，它们的实现都可以抽象为同一条路径：

1. 使用冻结或基本冻结的 CLIP ViT-B/16，把视频转换为时间对齐的 512-D snippet features；
2. 用 Transformer、TCN、GCN 或其组合建模 snippet 间时间关系；
3. 产生一个二分类 anomaly score，承担帧级时间定位；
4. 产生视觉-文本相似度或类别语义输出，承担异常含义和类别对齐；
5. 用 top-k/MIL 把 snippet 输出聚合到视频级标签训练；
6. 通过正常模式、双流解耦或语言条件弥补 MIL 的偏差。

| Baseline | 二分类定位证据 | 视觉-文本语义证据 | 时间/正常性设计 | 共同弱点 |
|---|---|---|---|---|
| DSANet | `classifier` 的 snippet logits | text adapter、视觉文本 logits | temporal block、SG-NM、DCSA、DNP | 初始 snippet 权重仍由模型自身分数产生 |
| DeSC | sensitivity stream 的 anomaly logits | semantic consistency stream | parallel TCN+GT、GMP、分流优化 | 两个专家仍从粗粒度视频标签启动 |
| LaGoVAD | binary head | class/caption similarity 与 language-guided fusion | temporal encoder、动态合成、negative loss | top-k 极值和合成候选质量决定监督上限 |

因此最稳定的通用接口不是修改 512-D feature，而是统一提供：`snippet_index -> positive confidence / negative confidence / uncertainty / concept responsibility`。每个 Baseline 保留作者输入、主干和原始损失，只增加片段监督与排序损失；需要解冻时，按统一规则解冻“距离最终 score 最近的时序适配块与分类头”，而不是要求三个模型解冻同名模块。

## 3. 哪些机制真正改变了 AUC/AP

### 3.1 互补分支交叉教学：对 XD AP 很强，对高位 UCF AUC 较弱

CPL-VAD 将 VadCLIP 的二分类定位分支和类别语义分支做 cross pseudo labeling，并用时间一致性模块过滤伪标签。相对 VadCLIP，它把 XD AP 从 84.51 提高到 88.53，即 **+4.02**；但 UCF AUC 只从 88.02 提高到 88.24，即 **+0.22** [4]。同一方法把 XD fine-grained AVG 从 24.70 提高到 33.53，把 UCF AVG 从 6.68 提高到 9.39。

这组结果说明语义与定位互教确实能找回完整事件，尤其适合 XD 的多类别、多片段和电影剪辑场景；但当 UCF AUC 已接近 88--90 时，仅做语义互教不足以整体重排全部正常/异常帧。对本项目而言，cross teaching 应保留，但 UCF 提分还必须加入 hard normal 与低分异常探索。

### 3.2 Hard normal 与弱标签污染分离：最直接改善全局排序

TLMA 明确把异常视频中的正常片段视为 Weakly Labeled Information。其 triplet learning 使用正常视频中“最像异常”的片段作为 hard-normal anchor；在 VideoMAEv2 特征的 MIL baseline 上，triplet 单独把 UCF AUC 从 85.69 提高到 88.37，即 **+2.68**，完整方法达到 89.47，即 **+3.78**；XD AP 从 83.61 提高到 86.78，即 **+3.17** [5]。随机、低分和高分正常 anchor 的 UCF AUC 分别为 89.04、89.06 和 89.47，支持“优先修正最危险的正常假阳性”。

D2MIL 给出更保守、也更接近插件场景的证据：先丢弃高损失疑似噪声，再用视觉语言模型找回被误删的 hard anomaly。在 UCF-Crime 上，它给不同 MIL Baseline 带来约 +0.36 到 +1.90 AUC；较强模型上的收益会收缩 [6]。两者共同说明，可靠提分来自同时处理“错误选中的正常片段”和“错误丢掉的困难异常片段”，而不是单向筛掉噪声。

### 3.3 探索低显著片段：主要提高 AP 与实际低 FPR recall

The Road Less Seen 不再只训练 top-k 极值，而是用 temporal clustering 覆盖不同时间/语义簇，并用 uncertainty sampling 探索低分但不确定的片段。UR-DMU 的 UCF AP 从 35.48 提高到 36.42；再融合 VLM 后达到 38.33，即总计 **+2.85 AP** [7]。单独聚类或单独不确定性反而降低 AP，二者结合才有效，说明“多样覆盖”和“困难探索”必须互相约束。

MuST-VAD 进一步显示指标为何会分化：一轮 LVLM--MIL 互学习只把 UCF AUROC 从 88.15 提高到 88.63，即 +0.48，却把 AP 从 37.25 提高到 42.46，即 **+5.21** [8]。使用随机窗口时 AUROC 降到 86.54；关键片段选择比大模型本身更重要。该方法需要 Qwen3-VL-8B 和四张 L40S，不符合本项目开销目标，但它证明了一个可用原则：训练应集中修正最高分 false positive 和最可信 anomaly candidate，这更直接影响 AP 的高分区域 precision。

### 3.4 时间边界完整性：能显著提高 segment mAP，但未必提高 frame AUC/AP 2--3 点

LEC-VAD 用 category-aware/category-agnostic 双结构、Gaussian mixture prior 和 prototype memory 学习事件完整性。其 coarse-grained CLIP 结果相对前一强方法约提高 0.75 XD AP 和 0.93 UCF AUC，而 fine-grained 提升更大，且 IoU 越严格收益越明显 [9]。LAS-VAD 用 anomaly-connected components 按时间邻接和语义关系划分视频片段，CLIP coarse-grained 相对 LEC-VAD 约提高 1.36 XD AP 和 0.89 UCF AUC，但 fine-grained AVG 提升更明显 [10]。REBA 的 gated residual multi-scale experts 同样把 UCF fine-grained AVG 从最佳单尺度 8.28 提高到 10.49，但 coarse AUROC 的改变很小 [11]。

这些工作支持 connected-component event growing 和 multi-scale temporal expert，却也给出边界：**只做平滑、连通或多尺度边界修正，很难单独带来 2--3 点 frame AUC/AP。** 它们应当用于把高质量种子扩展成完整事件，而不是负责发现种子。

### 3.5 直接优化 AUC/AP：只有在片段标签足够可靠时才值得使用

AUROC 本质上衡量随机异常帧排在随机正常帧之前的概率，适合用正负 pairwise ranking 代理；AP 对类别比例和高分区域 false positive 更敏感。理论工作表明，高 AUROC 并不保证高 PR-AUC，尤其在严重不平衡数据上 [12]。AP-Loss 和 AUPRC maximization 研究表明，直接排序优化能缓解极端前景/背景不平衡 [13,14]。

但这些损失不会修复错误伪标签；它们会更强地拟合伪标签。因此只能对独立教师给出的高置信 positive/hard-negative 集合使用，不能把整个异常视频的 snippet 当正样本直接优化 AP。为保持跨数据集统一，UCF 和 XD 都使用同一组合：可靠正负对的 pairwise AUC surrogate，加只在高置信集合上计算的 differentiable AP surrogate，而不是按数据集切换算法。

## 4. Hidden states 的解释证据与限制

### 4.1 中间层值得选择，但不能靠固定层号

HiProbe-VAD 在多模态大模型上用 KL divergence、Fisher-like discriminant ratio 和 entropy 选择异常信息最丰富的中间层。动态选层相对固定最后层提高 3.51 UCF AUC 和 2.87 XD AP，但绝对结果 86.72/82.15 仍低于强 WSVAD Baseline [15]。它证明“层选择有价值”，却没有证明轻量 probe 足以超过强 MIL 模型。

CLIP SAE 研究也发现 CLS token 在约第 6--7 层发生从稀疏到丰富的转变，不同任务的最佳 steering 层不同；SAE 特征比原始神经元提供约 10 倍概念覆盖，只有约 10--15% 的深层特征可稳定 steering，最优去偏层常在中间层 [16]。SpLiCE 则把 CLIP embedding 稀疏分解为人类概念组合，说明“可解释单元”更适合定义为稀疏概念方向，而不是任意原始坐标 [17]。

因此，本项目应先在 12 层上做低成本任务筛选，再只为排名最高的 2--3 层训练小型 Top-K SAE；不固定第 6、9、12 层，也不把 12 层全部拼接。

### 4.2 本项目能诚实声称的“神经元”是什么

建议将探测单元定义为：**选定 CLIP 层的 CLS residual stream 经 Top-K sparse autoencoder 分解后得到的一个稀疏 latent feature。** 每个 latent 通过三类证据解释：最大激活 snippet、关联的异常属性文本、对正常时序预测误差的贡献。相比 768 个原始维度，这种单元更接近 monosemantic functional feature。

必须保留一个方法边界：当前缓存只有每层 CLS，而没有同层全部 patch tokens。中间层 CLS 不能单独继续通过后续 CLIP blocks，因为自注意力还需要 patch tokens。因此不能把离线中间层消融称为严格 causal intervention。只有最后层经过 `ln_post + visual.proj` 的干预，或重新提取完整 token residual stream，才能进行严格下游因果验证。论文中应使用“functional concept feature / responsibility feature”，而不是对所有层都声称 causal neuron。

## 5. 建议主方法：TRACE-VAD

TRACE-VAD（Temporally Responsible Anomaly Concept Evidence）把 hidden-state 解释模块变成 Baseline-score-free 的片段监督教师，而不是额外视觉分支。

### 5.1 Stage A：跨层时间责任筛选

对每层 CLS 分别计算四项训练折内指标：

1. **normal predictability**：只在正常视频训练轻量 masked temporal predictor，衡量该层正常动态是否可预测；
2. **sparse innovation gap**：异常视频是否出现少量、连通的高预测误差，而正常视频没有同等强度的峰值；
3. **concept separability**：异常类别/属性的 MIL probe 在 held-out train videos 上是否可分；
4. **confounder penalty**：场景、镜头切换、拥挤、奔跑等 hard-normal 概念是否同样激活。

用训练视频五折结果选择 2--3 层，并加入层间冗余惩罚。层的选择依赖功能特征数量、稳定性和互补性，而不依赖 Baseline score，也不依赖固定层号。

### 5.2 Stage B：稀疏概念单元与两个独立教师

在选中层各训练一个 768 -> 2048 的 Top-K SAE。随后形成两个教师：

- **Semantic responsibility teacher**：用异常类别及其 action/object/context 属性文本训练 group-sparse MIL probe；同时加入 normal 和 hard-normal confounder concepts。输出 `semantic_evidence` 与激活的 SAE latent IDs。
- **Normal temporal teacher**：只用正常训练视频，在 SAE code 序列上做 masked center prediction。输出 `innovation_evidence`，即当前 snippet 无法由前后正常上下文解释的程度。

语义教师适合覆盖持续异常内部，时间教师适合发现开始/结束和突变；两者误差模式不同，避免了 CACC 中把两个弱信号直接相乘后同时衰减的问题。

### 5.3 Stage C：种子、hard normal 与事件完整性

阈值由 held-out normal videos 做 conformal calibration，控制正常 false-positive rate，而不是固定 top-p：

- positive seed：语义高且时序创新高；
- hard normal：来自正常视频，任一教师很高；
- reliable negative：两个教师都低；
- uncertain：二者冲突，前期完全忽略。

从 positive seed 构建视频内时间图。边只连接时间相邻或 SAE code 相似的 snippet；沿“语义支持或时序支持”区域扩展，遇到两个证据均落入 normal conformal set 时停止。该过程吸收 ACC、LEC-VAD 和 PseudoFormer 的事件完整性思想 [18]，但种子不来自 Baseline score。

训练后期可从 uncertain pool 中按 temporal cluster 每簇选一个片段探索，且必须由另一个教师或相邻连通证据确认，避免单独 uncertainty sampling 引入正常噪声。

### 5.4 Stage D：跨分支教学与指标感知训练

三个 Baseline 都有 binary localization output 和 vision-text semantic output。TRACE 统一提供 soft targets：

- Semantic -> Binary：把异常含义蒸馏给二分类定位分支，找回低显著但语义明确的异常；
- Temporal -> Semantic：用时间种子和连通边界限制语义分支，防止整段背景被识别成异常类别；
- Hard normal ranking：要求所有 positive seeds 高于正常视频里最高风险的 hard normals；
- Reliable AP/AUC loss：在高置信 positive/negative 上同时优化 pairwise AUC surrogate 和 AP surrogate；
- Event consistency：同一连通组件内部平滑，但组件边界两侧保持 margin。

原作者 MIL loss 始终保留。早期仅训练新增教师/监督头；通过候选门槛后，解冻各 Baseline 的分类头和最末时序适配块，学习率应比作者从头训练低 10--50 倍。CLIP image encoder 与原始 512-D feature 始终不改。DeSC 的两个专家继续独立 optimizer，DSANet 解冻最后 temporal/DCSA 路径，LaGoVAD 解冻最后 temporal/fusion 与 binary head；这是同一原则下的结构适配，不是三个不同方法。

UCF-Crime 与 XD-Violence 使用完全相同的层筛选、SAE、conformal calibration、候选生成和损失组合；数据集之间只替换作者本来就使用的异常类别文本和由各自训练分布估计的统计量，不手工设置不同阈值或模块。

### 5.5 推理与解释

正式 anomaly score 由训练后的 Baseline 输出，不做 hidden feature residual injection。TRACE 教师可在正式推理中关闭，因此不会增加主指标推理开销；若需要解释，额外运行小型 SAE/probe，输出：负责层、latent ID、关联异常属性、时间创新和事件组件边界。解释模块与最终分数共享训练证据，但不通过任意激活放大改变预测。

## 6. 为什么这一方案不同于已经失败的探测实验

| 已失败路线 | 实验暴露的问题 | TRACE 的结构修正 |
|---|---|---|
| CNCR 概念神经元残差路由 | 概念因果性成立，但正常/异常门差异小；关闭路由更好 | 不在推理期改 feature；概念只产生训练证据 |
| DANCE 跨层类别回路 | 类别 top-1 较高，但类别相关背景不等于异常时间位置 | 必须与 normal-only temporal innovation 交叉确认 |
| CACC 语义 × 偏离乘积门 | 两个弱连续分数相乘后变弱；层权重没有分化 | 独立教师、离散置信集合、冲突片段忽略，不做连续乘法 |
| 渐进解冻分类头/时序块 | 辅助目标错误时，解冻越深退化越明显 | Baseline 解冻前先过 candidate precision/event coverage 门槛 |

CACC 后验诊断中，语义分支以 1% 权重与作者分数融合时 AP 有互补性，但 UCF AUC 只微升；这与 CPL-VAD、MuST-VAD 的证据一致：语义更容易改善高分区域 AP，而要显著改善 UCF AUC，必须修正 hard normal 和被漏掉的异常片段的整体排序。

## 7. 开销估计

本方案复用现有 `[T, 12, 768]` hidden manifests 与 512-D CLIP features，不复制 9216-D 拼接文件，也不重新提取 patch tokens。

- 全层筛选：逐视频流式读取，主要是统计和小 probe；GPU 显存通常低于 4 GB。
- 每层 768 -> 2048 Top-K SAE：约 3.1M encoder/decoder 参数；选 2--3 层约 6.3--9.4M 参数。
- temporal predictor：建议投影到 128--256 维、2 层 TCN/Transformer，通常低于 2M 参数。
- 训练时不加载完整 CLIP image encoder，只读取缓存 hidden states；单卡 4090 可行。
- 最终 Baseline 推理可关闭 TRACE，主推理成本与作者模型接近；解释模式才增加 SAE/probe 的小额开销。

真实瓶颈会是磁盘随机读取，而不是显存。应复用 manifest、按视频顺序采样、使用进程内验证缓存，避免再次生成对齐 hidden feature 副本。

## 8. 实验门槛与优先级

### 8.1 在训练 Baseline 之前必须通过

1. 选中 SAE latents 在 held-out train folds 上显著优于随机 latents、全部原始维度和最后层 raw CLS probe；
2. normal temporal teacher 在 held-out normal videos 上达到稳定的 conformal coverage；
3. 固定配置的一次性 test diagnostic 中，positive seeds 相对 DSANet top-k 的 snippet precision 或 event coverage 至少提高 10 个百分点，且 normal-video top-1% false-positive rate 不升；该诊断只验证假设，不能用于扫描超参数；
4. semantic-only、temporal-only、intersection seed、event growth 四个输出必须分别报告 AUC/AP、top-q precision、event coverage 和 normal FPR；
5. 如果两个教师的错误高度相关，或 intersection seed 覆盖率过低，则停止，不进入昂贵 Baseline 微调。

### 8.2 推荐实验顺序

1. DSANet/UCF：先做固定 TRACE evidence diagnostic，不解冻 Baseline；
2. 只加入 hard-normal pairwise ranking，检验能否先获得至少 +0.5 AUC；
3. 加 cross teaching 与 event growth，目标累计至少 +1.0 AUC，且 AP 不下降；
4. 通过后迁移 DSANet/XD，重点验证 AP 是否达到 +2；
5. 最后扩展 DeSC 和 LaGoVAD，并保留统一算法与超参数规则。

### 8.3 必需消融

- raw neuron coordinates vs SAE latent features；
- fixed last layer vs selected single layer vs selected 2--3 layers；
- semantic teacher vs temporal teacher vs agreement seeds；
- top-k growth vs connected-component event growth；
- random normal vs low-risk normal vs hard normal；
- BCE/MIL only vs +AUC surrogate vs +AP surrogate vs combined；
- frozen Baseline vs head-only vs last-temporal partial unfreeze；
- selected latents vs equal-size random latents。

## 9. 风险与预期

最主要风险是只有 CLS token，没有局部 motion/patch evidence。TLMA 的 motion module、LAKE 的 patch-neuron gallery和局部异常分割工作都说明，细粒度前景信息能进一步减少背景假阳性；本方案只能通过时间预测和 hard-normal 概念间接补偿。若 TRACE 的独立候选门槛失败，应停止围绕 CLS 调 loss，而不是继续增加模块。

第二个风险是 UCF AUC 的高位饱和。CPL-VAD 在 XD AP 获得 +4.02，但在 UCF AUC 仅 +0.22；D2MIL 在强 Baseline 上常低于 +2；MuST-VAD 的 +5.21 主要发生在 AP。基于现有证据，合理预期是：

- DSANet/UCF AUC：先以 +0.8--1.5 为可信阶段目标；+2--3 属于需要候选质量显著领先 Baseline 才可能实现的进取目标；
- XD AP：cross teaching、hard normal 和可靠 AP ranking 共同达到 +2--4 更有文献依据；
- fine-grained temporal mAP：event growth 和多尺度边界机制更可能获得明显提升。

## 10. 结论

RQ1：三个 Baseline 的共性是“CLIP snippet feature -> temporal model -> binary localization + semantic alignment -> MIL”。最通用的突破口是片段监督接口，而不是 feature 注入接口。

RQ2：hard-normal 分离和困难异常找回最直接改变 AUC/AP；语义/定位 cross teaching 更容易显著提高 XD AP 和 fine-grained mAP；connected components、Gaussian prior 和 multi-scale experts主要提高事件完整性，单独对 frame AUC 的增益有限；直接 AUC/AP 优化只有在伪标签可靠时才有效。

RQ3：建议把“神经元”升级为选定中间层 CLS 的 SAE sparse concept feature，通过语义责任教师与 normal-only temporal predictor 产生 Baseline-score-free seeds，再进行 conformal hard-normal mining、event-complete graph propagation、跨分支教学和可靠集合上的指标感知排序。该方案复用现有 hidden states、单卡 4090 可运行、主推理可零额外开销，并且每个模块都对应已验证的误差来源。

## 11. 本轮实现边界

本轮可运行版本严格遵守项目中“神经元等于 hidden-state 原始维度”的定义，没有把 SAE latent 偷换成神经元。它实现的是 TRACE-Raw：先用异常/正常视频的稀疏响应差和 normal-only 邻域创新差给全部原始维度排序，再按有效神经元证据的累计覆盖率自动选 1--3 层，每层保留 64 个原始维度；小型时序探针只对这些维度建模。SAE 保留为后续对照实验，不是本轮主方法。

TRACE-Raw 已实现不依赖 Baseline 分数的双证据种子、normal quantile 校准、事件扩张、hard-normal ranking、可靠集合 AP surrogate、跨分支语义监督，以及分类头到最后时序块的渐进解冻。正式推理只使用适配后的 Baseline 分数，探针的独立指标作为解释和失效诊断报告。

## References

[1] Wenti Yin, Huaxin Zhang, et al., "Learning to Tell Apart: Weakly Supervised Video Anomaly Detection via Disentangled Semantic Alignment," arXiv:2511.10334, 2025.

[2] Hantao Zheng, Ning Han, Yawen Zeng, Hao Chen, "Decoupled Sensitivity-Consistency Learning for Weakly Supervised Video Anomaly Detection," arXiv:2603.19780, 2026.

[3] Zihao Liu, Xiaoyu Wu, Jianqin Wu, Xuxu Wang, Linlin Yang, "Language-Guided Open-World Video Anomaly Detection under Weak Supervision," ICLR, 2026.

[4] Dayeon Lee, Donghyeong Kim, Chaewon Park, Sungmin Woo, Sangyoun Lee, "Cross Pseudo Labeling for Weakly Supervised Video Anomaly Detection," ICASSP, 2026.

[5] Rong Xu, Runqi Wang, et al., "TLMA: Mitigating the Impact of Weakly Labeled Information for Video Anomaly Detection," CVPR, 2026.

[6] Yaxin Zhao, Yang Wang, et al., "Learning from Noisy Supervision: A Denoising-Debiasing Framework for Weakly Supervised Video Anomaly Detection," CVPR, 2026.

[7] Anusha Acharya, Hitesh Sapkota, Qi Yu, Xumin Liu, "The Road Less Seen: Segment Exploration for Weakly Supervised Video Anomaly Detection," CVPR, 2026.

[8] Satoshi Hashimoto, Hitoshi Nishimura, Mori Kurokawa, "MuST-VAD: Mutual Structured Learning for Video Anomaly Detection," arXiv:2608.06913, 2026.

[9] Yu Wang, Shiwei Chen, "Learning Event Completeness for Weakly Supervised Video Anomaly Detection," ICML, 2025.

[10] Yu Wang, Shengjie Zhao, "Weakly Supervised Video Anomaly Detection with Anomaly-Connected Components and Intention Reasoning," CVPR, 2026.

[11] Chengxi Chu, Nurul Japar, Chee Kau Lim, "REBA: Residual Mixture-of-Experts and Bidirectional Video-Text Alignment for Better Fine-grained Weakly Supervised Video Anomaly Detection," CVPR Findings, 2026.

[12] Martin Mihelich, François Castagnos, Charles Dognin, "Interplay of ROC and Precision-Recall AUCs: Theoretical Limits and Practical Implications in Binary Classification," ICML, 2024.

[13] Kean Chen, Jianguo Li, et al., "Towards Accurate One-Stage Object Detection with AP-Loss," CVPR, 2019.

[14] Guanghui Wang, Ming Yang, Lijun Zhang, Tianbao Yang, "Momentum Accelerates the Convergence of Stochastic AUPRC Maximization," AISTATS, 2022.

[15] Zhaolin Cai, Fan Li, Ziwei Zheng, Yanjun Qin, "HiProbe-VAD: Video Anomaly Detection via Hidden States Probing in Tuning-Free Multimodal LLMs," ACM Multimedia, 2025.

[16] Sonia Joseph, Praneet Suresh, et al., "Steering CLIP's Vision Transformer with Sparse Autoencoders," CVPR MIV Workshop, 2025.

[17] Usha Bhalla, Alex Oesterling, Suraj Srinivas, Flavio P. Calmon, Himabindu Lakkaraju, "Interpreting CLIP with Sparse Linear Concept Embeddings," arXiv:2402.10376, 2024.

[18] Ziyi Liu, Yangcen Liu, "Bridge the Gap: From Weak to Full Supervision for Temporal Action Localization with PseudoFormer," CVPR, 2025.
