# Idea 评估：责任层双专家

## 1. 第一印象

- Paper type：Novel Method。
- 一句话故事：现有视觉文本 VAD 把 CLIP 输出当成整体，而本方法先把责任探测选出的完整层训练成独立语义专家，再与 baseline 的时间专家交叉筛选片段，只用可靠共识适配最终二分类头。
- 结论：**Accept with Revisions，值得做，但必须先通过语义专家和标签质量闸门。**

旧方法“冻结文本相似度直接生成异常段”已经被 UCF 实验否定，本方案不是给旧方法换阈值，而是改变监督生成机制；KNN/DVS 暂时全部移除。

## 2. 致命缺陷审计

| # | 缺陷 | 严重度 | 防御 |
|---|---|---|---|
| 1 | 与 CPL-VAD 的“双分支交叉伪标签”很接近，若只讲双专家会缺少新意 | MAJOR | 主贡献必须落在责任探测选层、完整层可解释语义专家、跨 DSANet/DeSC/LaGoVAD 的统一 binary-head 接口；做随机层、最后层、分散神经元、全层四个对照 |
| 2 | 语义专家和共识标签尚无指标，不能提前声称提升 2～3 点 | MAJOR | 先跑 E0 梯度审计、E1 held-out 视频分类、E2 共识覆盖/分歧诊断、E3 head-only；任一失败即停止 |

最接近工作及差异：

| 工作 | 核心对象 | 与本方案的差异轴 |
|---|---|---|
| CPL-VAD，Lee et al., 2026 | VadCLIP binary/category 两分支交叉伪标签 | 本方案作用于责任探测选择的 CLIP 中间完整层，并适配三个异构 baseline；公开训练代码尚未发布 |
| TPWNG，Yang et al., 2024 | CLIP 域适配、正常视觉提示、伪标签 | 本方案不微调 CLIP，使用预计算中间 CLS 与小 Adapter；强调层级责任解释 |
| DSANet，Yin et al., 2026 | 正常模式与语义解耦 | 本方案把语义证据做成 baseline 外部独立专家，避免与时间目标联合优化 |
| DeSC，Zheng et al., 2026 | 敏感/稳定时间流独立训练 | 本方案借鉴“独立专家避免梯度冲突”，但专家分别承担时间定位与责任层语义 |
| MIST，Feng et al., 2021 | 两阶段稀疏自训练 | 本方案的伪标签由两个不同证据源共同确认，而不是单分支自训练 |

## 3. 生命周期与能力匹配

| 方面 | 当前条件 | 判断 |
|---|---|---|
| Idea 类型 | Innovative technique / application research 之间 | 应压缩成单一插件方法，不扩张到 KNN、光流、patch token |
| 生命周期 | 约 4～6 个月 | 视觉文本 VAD 更新较快，需要尽早得到 DSANet/UCF 决定性实验 |
| 每周有效时间 | 未提供 | 需要用户自行确认；当前按单人项目估计 |
| 能力 | 用户自述基础较弱，但已有完整数据、hidden states、权重和单卡环境 | 工程风险中等，必须使用阶段化命令和可读产物 |
| 硬件 | 单卡 RTX 4090 | 绿色：不反传 CLIP，只训练小 Adapter 或不足千级/万级的头部参数 |
| Fit | — | Yellow：研究问题匹配，主要风险是方法验证和跨 baseline 工程量 |

## 4. 五维评分

| 维度 | 分数 | 依据 | 提升建议 |
|---|---:|---|---|
| Higher | 7/10 | 机制依据：语义专家补充 baseline 时间证据，共识过滤避免旧方法错误伪标签；尚无新实验数据 | 首个决定性实验必须是 DSANet/UCF head-only，不用更多模块掩盖贡献 |
| Faster | 8/10 | 直接复用预计算 CLS；不提取 patch/光流，不反传 CLIP；推理最终只保留 baseline | 报告离线语义阶段时间、显存和最终推理增量 |
| Stronger | 8/10 | 机制依据：两个独立专家、分歧 ignore、纯正常视频提供确定负标签，直接针对伪标签噪声 | 做语义-only、baseline-only、直接平均、无精炼四个对照 |
| Cheaper | 8/10 | 不增加人工帧标注，不新增模态，单卡即可 | 记录 trainable parameter 与缓存大小 |
| Broader | 8/10 | 公共接口只有 `512-D snippets -> anomaly curve -> binary head`，适配三个结构不同的 baseline 和两个数据集 | 必须完成至少两个 baseline、两个数据集才能支撑通用性主张 |

高分均为**机制评分，尚未被新实验确认**，不代表已经获得指标提升。

## 5. 范式探测

| 问题 | 判断 | 原因 |
|---|---|---|
| First Principles | Yes | 挑战“CLIP 输出必须作为不可分整体、所有目标必须联合优化”的默认做法 |
| Elephant in the Room | Yes | 弱标签模型需要帧级判断，但伪标签经常只是模型自身峰值，社区普遍存在确认偏差 |
| Technology Cycle | Partial | CLIP 中间层解释与低成本缓存使独立层专家可行，但技术本身不是突然出现 |
| Hamming's Rule | No | 成功会改善一类 VAD 方法，但暂不足以改变整个领域问题排序 |

Disruptive potential：possible，更合适的论文定位仍是有依据的增量方法。

## 6. 可行性

| 风险 | 等级 | 缓解 |
|---|---|---|
| 计算 | 低 | 预计算 hidden states；CLIP 参数 0；语义 Adapter bottleneck=64；baseline 只训 binary head |
| 数据 | 低 | UCF/XD、CLS、作者权重均已有；XD 缺失 4 个训练视频继续记录并跳过 |
| 工程 | 中 | 三 baseline 的内部路径不同；通过外层 adapter 和梯度审计，禁止按相似模块名统一解冻 |
| 时间 | 中 | 先 DSANet/UCF；不过闸门不运行剩余五组，避免重复无效完整训练 |
| 新颖性 | 中高 | CPL-VAD 很接近；需要责任层对照、跨 baseline 和解释性可视化共同建立差异 |

## 7. 最终判断

**Accept with Revisions：worth pursuing, pending the validation experiment。**

最先完成三件事：

1. 跑语义专家 held-out train-video AUC/AP，确认它已经学到目标域语义，而不是再次输出整体负边际。
2. 跑梯度图和专家一致性图；确认共识不是空集，也不是 baseline 分数的简单复制。
3. 只训练 DSANet 的 513 参数 classifier。若 AUC 没有稳定超过作者权重或 AP 继续下降，停止本方向，不解冻 `mlp2`。

## 明确失败条件

- 语义专家验证 AUC 接近随机，或所有层权重塌缩但随机层同样有效；
- 异常视频共识正片段几乎为空，或两个专家相关性接近 1；
- head-only 只能降低训练损失，不能改善作者选模指标；
- 责任选层不优于最后层、随机层或全层平均。

这些条件会直接否定本方法的主要机制，不能用继续解冻更多层来掩盖。
