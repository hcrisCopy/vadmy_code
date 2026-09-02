# 弱监督视频异常检测 × 神经元定位：方案重置调研

> 日期：2026-09-02
> 用途：解释为什么 CVA-VAD v9 不 work、Universal 为什么能涨，以及下一版方法应保留什么。本文只记录已核实文献与项目事实；所有新公式均是待实验验证的假说。

## 1. 结论先行

当前最值得做的不是继续优化 contextual residual，而是把问题重新定义为：

> **冻结 WSVAD host 已经给出很强的 snippet 排序，但标量分数丢掉了 CLIP 内部仍可用于判断“这个高分该相信还是该否决”的证据。弱标签下需要定位的不是普通类别神经元，而是能作为异常 bag 证人、在正常 bag 上保持安静、并且补充 host 剩余错误的功能神经元。**

三个直接结论：

1. v9 的条件预测器学会了预测 hidden state，却没有学到独特检测信息。E1 已经否定当前 contextual-directional field，不能靠调窗口、预算或学习率救。
2. Universal 的主要增益不是某个神奇后处理，而是三类信息共同作用：强 host 排序、全层稀疏神经元证据、视频级正常/异常判别。事件滤波只是不稳定的工程补丁。
3. whole-video classification 在 WSVAD 中已有明确先例，不能作为创新。新颖性应放在 **weak-label witness neuron discovery** 与 **intervention-verified temporal tags**；视频级分类只作为把已验证信息转成稳定增益的桥。

建议下一版叫 **WITNESS-VAD**。只训练一次：冻结 host 和 CLIP，用一个稀疏神经元读出同时学习视频证人、单侧正常否决和局部异常修正。训练结束后才计算解释分数和 tag，不再训练—冻结—伪标签—再训练。

## 2. 冻结的研究问题

- RQ1：弱监督、长视频、正常/异常片段混杂时，现有 WSVAD 真正缺少什么能力？
- RQ2：沈飞团队的 layer/neuron localization、effect size 和 activation intervention 中，哪些能迁移，哪些会因视频级弱标签失效？
- RQ3：Universal 实际利用了什么信息，怎样用一个训练图保留它而不保留经验规则？
- RQ4：什么实验能同时证明 ≥1 pp 增益、神经元机制和解释可信，而不是只证明正常视频被整体压低？

## 3. 领域能力—问题—方法地图

| 任务能力 | 为什么 WSVAD 缺 | 现有主流做法 | 尚未补上的能力 |
|---|---|---|---|
| 从视频标签定位异常片段 | 异常 bag 大量片段仍正常；标签只说明至少有一个异常 | max/top-k MIL、特征幅值、伪标签、自训练、样本探索 | 哪些内部单元真正承担 bag 证据，哪些只是背景捷径 |
| 控制正常误报 | 正常 bag 的每个片段都是可靠负例，但 host 只输出压缩后的标量 | normal prototype、BatchNorm statistics、whole-video classification | 用 host 未读出的内部证据判断一次高分是否该被否决 |
| 覆盖完整异常事件 | top-k 偏向最显著片段，低分或模糊异常易漏 | completeness、uncertainty、connected components、多尺度时序 | 不借助帧标签时，不能把“连续”直接当作真实边界；必须把时间结构降为可检验归纳偏置 |
| 抗场景捷径 | 相机、背景、运动强度可能和视频标签相关 | unbiased MIL、normal guidance、semantic prompts | 证明选中神经元补的是 host 残余错误，而非重复数据集偏差 |
| 可解释与可干预 | heatmap/文本描述只说明相关，不说明该内部证据改变了判断 | scene graph、VLM verbalization、saliency | 对缓存的 CLS 坐标做等预算 erase/patch，并把作用绑定到具体视频和时间 |

这张图给出一个重要取舍：我们不再竞争“更好的 top-k”“更强的 temporal block”或“更多文本语义”。这些方向已经拥挤，而且项目失败表明它们不是当前 +1 pp 的可靠来源。

## 4. 沈飞路线真正提供了什么

### 4.1 DNA：从层到单元，再用多因素分数筛选

DNA 先用真实/伪造类别的 CLS 中心距离与注意分布变化定位层，再为每层训练线性 probe。单元分数为：

\[
S_{i,k}=\left|\bar g_{i,k}\,\bar a_{i,k}\,w_{i,k}\right|,
\]

其中激活、probe 权重和梯度敏感性必须同时大。它的价值不是把三个量相乘这个形式，而是要求一个候选单元同时满足“出现过、和任务方向一致、改变它会影响判断”。

不能直接照搬：DNA 的每张图都有可靠真假标签；WSVAD 的异常视频不能把所有 snippet 当异常。若直接做异常视频均值减正常视频均值，最容易找到的是视频/场景分类单元，而不是异常片段单元。

### 4.2 V-FIND：视频里已有神经元工作，但任务不是监控异常定位

V-FIND 已把 latent anchor neuron 用到生成视频伪造检测。它先按层级真实/伪造差异定位层，再用 `|activation| × |probe weight|` 和标准化 effect size 找单元，最后在稀疏子空间上训练线性分类器。它还做了同层、等预算随机控制和双向 activation swap。

因此不能声称“首次把神经元定位用于视频”。可防守边界是：

- V-FIND 是剪辑级真假分类，真假标签能覆盖整个 clip；
- 本项目是长监控视频的弱监督时间定位，异常标签只覆盖未知的少量片段；
- 本项目的神经元分数必须显式处理 latent witness、可靠正常片段和 frozen-host complementarity。

### 4.3 LAKE：正常数据能找敏感维度，但高方差不是普适答案

LAKE 用少量正常图像的通道方差选择敏感神经元，再在该子空间测量相对正常 gallery 的偏离。它说明“冻结表示里可挖出稀疏异常知识”是合理 tag，也明确把 temporal video 留作未来工作。

不能直接照搬：工业图像异常通常是局部结构缺陷，WSVAD 异常是行为与上下文关系；正常高方差维度可能只是相机运动、人物数量或场景变化。高方差只能是对照，不应成为最终神经元定义。

### 4.4 对本项目的准确迁移

保留三件事：

1. 从全层 CLS 坐标中寻找稀疏内部子空间；
2. 单元排序必须结合激活、任务敏感性和效应量；
3. 必须用等预算随机控制和 activation intervention 验证功能。

必须新增三项 WSVAD 适配：

1. 正类使用 latent witness responsibility，不能把异常视频所有片段当正类；
2. 正常视频提供 dense negative penalty；
3. 单元必须解释 host-only 视频预测的残差，避免只重复 host 已知信息。

## 5. Universal 到底拿到了什么

代码级事实如下：

1. `SparseNeuronExpert` 从 12 层 CLS 中每层选择稀疏坐标，用 MIL 产生独立神经元曲线。
2. evaluator 在训练数据上对 host/neuron/normality/context 曲线提取均值、方差、分位数、top 比例、变化率、相关和分歧等统计，再拟合两个视频级 Logistic Regression。
3. 只有两个视频分类器都判断偏正常时，`normal_shift` 才整体下压该视频；这是历史上最稳定、最大的增益来源。
4. 后续 max filter、median filter、dilation、Gaussian 和 0.5 snippet 前移提供了时间形态偏置，但 DSANet/XD 上部分单独消融为负，不能当作稳定机制。

它告诉我们的不是“保留 40 多个统计和 5 种滤波”，而是：

- host curve 的整体分布可识别容易误报的正常视频；
- CLIP 全层神经元曲线提供 host 分数没有完全利用的补充视角；
- 二者的 agreement/disagreement 比任一单独曲线更有用；
- 对强 host 应做条件修正，而不是重新训练一个检测器。

whole-video classification 已有 WACV 2024 workshop 工作系统验证，所以这部分是性能桥和必要对照，不是论文创新。

## 6. 为什么 v9 没 work

v9 的问题不是代码没连通。B1--B4 已证明 predictor、梯度、padding、checkpoint 和 auditor 都正确；E1 也证明 conditional NLL 更好。失败发生在信息假设：

1. **可预测性不等于异常判别性。** 一个 hidden 坐标能被上下文预测，并不代表预测残差和异常相关。
2. **当前上下文不是缺失信息源。** C1 全局 z-score 和 C4 target-visible controller 已解释或超过 C3 的极小变化。
3. **cross gate 没有得到可用证据。** 五组 `kappa_cross=0`，说明这个 evidence 无法驱动历史上真正有效的视频级抑制。
4. **两轴公式先于证据成立。** cross/within 分工在数学上整洁，但输入 evidence 不携带独特信息，最终只剩近零的 within 扰动。

所以必须删掉：masked context predictor、conditional residual、global scalar `kappa_cross/kappa_within`、“上下文违背”主张。它们可以保留为负结果和对照，不能进入新主方法。

## 7. 新假说：Witness Neurons

### 7.1 任务定义

正常 bag 满足：所有 snippet 都应为正常。异常 bag 只满足：至少存在一组能解释视频标签的 witness snippets。一个有用神经元还必须补充 frozen host 的剩余错误。

因此“异常神经元”定义为同时满足三项的 CLS 坐标：

1. **Witness responsibility**：在正 bag 的潜在证人片段上对 bag logit 有方向性贡献；
2. **Normal specificity**：在可靠正常 bag 上很少产生同方向贡献；
3. **Host complementarity**：其视频级摘要能解释 host-only 视频预测仍未解释的标签残差。

这三项分别对应弱标签混杂、正常误报和强 host 信息压缩，不是从单篇 baseline 人为制造的三个缺点。

### 7.2 一次训练的 detector

冻结 CLIP 与 host。对层 `l`、坐标 `d` 的 CLS hidden state 做固定 LayerNorm：

\[
x_{t,l,d}=\operatorname{LN}(h_{t,l,:})_d.
\]

每层用 straight-through Top-K gate 产生方向性层证据，再由很小的多尺度 temporal readout 得到神经元概率 `e_t`：

\[
z_t^e=f_\theta\!\left(\sum_l \pi_l
\sum_d m_{l,d}w_{l,d}x_{t,l,d}\right),\qquad
e_t=\sigma(z_t^e).
\]

`m`、`w`、层权重、temporal readout 与后续 head 在同一个 optimizer 中联合学习；没有独立 expert、teacher、student 或 correction checkpoint。

视频判别头只读取十个预先固定的透明统计：host 与 neuron curve 各自的 mean/std/top-10%/max，再加相关系数和平均绝对分歧：

\[
q_v=\sigma\bigl(g_\phi(\operatorname{Pool}_{10}(s^h_v,e_v))\bigr).
\]

最终只用一个 bag-state routed residual：

\[
\delta^-_v=\eta_N\min(0,\operatorname{logit}q_v),
\]

\[
\delta^+_{v,t}=q_v\eta_A\tanh r_\theta(s^h_{v,t},e_{v,t}),
\]

\[
s^{corr}_{v,t}=\sigma\left(\operatorname{logit}s^h_{v,t}
+\delta^-_v+\delta^+_{v,t}\right).
\]

人话解释：模型认为整段更像正常时，只允许统一下压；认为视频确有异常时，才允许神经元证据在时间上做局部修正。它把 Universal 的有效信息流保留下来，但把两个 LR、多专家和所有推理期滤波删除。

为使 complementarity 不只是训练后的解释，先从冻结 host 得到 bag score：

\[
p_v^h=P_k(s_v^h),\qquad r_v^h=\operatorname{sg}(|y_v-p_v^h|).
\]

再用 `r_h` 加权神经元 expert 的 bag loss。host 已经判断正确的视频权重小，host 的正常误报与异常漏报权重大；这样 gate 从训练开始就优先寻找 host 未解决的信息，不需要第二个 teacher 或伪标签。

训练目标包括视频 BCE、host-residual-weighted witness MIL、最终分数 MIL、正常 bag dense-negative BCE 和稀疏约束。它们在同一次 backward 中联合优化。UCF/XD 共享结构和超参数，seed 固定 42。

### 7.3 训练后的 WSVAD 神经元分数

训练结束后再计算解释，不反过来改变 detector。正 bag 的 witness responsibility 由最终 bag pooling 的梯度给出。对坐标 `u=(l,d)`：

\[
A_u=\mathbb E_{v:y=1,t}
\left[\rho_{v,t}\,|x_{v,t,u}|\right],
\]

\[
G_u=\mathbb E_{v:y=1,t}
\left[\rho_{v,t}\left|x_{v,t,u}
\frac{\partial z_v^{corr}}{\partial x_{v,t,u}}\right|\right],
\]

其中 `rho` 是 bag pooling 对各 snippet 的归一化责任。先用 host-only 标量头得到训练标签残差

\[
\varepsilon_v=y_v-\hat y_v^{host},
\]

再定义 host complementarity：

\[
C_u=\left|\operatorname{Corr}
\left(P_k(c_{v,:,u}),\varepsilon_v\right)\right|.
\]

正常 off-target cost 为：

\[
N_u=Q_{0.95,v:y=0,t}
\left(\left|x_{v,t,u}
\frac{\partial z_v^{corr}}{\partial x_{v,t,u}}\right|\right).
\]

最终解释排序：

\[
S_u=\frac{A_uG_uC_u}{N_u+\varepsilon}.
\]

这是对 DNA 三因子思想的 WSVAD 化，不声称是理论最优。它必须通过 `random`、`same-layer random`、`contribution-matched non-selected` 三类控制才能成立。

## 8. 三个创新点应该怎样写

### 创新一：Weak-label Witness Neurons

把异常神经元从“异常视频均值较大”重定义为“承担正 bag 证人责任、避开 dense normal、补充 host 标签残差”的内部单元。核心是弱标签本土化的神经元定义，不是 Top-K gate 本身。

### 创新二：Witness-Routed Frozen-Host Correction

一个联合训练图把稀疏神经元证据、视频状态和 host residual 连起来：normal-like bag 只能下压，anomaly-like bag 才能局部修正。whole-video classifier 是已有构件；创新是用 witness neuron 作为可审计 mediator，而不是宣称首次视频分类或首次残差融合。

### 创新三：Intervention-Verified Temporal Tags

对实际参与修正的 CLS 坐标做 erase/patch，并把 tag 写成四元组：

`<功能角色 | 视觉概念集合 | 时间形态 | 常见上下文>`。

例如 `VETO | {camera shake, dense crowd} | isolated burst | street`。CLIP 单神经元可能多义，因此不强行给一个唯一语义词。文本只在训练后命名，删除 tag 代码后分数必须逐点不变。

## 9. 最少但能讲清故事的实验

### 9.1 先做 Universal 信息验尸

只跑五行：Host、Universal full、去视频抑制、去神经元曲线、去 temporal rules。目的不是重新发表 Universal，而是确认 DSANet/UCF、DSANet/XD 的 +1 pp 主要来自哪条信息流。

### 9.2 新方法性能生死门

先只做 DSANet：

| 设置 | 回答的问题 |
|---|---|
| Host | 同口径起点 |
| Host + video-only | whole-video classifier 本身能拿多少；不是创新 |
| Host + neuron-only | 稀疏神经元曲线是否提供独立增量 |
| Full without host-residual weighting | 为什么必须优先学习 host 的剩余错误 |
| Full WITNESS-VAD | 三类信息是否在一次训练中恢复 ≥1 pp |

硬门槛：UCF frame AUC 和 XD frame AP 相对 paired executable host 都达到至少 +1.0 pp；若只一个数据集达到，不写 universal hard SOTA。

### 9.3 解释生死门

- selected erase 必须显著强于 same-layer random；
- selected patch 必须显著强于 contribution-matched non-selected；
- 效应必须发生在 adapter correction，不得说成改变了 frozen host 内部决策；
- held-out tag retrieval 要优于随机和 activation-only tag；
- 删除 tag 代码后预测逐点相同。

### 9.4 防止“只是压低正常视频”

主指标外只保留必要诊断：Cross-AUC、Macro Within-AUC、normal FPR@95% TPR、abnormal-video-only AUC/AP、平均校正量。若增益全部来自 Cross/FPR，论文只能说 false-alarm calibration，不能说 temporal localization 改善。

## 10. 反方审查

| 反方问题 | 当前回答 | 必须补的证据 |
|---|---|---|
| 只是把 Universal 重写了一遍 | 信息源相同，但训练图、神经元定义和验证对象改变 | behavioral parity + 模块数/训练阶段对比 |
| 视频分类早有人做 | 承认，它是性能桥 | video-only 对照；创新不落在此处 |
| 神经元只是 probe correlation | 用 residual complementarity 与 intervention | matched-contribution control、双向 patch |
| 单个 CLIP 维度是多义的 | tag 是组合集合，不声称一维一概念 | held-out tag fidelity 与反例展示 |
| 增益只靠压正常 | 指标分轴 | Cross/Within/FPR/abnormal-only |
| 多尺度 temporal readout 是新模块拼接 | 它不是贡献，只替代旧手工滤波 | 去 temporal readout；不得单独包装 |
| 一次训练是否真的一次 | 所有可训练参数同一 optimizer、同一 checkpoint | graph/optimizer/checkpoint 单元测试 |

## 11. 最终建议

值得继续，但必须换证据，不是换叙事。

- 保留：frozen host、12 层 CLS、directional sparse expert、训练标签驱动的视频级单侧抑制、干预验证。
- 重塑：把三条曲线和两个 LR 合成单个稀疏 expert + 单个 video head + 单个 residual；把神经元选择改成 witness/normal/complementarity 三条件。
- 删除：masked context predictor、two-axis global kappa、exact-OR/Markov chain、伪标签 student、评估时再拟合、全部 morphology 与 temporal advance。
- 文本定位：只做解释 tag；I3D baseline 的成功说明检测主干不能依赖文本语义才能成立。

## 12. 已核实的一手参考

1. Sultani, Chen, Shah. [*Real-World Anomaly Detection in Surveillance Videos*](https://openaccess.thecvf.com/content_cvpr_2018/html/Sultani_Real-World_Anomaly_Detection_CVPR_2018_paper.html). CVPR 2018.
2. Tian et al. [*Weakly-Supervised Video Anomaly Detection with Robust Temporal Feature Magnitude Learning*](https://openaccess.thecvf.com/content/ICCV2021/papers/Tian_Weakly-Supervised_Video_Anomaly_Detection_With_Robust_Temporal_Feature_Magnitude_Learning_ICCV_2021_paper.pdf). ICCV 2021.
3. Park et al. [*Normality Guided Multiple Instance Learning for Weakly Supervised Video Anomaly Detection*](https://openaccess.thecvf.com/content/WACV2023/html/Park_Normality_Guided_Multiple_Instance_Learning_for_Weakly_Supervised_Video_Anomaly_WACV_2023_paper.html). WACV 2023.
4. Lv et al. [*Unbiased Multiple Instance Learning for Weakly Supervised Video Anomaly Detection*](https://openaccess.thecvf.com/content/CVPR2023/html/Lv_Unbiased_Multiple_Instance_Learning_for_Weakly_Supervised_Video_Anomaly_Detection_CVPR_2023_paper.html). CVPR 2023.
5. Zhang et al. [*Exploiting Completeness and Uncertainty of Pseudo Labels for Weakly Supervised Video Anomaly Detection*](https://openaccess.thecvf.com/content/CVPR2023/html/Zhang_Exploiting_Completeness_and_Uncertainty_of_Pseudo_Labels_for_Weakly_Supervised_CVPR_2023_paper.html). CVPR 2023.
6. Tan et al. [*Overlooked Video Classification in Weakly Supervised Video Anomaly Detection*](https://openaccess.thecvf.com/content/WACV2024W/RWS/papers/Tan_Overlooked_Video_Classification_in_Weakly_Supervised_Video_Anomaly_Detection_WACVW_2024_paper.pdf). WACV Workshops 2024.
7. Chen et al. [*Prompt-Enhanced Multiple Instance Learning for Weakly Supervised Video Anomaly Detection*](https://openaccess.thecvf.com/content/CVPR2024/html/Chen_Prompt-Enhanced_Multiple_Instance_Learning_for_Weakly_Supervised_Video_Anomaly_Detection_CVPR_2024_paper.html). CVPR 2024.
8. Acharya et al. [*The Road Less Seen: Segment Exploration for Weakly Supervised Video Anomaly Detection*](https://openaccess.thecvf.com/content/CVPR2026/html/Acharya_The_Road_Less_Seen_Segment_Exploration_for_Weakly_Supervised_Video_CVPR_2026_paper.html). CVPR 2026.
9. Dou et al. [*DNA: Uncovering Universal Latent Forgery Knowledge*](https://arxiv.org/abs/2601.22515). arXiv:2601.22515, 2026.
10. Li et al. [*Latent Anomaly Knowledge Excavation: Unveiling Sparse Sensitive Neurons in Vision-Language Models*](https://arxiv.org/abs/2604.07802). arXiv:2604.07802, 2026.
11. Kan et al. [*V-FIND: Revealing the Intrinsic Forgery Knowledge Encoded in Video Forgery Detectors*](https://arxiv.org/abs/2608.03008). arXiv:2608.03008, 2026.
12. Gandelsman, Efros, Steinhardt. [*Interpreting the Second-Order Effects of Neurons in CLIP*](https://arxiv.org/abs/2406.04341). arXiv:2406.04341, 2024.
13. Doshi, Yilmaz. [*Towards Interpretable Video Anomaly Detection*](https://openaccess.thecvf.com/content/WACV2023/html/Doshi_Towards_Interpretable_Video_Anomaly_Detection_WACV_2023_paper.html). WACV 2023.
14. Ye et al. [*VERA: Explainable Video Anomaly Detection via Verbalized Learning of Vision-Language Models*](https://openaccess.thecvf.com/content/CVPR2025/papers/Ye_VERA_Explainable_Video_Anomaly_Detection_via_Verbalized_Learning_of_Vision-Language_CVPR_2025_paper.pdf). CVPR 2025.
