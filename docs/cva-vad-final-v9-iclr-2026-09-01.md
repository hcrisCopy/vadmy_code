# CVA-VAD v9：用上下文违背神经元审计冻结的弱监督视频异常检测器

> **状态更新（2026-09-02）：本方案已由 DSANet E1 正式实验判定 no-go。** C3 contextual-directional evidence 在 UCF/XD 均未赢 C1/C4，五组 `kappa_cross` 全为 0；不要继续调参或进入原 E2。后续正式方案见 `docs/witness-vad-final-v10-iclr-2026-09-02.md`，完整重置调研见 `docs/wsvad-neuron-research-2026-09-02.md`。本文保留为失败审计，不再修改其原公式。

> 目标会议：ICLR 2027。本文档是 v7 与 v8 对比后的唯一实施方案。
>
> 一句话主张：**弱监督视频异常检测不应抛弃强 host、再用粗标签重学一个检测器；它需要从冻结视觉表征中寻找正常上下文无法解释的定向神经元证据，据此审计 host 何时过火、何时需要视频内重排，并验证每次校正确由这些神经元驱动。**

## 1. v7 与 v8 的取舍结论

| 方面 | v8 更好的部分 | v8 的问题 | v9 决策 |
|---|---|---|---|
| 领域动机 | 弱监督、上下文条件性、正常视频可靠锚讲得比 v7 具体 | 把“事件性”强行推成第三模块 | 保留弱监督与上下文两条主命题；事件性由 host 和 context predictor 承担，不再新增后处理 |
| 旧实验归因 | 清楚指出 host、正常负证据、全层方向信息是真正信息源 | 重新把不稳定的 event propagation 写成“保底来源” | 只保留前三类稳定信息；event 仅作失败分析 |
| 沈飞信息差 | LAKE/DNA/V-FIND 的“定位—干预”迁移叙事适合 ICLR | Kneedle、集合差、效应量机械堆叠，形成多阶段发现流水线 | 迁移“内部知识定位与干预验证”的思想，不照搬所有算子 |
| host 处理 | 明确冻结并保留正式分数 | “有界残差结构上不可能掉点”是错误的 | host 始终冻结；有界和预算只限制风险，不承诺指标必然不降 |
| 神经元发现 | 强调全 12 层和方向性 | host top-k 伪标签 + Kneedle 字典仍受弱标签噪声影响，并重新出现训练/统计/冻结多阶段 | 稀疏 field 与校正器一次联合学习；解释排序在训练后计算，不参与反复选模 |
| 校准 | 正常负证据与旧增益对应得好 | “normality certificate”没有覆盖保证，不能叫 certificate；对 median 以下证据直接解释成正常也不严谨 | 改成 normal-support audit，不声称统计证书；视频级校正只允许向下抑制 |
| 时间建模 | 认识到异常具有时间结构 | 任意的 `0.35 × q75`、clip 7–21 和 median filter 是新后处理 | 删除整个 event-scale 模块；视频内校正用零均值结构隔离时间重排 |
| 可解释性 | 干预和跨 host 叙事好 | 与 event scale 拼成一个创新，逻辑不闭合 | 单独成为第三创新：解释“为什么改分”，用 erase/patch/匹配控制验证 |
| 评估 | 有 go/no-go 意识 | 没有拆开 pooled 指标中的跨视频与视频内贡献 | 继承 v7 的 Cross/Within 分解，防止正常视频压分冒充定位改善 |

结论：**论文叙事以 v8 为底，方法骨架以 v7 为底，再做两处关键收缩：视频级只做单侧正常抑制；第三创新只做可验证校正，不做事件后处理。**

## 2. ICLR 定位

本文是 **cross-domain technique paper**：把神经元级内部知识定位与干预验证，从图像异常/伪造检测迁移到 WSVAD，并完成 context-conditional、coarse-label 和 frozen-host 三项领域适配。

ICLR 版本不能写成“在 DSANet 上加模块并提高 1 点”。核心科学问题应是：

> **Can context-conditioned neurons expose corrective evidence that is lost in the scalar score of a frozen weakly supervised video anomaly detector?**

ICLR 关心的是 learned representation 是否包含这种增量知识、这种知识能否安全改变决策、干预是否支持功能解释。官方 AUC/AP 超 host 至少 1 点是方法有效性的硬证据，但不是研究问题本身。

## 3. 从任务出发的三个能力缺口

### 缺口一：现有表示只能说“像不像异常”，不能说“相对当前上下文是否反常”

WSVAD 中同一动作是否异常取决于场景和前后行为。raw activation、全局均值偏离和类别均值差都是边际统计，不能回答某个 CLS 坐标在当前正常时间上下文中本应怎样响应。正常视频的所有 snippet 均可视为正常，这是弱监督设定中唯一可靠的细粒度支撑。

### 缺口二：强 host 给出标量判断，却没有自我审计能力

host score 已经包含时序网络、MIL 和正常性模块学到的强排序，抛弃它会重演 P1。另一方面，单个分数压缩了上游表征，无法表达“host 高分是否得到内部违背证据支持”。旧系统最大的稳定收益来自正常负证据，说明真正缺失的是对 host 误报的刹车，而不是另一个全量 anomaly scorer。

### 缺口三：内部神经元解释尚未与一次具体决策修正绑定

LAKE、DNA 和 V-FIND 表明异常或伪造知识可集中在稀疏内部单元，并可通过干预验证；但它们主要处理 input-intrinsic 线索。WSVAD 需要的是 context-conditional 单元，而且解释必须回答：哪个方向单元使哪个 snippet 的 host 分数上调或下调。单纯 heatmap、类别探针或给神经元命名都不足以证明这一点。

## 4. 三个创新点

### 创新一：Masked Contextual Violation Field

从正常视频学习每个 CLS 神经元在被遮蔽目标之外的时间上下文条件分布，以正、负标准化预测残差定义方向违背单元。稀疏 field 不是先用 host top-k 生成伪标签再固定，而是在保留 host 的校正目标下联合学习，因此选中的是对 host 剩余错误有用的 correction neurons。

### 创新二：Asymmetric Two-Axis Host Auditing

将 host logit 的校正拆为两个职责正交的项：

- cross-video audit：只有“host 高分但内部证据仍符合正常支撑”时才允许向下抑制；
- within-video redistribution：以严格零均值的局部项重新分配同一视频内的时间排序。

两项均有界并受统一校正预算约束。这样既保留旧系统最有效的正常抑制，又能明确区分官方 pooled gain 来自跨视频校准还是时间定位。

### 创新三：Intervention-Verified Correction Fields

用某个方向单元对实际校正量的梯度贡献定义解释对象，再通过 context replacement、readout erase、CLIP activation patching 和三类匹配控制验证。文本只负责给通过验证的 field 生成组合 tag，不参与检测。

三项创新只有一条因果链：

\[
\text{normal context expectation}
\rightarrow
\text{directional violation evidence}
\rightarrow
\text{bounded host correction}
\rightarrow
\text{intervention-verified explanation}.
\]

## 5. 输入与固定边界

对视频 \(v\)：

\[
H_v\in\mathbb R^{T_v\times12\times768},\quad
s_v^h\in(0,1)^{T_v},\quad
y_v\in\{0,1\},\quad
M_v\in\{0,1\}^{T_{max}}.
\]

- 神经元是 CLIP ViT-B/16 层级 CLS 坐标 \(u=(l,d)\)；
- \(s^h\) 是 host 原 evaluator 使用的正式 detection score；
- \(M_{v,t}=1\) 表示真实 snippet，0 表示 batch padding，不是异常 mask；
- \(G_t=\{j:|j-t|\le g\}\) 是上下文预测时不可见的目标保护区间；
- CLIP 和 host 从开始到结束始终冻结；文本不进入训练或推理。

可训练部分只有 context predictor、稀疏 field 权重和两个校正强度。一次前向、一次反向、一个 optimizer，不存在“训 detector—冻结—再训校正头”的切换。

## 6. 方法

### 6.1 正常上下文条件分布

每层先做固定 LayerNorm：

\[
x_{v,t,l,:}=\operatorname{LN}(h_{v,t,l,:}).
\]

只用 \(G_t\) 外的有效位置预测目标神经元：

\[
(\mu_{v,t,u},\log\sigma_{v,t,u})
=g_\psi\!\left(H_{v,\{j:M_{v,j}=1,j\notin G_t\}}\right).
\]

query 只含位置编码，不能读取目标内容。正常视频上的条件 NLL 为：

\[
\mathcal L_{ctx}
=\frac{1}{|\Omega_N|}
\sum_{v:y_v=0}\sum_{t,u}M_{v,t}
\left[
\frac{(x_{v,t,u}-\mu_{v,t,u})^2}{2\sigma_{v,t,u}^2}
+\log\sigma_{v,t,u}
\right].
\]

校正损失读取 \(\operatorname{sg}(\mu),\operatorname{sg}(\sigma)\)，不能用异常 bag 的粗标签篡改正常参考系；\(\mathcal L_{ctx}\) 在同一次反向中正常更新 predictor。

实现采用共享 masked temporal encoder 加低秩 layer-specific heads，不为 9216 个坐标各建独立网络；否则参数量本身会成为替代 detector。

### 6.2 定向违背单元与联合稀疏 field

\[
r_{v,t,u}=\frac{x_{v,t,u}-\operatorname{sg}(\mu_{v,t,u})}
{\operatorname{sg}(\sigma_{v,t,u})+\epsilon},
\]

\[
v_{v,t,u,+}=\operatorname{ReLU}(r_{v,t,u}-\delta),\qquad
v_{v,t,u,-}=\operatorname{ReLU}(-r_{v,t,u}-\delta).
\]

\(\delta\) 是所有数据集共用的尾部容忍界；其值只在训练协议中确定，不从测试集选择。方向权重使用稀疏归一化：

\[
\pi=\operatorname{entmax}_{1.5}(\omega),\qquad
a_{v,t}=\sum_{u,q\in\{+,-\}}\pi_{u,q}v_{v,t,u,q}.
\]

用正常训练视频的 running median/MAD 标准化：

\[
e_{v,t}=\frac{a_{v,t}-m_N}{1.4826\,\operatorname{MAD}_N+\epsilon}.
\]

每个 batch 只在 normal snippets 上计算 stop-gradient batch median/MAD，并更新 running statistics；推理时固定最终 running values。这样统计量会随 \(\pi\) 的学习同步更新，却不会形成额外训练阶段。这里不使用 host top-k 发现池、不使用 Kneedle、不在训练中固定字典。\(\pi\) 与校正器联合学习；训练结束后再根据实际校正贡献排序解释 field。

### 6.3 单侧视频级 normal-support audit

先得到 host 的视频级异常置信：

\[
h_v=P_k(s^h_{v,1:T_v}),
\]

再由神经元证据计算视频级违背摘要，并用正常训练视频的稳健统计量标定：

\[
q_v=P_k(e_{v,1:T_v}),\qquad
z_v=\frac{q_v-m_q^N}{1.4826\,\operatorname{MAD}_q^N+\epsilon}.
\]

正常支撑强度为：

\[
n_v=\sigma\!\left(\tau_N-z_v\right).
\]

\(\tau_N\) 来自 normal-video \(z_v\) running reservoir 的预先固定分位规则，随训练统计更新，测试前锁定。视频级校正只允许向下：

\[
\Delta_v^{cross}
=-\alpha_v\kappa_v\,h_v n_v\le0,
\qquad \kappa_v\in[0,1].
\]

只有 host 自己给出较高视频分数、而内部神经元仍显示正常支撑时，抑制才明显。这直接对应旧系统中最稳定的正常负证据，并避免没有实验支持的正向 video shift。

### 6.4 零均值视频内重排

\[
u_{v,t}=\alpha_t\kappa_t\tanh(e_{v,t}),
\qquad \kappa_t\in[0,1],
\]

\[
\Delta_{v,t}^{within}
=u_{v,t}-\frac{\sum_jM_{v,j}u_{v,j}}{\sum_jM_{v,j}}.
\]

因此 \(\sum_tM_{v,t}\Delta_{v,t}^{within}=0\)。它在 **logit space** 中与视频常数方向正交，不能直接加入整段常数偏移；由于后续 sigmoid 非线性，它不保证视频平均概率逐点守恒，所以仍需用 A3 的 Cross/Within 指标实测归因，不能只凭公式宣称纯定位。

最终分数：

\[
b_{v,t}=\operatorname{logit}(\operatorname{clip}(s^h_{v,t},\varepsilon,1-\varepsilon)),
\]

\[
s_{v,t}^{corr}=\sigma\!\left(
b_{v,t}+\Delta_v^{cross}+\Delta_{v,t}^{within}
\right).
\]

\(\kappa_v,\kappa_t\) 用 projected gradient 更新：它们以 0 初始化，每次 optimizer step 后投影到 \([0,1]\)。因此初始输出逐点严格等于 host，同时在 0 处仍有一阶梯度，不会出现 \(\tanh^2(\gamma)\) 的零梯度死点。\(|\Delta^{cross}|\le\alpha_v\)，\(|\Delta^{within}|\le2\alpha_t\)。这些上界限制最坏改动，但不构成性能不下降保证。

### 6.5 非对称弱监督与校正预算

正常 bag 提供 dense negative supervision；异常 bag 只提供 top-k positive supervision：

\[
\mathcal L_{ws}
=-\sum_v\left[
y_v\log P_k(s_v^{corr})
+(1-y_v)\frac{\sum_tM_{v,t}\log(1-s_{v,t}^{corr})}{\sum_tM_{v,t}}
\right].
\]

定义平均改动：

\[
\mathcal C
=\frac1N\sum_v\left[
|\Delta_v^{cross}|+
\frac{\sum_tM_{v,t}|\Delta_{v,t}^{within}|}{\sum_tM_{v,t}}
\right].
\]

最终目标：

\[
\mathcal L
=\mathcal L_{ws}
+\lambda_{ctx}\mathcal L_{ctx}
+\lambda_\rho\max(0,\mathcal C-\rho).
\]

\(\rho\) 是唯一的 correction budget。它回答“允许审计器改动 host 到什么程度”，替代旧系统中大量 fusion、suppression 和 event 参数。

## 7. 可解释性：从神经元 tag 到校正验证

解释对象是方向单元对一次实际校正的贡献：

\[
C_{v,t,u,q}
=v_{v,t,u,q}
\frac{\partial(\Delta_v^{cross}+\Delta_{v,t}^{within})}
{\partial v_{v,t,u,q}}.
\]

按 \(|C|\) 聚合得到 correction field，而不是把单个不稳定坐标写成固定语义神经元。tag 描述：

\[
[\text{suppress/raise host}]
+[\text{actor/action}]
+[\text{scene/temporal context}].
\]

文本或 VLM 只提出候选 tag；独立 probe set 上由 held-out fidelity 决定是否保留。每个主文展示 field 必须同时通过：

1. context replacement：目标 \(x_t\) 不变，仅替换匹配 donor context，观察 \(\mu\to r\to e\to\Delta\)；
2. readout erase：置零目标 field，目标位置校正变化强于非目标位置；
3. activation patch：重放 CLIP，将目标坐标替换为匹配正常 donor 激活；
4. 三类控制：同层随机坐标、贡献幅值匹配但未选中的坐标、随机正常 donor；
5. 删除全部 tag/文本代码，检测输出逐点不变。

只能声称 field 对模型校正具有功能贡献，不能声称现实世界因果或 seed-invariant 单神经元语义。

## 8. 论文逻辑骨架

| 环节 | 内容 |
|---|---|
| Research background | WSVAD 用廉价视频标签定位异常，但强 detector 的标量输出缺乏内部证据审计，部署时正常误报又无法解释 |
| Limitation 1 | 边际激活或全局正常偏离不能表达 context-conditional anomaly |
| Limitation 2 | 重新训练 detector 会丢失 strong host；无约束融合会破坏已有排序 |
| Limitation 3 | 现有解释没有绑定一次具体分数修正，也缺少内部干预控制 |
| Key idea | 用正常上下文无法解释的方向神经元残差，作为冻结 host 的增量审计证据 |
| Challenge 1 | 仅靠正常 bag 学习无局部泄漏的神经元条件预期 |
| Challenge 2 | 在粗标签下修正 host，同时隔离跨视频校准与视频内定位 |
| Challenge 3 | 证明被解释的 field 真正驱动校正，而不是相关性热图 |
| Module 1 | Masked Contextual Violation Field |
| Module 2 | Asymmetric Two-Axis Host Auditing |
| Module 3 | Intervention-Verified Correction Fields |
| Contribution 1 | context-conditional correction neurons 的 WSVAD 定义与学习机制 |
| Contribution 2 | host-preserving、单侧且两轴解耦的校正机制 |
| Contribution 3 | 与具体校正绑定的语义 tag 和受控干预证据 |

四条一致性检查全部闭合：limitations 对应 key idea，三个 challenge 均由 key idea 自然产生，三个 module 一一解决 challenge，三条 contribution 均有方法和实验支撑入口。

## 9. 新颖性边界

| 最近工作 | 已完成部分 | v9 必须证明的差异 |
|---|---|---|
| BN-WVAD | 用 marginal feature-from-mean 修正受噪声影响的 classifier | masked conditional residual 必须优于同 readout 下的 marginal DFM |
| RPC | 用正常 prototype deviation 校准冻结 pose-flow VAD | WSVAD coarse labels、方向 CLS neurons、两轴分解与干预解释 |
| LAKE | 用正常支撑挖掘 VLM anomaly-sensitive neurons | 从 input-intrinsic marginal sensitivity 改为 temporal context-conditional correction evidence |
| DNA / V-FIND | 定位 forgery neurons 并做干预 | correction-specific attribution、WSVAD coarse labels 与 frozen-host auditing |
| SteerVAD | 在冻结 MLLM 中定位 heads 并用全局上下文 steering | masked normal conditional distribution、CLS-coordinate field 和 host score correction |
| ConFidNet / SelectiveNet | 学习 failure confidence 或 risk-coverage | WSVAD 的非对称监督、直接 correction、Cross/Within 分解 |

不能声称“首次异常神经元”“首次 context-aware VAD”“首次冻结 VAD 校准”或“严格统计证书”。

## 10. 实验方案：只做能支撑三条贡献的实验

### 10.1 R0：身份与评估口径

- 逐视频缓存正式 host score，并与 12 层 hidden、官方 feature length、frame GT 对齐；
- \(\kappa_v=\kappa_t=0\) 时逐点等于 host；
- 先对 host 输出计算 pooled、Cross、Within 和正常视频误报指标，建立统一 evaluator。

### 10.2 R1：创新一的决定性对照

所有变体使用同一个 correction readout，只替换证据：

| ID | 证据 |
|---|---|
| C0 | raw directional activation |
| C1 | global directional normal z-score / BN-WVAD-style DFM |
| C2 | masked contextual absolute residual |
| C3 | masked contextual directional residual |
| C4 | SteerVAD-style global-context controller |

创新一成立必须同时满足：C3 的 normal conditional NLL 优于 C1；context replacement 会改变 C3 而不改变目标 raw activation；C3 在最终校正指标上优于 C0/C1/C2/C4。

### 10.3 R2：创新二的核心消融

| ID | 模型 | 回答的问题 |
|---|---|---|
| A0 | frozen Host | 正式起点 |
| A1 | Host + contextual field + symmetric single residual | 普通残差融合能做到哪里 |
| A2 | Host + cross-video audit only | 正常误报抑制贡献 |
| A3 | Host + within-video redistribution only | 纯时间重排贡献 |
| A4 | A2 + A3 | 两轴是否互补 |
| A5 | A4 去 correction budget | 预算是否保护 strong host |
| A6 | A4 把单侧 cross correction 改成双向 | 旧实验提示的单侧性是否真实 |

主结果先跑 UCF-Crime/XD-Violence × DSANet/DeSC。主线成立后，再加 LaGoVAD 或 VadCLIP 作为第三 host；不先跑四 host 的笛卡尔积。

### 10.4 R3：创新三的验证

- top correction fields 的 held-out tag fidelity，全部报告而非挑成功案例；
- context replacement 的 \(\Delta\mu,\Delta r,\Delta e,\Delta s\)；
- readout erase 与 activation patch 对比三类控制；
- selected field、same-layer random、contribution-matched non-selected 的等预算比较；
- 跨 host：在 source host 学习 context predictor 与 field，target host 只拟合两个校正强度；另报告完全零拟合 transfer 作为压力测试。

### 10.5 指标按能力分轴

| 指标 | 证明什么 |
|---|---|
| 官方 pooled AUC/AP | 总体排名与 +1 性能目标 |
| Cross-AUC / video-constant AP | 视频级 normal-support audit |
| Macro Within-AUC | 异常视频内部时间排序 |
| normal FPR@固定 TPR、正常视频 top-score | 实际误报抑制 |
| 平均 \(|\Delta|\)、gain-budget curve | 校正是否保守 |
| erase/patch effect 与匹配控制 | 神经元功能特异性 |

如果 A2 提升 pooled/Cross、A3 不提升 Within，论文只能声称可靠性校准；如果 A3 提升 Within，才能声称时间定位改善。mAP 仅作补充，不作为主目标。

## 11. Reviewer 会直接攻击的点与预先防守

| 攻击 | 防守实验 |
|---|---|
| “只是 BN-WVAD 的 DFM 换成 CLIP” | C1 vs C3，同一 correction readout；context replacement |
| “只是冻结 detector 后做 score calibration” | RPC-style prototype control；within zero-mean branch；neuron intervention |
| “pooled AUC 靠正常视频整体压分” | Cross/Within/A2/A3 分解，主动披露贡献来源 |
| “所谓神经元只是相关 feature dimensions” | contribution-matched control、readout erase、backbone patch |
| “field 只对一个 host 有效” | source-field → target-host transfer |
| “方法仍是多阶段拼装” | 一次联合训练；正常 running stats 只作固定标定；文本与干预均在检测之后 |
| “为什么不用文本” | 检测任务不需要类别命名；文本删除不改变输出；I3D/纯视觉强方法作为领域证据 |

## 12. Go/No-Go

1. **G0 Host identity**：不能逐点复现 host，停止。
2. **G1 Context mechanism**：C3 不优于 C1/C4，删除 contextual claim，当前 ICLR 故事不成立。
3. **G2 Performance**：A4 不优于 host，停止 tag 工作；不要靠后处理补分。
4. **G3 Attribution**：A4 不优于 A2/A3 或 gain 只来自 A2，按真实结果收缩贡献。
5. **G4 Intervention**：patch/erase 不强于匹配控制，tag 降级为 descriptor。
6. **G5 Generality**：第二 host 失败，不能声称 host-agnostic auditor。

## 13. Idea 评估

### Fatal flaws

| 风险 | 严重度 | 防守 |
|---|---|---|
| 与 BN-WVAD/RPC/SteerVAD 的方法边界可能不够大 | MAJOR | C1/C4/RPC-style 直接同口径对照，加 intervention 与两轴分解 |
| pooled metric 可能把视频级 suppression 误写成定位 | MAJOR | Cross/Within 分解与 A2/A3 独立消融 |

### 五维判断

| 维度 | 判断 | 依据 |
|---|---:|---|
| Higher | 7/10，机制成立但尚未由 v9 实验确认 | 旧系统已证明 host + 正常负证据 + 方向 neurons 有增益；v9 的干净归因仍待 A0–A4 |
| Faster | 6/10 | host/CLIP 缓存复用，adapter 轻量；不是核心 claim |
| Stronger | 8/10，机制性 | 有界预算、单侧 audit、第二 host 与 transfer 直接验证 |
| Cheaper | 7/10 | 不新增 snippet 标注，不重训 host；context predictor 只用已有正常视频 |
| Broader | 8/10，机制性 | 把 neuron discovery/intervention 迁移到 context-conditional WSVAD，并探索跨 host |

Verdict：**Accept with Revisions，worth pursuing pending C3>C1/C4 and A4>A0。** v8 的 event-scale 版本不建议继续；v9 的 ICLR 潜力来自 representation auditing，而不是后处理增益。

## 14. ICLR 提交约束

ICLR 2027 abstract deadline 为 2026-09-18，paper deadline 为 2026-09-25，主文最多 9 页。只有在 G0–G3 已完成、核心结果可复现时才应提交；否则公开 OpenReview 记录会留下一个未完成版本。

ICLR 2027 要求在论文和提交表单中披露 AI 用途。当前这类“文献整理、研究方法反馈、数学公式与实验设计辅助”属于官方明确要求披露的范围；最终作者必须逐条验证本文档、实验代码、引用和论文表述，并承担全部责任。

## 15. 核实入口

- [ICLR 2027 Call for Papers](https://www.iclr.cc/Conferences/2027/CallForPapers)
- [ICLR 2027 Author Guidelines](https://iclr.cc/Conferences/2027/AuthorGuidelines)
- [ICLR 2027 AI Policy for Authors](https://iclr.cc/Conferences/2027/AIPolicyForAuthors)
- [BN-WVAD](https://arxiv.org/abs/2311.15367)
- [RPC: Reliability-Aware Prototype Calibration](https://arxiv.org/abs/2606.20312)
- [LAKE](https://arxiv.org/abs/2604.07802)
- [DNA](https://arxiv.org/abs/2601.22515)
- [V-FIND](https://arxiv.org/abs/2608.03008)
- [SteerVAD](https://arxiv.org/abs/2602.24021)
- [Frame-Level Evaluation in WSVAD Mostly Measures Video-Level Ranking](https://arxiv.org/abs/2608.21854)
- [Auditing Frame-Level AUC in WSVAD](https://arxiv.org/abs/2608.11985)
