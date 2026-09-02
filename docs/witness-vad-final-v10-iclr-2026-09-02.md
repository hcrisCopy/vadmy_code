# WITNESS-VAD v10：用弱标签证人神经元审计冻结的视频异常检测器

> 目标会议：ICLR 2027。
> 状态：F0--F2 已通过；F3 已判定 NO-GO；F3.1 已排除“只因 eta_A 太小”，下一步必须重构 witness evidence/router，尚未进入 F4。CVA-VAD v9 的 contextual-directional evidence 已由 E1 判定 no-go。
> 硬目标：同一权重、同一 evaluator 的 paired executable baseline 上，DSANet/UCF frame AUC 与 DSANet/XD frame AP 均至少 `+1.0 pp`。

## 1. 一句话主张

> **弱监督 VAD 缺的不是另一个从粗标签重学的检测器，而是识别哪些冻结视觉神经元真正承担异常视频的证人责任、哪些能否决正常视频的误报、哪些补充了强 host 已丢失的信息；这些神经元应在一次训练中直接路由 host 校正，并由时间定位的 activation intervention 验证。**

## 2. 为什么 v9 必须结束

v9 的 masked context predictor 在 UCF/XD 都学到了更好的 conditional NLL，但 C3 detection gain 近零，且没有稳定赢 C1/C4。五组 `kappa_cross` 全部回到 0。结论很明确：**“能预测上下文”没有转化为“拥有 host 缺失的异常证据”。**

因此 v10 不调 v9 的窗口、overlap、budget、学习率或 smoothing；直接删除 contextual predictor、directional residual field 和 two-axis kappa。

F0 对历史 Universal 做了同缓存 leave-one-out 验尸。Full 相对 host 在 UCF AUC
提升 1.058 pp、XD AP 提升 1.215 pp；删除局部神经元校正分别损失 0.709/1.018 pp，
删除视频级 suppression 分别损失 0.242/0.776 pp，删除手工 temporal rules 分别损失
0.292/0.259 pp。只保留 suppression 时，两数据集的视频内排序都弱于 host。结论是：
**必须保留“视频级判断是否需要改 + 神经元决定在哪里改”的信息分工，但不继承手工
temporal morphology。**

## 3. 领域能力缺口

### 缺口一：视频标签无法直接定义异常神经元

正常视频的所有 snippet 都是可靠负例；异常视频只有“至少有异常”的承诺，大部分 snippet 仍可能正常。图像分类式的正负均值、Fisher ratio 或 effect size 会把整个异常视频当正类，容易找到场景和视频类别单元，而不是异常时刻的功能单元。

### 缺口二：强 host 的标量输出丢失了可纠错内部证据

DSANet 已经有强时序与正常性建模，不能替换。Universal 的历史增益说明，host curve 与全层神经元 curve 的视频级分布和分歧仍能识别一部分正常误报。缺的是“哪些内部证据有资格修正 host”，不是更大的 detector。

### 缺口三：WSVAD 的解释没有绑定到一次真实校正

语义文本、scene graph 和 heatmap 可以描述异常，但不能证明某个 CLS 坐标导致了分数变化。沈飞路线证明稀疏单元与 activation swap 是有效范式；WSVAD 还缺 latent witness、dense normal 和时间定位三项适配。

## 4. 三个创新点

### 创新一：Weak-label Witness Neuron

把神经元定义为同时满足三项的 CLIP CLS 坐标：

1. 对正 bag 的 latent witness snippets 有方向性责任；
2. 在正常 bag 上 off-target contribution 小；
3. 能解释 host-only 视频预测的标签残差。

这不是“异常视频均值较大”的普通 probe，而是针对 WSVAD 标签结构定义的功能神经元。

### 创新二：Witness-Routed Frozen-Host Correction

一个联合训练图产生神经元曲线、视频状态和最终 residual。normal-like bag 只允许整体下压；anomaly-like bag 才允许局部神经元修正。视频分类是已有技术，本创新只主张：**把 weak-label witness neuron 作为 frozen-host correction 的可审计 mediator。**

### 创新三：Intervention-Verified Temporal Tag

解释对象是实际 correction contribution。对选中 CLS 坐标做 erase 和双向 activation patch，与同层随机、等预算随机、贡献匹配未选中坐标比较。tag 是 `<功能角色 | 概念集合 | 时间形态 | 上下文>`，允许一个坐标多义；文本不参与检测。

三条贡献是一条链，不是三块拼装：

\[
\text{weak bag semantics}
\rightarrow \text{witness neurons}
\rightarrow \text{routed host correction}
\rightarrow \text{verified temporal tags}.
\]

## 5. 固定边界

对视频 `v`：

\[
H_v\in\mathbb R^{T_v\times12\times768},\quad
s^h_v\in(0,1)^{T_v},\quad y_v\in\{0,1\}.
\]

- 神经元定义固定为 CLIP ViT-B/16 每层 CLS hidden state 的一个维度 `(layer, dimension)`；
- host score 必须是 baseline 官方 evaluator 使用的正式 score；
- CLIP 与 host 全程冻结；
- 所有新参数从第一个 step 到最后一个 step 使用同一个 optimizer；
- 训练后才做 attribution、intervention 和 tag，不把解释阶段变成第二个训练阶段；
- 文本、测试 GT、测试视频标签不进入推理公式。

## 6. 方法

### 6.1 全层稀疏方向证据

每个 hidden token 做固定 LayerNorm：

\[
x_{v,t,l,:}=\operatorname{LN}(h_{v,t,l,:}).
\]

每层使用与 Universal 已验证实现一致的 straight-through Top-K gate，方向由可学习有符号权重给出：

\[
a_{v,t,l}=\sum_d m_{l,d}w_{l,d}x_{v,t,l,d}.
\]

再学习层权重与一个小型多尺度 temporal readout，输出神经元概率：

\[
z^e_{v,1:T}=f_\theta([\pi_1a_{:,1},\ldots,\pi_{12}a_{:,12}]),
\qquad e_{v,t}=\sigma(z^e_{v,t}).
\]

`f_theta` 固定复用 Universal 已验证的最小时序结构：`Conv1D(k=3,d=1,12→64) → GELU → Conv1D(k=3,d=2,64→64) → GELU → Conv1D(1×1,64→1)`。它负责替代 Universal 的 max/median/Gaussian/advance，不作为单独创新，也不做结构搜索。禁止在推理后再做任何滤波。

### 6.2 透明的视频状态头

固定十维 pooling：

\[
\Phi_v=[\mu,\sigma,P_{10\%},\max]_{s^h}
\oplus[\mu,\sigma,P_{10\%},\max]_e
\oplus[\operatorname{corr}(s^h,e),\mathbb E|s^h-e|].
\]

视频异常概率为：

\[
q_v=\sigma(w_q^\top\Phi_v+b_q).
\]

这十维特征只保留 Universal 已证明有用的“分布 + top evidence + agreement”信息，不再使用 40 多维分位数、两套 LR 和评估期现场拟合。

### 6.3 单一 witness-routed residual

视频判为 normal-like 时，只允许统一下压：

\[
\delta_v^-=\eta_N\min(0,\operatorname{logit}q_v),\qquad \eta_N=\operatorname{softplus}(\tilde\eta_N)>0.
\]

视频判为 anomaly-like 时，神经元证据才获得局部修正权：

\[
u_{v,t}=\tanh r_\psi(s^h_{v,t},e_{v,t}),\qquad
\delta_{v,t}^+=q_v\eta_A\left(u_{v,t}-\frac{1}{T_v}\sum_j u_{v,j}\right),
\qquad \eta_A=\operatorname{softplus}(\tilde\eta_A)>0.
\]

其中 `r_psi` 由视频内标准化的 witness evidence 直连项与逐 snippet 小型 MLP 残差相加。
直连项保留 Universal 已验证的 neuron ordering 信息，避免强 host 下局部头因梯度太弱而忽略
witness；MLP 只学习 host 与 witness 的局部交互，不再增加第二套时序网络。

最终：

\[
s^{corr}_{v,t}=\sigma\left(
\operatorname{logit}s^h_{v,t}+\delta_v^-+\delta_{v,t}^+
\right).
\]

异常分支显式做视频内零均值，因此它不能偷偷退化成第二个全局偏置；跨视频下压只能由正常分支完成，局部重排只能由神经元分支完成。这不是 v9 的两个独立 global kappa。`q_v` 有直接视频标签监督；两条行为由同一 bag-state 路由，`eta` 不做零点投影。

### 6.4 Host-residual witness weighting

从冻结 host 直接得到 bag score 与未解释风险：

\[
p_v^h=P_k(s_v^h),\qquad
r_v^h=\operatorname{sg}(|y_v-p_v^h|).
\]

`r_h` 大表示 host 对该训练视频犯错或不确信。神经元 expert 的 bag loss 按 `r_h` 加权，
使 gate 优先学习 host 的正常误报和异常漏报，而不是重复 host 已经会的容易样本。它只使用
训练标签和冻结 host score，不生成 snippet 伪标签，也不引入第二阶段。

### 6.5 一次训练目标

\[
\mathcal L=
\mathcal L_{video}(q_v,y_v)
+\lambda_W r_v^h\mathcal L_{mil}(e_v,y_v)
+\lambda_{mil}\mathcal L_{mil}(s^{corr}_v,y_v)
+\lambda_N\mathbf1[y_v=0]\frac1{T_v}\sum_t
\left[-\log(1-e_{v,t})-\log(1-s^{corr}_{v,t})\right]
+\lambda_S\mathcal L_{sparse}.
\]

- `L_video` 训练视频状态，不用测试标签；
- `r_h L_mil(e)` 让神经元优先解释 host 的剩余错误；
- `L_mil` 保持 baseline 正式 top-k/MIL 口径，不再尝试 exact OR；
- dense normal loss 同时约束 neuron curve 与 corrected curve，使用弱监督中唯一可靠的 snippet 级负证据；
- sparse loss 只约束 gate，不引入新 teacher 或伪标签。

## 7. Witness neuron 定位公式

解释在完整模型训练后一次计算。令 `z_v_corr` 为 corrected bag logit，`rho_vt` 为 bag pooling 对 snippet 的归一化责任。

正 bag 的 witness activation 与梯度贡献：

\[
A_u=\mathbb E_{y=1}[\rho_{v,t}|x_{v,t,u}|],
\]

\[
G_u=\mathbb E_{y=1}\left[\rho_{v,t}\left|
x_{v,t,u}\frac{\partial z_v^{corr}}{\partial x_{v,t,u}}
\right|\right].
\]

先用 host pooled score 拟合一个仅用于解释的 host-only scalar calibration，得到训练标签残差：

\[
\varepsilon_v=y_v-\hat y_v^{host}.
\]

单元对 host 残差的补充性：

\[
C_u=\left|\operatorname{Corr}(P_k(c_{v,:,u}),\varepsilon_v)\right|.
\]

正常 off-target cost：

\[
N_u=Q_{0.95,y=0,t}\left(
\left|x_{v,t,u}\frac{\partial z_v^{corr}}{\partial x_{v,t,u}}\right|
\right).
\]

最终排序：

\[
S_u=\frac{A_uG_uC_u}{N_u+\varepsilon}.
\]

`A/G/C/N` 全部保存，不能只保存乘积；否则无法知道某个坐标因为什么入选。

## 8. 可解释 tag

每个 top witness neuron 保存：

- 功能角色：`WITNESS`、`VETO` 或 `MIXED`，由正/负 correction contribution 决定；
- top activating snippet 联系表与视频帧 montage；
- 时间形态：isolated、burst、persistent、transition；
- 上下文：场景/运动/物体的组合描述；
- 多个候选概念及 held-out fidelity，不给 polysemantic 坐标强行贴唯一词。

只允许声称“这些坐标对 adapter 的校正有功能作用”。因为 host score 是缓存输入，不能声称 intervention 改变了原 host 内部机制。

## 9. 与相邻工作的精确边界

| 工作 | 已做到 | 本文必须不同 |
|---|---|---|
| RTFM / UMIL / completeness | MIL 选择、去偏、伪标签完整性 | 不再 claim 新 MIL；研究内部 witness responsibility |
| whole-video classification | 视频级监督可减少正常误报 | 作为非创新的性能桥，并用 video-only 对照扣除 |
| LAKE | 正常图像中找敏感维度 | 长视频、弱标签、host complementarity、时间干预 |
| DNA | layer probe + activation/weight/gradient | positive snippet 未知，必须 latent responsibility + dense normal |
| V-FIND | 视频伪造 latent neurons + swap | 不是首次 video neurons；本文是长视频 WSVAD 时间定位 |
| VERA 等 explainable VAD | 文本语义解释 | 本文解释 frozen CLS coordinates 对具体校正的功能作用 |

## 10. 只做这些核心实验

### 10.1 Universal 信息验尸

| ID | 设置 | 目的 |
|---|---|---|
| U0 | Host | paired 起点 |
| U1 | Universal full | 复核目标上限 |
| U2 | U1 去 video suppression | 视频级判别贡献 |
| U3 | U1 去局部 neuron correction | 神经元对时间定位的贡献 |
| U4 | U1 去 temporal rules | 手工时间规则贡献 |

只在现有缓存上跑，不重新训练，不增加组合。

### 10.2 v10 性能与机制消融

| ID | 设置 | 唯一回答 |
|---|---|---|
| W0 | Host | 正式起点 |
| W1 | Host + video-only | 已有 whole-video classification 能拿多少 |
| W2 | Host + neuron-only | 神经元曲线有无独立增量 |
| W3 | Full 去 host-residual witness weighting | 为什么必须学习 host 剩余错误 |
| W4 | Full 去 dense-normal loss | 正常标签不对称是否必要 |
| W5 | Full 去 temporal readout | 旧 morphology 是否可由联合学习替代 |
| W6 | Full WITNESS-VAD | 是否恢复至少 +1 pp |

不跑 context 窗口、event width、滤波器、所有层组合或 correction budget 网格。

### 10.3 解释消融

| ID | 干预 | 对照 |
|---|---|---|
| I1 | selected erase | global random、same-layer random |
| I2 | matched-direction activation patch | random donor、same-layer random coordinate |
| I3 | selected patch | contribution-matched non-selected coordinate |
| I4 | held-out tag retrieval | activation-only tag、random tag |

### 10.4 主指标与诊断

- 硬性能：UCF frame AUC、XD frame AP；
- 模型选择：与 DSANet/VadCLIP 官方实现一致，UCF 按测试 AUC、XD 按测试 AP 选择 best epoch；固定 20 轮并公开完整选择轨迹；
- 归因：Cross-AUC、video-constant AP、Macro Within-AUC、abnormal-video-only AUC/AP；
- 误报：normal FPR@95% TPR、normal top-score；
- 修正规模：mean/max absolute logit correction；
- 解释：erase/patch effect gap、flip rate、tag fidelity。

## 11. Go/No-Go

1. `W1` 已经达到全部增益：不能把性能归因给神经元；方法降级为普通视频校准，不投当前故事。
2. `W2` 无增益且 `W6-W1 < 0.2 pp`：neuron mediator 没有性能价值，停止。
3. `W6` 在 UCF/XD 任一数据集低于 `+1.0 pp`：不满足本项目硬 SOTA 目标，先只检查 Universal 信息是否在一次训练中丢失，不扩实验。
4. selected intervention 不强于 contribution-matched control：删除 functional neuron claim，不用漂亮热图掩盖。
5. 增益只来自 Cross/FPR，Within 与 abnormal-only 不变或下降：只允许写 normal false-alarm auditing，不写定位改善。
6. DSANet 通过后才接第二 host；第二 host 不通过则不写 universal/host-agnostic。

## 12. 论文三张核心图

1. **Figure 1：同一个 host 高分的两种结局。** 正常视频中 VETO neurons 否决高分；异常视频中 WITNESS neurons 支持局部片段。并排展示 host、neuron、q、correction、GT。
2. **Figure 2：方法。** 12 层 CLS → sparse witness expert → video state → one routed residual → score。只画一个训练箭头和一个 optimizer。
3. **Figure 3：功能验证。** selected vs contribution-matched patch 的时间曲线、logit shift、flip rate和 tag montage。

## 13. 最终取舍

| 保留 | 重塑 | 删除 |
|---|---|---|
| 正式 frozen host | 三个 neuron streams → 一个 sparse expert | masked context predictor |
| 12 层 CLS 坐标 | 两个视频 LR → 一个联合 video head | two-axis global kappa |
| directional neuron weights | morphology → learned temporal readout | exact-OR / Markov chain |
| 单侧正常下压 | 普通 neuron rank → witness/normal/complementarity | teacher/student/伪标签 |
| activation intervention | 单词 tag → 多义组合 temporal tag | 推理期 max/median/Gaussian/advance |

完整调研与证据边界见 `docs/wsvad-neuron-research-2026-09-02.md`。

正式搭建顺序、固定超参数、分阶段生死门和最小充分消融见 `docs/witness-vad-build-ablation-guide-2026-09-02.md`。
