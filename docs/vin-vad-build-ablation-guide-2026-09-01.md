# CVA-VAD v9 搭建与核心实验指南

> 唯一方法依据：`docs/cva-vad-final-v9-iclr-2026-09-01.md`
>
> 本指南只回答四件事：先写什么、每个模块如何验收、哪些实验决定 claim 是否成立、结果出来后论文能说什么。

## 1. 先锁定论文要证明的三件事

| Claim | 方法对象 | 必须由什么证据证明 |
|---|---|---|
| C1：上下文条件的方向违背是 host 标量分数之外的增量信息 | Masked Contextual Violation Field | contextual directional residual 优于 raw activation、global deviation、absolute residual 和 global-context control；context replacement 能改变残差链 |
| C2：冻结 host 应被审计，而不是被替换；跨视频误报抑制与视频内定位必须分开 | Asymmetric Two-Axis Host Auditing | cross-only 改善 Cross/正常误报，within-only 改善 Within，二者组合优于单分支；单侧与预算设计优于对应替代项 |
| C3：被命名的神经元 field 确实驱动了具体校正 | Intervention-Verified Correction Fields | selected field 的 erase/patch effect 强于等预算匹配控制；删除 tag 代码不改变检测输出 |

整篇论文只有一条链：

\[
\text{normal context expectation}
\rightarrow \text{directional violation}
\rightarrow \text{host correction}
\rightarrow \text{intervention verification}.
\]

任何代码或实验若不服务这条链，不进入主项目。

## 2. 固定输入、输出和训练边界

```text
输入
H      [B,T,12,768]  冻结 CLIP ViT-B/16 每层 CLS
S_host [B,T]         baseline 官方 evaluator 使用的 snippet score
y      [B]           视频级标签，0 正常，1 异常
M      [B,T]         有效位置 mask；1=真实 snippet，0=padding

输出
S_corr [B,T]         校正后的 snippet score
```

`M` 只处理变长序列和 padding，不是异常区域标注。`G_t` 是预测位置 `t` 时不可见的时间保护区间，也不是监督标签。

训练边界：

- CLIP 和 host 全程冻结；训练对象只有 context predictor、field 权重 `omega`、`kappa_cross`、`kappa_within`；
- 一个训练过程、一个 optimizer；`L_ctx` 更新 predictor，`L_ws` 通过 stop-gradient 的 `mu/sigma` 更新 field 与 auditor；
- 正常 running median/MAD 和 `tau_N` 只由训练集正常视频更新，验证和测试时固定；
- tag、文本模型和 intervention 都在检测模型训练完成后运行，不能反馈到检测分数；
- 不接回旧方案的 detector、event chain、exact OR、平滑、膨胀、提前量、Kneedle 或 host top-k 神经元伪标签。

## 3. 代码按六个可验收单元搭建

建议在现有 `vin_vad/` 上实现：

```text
vin_vad/
├── data.py                 # 对齐 H、S_host、y、M
├── context_predictor.py    # masked conditional Gaussian
├── violation_field.py      # directional residual、entmax field、normal 标定
├── host_auditor.py         # cross / within 两轴校正
├── losses.py               # asymmetric MIL、L_ctx、correction budget
├── model.py                # 唯一前向链
├── train.py                # 单 optimizer 与 projected update
├── evaluate.py             # pooled / Cross / Within / normal FPR
├── interventions.py        # replacement、erase、activation patch
├── tags.py                 # 训练后 field 命名与 held-out fidelity
└── tests/
```

旧 `base_tcn.py`、`fixed_emission.py`、`event_chain.py` 只用于复现失败方案，不允许被 `model.py` 导入。

### B0：数据与 host identity

先完成 `data.py` 和 `evaluate.py`，此时不写新模型。

必须实现：

1. 逐视频读取并裁齐 `H` 与正式 `S_host`；时间长度以 baseline 官方 feature 为准；
2. 明确区分空间 view 与时间 snippet；
3. 缺失、重复、长度不一致写入 `audit.json`，测试视频出现异常直接报错；
4. evaluator 一次输出 pooled、Cross、Macro Within、normal FPR 和 normal top-score；
5. 保存逐视频 score，而不是只保存最终 AUC。

单元测试与验收：

- `S_corr=S_host` 时，复现 baseline evaluator；
- `H.shape[0] == S_host.shape[0] == M.sum()` 对每个视频成立；
- 随意改变 padding 数值或 batch 内最长视频，不改变任何有效位置输出；
- frame expansion 后，预测长度和每个视频边界与官方 GT 完全一致。

评估口径从 B0 起锁死：UCF 主指标是官方 pooled AUC，XD 主指标是官方 pooled AP；两者都额外报告 pooled AUC 的精确视频身份分解

\[
\operatorname{PooledAUC}=w\operatorname{WithinAUC}+(1-w)\operatorname{CrossAUC},
\quad w=\frac{\sum_v a_vn_v}{(\sum_v a_v)(\sum_v n_v)}.
\]

同时报告每个 mixed-label 视频等权的 Macro Within-AUC、video-mean constant 对照，以及 normal-video frame FPR@95% TPR。Cross/Within 只分解 AUC；不要伪造一个“Cross-AP”。XD 的 AP 通过 video-constant AP 判断其收益是否主要来自视频级排序。

任何一项失败都先修数据，不开始 B1。

#### B0 正式结果（DSANet，2026-09-02）

| Dataset | 官方主指标 | Cross-AUC | Within-AUC | Macro Within-AUC | video-constant 对照 | normal frame FPR@95% TPR | identity error |
|---|---:|---:|---:|---:|---:|---:|---:|
| UCF-Crime | AUC 89.445 | 89.506 | 73.675 | 69.913 | AUC 85.371 | 10.199% | 0 |
| XD-Violence | AP 86.951 | 95.408 | 85.468 | 79.845 | AP 77.424 | 1.474% | 0 |

B0 结论：**通过，可以进入 B1，但本阶段结束时先停下汇报。** UCF 1610/290 个训练/测试视频与 1,109,888 帧 GT 对齐；XD 3950/800 个可用训练/测试视频与 2,331,296 帧 GT 对齐。XD 训练集缺 4 个 hidden，已写入 `audit.json` 并跳过；测试集没有缺失。两套数据的 pooled AUC 分解误差均为 0，host score 与 identity correction 逐点完全相同。

不要横向比较两个数据集的 FPR 数值：95% TPR 对应的阈值和正样本分布不同。B0 只把它们锁为各自数据集后续 A0–A6 的固定参照。

### B1：无泄漏的正常上下文预测器

在 `context_predictor.py` 实现共享 masked temporal encoder 和低秩 layer-specific heads：

\[
(\mu_{t,u},\log\sigma_{t,u})
=g_\psi(H_{\{j:M_j=1,j\notin G_t\}}).
\]

query 只能包含位置编码；key/value 必须排除 `G_t` 和 padding。`L_ctx` 只在正常视频有效位置计算 Gaussian NLL。

必须测试：

1. **泄漏测试**：只改写 `G_t` 内的 hidden，目标位置 `mu/sigma` 逐点不变；
2. **padding 测试**：只改 padding，预测不变；
3. **梯度测试**：`L_ctx` 能更新 predictor；`L_ws` 不能经 `mu/sigma` 更新 predictor；
4. **尺度测试**：`sigma` 有数值下界，无 NaN/Inf；
5. **预测测试**：在正常验证视频上，conditional NLL 优于逐神经元 global mean/scale。

B1 有效的唯一含义是：上下文确实能预测正常神经元响应。这里不看检测 AUC，也不加入异常视频局部伪标签。

#### B1 正式结果（DSANet，2026-09-02）

| Dataset | held-out conditional NLL | global Gaussian NLL | 相对改善 | 最优 epoch | train/val 正常视频 |
|---|---:|---:|---:|---:|---:|
| UCF-Crime | -1.243614 | -1.166209 | 6.64% | 10 | 720 / 80 |
| XD-Violence | -1.306811 | -1.130192 | 15.63% | 10 | 1841 / 205 |

正式配置使用 1091728 个可训练参数，batch size 8，最多 10 epochs，seed 42。
训练时每个正常视频每轮抽取一个由 `seed + epoch + video key` 决定的窗口；
验证集每个视频固定取一个中心窗口，最多 256 个 snippet。训练与验证按视频拆分，
全程只读官方训练集正常视频，`test_split_used=false`。泄漏、padding、梯度隔离、
sigma 数值边界和短序列 guard 共 5 项测试全部通过。

B1 结论：**通过，但只证明正常神经元响应存在可学习的上下文结构。** 两个数据集
都在 held-out 正常视频上优于逐神经元全局高斯，因此可以进入 B2；本阶段结束时先
停下汇报。不要为 B1 增加检测 AUC、异常伪标签或更多 predictor 结构消融，这些都
不能回答审稿人的核心问题。后续真正决定论文是否成立的是 C0--C4：方向性 evidence
能否在固定 host 上提供增量检测信息。

远程产物位于 `../vadmy_data/vin_vad/dsanet/b1/<dataset>/`；正式命令和查看方式见
`run_instructions/RUN_VIN_VAD_B1_DSANET.md`。

### B2：方向违背与稀疏 correction field

在 `violation_field.py` 实现：

\[
r_{t,u}=\frac{x_{t,u}-\operatorname{sg}(\mu_{t,u})}
{\operatorname{sg}(\sigma_{t,u})+\epsilon},
\]

\[
v_{t,u,+}=\operatorname{ReLU}(r_{t,u}-\delta),\qquad
v_{t,u,-}=\operatorname{ReLU}(-r_{t,u}-\delta),
\]

\[
\pi=\operatorname{entmax}_{1.5}(\omega),\qquad
a_t=\sum_{u,q}\pi_{u,q}v_{t,u,q}.
\]

只用训练集正常 snippet 的 running median/MAD 得到标准化 evidence `e_t`。`pi` 从同值 `omega` 初始化，在校正目标下联合学习，不提前固定 top-K。

必须测试：

- 正负方向不会对同一个非零残差同时激活；
- `pi>=0` 且 `sum(pi)=1`；
- normal running statistics 不读取异常视频、padding、验证集或测试集；
- 保存 `r`、`v`、`pi`、`e` 后可逐项复算 auditor 输出；
- 所有 evidence 共享同一输入/输出接口和 auditor 配置；C0–C4 分别训练，但只允许改变表中指定的 evidence/controller。

B2 不能靠“异常视频的 `e` 更大”验收。它是否提供增量检测信息，只由后面的 C0–C4 决定。

实现纪律：running statistics 使用 stop-gradient 的 batch normal median/MAD 做
EMA，正式 momentum 固定为 0.05，不做数据集级扫描。B2 的 `omega` 按方案从同值
初始化，所以初始 `pi` 合法但稠密；稀疏 support 必须由 B4 的校正目标联合学出，
禁止在 B2 根据异常视频或测试指标预选 top-K。B2 正式命令与产物见
`run_instructions/RUN_VIN_VAD_B2_DSANET.md`。

#### B2 正式结果（DSANet，2026-09-02）

| Dataset | normal train videos | normal snippets | running median | running MAD | direction overlap | `pi` sum error | activation/evidence 复算最大误差 |
|---|---:|---:|---:|---:|---:|---:|---:|
| UCF-Crime | 720 | 109,811 | 0.079880 | 0.013631 | 0 | 0 | 4.47e-8 / 2.21e-6 |
| XD-Violence | 1,841 | 353,820 | 0.073371 | 0.011415 | 0 | 0 | 3.73e-8 / 2.26e-6 |

B2 单元测试 5 项全部通过：正负方向互斥，`entmax-1.5` 非负、归一且可产生
稀疏 support，running statistics 只读取 normal valid snippets，保存项可逐步复算，
padding 不改变有效输出或统计量。正式审计只读取 B1 的 `train_normal.csv`，没有读取
异常、验证或测试视频。初始 `omega` 全相等，因此 18,432 个方向坐标初始都在 support
内；这符合“联合学习、不预选 top-K”的方案，不能把初始稠密误写成失败。

B2 结论：**通过，可以进入 B3，但本阶段结束时先停下汇报。** 这一步只证明方向
残差与正常标定实现正确、无泄漏、可审计，不证明 detection gain，也不证明最终 field
会变稀疏。远程产物位于 `../vadmy_data/vin_vad/dsanet/b2/<dataset>/`；正式命令和
查看方式见 `run_instructions/RUN_VIN_VAD_B2_DSANET.md`。

### B3：两轴 host auditor

在 `host_auditor.py` 独立实现两个分支。

Cross-video branch：

\[
h_v=P_k(S_v^{host}),\quad q_v=P_k(e_v),
\]

\[
z_v=\frac{q_v-m_q^N}{1.4826\operatorname{MAD}_q^N+\epsilon},\qquad
n_v=\sigma(\tau_N-z_v),
\]

\[
\Delta_v^{cross}=-\alpha_v\kappa_v h_v n_v\le 0.
\]

Within-video branch：

\[
u_{v,t}=\alpha_t\kappa_t\tanh(e_{v,t}),
\]

\[
\Delta_{v,t}^{within}=u_{v,t}-\operatorname{MaskedMean}(u_v).
\]

最终只在 logit space 相加：

\[
S^{corr}_{v,t}=\sigma\!\left(
\operatorname{logit}(S^{host}_{v,t})+
\Delta_v^{cross}+\Delta_{v,t}^{within}
\right).
\]

`kappa_cross` 和 `kappa_within` 初始化为 0；每次 optimizer step 后投影到 `[0,1]`。

必须测试：

1. 两个 `kappa=0` 时，输出与 host 逐点相等，且参数在 0 处梯度非零；
2. `Delta_cross<=0`，并且在同一视频所有有效位置取同一值；
3. `MaskedMean(Delta_within)=0`；
4. `abs(Delta_cross)<=alpha_v`，`abs(Delta_within)<=2*alpha_t`；
5. 关闭一个 branch 不改变另一个 branch 的数值；
6. padding 不进入 pooling、中心化和校正预算。

不要用“zero-mean 所以只改善定位”作为验收。sigmoid 后并不严格保持视频平均概率；必须通过 A2/A3 的 Cross/Within 指标判断真实作用。

B3 只做结构审计，正式命令中的 `alpha/kappa` 是测试公式边界用的探针值，不是
模型选择结果，也不允许据此写性能结论。正式命令和产物见
`run_instructions/RUN_VIN_VAD_B3_DSANET.md`。

### B4：统一训练目标

在 `losses.py` 和 `train.py` 实现：

\[
\mathcal L
=\mathcal L_{asym\text{-}MIL}
+\lambda_{ctx}\mathcal L_{ctx}
+\lambda_\rho\max(0,\mathcal C-\rho).
\]

- 正常 bag：所有有效 snippet 都是 dense negative；
- 异常 bag：只有视频级标签，使用 top-k positive MIL；
- correction budget 同时约束 cross 常数偏移和 within 平均绝对改动；
- 每个 checkpoint 保存 predictor、`omega`、两个 `kappa`、normal running statistics、配置和数据 manifest。

训练日志只保留能诊断方法的量：三项 loss、两个 `kappa`、平均绝对 correction、budget violation、field support size、正常/异常视频 evidence 摘要。不要为旧模块继续加日志。

### B5：干预和 tag

仅在完整检测模型确定后实现。贡献定义为：

\[
C_{t,u,q}=v_{t,u,q}
\frac{\partial(\Delta^{cross}+\Delta_t^{within})}
{\partial v_{t,u,q}}.
\]

`interventions.py` 必须提供四种等预算操作：

- selected field erase；
- 同层随机 field erase；
- 贡献幅值匹配但未选中 field erase；
- matched-normal activation patch 与随机 donor patch。

context replacement 必须固定目标 `x_t`，只替换上下文，并保存 `mu -> r -> e -> Delta -> S_corr` 的整条变化。`tags.py` 只给通过干预的 field 生成组合 tag；删除整个文件后，`S_corr` 必须逐点不变。

## 4. 最小实验顺序

不要先跑完整消融表。下面每一关决定下一关是否值得做。

### E0：身份与评估口径

| ID | 设置 | 必须确认 |
|---|---|---|
| A0 | frozen host 原始输出 | baseline 分数、时间对齐、pooled/Cross/Within/normal FPR 全部可信 |

E0 通过后保存不可改动的 host score cache 和 evaluator 版本。后续所有方法共享它们。

### E1：创新一的生死对照

固定同一个 host auditor、loss、budget、训练数据和 evaluator，只替换 evidence：

| ID | Evidence | 它排除什么解释 |
|---|---|---|
| C0 | raw directional activation | 不是简单挑高激活维度 |
| C1 | global directional normal z-score / DFM | 不是普通边际正常偏离 |
| C2 | masked contextual absolute residual | 方向拆分确有必要 |
| C3 | masked contextual directional residual | 完整创新一 |
| C4 | global-context controller | 不是任意 context encoder 都能做到 |

C3 必须同时满足：

1. 正常验证视频 conditional NLL 优于 global baseline；
2. detection 指标优于 C0、C1、C2、C4；
3. context replacement 在目标 raw activation 不变时，能稳定改变 `mu/r/e/correction`。

若只满足第 2 条，可能只是更复杂的 readout；若只满足第 1 条，predictor 更准但没有提供检测增量。两者都不能支撑 C1 claim。

### E2：创新二的职责分解

先只跑四项：

| ID | 设置 | 主看指标 | 可以得出的结论 |
|---|---|---|---|
| A0 | frozen host | 所有指标 | 正式起点 |
| A2 | C3 field + cross only | Cross、normal FPR、normal top-score | 是否真正抑制正常误报 |
| A3 | C3 field + within only | Macro Within，同时检查 Cross | 是否真正改善视频内定位 |
| A4 | cross + within | pooled、Cross、Within、平均改动 | 两轴是否互补 |

结果解释必须按下面写：

- A2 改善 Cross/normal FPR：可以讲 normal-support audit；
- A3 改善 Within：可以讲 temporal redistribution；
- A3 只改善 Cross：不能把 zero-mean 公式写成定位贡献；
- A4 不优于最好单分支：两轴不互补，删除无效分支；
- A4 的收益只来自 A2：论文收缩为可靠性校准，不声称改善时间定位。

A4 确实优于 A0 和两个单分支后，再补三个必要结构消融：

| ID | 改动 | 回答的 reviewer 问题 |
|---|---|---|
| A1 | symmetric single residual | 为什么不是普通 residual fusion |
| A5 | A4 去 correction budget | budget 是否限制破坏 strong host |
| A6 | cross 改为双向校正 | 为什么 cross 只允许正常支撑下压分 |

`entmax -> softmax` 只在论文强调“稀疏 field”时补做；它不决定两轴 auditor 是否成立。

### E3：最近方法的同口径控制

这是新颖性实验，不是普通 baseline 表。固定 A0 host、hidden、loss、budget 和 evaluator，只替换 evidence/controller：

| Control | 对照目的 |
|---|---|
| BN-WVAD-style marginal DFM | 排除“只是正常均值偏离” |
| RPC-style normal prototype deviation | 排除“只是冻结模型后校准” |
| SteerVAD-style global-context controller | 排除“只是加一个 context 模块” |
| CVA-VAD C3 field + A4 auditor | 完整方法 |

公开论文数字不能替代这个实验，因为 backbone、host、输入与 evaluator 均不同。

### E4：创新三的功能验证

只做能区分“功能单元”和“相关维度”的实验：

| ID | 实验 | 必须比较 |
|---|---|---|
| I1 | context replacement | matched context donor vs random donor |
| I2 | readout erase | selected vs same-layer random vs contribution-matched non-selected |
| I3 | CLIP activation patch | selected matched-normal patch vs 等预算 controls |
| I4 | held-out tag fidelity | 全部预先固定的 top fields，不挑案例 |

关键量不是 heatmap 漂不漂亮，而是 `abs(Delta change)`、目标/非目标位置差异，以及 selected-control effect gap。I2/I3 不强于贡献匹配控制时，只能把 tag 称为描述，不能称为功能解释。

### E5：最小泛化验证

主实验先完成 UCF-Crime、XD-Violence 和两个已接入 host。不要先扩成多 host 笛卡尔积。

跨 host 只验证两个问题：

1. source context predictor 与 field 固定，target 只学习两个 `kappa`；
2. 完全零拟合 transfer 作为压力测试。

第一项成立才可以声称 correction field 可迁移；第二项失败不影响主方法，但必须如实报告。

## 5. Reviewer 最可能问什么，哪一个实验回答

| Reviewer 质疑 | 唯一核心回答 |
|---|---|
| 只是把 BN-WVAD 的 DFM 换成 CLIP hidden | C1 vs C3，同 readout；再加 context replacement |
| 更强的 context 网络带来提升，并非 conditional violation | C3 vs C4；目标泄漏测试；正常 conditional NLL |
| 只是 frozen detector 后做 calibration | A1 vs A4；A3 的 Within；I2/I3 神经元干预 |
| pooled AUC 只是把正常视频整体压低 | A2/A3 分解与 Cross/Within/normal FPR |
| 两个 branch 是人为拼接 | A2、A3、A4 的职责和互补性 |
| 单侧抑制和 budget 是拍脑袋 | A5、A6；同时报告平均 correction |
| 所谓解释只是相关 heatmap | contribution-matched control、readout erase、activation patch |
| 只对一个 host 有效 | 第二 host 和 source-field -> target-host transfer |
| 文本偷偷参与检测 | 删除 `tags.py` 后逐点 identity test |

## 6. 指标不能混着解释

| 指标 | 只允许支撑什么 |
|---|---|
| pooled AUC/AP | 总体排序性能，不单独证明时间定位 |
| Cross-AUC / video-constant AP | 视频级 normal-support audit |
| Macro Within-AUC | 异常视频内部时间排序 |
| normal FPR@固定 TPR、normal top-score | 正常误报抑制 |
| mean absolute correction、gain-budget curve | 对 frozen host 的改动幅度与收益关系 |
| conditional NLL、replacement chain | 上下文条件预期机制 |
| erase/patch effect gap | 神经元 field 的功能特异性 |
| held-out tag fidelity | tag 能否稳定描述已验证 field，不证明检测能力 |

禁止用 pooled AUC 上升直接写“定位更准”，也禁止用零均值公式代替 Macro Within 实验。

## 7. 论文最终只需要的结果

### 表 1：主结果

每个 dataset-host 组合只列：Host、完整方法、pooled、Cross、Within、normal FPR、平均绝对 correction。

### 表 2：机制与结构消融

上半部分 C0–C4；下半部分 A0、A2、A3、A4、A1、A5、A6。每一行对应一个明确 claim，不再增加旧模块。

### 表 3：解释与迁移

列 context replacement、erase、patch、三类控制、tag fidelity、source-to-target transfer。

### 图 1：动机例子

一个正常误报案例展示 host 高分但 normal support 强，因此 cross branch 下压；一个异常视频展示 contextual violation 如何经 within branch 改变时间排序。

### 图 2：方法总图

只画四步：masked normal expectation -> directional field -> cross/within correction -> intervention-verified field。

### 图 3：机制证据

同一视频对齐展示 raw activation、预测区间、directional violation、两项 correction、最终 score，并加入 selected 与 matched-control patch 结果。

## 8. 实际执行清单

```text
[x] B0 数据审计、host identity、统一 evaluator（DSANet；UCF/XD 均通过）
[x] B1 leakage/padding/gradient/NLL 测试（DSANet；UCF/XD 均通过）
[x] B2 residual/entmax/running-stat 测试（DSANet；UCF/XD 均通过）
[ ] B3 identity/sign/zero-mean/bound/branch-independence 测试
[ ] B4 单 optimizer 训练与完整 checkpoint
[ ] E1 C0-C4，决定 contextual-directional claim
[ ] E2 A0/A2/A3/A4，决定 cross/within 两条职责
[ ] E2 成立后补 A1/A5/A6
[ ] E3 三个最近邻同口径控制
[ ] E4 replacement/erase/patch/匹配控制
[ ] E5 第二 host 与最小 transfer
[ ] 最后才做 tag 和论文可视化
```

停止增加实验的标准很简单：三条 claim 都已经有一个直接对照、一个机制量和一个失败可解释的结果。不要做事件窗口扫描、所有层组合、预算笛卡尔积、额外 smoothing、装饰性 heatmap，或把旧方案模块逐个接回来。

## 9. 结果出来后的诚实收缩规则

- C3 不优于 C1/C4：不能讲 context-conditional correction neurons；
- A2 有效、A3 无效：保留 cross audit，论文只讲正常误报审计；
- A3 有效、A2 无效：删除 cross branch，论文只讲 context-driven temporal redistribution；
- A4 不优于单分支：不讲 two-axis synergy；
- erase/patch 不强于匹配控制：保留 attribution 可视化，删除 functional neuron claim；
- 第二 host 不成立：不讲 host-agnostic；
- tag fidelity 不成立：删除语义 tag，不影响检测模型。

这些不是额外补救路线，而是每项创新在实验上不成立时必须同步缩小的论文表述。
