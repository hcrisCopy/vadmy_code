# CVA-VAD v9 项目搭建与 ICLR 核心实验指南

> 配套方案：`docs/cva-vad-final-v9-iclr-2026-09-01.md`
>
> 只做三条证据链：contextual residual 是否是真增量信息；单侧跨视频审计和视频内重排各自解决什么；神经元 field 是否真实驱动校正。

## 1. 输入与红线

\[
H\in\mathbb R^{B\times T\times12\times768},\quad
S^h\in(0,1)^{B\times T},\quad
y\in\{0,1\}^{B},\quad
M\in\{0,1\}^{B\times T}.
\]

- `H`：冻结 CLIP 12 层 CLS；神经元为 `(layer, dim)`；
- `S_h`：baseline 正式 detection score；
- `y`：视频级标签；
- `M=1`：真实有效 snippet，`M=0`：padding。不是异常 mask；
- `G_t`：预测 `t` 时隐藏的目标保护区间。

红线：

- host 与 CLIP 永久冻结，但不是阶段性反复冻结；
- 不训练新 detector 替代 host；
- 不使用 host top-k 发现伪标签神经元字典；
- 不使用 exact OR、event chain、median/max/dilation/smoothing/advance；
- 不使用 Kneedle 固定字典或数据集名分支；
- 文本、tag 与 intervention 不进入检测分数；
- 测试集不决定任何阈值、预算、field 或 checkpoint。

## 2. 最小代码结构

```text
vin_vad/
├── data.py
├── context_predictor.py
├── violation_field.py
├── host_auditor.py
├── losses.py
├── model.py
├── train.py
├── evaluate.py
├── interventions.py
├── tags.py
└── tests/
```

`base_tcn.py` 和 `event_chain.py` 只保留 P1 失败复现，不进入 v9 import graph。

## 3. P0：先锁死数据、host 与 evaluator

时间长度服从 baseline 官方 feature，不由原视频帧数单独推算。hidden 多出的尾 snippet 逐视频裁掉；`__0`--`__9` 是空间 views，不是时间块；XD 缺失的 4 个训练 hidden 写入 `audit.json`，测试视频不允许缺失。

必须通过：

1. `kappa_cross=kappa_within=0` 时，输出与 host 逐点相等；
2. 每个视频 `H.shape[0] == S_h.shape[0] == M.sum()`；
3. padding 内容和 batch 最大长度不影响有效输出；
4. frame expansion 后与官方 GT 总长度和逐视频边界一致；
5. host-only 复现 executable baseline；
6. evaluator 同时产出 pooled、Cross、Macro Within 和 normal false-alarm 指标。

P0 不过，不开始训练。

## 4. P1：Masked Contextual Violation Field

### 4.1 先实现无泄漏 predictor

只用正常视频训练 Gaussian conditional NLL。query 只有位置编码，key/value 排除 `G_t` 和 padding。

结构使用共享 masked temporal encoder + 低秩 layer-specific heads；禁止为每个坐标建立独立预测网络。

单元测试：

1. 改写 `G_t` 内 hidden，目标 `mu/sigma` 不变；
2. 改 padding，目标预测不变；
3. `L_ctx` 对 predictor 有梯度；`L_ws` 经 stop-gradient 后对 `mu/sigma` 路径无梯度；
4. `sigma` 有下界且没有整体膨胀；
5. normal conditional NLL 优于 global mean/scale。

### 4.2 联合学习稀疏方向 field

\[
r=(x-\operatorname{sg}(\mu))/(\operatorname{sg}(\sigma)+\epsilon),
\]

\[
v^+=\operatorname{ReLU}(r-\delta),\qquad
v^-=\operatorname{ReLU}(-r-\delta),
\]

\[
\pi=\operatorname{entmax}_{1.5}(\omega),\qquad
a_t=\sum_{u,q}\pi_{u,q}v_{t,u,q}.
\]

`pi` 与 auditor 一次联合训练，不提前选 top-K。记录 `pi` 非零数、有效 field 大小、层级质量和正常/异常 `e_t` 分布即可。

normal snippets 的 batch median/MAD 只作 stop-gradient 标定并更新 running values；推理时固定。`tau_N` 由 normal-video `z_v` running reservoir 的固定分位规则得到，不使用测试集。

## 5. P2：Asymmetric Two-Axis Host Auditor

### 5.1 跨视频单侧审计

\[
h_v=P_k(S^h_v),\qquad
q_v=P_k(e_v),
\]

\[
z_v=(q_v-m_q^N)/(1.4826\operatorname{MAD}_q^N+\epsilon),
\]

\[
n_v=\sigma(\tau_N-z_v),
\]

\[
\Delta_v^{cross}=-\alpha_v\kappa_v\,h_v n_v,\qquad \kappa_v\in[0,1].
\]

它必须恒小于等于 0，只在 host 高、normal support 强时明显。

### 5.2 视频内零均值重排

\[
u_{v,t}=\alpha_t\kappa_t\tanh(e_{v,t}),\qquad \kappa_t\in[0,1],
\]

\[
\Delta_{v,t}^{within}=u_{v,t}-\operatorname{MaskedMean}(u_v).
\]

零均值发生在 logit correction 上，不等价于 sigmoid 后视频平均概率严格不变；A3 仍需同时报告 Cross 与 Within，确认它实际改变了哪一轴。

### 5.3 最终输出与目标

\[
S^{corr}=\sigma(\operatorname{logit}(S^h)+\Delta^{cross}+\Delta^{within}),
\]

\[
\mathcal L=\mathcal L_{asym\text{-}MIL}
+\lambda_{ctx}\mathcal L_{ctx}
+\lambda_\rho\max(0,\mathcal C-\rho).
\]

单元测试：

1. 两个 `kappa=0` 时严格等于 host，且 projected-gradient 参数在 0 处梯度非零；
2. `Delta_cross<=0`，同一视频所有位置相同；
3. `MaskedMean(Delta_within)==0`；
4. `abs(Delta_cross)<=alpha_v`，`abs(Delta_within)<=2*alpha_t`；
5. 删除 cross branch 不改变 within branch，反之亦然；
6. padding 不进入 pooling、中心化、running stats 和 loss。

## 6. P3：先跑最小生死实验

不要一开始跑全表。按下面顺序：

### Gate 1：context 是否成立

| ID | 证据 | 必须回答 |
|---|---|---|
| C0 | raw directional activation | 高激活是否足够 |
| C1 | global directional normal z-score / DFM | 边际正常偏离是否足够 |
| C2 | contextual absolute residual | 上下文是否有效 |
| C3 | contextual directional residual | 方向是否必要 |
| C4 | global-context controller | masked conditional 是否优于 SteerVAD-style 近邻 |

C3 不优于 C1/C4，ICLR 主故事停止。

### Gate 2：两轴校正是否成立

先跑 A0、A2、A3、A4：

| ID | 模型 | 看什么 |
|---|---|---|
| A0 | frozen Host | 正式起点 |
| A2 | contextual field + cross-only | pooled/Cross/normal FPR |
| A3 | contextual field + within-only | Macro Within |
| A4 | cross + within | 总体互补与 +1 目标 |

A4 有效后再补：

| ID | 模型 | 消融目的 |
|---|---|---|
| A1 | symmetric single residual | 两轴非对称设计是否必要 |
| A5 | A4 去 budget | correction budget 是否保护 host |
| A6 | cross branch 改成双向 | 单侧正常抑制是否符合旧实验信息 |
| A7 | entmax 改 softmax | sparse field 是否必要 |

先跑 UCF/XD × DSANet/DeSC；主线成立后只加一个第三 host。

## 7. P4：最危险近邻必须同口径比较

同一 host、hidden、loss、budget 和 evaluator 下，只替换 evidence/controller：

1. BN-WVAD-style marginal DFM；
2. RPC-style normal prototype deviation；
3. SteerVAD-style global-context controller；
4. CVA-VAD masked contextual directional field。

不能用各论文公开数字代替这个直接对照。CVA-VAD 不优于这些近邻时，不能靠写作放大新颖性。

## 8. P5：最后做 Intervention-Verified Fields

解释贡献：

\[
C_{t,u,q}=v_{t,u,q}\frac{\partial(\Delta^{cross}+\Delta_t^{within})}{\partial v_{t,u,q}}.
\]

只在 A4 成立后执行：

1. 训练完成后按 `abs(C)` 固定 top correction fields；
2. context replacement：固定目标 `x_t`，替换匹配 donor context；
3. readout erase；
4. 重放 CLIP 做 matched-normal activation patch；
5. 同层随机、贡献匹配未选中、随机 donor 三类控制；
6. 独立 probe set 做组合 tag held-out fidelity；
7. top fields 全部报告，不挑成功案例。

跨 host 只做两个协议：

- source context predictor + field 固定，target 只拟合 `kappa_cross/kappa_within`；
- 完全零拟合作为压力测试，不作为必须成功的主 claim。

## 9. 指标和结论绑定

| 指标 | 对应 claim |
|---|---|
| pooled AUC/AP | 总体性能与 +1 目标 |
| Cross-AUC / video-constant AP | 视频级可靠性校准 |
| Macro Within-AUC | 视频内部时间定位 |
| normal FPR@固定 TPR、normal top-score | 误报抑制 |
| mean abs correction、gain-budget curve | host-preserving 程度 |
| context replacement 链式变化 | contextual mechanism |
| erase/patch vs controls | neuron functional contribution |

直白判定：

- A2 涨 pooled、A3 不涨 Within：只讲 reliability calibration；
- A3 涨 Within：才讲 temporal localization；
- A4 不优于 A0：停止解释实验，不加后处理救分；
- C3 不优于 C1：退化为普通 normal deviation，ICLR 新颖性不足；
- patch 不强于贡献匹配控制：tag 只是 descriptor；
- 第二 host 失败：不讲 host-agnostic。

## 10. 论文只需要三表三图

三表：

1. 主结果：host、pooled、Cross、Within、normal FPR；
2. C0--C4 与 A0--A7 核心消融；
3. tag fidelity、erase、patch、匹配控制和跨 host transfer。

三图：

1. Figure 1：强 host 高分但内部 normal support 强，auditor 抑制误报；另一例内部 contextual violation 强，within branch 定位事件；
2. 方法图：normal expectation → directional field → cross/within correction → verified field；
3. 机制图：实际响应、预测区间、violation、两项 correction、最终 score，加 patch 对照。

不做七张装饰图、事件窗扫描、所有层遍历、所有预算笛卡尔积或额外 smoothing baseline。

## 11. ICLR 交付闸门

提交前至少具备：

1. G0 数据与 host identity 全通过；
2. C3 明确优于 C1/C4；
3. A4 优于 A0，且知道增益来自 Cross 还是 Within；
4. 两个数据集、两个 host 的完整结果；
5. 至少一组受控 patch/erase 结果；
6. 文档、代码、公式、数字和引用由作者逐项核实；
7. 按 ICLR 2027 要求在论文与提交表单披露 AI 用途。
