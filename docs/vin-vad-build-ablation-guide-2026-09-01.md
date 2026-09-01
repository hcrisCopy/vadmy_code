# ViN-VAD 项目搭建与核心消融指南

> 配套方案：`docs/ws-vad-ultimate-v6-2026-08-31.md`
>
> 目标不是把模块都跑一遍，而是用最短证据链证明：**异常应建模为上下文违背；视频弱标签应通过事件模型归因；神经元标签是经过验证的功能解释。**

## 1. 论文故事只看三条证据链

| 论文主张 | 审稿人真正要看的证据 | 不通过时怎么写 |
|---|---|---|
| 异常是 contextual violation，不是 activation peak | raw activation、global residual、contextual residual 的直接对照；只换上下文时 residual 和 posterior 随之改变 | 若 contextual 不优于 global/raw，删除“contextual violation”主张 |
| OR-chain 比 top-k 更符合弱标签事件语义 | 相同 emission 下，exact OR、persistence、length calibration 分别带来什么 | 若只改善曲线平滑而不改善定位/长度偏差，降级为时序建模模块 |
| tag 是 verified explanation | held-out semantic fidelity + readout erase + backbone patch controls | 任何一项不通过，只叫 descriptor，不叫 verified explanation |

三条停止规则：

1. Event chain 不成立，不做 violation branch。
2. Full violation 不优于 raw/global/absolute 对照，不做 tag。
3. Patching 不强于匹配控制，不讲神经元因果解释。

## 2. 项目边界

输入固定为：

\[
H\in\mathbb R^{B\times T\times12\times768},\qquad
y\in\{0,1\}^{B},\qquad
M\in\{0,1\}^{B\times T}.
\]

- `H`：冻结 CLIP ViT-B/16 每隔 16 帧采样得到的 12 层 CLS 状态；
- `y`：视频级标签；
- `M`：有效 snippet mask。

输出只有 `video_prob`、`snippet_prob` 和机制诊断量。

硬约束：

- 不读取已有 VAD detector 分数，不做伪标签，不让文本参与检测。
- 不做 smoothing、膨胀、前移或测试后校准。
- UCF-Crime、XD-Violence 使用官方 train/test，不造 validation split。
- 所有消融固定相同 TCN、训练轮数、默认 seed、evaluator 和 checkpoint 规则。
- 当前输入是稀疏采样的层级视觉状态，不写成原生运动特征。

推荐独立目录：

```text
vin_vad/
├── data.py
├── base_tcn.py
├── event_chain.py
├── context_predictor.py
├── violation_field.py
├── model.py
├── losses.py
├── train.py
├── evaluate.py
├── interventions.py
├── tags.py
└── tests/
```

旧工程只复用数据读取和官方 evaluator，不复用旧 fusion、expert、score correction 或后处理。

## 3. 模块搭建顺序

### P0：先锁死数据和 evaluator

只做三件事：

1. 检查 `hidden.shape == [T,12,768]`，保存真实 `frame_indices`、视频长度和 mask。
2. 所有模型使用同一个 snippet-to-frame 映射，尾段按真实帧数截断。
3. 人工构造一条 snippet 曲线，确认展开后的帧位置、长度和 GT 完全对齐。

**DSANet 正式口径写死：** 每个视频的有效 snippet 数以 DSANet 官方 CLIP 特征为准，不能仅由原视频
`num_frames` 反推。官方文件实际同时存在 `floor(num_frames / 16)` 和 `ceil(num_frames / 16)` 两种长度；
现有 hidden cache 固定按 `range(0, num_frames, 16)` 提取，因此有些视频会比官方特征多一个尾 snippet。
数据层必须逐视频裁到官方特征长度。测试展开长度固定为 `valid_snippets * 16`，所有测试视频之和必须与
官方 GT 逐点等长；禁止把全部 hidden 直接 `repeat(16)` 后再靠全局截断凑长度，因为这会让后续视频错位。

依据不是经验判断：VadCLIP 的 `list/make_gt_ucf.py` 与 `list/make_gt_xd.py` 直接按每个 `__0.npy` 的
`fea.shape[0] * 16` 生成 GT；DSANet 与 VadCLIP 的 `process_split` 实现相同，仓库内对应 UCF/XD GT
文件也逐字节一致。`__0`–`__9` 来自 VadCLIP 的十种空间 crop，同一视频的时间长度必须一致且只计一次。

`raw_num_frames` 与真实 `frame_indices` 仍需保存，供训练尾段边界和机制可视化使用。训练集也按 DSANet
实际特征长度保留 `floor` 或 `ceil` 个 snippet；若保留不完整尾 snippet，其右边界使用真实
`num_frames`。测试 AUC/AP 则一律服从官方 evaluator 的有效帧域，不擅自重造 GT。

当前 XD hidden cache 缺 4 个训练视频。P0 允许用显式参数跳过这些训练项，并把完整 key 列表写入
`audit.json`；提取日志显示原因是原始 video root 没有匹配到这 4 个视频，不是 hidden 解码失败。测试视频
不允许缺失。论文的数据实现细节需如实披露这一点，不能写成“训练集无缺失”。

DSANet 的 P0 正式命令：

```bash
bash run_instructions/run_vin_vad_p0_dsanet.sh
```

输出：`../vadmy_data/vin_vad/dsanet/p0/<dataset>/audit.json`、`train.csv`、`test.csv` 和
`alignment_probe.npz`。中断后直接重跑会续审；仅在确认要清空旧 P0 产物时运行
`CLEAN=1 bash run_instructions/run_vin_vad_p0_dsanet.sh`。

必须通过：改变 padding 内容或 batch padding 长度，loss 和有效位置输出不变。

### P1：基础 TCN 与 event chain

先做最终层 CLS + 三层 residual TCN：

\[
a_{1:T}=b_\theta(\operatorname{LN}(h_{1:T,L,:})).
\]

先接 top-k MIL，再替换成 OR event chain。不要同时加入 violation。

event chain 必须先过单元测试：

- 对 \(T\le12\) 枚举全部 \(2^T\) 序列，`logZ0/logZ1` 和 snippet marginal 与 DP 一致；
- emission、\(\rho\)、\(\kappa\) 梯度有限且无 NaN；
- padding 不改变有效位置结果；
- 固定同一组 emissions，孤立峰、连续中等证据和不同视频长度产生符合设计的 posterior。

通过标准：E0→E1→E2→E3 的变化能分别解释 exact OR、事件持续性和长度校准，而不是只让曲线更平滑。

**正式训练口径写死：**

- E0–E3 共用宽度 512、kernel 3、dilation 1/2/4 的三层 residual TCN，dropout 0.1；
- 训练时沿用 DSANet 的均匀分桶平均，将长视频压到 256 snippets；测试保持完整时间长度；
- seed 234、10 epochs、AdamW、weight decay 0.01；UCF 使用 batch 64/类、lr 7e-5，XD 使用
  batch 96/类、lr 1e-5，这些是 DSANet 官方训练参数，不是按结果调参；
- 最终 epoch 是唯一正式 checkpoint，不用测试集选 epoch；E0–E3 都不使用 smoothing 或测试后校准；
- E1 的独立状态先验、E2 的 constant onset、E3 的 length-calibrated onset 均从“256 snippets
  期望约一个 onset”初始化；E2/E3 persistence 统一从 0.9 初始化并参与学习。

DSANet 的 P1 正式命令：

```bash
bash run_instructions/run_vin_vad_p1_dsanet.sh
```

输出：`../vadmy_data/vin_vad/dsanet/p1/summary.csv`、`fixed_emission.json`，以及
`<dataset>/<E0-E3>/train_summary.json`、`history.json`、`model_final.pt`、`evaluation/metrics.json`、
`evaluation/per_video.csv` 和逐视频 `curves/`。直接重跑会按 epoch/视频复用；确认要清空 P1 后运行
`CLEAN=1 bash run_instructions/run_vin_vad_p1_dsanet.sh`。

审稿人只看三个判断：E1 是否优于 E0；E2 是否在定位指标和碎片数上优于 E1；E3 是否在不损害
frame AUC/AP 的前提下降低正常视频分数与长度的相关性。若 E3 只让曲线更顺、不改善这三点，event
story 不成立，按停止规则不进入 P2。

### P2：normal-context predictor

实现两层 masked cross-attention。query 只有位置编码，任何计算路径都不能读取目标保护区间 \(G_t\)。只有正常训练视频进入 \(\mathcal L_{ctx}\)。

必须通过：

- 随机改写 \(G_t\) 内输入，目标位置的 \(\mu_t,\sigma_t\) 逐点不变；
- 改写 padding，预测不变；
- bag loss 对 \(\mu,\sigma\) 无梯度，context loss 有梯度；
- \(\sigma\) 不整体膨胀，normal residual 没有明显失控。

怎么看有效：冻结模型后，在 test normal subset 上，conditional NLL 优于 global per-neuron mean/scale。这个结果只用于机制分析，不用于选 checkpoint。

### P3：directional violation field

严格按最终方案实现：

\[
r=\frac{x-\operatorname{sg}(\mu)}{\operatorname{sg}(\sigma)+\epsilon},\qquad
v^+=\operatorname{ReLU}(r-\delta),\quad
v^-=\operatorname{ReLU}(-r-\delta),
\]

\[
\pi=\operatorname{entmax}_{1.5}(\omega),\qquad
e_t=\sum_u\pi_uv_{t,u},\qquad
\eta_t=a_t+\operatorname{softplus}(\beta)\tanh(\bar e_t).
\]

只记录五个诊断量：`beta`、entmax 非零数、\(N_{eff}\)、各层权重、正常/异常的 \(\bar e_t\) 分布。

必须通过：

- `pi >= 0` 且 `sum(pi) == 1`；
- 去掉 violation 后严格退化为同一个 TCN + E3；
- \(\beta\) 没有塌到 0；
- V4 优于 raw activation、global residual 和 absolute residual。

### P4：统一训练

目标只有：

\[
\mathcal L=\mathcal L_{bag}+\lambda_{ctx}\mathcal L_{ctx}.
\]

一次前向、一次反向、一个 optimizer。固定训练轮数并统一使用同一 checkpoint 规则。不要为不同消融挑各自最好看的 epoch。

训练失败只检查：posterior 是否全 1、\(\sigma\) 是否膨胀、\(\beta\) 是否为 0、\(N_{eff}\) 是否退化。不要先加新 loss。

### P5：最后才做 tag 和 intervention

只有 V4 已经成立才开始：

1. 按训练集上的 \(\pi_u\mathbb E[v_u]\) 固定选 top-K units/fields。
2. 开放词表模型只提出候选 actor/action/scene/temporal-relation。
3. 人工标注独立 probe set；按视频切 discovery/held-out。
4. 搜索有限深度组合 tag，held-out 上计算 fidelity。
5. 做 readout erase。
6. 重放 CLIP 做 normal-donor activation patch。
7. 与同层随机坐标、幅值匹配未选中坐标、随机正常 donor 比较。

必须报告 top-K 的全部结果和通过比例，不能只挑成功案例。

## 4. 核心消融：只跑这张表

| ID | 模型 | 证明什么 |
|---|---|---|
| E0 | final-layer TCN + top-k MIL | 常规弱监督归因基线 |
| E1 | final-layer TCN + independent exact OR | 对全部合法实例分配求和是否优于 top-k |
| E2 | E1 + Markov persistence，constant onset | 连续事件先验是否有用 |
| E3 | E2 + length-calibrated onset | 长视频先验偏差是否被解决 |
| V1 | E3 + raw directional activation field | activation 本身是否已经足够 |
| V2 | E3 + global directional normal residual | 正常锚有用，但没有上下文时能做到哪里 |
| V3 | E3 + contextual absolute residual | 上下文有用，但方向性是否必要 |
| V4 | E3 + contextual directional violation field | Full ViN-VAD |
| V5 | V4，entmax 换 dense softmax | 稀疏 field 是否必要 |

对照口径写死：V1 对 layer-normalized \(x\) 直接取正/负 ReLU，V2 的 mean/scale 只来自训练集正常视频，V3 使用 \(\operatorname{ReLU}(|r|-\delta)\)。V1–V5 复用同一个 TCN、event chain、readout/fusion 和 evaluator。报告参数量即可，不添加无语义的“参数匹配网络”。

主比较关系：

- Event story：E0 → E1 → E2 → E3。
- Violation vs activation：V1 vs V4。
- Context：V2 vs V4。
- Direction：V3 vs V4。
- Sparse field：V5 vs V4。

只有当论文明确强调“中间层比最终层重要”时，再补 `V4-final-layer-only`。否则不做。

单 seed、相同预算即可。若某个核心差值非常小或训练明显不稳定，只补 Full 与该直接对照，不重跑整张表。

## 5. 三项真正把故事讲实的机制实验

### 5.1 Context-matched intervention

保持目标 snippet 不变，只替换保护区间外的上下文。donor 预先按长度、目标位置和可见 actor/action 匹配，不能按模型输出挑。

报告：

\[
\Delta\mu,\quad\Delta r,\quad\Delta e,\quad\Delta s.
\]

再画一组正常/异常 context-matched pair：raw activation 相近，但正常预测区间、directional violation 和 posterior 不同。

这是“contextual”最重要的证据。纯 temporal shuffle 只作为辅助压力测试，因为它可能制造分布外输入。

### 5.2 Fixed-emission event experiment

输入完全相同的 emissions，只替换 independent OR、constant-onset chain 和 length-calibrated chain。

必须展示：

- 孤立高峰与连续中等证据得到不同事件后验；
- 复制成更长序列后，E3 的视频异常先验不会仅因长度上升；
- 在真实测试视频上，E3 相比 E0/E1 降低正常视频分数与视频长度的相关性，并减少碎片化。

合成 emissions 只验证算法行为，不能当检测性能。

### 5.3 Verified neuron intervention

每个展示的 unit/field 同时给出：

- compositional tag 与最佳 atomic tag 的 held-out fidelity；
- tag 对应位置和非对应位置的 readout erase effect；
- normal-donor backbone patch effect；
- 同层随机、幅值匹配、随机 donor 三类控制。

只有目标 patch 稳定强于控制，才能写“该 field functionally contributes to the detected violation”。不能写“这是偷窃神经元”。

## 6. 怎么判断三项创新成立

### Event inference 成立

- E3 相比 E0 在官方 frame-level AUC/AP 上有实质改善，且不是靠后处理得到；
- E2 优于 E1，说明 persistence 有用；
- E3 相比 E2 降低正常视频分数与长度的相关性；
- posterior 没有靠占满整段视频作弊。

### Contextual violation 成立

- V4 优于 E3、V1、V2、V3；
- \(\beta\) 不接近 0；
- context-matched swap 会沿着 \(\mu\rightarrow r\rightarrow e\rightarrow s\) 改变结果；
- UCF-Crime 与 XD-Violence 的关键比较方向基本一致。若只在一个数据集成立，就收缩泛化表述。

### Verified tags 成立

- compositional fidelity 在 held-out probe 上优于 atomic tag；
- erase 对 tag 对应位置影响更大；
- backbone patch 强于三类控制；
- 移除所有文本模块后 detector 输出逐点不变。

## 7. 论文最终只需要这些结果

1. **主结果表**：UCF-Crime AUC、XD-Violence AP。外部数字若 backbone/采样不同，明确标出，不据此声称纯方法优势。
2. **核心消融表**：E0–E3、V1–V5。
3. **机制主图**：raw response、normal prediction interval、directional violation、event posterior 四条轨迹，加 context-matched pair。
4. **解释验证表/图**：held-out fidelity、erase、patch 与控制、top-K 验证成功率。

不需要：所有模块笛卡尔积、每层遍历、guard radius 大表、\(\delta\) 大表、多 seed 全重跑、额外 smoothing baseline、装饰性神经元截图。

## 8. 最短执行顺序

1. P0 数据/evaluator。
2. E0–E3；event story 不成立就停。
3. predictor 泄漏与梯度测试。
4. V1–V5；V4 不成立就停。
5. context-matched intervention 和机制主图。
6. 最后做 probe、tag、erase、patch。
7. 汇总四张表/图，按实际证据强度写 claim。
