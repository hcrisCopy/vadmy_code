# WITNESS-VAD v10 项目搭建与核心消融指南

> **方法依据**：docs/witness-vad-final-v10-iclr-2026-09-02.md
>
> **实验对象**：先在 DSANet 上跑通 UCF-Crime 与 XD-Violence；DSANet 通过后再迁移其他 baseline。
>
> **执行原则**：每完成一个阶段必须停下汇报。没有通过当前阶段的生死门，不进入下一阶段。
>
> **边界**：只在本仓库写代码；训练、测试和结果分析都在远程服务器执行；产物统一写到 ../vadmy_data。
>
> **当前进度**：v9 的 B0--E1 已归档；v10 的 F0--F2 已通过；F3 已按 baseline-compatible checkpoint 选择协议完成并判定 NO-GO，尚未进入 F4。

---

## 0. 我们究竟要证明什么

论文只有三条核心 claim，所有实验都服务于它们：

1. **异常不是一个类别，而是少量可定位的 witness neurons 共同作证。**
   - neuron-only 必须在两个数据集都带来正增益。
   - 去掉 host residual weighting 或 normal dense constraint 后必须退化。
2. **视频级证据只决定“要不要改”，神经元证据决定“在哪里改”。**
   - Full 必须显著好于 video-only。
   - 整个方法必须一次训练、一个 optimizer、一个 checkpoint，禁止交替训练和冻结拼接。
3. **神经元证据能够形成可验证的时间标签，而不是事后画热力图。**
   - 擦除高分 witness 后异常分数下降。
   - 正常→异常 patch 后分数上升，异常→正常 patch 后分数下降。
   - 删除某个 tag 对应的神经元后，只有对应的时间行为应明显减弱。

如果某个实验不能支撑以上三条中的任何一条，就不做。

---

## 1. 固定输入、输出与不可改边界

### 1.1 输入

- hidden states：H ∈ R^[B,T,12,768]，CLIP ViT-B/16 每个 snippet 的 12 层 CLS hidden states。
- host score：s_host ∈ [0,1]^[B,T]，直接复用冻结 DSANet 的测试分数。
- video label：y ∈ {0,1}。
- valid mask：m ∈ {0,1}^[B,T]。

### 1.2 输出

- witness evidence：E ∈ [0,1]^[B,T]。
- video route：q ∈ [0,1]。
- corrected score：s_corr ∈ [0,1]^[B,T]。
- neuron statistics：A、G、C、N、S。
- temporal tags：由响应原型与干预结果共同定义，不允许人工命名后反推证据。

### 1.3 禁止修改

- 不改 DSANet、RTFM 等 baseline 的训练和测试逻辑。
- 不重新训练 CLIP，不增加文本分支。
- 不引入 v9 的 field、budget、kappa_cross、boost/suppress 双门。
- 不恢复 Universal 的两阶段冻结、两套学习率、40+ 手工统计量、时长分支和多种平滑后处理。
- checkpoint 选择严格对齐 DSANet/VadCLIP 官方实现：UCF 按测试 AUC、XD 按测试 AP 选 best epoch；保存全部 epoch 指标，禁止事后改选择指标。

---

## 2. 正式超参数：从 Universal 继承有效经验，但去掉拼装

以下配置是 DSANet 正式起点，不做大网格：

| 项目 | 固定值 |
|---|---:|
| seed | 42 |
| 每层 active neurons | 32 |
| layer count | 12 |
| temporal width | 64 |
| temporal encoder | Conv1D k=3,d=1 → GELU → Conv1D k=3,d=2 → GELU → Conv1D 1×1 |
| max snippets | 256 |
| MIL top-k | max(1, floor(T/16)+1) |
| epochs | 20 |
| batch size | 8 |
| optimizer | AdamW |
| learning rate | 3e-4 |
| weight decay | 1e-4 |
| scheduler | cosine |
| ranking margin / weight | 0.5 / 0.5 |
| smoothness weight | 0.02 |
| λ_video / λ_W / λ_mil | 1.0 / 1.0 / 1.0 |
| λ_N / λ_S | 0.5 / 1e-3 |
| η_N / η_A | 1.0 / 0.25 |
| video summary | 10 维：host 与 neuron 各 mean/std/top10/max，加 corr 与 MAE |
| model selection | UCF test AUC best / XD test AP best（对齐 DSANet/VadCLIP） |

从 Universal 只继承三个已验证且不引入拼装的数值细节：active-neuron 求和除以
`sqrt(32)`、softplus ranking margin、视频内标准化 neuron evidence 的局部直连。
它们分别稳定尺度、保留难样本梯度、保证 witness 不会被强 host 忽略；不恢复测试期拟合和手工后处理。

允许的调参只有：

- 若 Full 与 video-only 差距不足，仅试 η_A ∈ {0.15, 0.25, 0.35}。
- 若 neuron-only 不涨，仅试 λ_W ∈ {0.5, 1.0, 2.0}。
- 每次只改一个量，先 UCF，再用同一配置测 XD。不得为两个数据集写不同分支。

---

## 3. 文件与产物规划

### 3.1 新增代码

~~~text
vin_vad/
  universal_autopsy.py
  witness_neurons.py
  witness_temporal.py
  witness_router.py
  witness_losses.py
  witness_model.py
  train_witness.py
  evaluate_witness.py
  score_witness_neurons.py
  intervene_witness.py
  tag_witness.py
tests/
  test_witness_neurons.py
  test_witness_router.py
  test_witness_losses.py
  test_witness_intervention.py
run_instructions/
  RUN_WITNESS_VAD_F0_DSANET.md
  RUN_WITNESS_VAD_DSANET.md  # F1--F5 完成后生成总入口
  run_witness_vad_f0_dsanet.sh
  run_witness_vad_f1_dsanet.sh
  run_witness_vad_f2_dsanet.sh
  run_witness_vad_f3_dsanet.sh
  run_witness_vad_f4_dsanet.sh
  run_witness_vad_f5_dsanet.sh
  run_witness_vad_dsanet_all.sh
~~~

每个 shell 脚本必须把 Python 参数显式写全；RUN_WITNESS_VAD_DSANET.md 必须原样复制同一条命令，不能依赖代码默认值偷偷补参数。

### 3.2 远程输出

~~~text
../vadmy_data/witness_vad/dsanet/
  f0_universal_autopsy/
  f1_smoke/
  f2_train_contract/
  f3_performance/
  f4_ablation/
  f5_interpretability/
~~~

每阶段至少包含：

- command.txt：实际执行命令。
- config.json：完整配置。
- metrics.json：机器可读指标。
- summary.md：人能直接读懂的结论。
- stdout.log：完整日志。

训练阶段额外保存 checkpoints/last.pt、optimizer 状态、scheduler 状态和 epoch。

---

## 4. 远程统一执行规则

每次登录后：

~~~bash
cd <remote_repo>/vadmy_code
source /etc/network_turbo
git pull origin main
conda activate <vad_env>
~~~

续跑：

~~~bash
bash run_instructions/run_witness_vad_fN_dsanet.sh --resume
~~~

重跑某阶段：

~~~bash
bash run_instructions/run_witness_vad_fN_dsanet.sh --clean
~~~

clean 只能删除该阶段精确目录，脚本必须先打印绝对路径并校验它位于 ../vadmy_data/witness_vad/dsanet/ 下。

---

## 5. F0：Universal 提点验尸

### 目的

不猜 Universal 为什么涨点，直接量出信息来源。只跑缓存推理，不重新训练。
F0 沿用历史 Universal 的 seed 234；它不是 v10 训练。F1 之后的新模块仍固定 seed 42。

### 比较

| 编号 | 配置 | 回答的问题 |
|---|---|---|
| U0 | host only | 基线 |
| U1 | Universal full | 已知涨点复现 |
| U2 | 去视频级 suppression | 增益是否主要来自整体压低正常视频 |
| U3 | 去 neuron-derived local correction | 神经元是否真的提供定位信息 |
| U4 | 去手工 temporal rules | 旧方案涨点是否依赖 median/Gaussian/dilation/advance |

### 执行

~~~bash
bash run_instructions/run_witness_vad_f0_dsanet.sh
~~~

完整参数与续跑、清理命令见
run_instructions/RUN_WITNESS_VAD_F0_DSANET.md。

### 看哪里

~~~text
../vadmy_data/witness_vad/dsanet/f0_universal_autopsy/summary.md
../vadmy_data/witness_vad/dsanet/f0_universal_autopsy/comparison.csv
../vadmy_data/witness_vad/dsanet/f0_universal_autopsy/information_source.json
~~~

### 生死门

- U1 必须复现历史结果，误差 ≤ 0.1pp。
- 必须明确主增益来自 video suppression、local correction 或二者组合。

### 阶段汇报

停下来报告：U0–U4 两数据集指标、最大增益来源、远程产物路径。

### F0 已完成记录（2026-09-02）

正式命令：

~~~bash
bash run_instructions/run_witness_vad_f0_dsanet.sh --clean
~~~

| 数据集 | Host | Full | Full 增益 | 去 suppression 损失 | 去局部神经元损失 | 去 temporal rules 损失 |
|---|---:|---:|---:|---:|---:|---:|
| UCF AUC | 89.445 | 90.503 | +1.058 | 0.242 | 0.709 | 0.292 |
| XD AP | 86.951 | 88.166 | +1.215 | 0.776 | 1.018 | 0.259 |

- 4 个单元测试全部通过；两数据集的 Full 均以小于 0.001 pp 的误差复现历史结果。
- UCF 的主导来源是局部神经元校正；XD 是局部校正与视频级 suppression 的组合。
- 只保留 suppression 时，UCF/XD 的 Macro-Within-AUC 都低于 host。不能把“压低正常”
  写成定位贡献；局部神经元校正才负责补回视频内排序。
- 手工 temporal rules 只有约 0.26--0.29 pp 的 pooled 增益，而且没有稳定改善
  Macro-Within-AUC。v10 不复制这些规则，只保留可学习的最小时序 readout。
- **裁决：PASS，允许进入 F1；本轮按约定停在 F1 开始前。**

远程结果：

~~~text
../vadmy_data/witness_vad/dsanet/f0_universal_autopsy/
~~~

---

## 6. F1：单体结构与梯度闭环

### 搭建顺序

1. 复用 B0 cache loader 与统一 evaluator。
2. 实现 signed top-k witness neurons：每层 32 个，保留正负权重。
3. 实现固定 d=1/d=2 的最小时序编码器。
4. 实现 10 维 video summary 与单一 route q。
5. 实现 routed residual：
   - 正常路由只允许 uniform suppression。
   - 异常路由只允许视频内零均值的 neuron-derived local correction，不能退化成第二个全局偏置。
6. 实现 video loss、host-residual-weighted neuron MIL、final MIL、normal dense loss、sparsity loss。
7. 合并成一次 forward、一次 backward、一个 optimizer。

### 必过测试

- η_N=η_A=0 时 s_corr 与 s_host 完全一致。
- normal route 不能产生正增量；anomaly route 在每个视频上必须严格零均值，不能产生全局常数偏移。
- padding 不进入 pooling、top-k、smoothness 和指标。
- 每层 active neuron 数恰为 32，signed 权重可导。
- 每个 loss 都能把非零梯度传回 witness 参数。
- neuron-only 推理不得读取 host score 作为输入特征。
- 删除某个 tag 的 mask 只影响对应 neuron 子集。

### 执行

~~~bash
bash run_instructions/run_witness_vad_f1_dsanet.sh
~~~

### 看哪里

~~~text
../vadmy_data/witness_vad/dsanet/f1_smoke/summary.md
../vadmy_data/witness_vad/dsanet/f1_smoke/test_report.txt
../vadmy_data/witness_vad/dsanet/f1_smoke/gradient_report.json
~~~

### 生死门

所有测试通过；否则不允许上 GPU 正式训练。

### 阶段汇报

停下来报告：测试数量、失败项、梯度是否覆盖所有模块、远程产物路径。

### F1 已完成记录（2026-09-02）

正式通过命令：

~~~bash
bash run_instructions/run_witness_vad_f1_dsanet.sh --resume
~~~

- 10 个单元测试全部通过，失败 0 项。
- UCF/XD 的真实 B0 训练清单均抽查一支正常、一支异常视频；hidden `[T,12,768]`
  与 host score 长度完全对齐。
- 每层恰有 32 个 active neuron；signed weight、tag mask、padding 隔离均通过。
- `eta_N=eta_A=0` 的 host 恒等误差为 0；正常分支最大增量为 `-0.1897`，没有正向加分；
  异常分支视频内均值绝对误差为 `6.40e-10`，不能充当全局偏置。
- video、host-residual witness MIL、final MIL、dense normal、sparse 五项损失都能回传到
  witness 参数。联合损失对 gate、signed weight、层权重、temporal readout、video head、
  local head、`eta_N`、`eta_A` 八组参数的梯度均非零。
- 只有一个 `AdamW`、一次 forward/backward/step；总训练参数 33,339。
- 首次远程运行发现优化一步后的零强度路径有 `5.96e-8` 浮点往返误差；已改成显式 bitwise
  host identity 并增加回归测试，复跑通过。
- **裁决：PASS，结构和梯度闭环成立，允许进入 F2；本轮按约定停在 F2 开始前。**

远程结果：

~~~text
../vadmy_data/witness_vad/dsanet/f1_smoke/
~~~

---

## 7. F2：训练、续跑与确定性

F2 只在 UCF 训练清单上做一次小规模工程验证。续跑和确定性是数据集无关的代码契约，
不在 XD 重复浪费算力；UCF/XD 的完整训练和指标统一留到 F3。

### 必须实现

- tqdm 显示 epoch、batch、总 loss 与各子 loss。
- 每个 epoch 保存 last.pt；中断后恢复 model、optimizer、scheduler、epoch。
- config.json、command.txt、git commit hash 自动落盘。
- 同 seed 的两次 smoke run 首个 epoch 指标近似一致。
- train 阶段不读取测试 GT，不在测试集挑 checkpoint。

### 执行

~~~bash
bash run_instructions/run_witness_vad_f2_dsanet.sh
~~~

### 看哪里

~~~text
../vadmy_data/witness_vad/dsanet/f2_train_contract/summary.md
../vadmy_data/witness_vad/dsanet/f2_train_contract/resume_report.json
../vadmy_data/witness_vad/dsanet/f2_train_contract/checkpoints/last.pt
~~~

### 生死门

训练可中断续跑，续跑后 epoch 与学习率连续；两个确定性 smoke run 的首轮 loss 相对误差 ≤ 1%。

### 阶段汇报

停下来报告：续跑是否成功、checkpoint 位置、显存和单 epoch 时间、确定性误差。

### F2 已完成记录（2026-09-02）

正式通过命令：

~~~bash
bash run_instructions/run_witness_vad_f2_dsanet.sh --resume
~~~

- 14 个单元测试全部通过，失败 0 项。
- 使用 UCF 真实训练清单中固定的 8 支正常、8 支异常视频，batch size 8、seed 42；
  不读取测试清单、测试 GT，也不按测试指标挑 checkpoint。
- 连续训练跑 2 个 epoch；另一条 run 在 epoch 1 计划中断并从 `last.pt` 恢复到 epoch 2。
- 两条 run 的首轮总 loss 都是 `2.1401556730`，相对误差为 0。
- epoch 1 结束学习率与 epoch 2 恢复学习率都是 `1.5e-4`，连续性误差为 0。
- 连续训练与续跑训练的最终参数最大绝对误差为 0；checkpoint 完整包含 model、optimizer、
  scheduler、epoch、history 以及 Python/NumPy/PyTorch/CUDA RNG 状态。
- 峰值显存 `164.0 MiB`，平均每个 smoke epoch `1.30 s`。
- 首次续跑准确暴露了 RNG tensor 被 `map_location=cuda` 搬错设备的问题；修复后直接从已保存的
  epoch 1 恢复成功，没有重新训练首轮。
- **裁决：PASS，训练、续跑和确定性契约成立，允许进入 F3；本轮按约定停在 F3 开始前。**

远程结果：

~~~text
../vadmy_data/witness_vad/dsanet/f2_train_contract/
~~~

---

## 8. F3：性能生死门

只跑四个正式变体：

| 编号 | video route | neuron localization | 用途 |
|---|---:|---:|---|
| W0 | 否 | 否 | frozen host |
| W1 | 是 | 否 | video-only |
| W2 | 否 | 是 | neuron-only |
| W6 | 是 | 是 | Full |

W1 只能使用 host 的视频级统计；W2 仍以 host score 作为最终残差基底，但不能把它输入 neuron 分支。四组共用数据、seed、epoch 和 evaluator。

每轮都保存 checkpoint 并用同一 evaluator 测试；UCF 只按 frame AUC、XD 只按 frame AP
选择 best epoch。`selection_curve.csv` 必须保留 20 轮完整轨迹，不能看到结果后改选择指标。

### 执行

~~~bash
bash run_instructions/run_witness_vad_f3_dsanet.sh
~~~

### 看哪里

~~~text
../vadmy_data/witness_vad/dsanet/f3_performance/summary.md
../vadmy_data/witness_vad/dsanet/f3_performance/main_results.csv
../vadmy_data/witness_vad/dsanet/f3_performance/error_decomposition.json
../vadmy_data/witness_vad/dsanet/f3_performance/<dataset>/<variant>/selection/selection_curve.csv
../vadmy_data/witness_vad/dsanet/f3_performance/<dataset>/<variant>/selection/selection.json
../vadmy_data/witness_vad/dsanet/f3_performance/<dataset>/<variant>/training/checkpoints/best.pt
~~~

### 生死门

同时满足才进入后续实验：

- W6 − W0：UCF-Crime AUC ≥ +1.0pp。
- W6 − W0：XD-Violence AP ≥ +1.0pp。
- W6 − W1：两个数据集都 ≥ +0.2pp，证明不是只靠压低正常视频。
- W2 − W0：两个数据集都 > 0，证明神经元分支本身有信息。
- normal FPR 改善不能伴随异常视频内排序的明显下降；同时报告 within-video AUC、abnormal-only AUC/AP。

这里预先把“明显下降”定义为 Macro-Within-AUC 下降超过 0.2 pp，避免结果出来后改口径。

若失败，只允许按第 2 节的小范围单变量规则调整一次。仍失败则停止：方法尚未 work，不做可解释性包装。

### 阶段汇报

停下来报告 W0/W1/W2/W6、是否跨过 +1pp、增益来自跨视频还是视频内排序、下一步 go/no-go。

### 已完成记录（2026-09-02）

正式复现命令：

~~~bash
bash run_instructions/run_witness_vad_f3_dsanet.sh --clean
~~~

四组均从同一 host cache 训练 20 epoch；每轮保存 checkpoint，并按 UCF frame AUC、XD frame AP 选择 test-best epoch，平局取更早 epoch。结果：

| Dataset | W0 | W1 | W2 | W6 | W6-W0 | W6-W1 | W2-W0 |
|---|---:|---:|---:|---:|---:|---:|---:|
| UCF AUC | 89.445 | 89.449 | 89.451 | 89.476 | +0.032 | +0.027 | +0.007 |
| XD AP | 86.951 | 86.979 | 87.420 | 87.452 | +0.501 | +0.473 | +0.469 |

best epoch：UCF 的 W0/W1/W2/W6 为 20/3/2/20；XD 为 20/20/16/19。

- **裁决：NO-GO。** W6 在两个数据集都没有达到 `+1.0 pp`，UCF 也没有达到 W6-W1 `+0.2 pp`；因此不得进入 F4。
- UCF 的 W6 主要改善跨视频排序与正常误报，异常视频内 AUC/AP 反而略降，不能宣称定位更好。W2 虽使 Macro-Within-AUC 提升约 `0.322 pp`，但主 AUC 只提升 `0.007 pp`。
- XD 的 neuron route 有真实信号：W2 提升 `0.469 pp`，视频内指标和正常 FPR 同时改善；但 W6 只比 W2 再多约 `0.033 pp`，video route 基本没有增加有效信息。
- 按预注册规则，下一步最多允许一次小范围单变量调整；本阶段没有启动调参。

远程结果：

~~~text
../vadmy_data/witness_vad/dsanet/f3_performance/
~~~

---

## 9. F4：审稿人真正需要的结构消融

在 W6 基础上只做三组：

| 编号 | 删除内容 | 审稿问题 |
|---|---|---|
| W3 | host residual weighting r_h | witness 是否真的针对 baseline 残差学习 |
| W4 | normal dense constraint | 正常数据是否提供了关键的负证据约束 |
| W5 | temporal readout，改为逐 snippet 线性头 | 时间上下文是否必要 |

### 执行

~~~bash
bash run_instructions/run_witness_vad_f4_dsanet.sh
~~~

### 看哪里

~~~text
../vadmy_data/witness_vad/dsanet/f4_ablation/summary.md
../vadmy_data/witness_vad/dsanet/f4_ablation/core_ablation.csv
~~~

### 生死门

- W6 应在两个数据集都优于 W3、W4、W5。
- 若某项在两个数据集都无影响，删掉该设计和对应 claim，不用话术硬保。
- 不做层数、宽度、激活函数、所有 loss 权重的穷举表。

### 阶段汇报

停下来报告每项删除造成的变化，以及论文中应保留或删除的机制。

---

## 10. F5：神经元因果证据与 temporal tags

### 10.1 神经元打分

按最终方案计算：

- A：异常样本激活强度。
- G：对输出的梯度敏感性。
- C：与异常时间位置的一致性。
- N：正常样本激活惩罚。
- S = A·G·C/(N+ε)。

先固定 hash 后的评测 neuron 集合，再运行任何干预，防止挑案例。

### 10.2 必做对照

| 方法 | 作用 |
|---|---|
| WITNESS score | 主方法 |
| activation × gradient | 神经元解释基线 |
| same-layer random | 随机对照 |
| contribution-matched | 排除只是挑了大权重神经元 |

### 10.3 三个干预

1. Erase：置零 top witness neurons，异常分数应显著下降。
2. Normal→Anomaly patch：把正常 snippet 的对应神经元替换为异常原型，分数应上升。
3. Anomaly→Normal patch：反向替换，分数应下降。

### 10.4 Temporal tag 定义

tag 来自可计算的响应形状，例如：

- onset：边界前后响应跃升。
- sustain：异常区间持续高响应。
- burst：短时高峰。
- recovery：异常结束后快速回落。

每个 tag 必须同时给出：神经元集合、时间响应原型、代表视频、删除该集合后的定量变化。只画热力图不算证据。

### 执行

~~~bash
bash run_instructions/run_witness_vad_f5_dsanet.sh
~~~

### 看哪里

~~~text
../vadmy_data/witness_vad/dsanet/f5_interpretability/summary.md
../vadmy_data/witness_vad/dsanet/f5_interpretability/neuron_ranking.csv
../vadmy_data/witness_vad/dsanet/f5_interpretability/intervention_results.csv
../vadmy_data/witness_vad/dsanet/f5_interpretability/tags/
~~~

### 生死门

- WITNESS 的 erase 降幅显著大于 random 和 contribution-matched。
- 两个 patch 方向都正确，并在预先固定的视频集合上统计。
- 至少两个 tag 的删除结果具有可区分的时间效应。
- 若只有漂亮图、没有干预差异，解释性 claim 降级为相关性观察。

### 阶段汇报

停下来报告：排名基线对比、三种干预效应量、可靠 tag 数量、代表图路径。

---

## 11. 何时迁移其他 baseline

只有 F3–F5 全部通过后：

1. 选择第二个结构明显不同的 host，完整复现 W0/W1/W2/W6。
2. 超参数完全沿用 DSANet，不重新搜索。
3. 其余 baseline 只跑 W0 与 W6，进入主结果表。
4. 核心结构消融只在 DSANet 做，不在所有 baseline 重复。

这一步证明的是 host-agnostic，不是靠 DSANet 特调。

---

## 12. ICLR 最小充分证据包

| 论文 claim | 最小证据 |
|---|---|
| 硬性能 | W6 在 UCF AUC、XD AP 都比 W0 ≥ +1pp |
| 不是正常视频压分 trick | W6 > W1；报告 within-video 与 abnormal-only 指标 |
| witness neurons 有独立信息 | W2 > W0；W6 > W3/W4 |
| 时序建模必要 | W6 > W5 |
| 神经元定位可信 | WITNESS erase > 两个匹配对照 |
| 解释具有方向性 | 两个 patch 方向正确 |
| tag 不是命名游戏 | tag 删除产生可区分时间效应 |
| 方法不是训练拼装 | 单次训练、单 optimizer、单 checkpoint |
| 可迁移 | 第二个 host 不调参仍有稳定增益 |

---

## 13. 明确不做

- 不做几十个 loss 权重和网络宽度表。
- 不做没有 matched control 的 neuron visualization。
- 不用正常视频整体压低单独冒充定位能力。
- checkpoint 选择严格沿用 host baseline 的固定主指标与 test-best 规则；不挑 seed、不在结果出来后更换选择指标。
- 不在 F3 失败后继续堆模块。
- 不把 Universal 的工程 trick 全搬回来；只保留被 F0 证实的信息来源与简洁超参数经验。

---

## 14. 最终一键入口

待 F0–F5 的分阶段脚本全部验证后，提供：

~~~bash
bash run_instructions/run_witness_vad_dsanet_all.sh
~~~

一键脚本必须：

- 依次执行 F0→F5。
- 每阶段写独立日志与状态文件。
- 已完成阶段自动跳过。
- 生死门失败立即停止，并在 summary.md 写清失败条件。
- 最后输出所有产物绝对路径和一张总表。

这份指南是搭建与裁决标准。方法公式、研究动机和论文表述以最终方案文档为准。
