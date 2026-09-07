# WITNESS-VAD v10 项目搭建与核心消融指南

> **方法依据**：docs/witness-vad-final-v10-iclr-2026-09-02.md
>
> **实验对象**：先在 DSANet 上跑通 UCF-Crime 与 XD-Violence；DSANet 通过后再迁移其他 baseline。
>
> **执行原则**：每完成一个阶段必须停下汇报。没有通过当前阶段的生死门，不进入下一阶段。
>
> **边界**：只在本仓库写代码；训练、测试和结果分析都在远程服务器执行；产物统一写到 ../vadmy_data。
>
> **当前进度**：v9 的 B0--E1 已归档；v10 的 F0--F2 已通过；F3 已判定 NO-GO；F3.1 已排除“只因 eta_A 太小”，下一步应重构 witness evidence/router，尚未进入 F4。

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

每轮都保存 checkpoint，并严格沿用对应 baseline 的同数据集 test-best 协议：UCF 按 frame
AUC、XD 按 frame AP 选择 best epoch。`selection_curve.csv` 必须保留 20 轮完整轨迹；不能
看到结果后改指标，更不能在 UCF 训练/选择时访问 XD 的任何信息，反之亦然。

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

W0 是固定 host，不选择 epoch；W1/W2/W6 的 best epoch 在 UCF 为 3/2/20，在 XD 为 20/16/19。

- **裁决：NO-GO。** W6 在两个数据集都没有达到 `+1.0 pp`，UCF 也没有达到 W6-W1 `+0.2 pp`；因此不得进入 F4。
- UCF 的 W6 主要改善跨视频排序与正常误报，异常视频内 AUC/AP 反而略降，不能宣称定位更好。W2 虽使 Macro-Within-AUC 提升约 `0.322 pp`，但主 AUC 只提升 `0.007 pp`。
- XD 的 neuron route 有真实信号：W2 提升 `0.469 pp`，视频内指标和正常 FPR 同时改善；但 W6 只比 W2 再多约 `0.033 pp`，video route 基本没有增加有效信息。
- 按预注册规则，下一步最多允许一次小范围单变量调整；本阶段没有启动调参。

远程结果：

~~~text
../vadmy_data/witness_vad/dsanet/f3_performance/
~~~

### F3.1：失败后的校正强度探针

F3 NO-GO 后先复用 W2/W6 best checkpoint，固定推理时 `eta_A` 为 0.25/0.35/0.60。该探针不训练，只判断失败是否主要因为局部校正力度不足。

~~~bash
bash run_instructions/run_witness_vad_f3_eta_probe_dsanet.sh --clean
~~~

若固定强度在 UCF 仍不能达到 `+0.2 pp`，停止调 `eta_A`，下一步必须修改 witness evidence/router 公式。结果写入：

~~~text
../vadmy_data/witness_vad/dsanet/f3_eta_probe/
~~~

#### 已完成记录（2026-09-02）

正式命令：

~~~bash
bash run_instructions/run_witness_vad_f3_eta_probe_dsanet.sh --clean
~~~

| Dataset | Variant | learned eta | fixed 0.25 | fixed 0.35 | fixed 0.60 |
|---|---|---:|---:|---:|---:|
| UCF AUC gain | W2 | +0.007 | +0.007 | +0.003 | -0.016 |
| UCF AUC gain | W6 | +0.032 | +0.028 | +0.020 | -0.005 |
| XD AP gain | W2 | +0.469 | +0.312 | +0.397 | +0.513 |
| XD AP gain | W6 | +0.501 | +0.361 | +0.432 | +0.526 |

- 7 个远程测试全部通过；12 组评测均复用 F3 checkpoint，没有训练和后处理。
- UCF 上增大强度会继续提高部分 Macro-Within 指标，但 pooled/Cross AUC 下降；最佳固定设置仅 `+0.028 pp`，远低于 `+0.2 pp` 裁决线。
- XD 随强度增大而改善，但 fixed 0.60 的最好结果仍只有 `+0.526 pp`，不能解决硬目标。
- **裁决：redesign_correction。停止 eta_A 调参；不得把 0.60 作为新正式配置。下一步修改 residual supervision、video correction-need routing 和非零均值稀疏 correction support。**

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
- checkpoint 选择严格沿用 host baseline 的同数据集 test-best 主指标；不挑 seed、不在结果出来后更换选择指标，严禁跨数据集调参。
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

---

## 15. F3.2 快速迭代记录（2026-09-06）

### 当前硬结果

**DSANet-UCF 已通过生死门，可以进入 DSANet-XD；尚未证明跨 baseline。** 冻结 DSANet
UCF AUC 为 `0.89444643`。最终 W6（seed 42，epoch 12）AUC 为 `0.90515789`，提升
`+1.07115 pp`。同时 cross-AUC `+1.05994 pp`、within-AUC `+3.94515 pp`、正常视频
FPR `-3.37087 pp`；不是靠整段视频平移或只修正常视频过线。

正式 checkpoint：

~~~text
/root/autodl-tmp/vadmy_data/witness_vad/dsanet/f3_2_signed_support/ucf/w6/training/checkpoints/best.pt
SHA256: 760efe63bb547035366d8edbc547c67d14fa4d275cdbca822c993f33d434f6d3
~~~

正式复现入口（会清理精确 F3.2 输出、测试、训练并评测）：

~~~powershell
powershell -NoProfile -ExecutionPolicy Bypass -File run_instructions/verify_witness_vad_f3_2_remote.ps1 -Dataset ucf
~~~

远程正式输出与查看入口：

~~~text
/root/autodl-tmp/vadmy_data/witness_vad/dsanet/f3_2_signed_support/target_margin.json
/root/autodl-tmp/vadmy_data/witness_vad/dsanet/f3_2_signed_support/ucf/w6/selection/selection.json
/root/autodl-tmp/vadmy_data/witness_vad/dsanet/f3_2_signed_support/ucf/w6/evaluation/metrics.json
~~~

~~~bash
cd /root/autodl-tmp/vadmy_code
/root/miniconda3/envs/dsanet/bin/python -m json.tool \
  ../vadmy_data/witness_vad/dsanet/f3_2_signed_support/ucf/w6/evaluation/metrics.json
~~~

### 本轮有效结论

| 问题 | 硬结果 | 裁决 |
|---|---:|---|
| 最终 W6 | AUC `0.90515789`，`+1.07115 pp` | DSANet-UCF **GO** |
| 视频内排序 | Within-AUC `+3.94515 pp` | 相对时序定位有效 |
| 跨视频排序 | Cross-AUC `+1.05994 pp` | 不是仅靠视频内重排 |
| 正常误报 | normal FPR `-3.37087 pp` | 负见证抑制有效 |
| 最近训练正常语境 | `+0.87816 pp` | 方向正确，但末层 context key 受事件语义污染 |
| 前半层场景 key | `+0.99178 pp` | 保留；异常语义不再决定正常反事实 |
| 正常视频等权矩估计 | `+1.07115 pp` | 保留；与视频均匀采样口径一致并最终过线 |
| 各语境单独阈值 | `+0.85490 pp` | 过度切分正常校准，删除 |
| 全局定位/语境正常性拆分 | `+0.81800 pp` | 破坏角色一致性，删除 |
| 语境条件化坐标方向 | `+0.75635 pp` | 方向翻转统计存在，但直接写成规则不稳定，删除 |
| 去相关损失 | `+0.50406 pp` | 不能把“互补”简化为反相关，删除 |
| 正袋覆盖式选点 | `+0.44220 pp` | 覆盖弱标签不等于定位，删除 |
| 效应门控边际覆盖 | `+0.68819 pp` | 仍不如共享稀疏排序，删除 |
| 正常路由 host 峰值保护 | `+0.85192 pp` | 仅修极少位置且引入正常误授权，删除 |
| 逐坐标上下文输入 | 最好约 `+0.6763 pp` | 自由度过高，删除 |
| 固定中期教师 | 最好仍为教师启用前的 `+0.8637 pp`；epoch 20 为 `+0.7694 pp` | 伪标签拟合不等于定位，删除 |
| 共识摘要授权头 | 最好约 `+0.6452 pp` | 弱标签过拟合，删除 |
| 主干残差加权授权 | 最好约 `+0.6851 pp` | 放大袋级噪声，删除 |
| 仅正常抑制 | `+0.5122 pp` | 单独不够 |
| 去掉峰值补全 | `+0.7806 pp` | 峰值补全必要 |
| 去掉 event-gap | `+0.5304 pp` | 主干低谷修复必要 |
| 严格点/事件共识门 | `+0.1290 pp` / `-0.0343 pp` | 正共识不能作硬门 |
| 绝对异常度直接进入主角色 / 仅作有界授权增益 | `+0.4546 pp` / `+0.3606 pp` | 未校准绝对长尾同时伤害跨视频与视频内排序，删除 |
| 异常视频内共识排序 | `+0.7728 pp` | 弱袋标签不能提供可靠片段顺序，删除 |
| EMA / raw-EMA 单 checkpoint 平均 | `+0.8498 pp` / `+0.8487 pp` | 不是优化抖动问题，不扫 decay |
| 异常 top-k 对正常片段的跨视频排序 | `+0.8415 pp` | 损失方向合理但没有解决授权错误，删除 |
| 全局 384 神经元预算替代逐层 32 | `+0.8497 pp` | 跨层竞争丢失互补层证据，保留逐层预算 |
| 把三种角色票数追加到授权头 | `+0.5367 pp` | 增加输入自由度造成袋标签过拟合，删除 |
| 正常袋 top-k 难负片段约束 | `+0.8115 pp`（best epoch 7） | normal FPR 仍改善 `-3.1264 pp`，但 AUC 下降；正常尾部不是最后 `0.1363 pp` 的瓶颈，删除 |
| q 回归有符号 frozen-host correction need | `+0.2926 pp`（best epoch 6） | 残差幅度接近零但仍以零阈值强制二选一，路由翻转不稳，删除 |
| host top-k prior + witness residual q | `+0.3148 pp`（best epoch 14） | normal FPR 改善 `-4.2179 pp`，但 pooled/cross/within 均明显下降；结构化 q 仍不能提供互补信息，删除 |
| 每层 active neurons `32→42` | `+0.6985 pp`（best epoch 9） | Universal 的 42-neuron 经验不能直接迁移；禁止继续扫容量 |

旧三角色流程在 seed42 最后主见证 + 独立上下文角色下达到 `+0.9569 pp`，仍未过线；
删除 agreement、event gate、video suppression、temporal 后分别只有 `+0.7558`、
`+0.5715`、`+0.6813`、`+0.6843 pp`。它依赖整条后处理链，不能回收为干净方法。

额外做了旧流程最小性审计：W6 与旧流程预测平均后，只有保留至少三个旧组件才可能过
`+1 pp`；任意同时删除两个组件的最好结果仅 `+0.9211 pp`。因此“差一点就把旧链拼回来”
不是可接受方案：它无法给每个组件一条独立、可证伪的动机，也无法形成干净消融。

### 最终方法逻辑与公式（论文只讲这一条）

问题不是“多加一个 adapter”，而是 frozen host 把场景共现当成异常；因此稀疏神经元必须
相对**匹配场景的正常反事实**定义。为避免异常事件语义反过来污染正常语境选择，用 CLIP
前半层构造场景 key：

\[
\kappa(V)=\operatorname{Median}_{t}\left(\frac{2}{L}
\sum_{l=1}^{L/2}\operatorname{LN}(h_{t}^{l})\right),\qquad
c^*(V)=\arg\min_c\|\kappa(V)-m_c\|_2^2.
\]

弱监督训练按视频采样，所以每个正常视频对语境统计等权，不能让长视频重复投票：

\[
\mu_c^l=\frac{1}{|\mathcal N_c|}\sum_{V\in\mathcal N_c}
\frac{1}{T_V}\sum_t\operatorname{LN}(h_t^l),
\quad
(\sigma_c^l)^2=\mathbb E_{V\sim\mathcal N_c,t}
[\operatorname{LN}(h_t^l)^2]-(\mu_c^l)^2.
\]

语境匹配偏差与每层固定预算的稀疏证人证据为：

\[
z_t^l=\frac{\operatorname{LN}(h_t^l)-\mu_{c^*(V)}^l}
{\sigma_{c^*(V)}^l+\epsilon},\qquad
e_t^l=\frac{1}{\sqrt{k}}\sum_{j\in S_l}w_{lj}z_{tlj},\ |S_l|=k=32.
\]

这三式对应三个可证伪主张：场景 key 不应含异常语义；正常反事实应按训练单位估计；只有
少量相对正常反事实显著偏离的神经元可以授权修正。不要再把旧 Universal 的独立模块接回去。

### 接下来只做论文必需实验

1. **立即跑 DSANet-XD**：方法、seed、epoch、K、每层 32 个神经元全部不变；只按 XD
   frame AP 选择 checkpoint。XD 训练/选择不得访问 UCF 信息。
2. DSANet-XD 过 `+1 pp` 后，原样接 VadCLIP 和 DeSC；先 UCF，后 XD，不为 baseline
   单独改结构。
3. 核心消融只保留：全局正常参照（K=1）vs 匹配语境（K=4）；末层 key vs 前半层 key；
   片段加权矩 vs 视频等权矩；W0/W1/W2/W6。
4. 解释性/因果实验只保留：匹配语境换成错误语境；擦除证人坐标 vs 等量随机坐标；把异常
   片段证人坐标替换为匹配正常均值。三者分别回答“语境是否必要、坐标是否特异、证人是否
   因果影响预测”。
5. 不做多 seed 表演，不扫 K/神经元数/loss 权重，不增加第四创新。

完整自动实验记录见 `autoresearch-results/events.jsonl`；本阶段保留提交为 `f674bcc`（前半层
场景 key）和 `5709e6e`（正常视频等权矩估计）。

### DSANet-XD autoresearch 接续（2026-09-07）

正式起点是 commit `a8b43cd`：host AP `86.95090%`，W6 AP `87.29935%`，增益
`+0.34845 pp`，距 `+1 pp` 目标 `0.65155 pp`。完整命令输出和配置保存在
`autoresearch-results/archive/20260906-191324/logs/0002-verify.json`，正式结果回执为
`run_instructions/retained_witness_vad_f3_2_xd.json`。

第一条结构替换删除 primary/normality/context 三角色投票和 agreement，只保留同一组有符号
反事实 witness 的视频内相对项、时序上下文项与训练正常分布校准的绝对项。seed 42、20 epoch
正式结果：best epoch 15，AP `87.23914%`，增益 `+0.28824 pp`；cross-AUC
`95.56849%`，within-AUC `85.87847%`，normal FPR 改善 `0.06642 pp`。结果优于 host，
但弱于正式起点，因此 autoresearch 已 discard（trial `974e17c`，revert `d9c9a8b`）。

失败本质：离线冻结投影诊断能达到 AP `87.66759%`，而让异常袋的存在性 MIL 继续改写
witness 坐标权重后只达到 `87.23914%`。正常袋能密集定义匹配反事实；异常袋只有视频级
存在性标签，不足以持续重定义片段级神经元语义。下一条只做 matched control：冻结训练集
统计得到的坐标、方向和效应权重，仅学习同一证据的时序读出与 frozen-host correction；不扫
权重、不加新模块。

远端 trial（下一次正式 run 前会自动归档到 `diagnostics/formal_trials/`）：

~~~text
/root/autodl-tmp/vadmy_data/witness_vad/dsanet/f3_2_signed_support/xd/w6/evaluation/metrics.json
/root/autodl-tmp/vadmy_data/witness_vad/dsanet/f3_2_signed_support/xd/w6/selection/selection_curve.csv
/root/autodl-tmp/vadmy_data/witness_vad/dsanet/f3_2_signed_support/xd/w6/training/checkpoints/best.pt
~~~

实际复现与查看命令：

~~~powershell
powershell -NoProfile -ExecutionPolicy Bypass -File run_instructions/verify_witness_vad_f3_2_remote.ps1 -Dataset xd
~~~

~~~bash
/root/miniconda3/envs/dsanet/bin/python -m json.tool \
  /root/autodl-tmp/vadmy_data/witness_vad/dsanet/f3_2_signed_support/xd/w6/evaluation/metrics.json
~~~

第二条 matched control 固定训练集统计得到的 witness support、方向和效应权重，其余设置与
第一条完全一致。best epoch 2，AP `87.20845%`，增益 `+0.25756 pp`；cross-AUC
`95.51900%`，within-AUC `85.92948%`，normal FPR 改善 `0.05364 pp`。它比可训练
signed chain 再低 `0.03068 pp`，autoresearch 已 discard（trial `cbdd04c`，revert
`2877477`）。因此“不可靠的神经元微调”不是缺失机制，停止围绕冻结范围做实验。

两条正式失败与离线诊断的唯一结构差异是 temporal context：离线 `87.66759%` 使用同一
witness 的 ReLU 正向偏离构造 context，而两条正式 trial 把 context 也改成 signed，造成
时间窗口内正负抵消。下一条只恢复**同坐标双统计**：signed 投影决定校正方向，rectified
幅度提供不抵消的时序支持；它们仍属于同一 witness 证据链，不恢复三角色投票或 agreement。

第三条正式实验完成了上述同坐标双统计：固定同一组 witness，signed 瞬时偏离负责校正
方向，rectified 非负幅度负责 temporal context。seed 42、20 epoch，best epoch 2，AP
`87.28811%`，比 host 提升 `+0.33721 pp`；pooled/cross/within/macro AUC 分别为
`95.53559% / 95.54227% / 86.14622% / 80.24726%`，normal FPR 改善 `0.00766 pp`。
它仍比保留起点低 `0.01124 pp`，距 `+1 pp` 目标 `0.66279 pp`，因此 autoresearch 已
discard（trial `7532494`，revert/current HEAD `8fda0bb`）。

这条结果排除了“signed context 的正负抵消就是全部瓶颈”：改为 rectified context 仅从前两
条正式实验的 `+0.258/+0.288 pp` 回升到 `+0.337 pp`，仍无法复现离线冻结诊断的
`+0.71669 pp`。下一窗口不要直接开第四次训练。先在当前第三条 trial 的 checkpoint/逐 epoch
输出上做**只读分量审计**，分别重算 signed instantaneous、rectified context、absolute
normal calibration 及其组合，定位离线公式与正式图之间究竟是哪一项失真；诊断不得把 test
标签用于拟合系数。只有审计指出一个可证伪的结构缺口后，才允许再做一次正式 `finish`。

第三条 trial 当前远端产物（它属于失败 trial，不是 retained）：

~~~text
/root/autodl-tmp/vadmy_data/witness_vad/dsanet/f3_2_signed_support/xd/w6/evaluation/metrics.json
/root/autodl-tmp/vadmy_data/witness_vad/dsanet/f3_2_signed_support/xd/w6/selection/selection.json
/root/autodl-tmp/vadmy_data/witness_vad/dsanet/f3_2_signed_support/xd/w6/training/checkpoints/best.pt
~~~

前两条失败 trial 已自动归档：

~~~text
/root/autodl-tmp/vadmy_data/witness_vad/dsanet/diagnostics/formal_trials/20260906T200609Z-974e17cebe5f/
/root/autodl-tmp/vadmy_data/witness_vad/dsanet/diagnostics/formal_trials/20260906T204530Z-cbdd04cdb6f1/
~~~

注意：UCF 的 `+1.07115 pp` 结果和 SHA256 仍是有效正式记录，但原 UCF checkpoint 已被后续
XD 正式验证清理共享输出根时移除。不要声称原路径仍有文件；XD 结构定型后，必须用最终同一
结构复跑 UCF，并把新 checkpoint 移入不会被下一数据集清理的受保护目录。

### DSANet-XD 固定 checkpoint 分量审计（2026-09-07）

本阶段只复用第三条失败 trial 的 epoch-2 checkpoint；没有重新训练，没有用测试标签拟合
系数、阈值或结构。测试标签只计算最终指标。审计产物统一位于：

~~~text
../vadmy_data/witness_vad/dsanet/diagnostics/
~~~

| 固定公式 | XD AP | 相对 host | 裁决 |
|---|---:|---:|---|
| 正式 learned full | 87.28811 | +0.33721 pp | 参考 |
| signed + absolute | 87.35148 | +0.40059 pp | 最强简单分量，但不足 |
| signed + 确定性 rectified context + absolute | 87.39086 | +0.43996 pp | 本阶段最好，仍不足 |
| 绝对证据逐片段 CDF 授权 | 87.27911 | +0.32821 pp | 只改善 FPR，删除 |
| `q` 乘法正授权 | 87.23001 | +0.27911 pp | 会抹掉可靠正见证，删除 |
| `q` 追加正授权 | 87.31934 | +0.36844 pp | 增益太小，删除 |
| host-to-top-k 反事实补偿 | 87.34102 | +0.39012 pp | 改善正常 FPR，但不足 |
| Entmax 稀疏责任 | 86.02970 | -0.92120 pp | 稀疏化不是瓶颈，删除 |
| host-witness 冲突专修 | 87.03087 | +0.07997 pp | 丢失一致位置的有效排序，删除 |
| OR-Markov 责任链 | 87.28864 | +0.33774 pp | 未增加信息，删除 |
| 训练持续度 11 的 median 投影 | 87.29810 | +0.34720 pp | 时序平滑不是瓶颈，删除 |

关键诊断不是“还差一个门”：learned video route 的视频级 AUC 已达 `98.78067%`，但任何
路由重写都没有显著提高 frame AP。相反，绝对 witness 的视频级 mean/top-k AUC 分别为
`87.70678% / 88.48864%`，而视频内标准化 relative top-k 的视频级 AUC 只有 `34.52899%`。
当前神经元主要会区分视频，却没有提供足够稳定的片段证词；固定这组 witness 后，局部公式的
实测上限明显低于 `+1 pp`。因此停止 q、eta、平滑、稀疏度和后处理实验。

下一条正式假设只改变 witness 的**训练袋可信度定义**。当前均值/方差效应可能被少数极端
异常视频主导；改为每个方向上的稳健效应：

\[
C_{lj}^{\pm}=\frac{\operatorname{Median}_{V:Y=1}u_{Vlj}^{\pm}
-\operatorname{Median}_{V:Y=0}u_{Vlj}^{\pm}}
{\operatorname{MAD}_{V:Y=1}u_{Vlj}^{\pm}
+\operatorname{MAD}_{V:Y=0}u_{Vlj}^{\pm}+\epsilon},
\]

其中 `u` 仍是每个训练视频的 MIL top-k 方向偏离。每层只保留 `C` 最大的 32 个坐标；其余
模型、seed 42、20 epoch、损失和 DSANet 评测协议全部不变。这不是加模块，而是要求一个
可解释 witness 必须在多数异常袋中稳定作证、在正常袋中保持沉默。若 DSANet-XD 不能超过
保留起点 `87.29935%`，立即删除该定义，不继续改分位数或 MAD 系数。

### DSANet-XD 稳健袋可信度正式结果（2026-09-07）

第四条正式实验只把方向 witness 的训练袋效应从均值/方差改为 median/MAD；模型、损失、
seed 42、20 epoch、每层 32 个坐标和评测协议均未改变。best epoch 1，XD AP
`87.30967%`，相对 host `86.95090%` 提升 `+0.35877 pp`，仅比此前保留结果
`87.29935%` 高 `+0.01032 pp`，距 `+1 pp` 目标仍差 `0.64123 pp`。normal-frame FPR
反而增加 `+0.22351 pp`。控制器按单一主指标保留 commit `630a938`，但方法裁决是：
**没有解决 XD 定位瓶颈，不把 median/MAD 包装成独立创新，也不继续扫分位数或尺度下限。**

远端正式产物：

~~~text
/root/autodl-tmp/vadmy_data/witness_vad/dsanet/f3_2_signed_support/xd/w6/evaluation/metrics.json
/root/autodl-tmp/vadmy_data/witness_vad/dsanet/f3_2_signed_support/xd/w6/selection/selection_curve.csv
/root/autodl-tmp/vadmy_data/witness_vad/dsanet/f3_2_signed_support/xd/w6/training/checkpoints/best.pt
~~~

本地完整命令输出为 `autoresearch-results/logs/0004-verify.json`。复现和查看命令：

~~~powershell
powershell -NoProfile -ExecutionPolicy Bypass -File run_instructions/verify_witness_vad_f3_2_remote.ps1 -Dataset xd
~~~

~~~bash
/root/miniconda3/envs/dsanet/bin/python -m json.tool \
  /root/autodl-tmp/vadmy_data/witness_vad/dsanet/f3_2_signed_support/xd/w6/evaluation/metrics.json
~~~

#### 为什么 UCF 过线而 XD 卡住

1. UCF 的 AUC 从 `89.44464%` 到 `90.51579%`，提升 `+1.07115 pp`；XD 评 AP，当前只从
   `86.95090%` 到 `87.30967%`。AP 更依赖高分端精确排序，也更容易被正常高分误报伤害。
2. XD 固定 checkpoint 分量审计中 signed local witness 单独贡献 `+0.37664 pp`，而
   normal/context/absolute 单独只有 `+0.02342/-0.01754/+0.06007 pp`。核心方向有效，
   但视频授权和上下文没有补足局部排序。
3. learned video route 的视频级 AUC 已达 `98.78067%`，而 relative witness top-k 的视频级
   AUC 只有 `34.52899%`。模型已经会判断“视频是否异常”，但不能稳定指出“异常在哪”。
4. 第三、第四条正式实验都在早期 epoch 取得最佳 AP，此后训练 loss 继续下降而 AP 下降，
   直接证实训练目标与最终逐帧排序错配。
5. median/MAD 只带来 `+0.01032 pp`，说明少数异常视频主导效应不是主因。异常袋只保证
   存在证词；强求多数异常袋出现同一坐标，反而会压制类别特异、稀有但真实的异常证词。

#### 下一条唯一正式假设：存在性候选对可靠正常反事实的片段排序

当前 `final_loss` 只对 corrected score 做 top-k bag BCE；跨袋 ranking 只施加在 witness role
的 bag 均值上。下一条不加网络模块，只让最终校正直接学习 pooled AP 所需的弱监督排序：

\[
\mathcal L_{\mathrm{CF-rank}}=
\frac{1}{|P||N|}\sum_{p\in P,n\in N}\operatorname{softplus}(m-p+n),
\quad
P=\operatorname{TopK}(S_{Y=1}),\;N=\operatorname{TopK}(S_{Y=0}).
\]

`P` 仅把异常袋当前 top-k 当作存在性候选，不把异常袋其余片段伪标为正常；`N` 只使用正常
袋最难片段，仍是可靠密集负监督。该项作用在最终 corrected score，而不是新增 expert、router
或后处理。沿用已有 `rank_margin=0.5` 和 `rank_weight=0.5`，不增加超参、不扫权重。判据：
先在 DSANet-XD 超过 `87.30967%`；若仍不能明显缩小到 `+1 pp` 的差距，删除该目标并停止
所有见证选择/路由微调，重新审视最终校正参数化。

### DSANet-XD 最终分数反事实排序正式结果（2026-09-07）

第五条正式实验在 commit `440953e` 上给 corrected score 增加存在性候选对正常难负例的
pairwise ranking；没有新增网络模块或超参。seed 42、20 epoch 的 best 仍为 epoch 1，XD AP
`87.31431%`，相对 host 提升 `+0.36341 pp`，仅比第四条多 `+0.00464 pp`，距目标仍差
`0.63659 pp`。within-AUC 提升 `+0.76272 pp`，但 cross-AUC 仅提升 `+0.11309 pp`，
normal-frame FPR 恶化 `+0.19797 pp`。控制器按主指标保留该 commit，但方法裁决是：
**排序损失微调失败；不扫 margin/weight，不把该项写成创新。**

远端产物与复现命令仍是上一节列出的固定路径和命令；本地完整输出为
`autoresearch-results/logs/0005-verify.json`。

#### 通用性状态纠正（必须遵守）

历史 UCF `+1.07115 pp` 对应方法链截至 commit `5709e6e`。XD 正式起点 `a8b43cd` 虽然包含
该祖先，但中间的 `b6badcc`、`a8b43cd` 已继续修改 `train_witness.py` 和
`witness_router.py`；之后还有 `630a938`、`440953e`。因此历史 UCF 数字与当前 XD 数字
**不是同一最终 commit，禁止拼在一起声称通用方法已经在两个数据集过线。**

从现在起，DSANet 完成门只有一个：同一最终 commit、同一结构、同一公式和同一固定超参，
分别只用 UCF/XD 各自训练集拟合数据统计，然后在 UCF AUC 与 XD AP 上均提升至少 `1 pp`。
允许训练得到的数据集内正常参考和 witness 不同；不允许数据集专属模块、公式或手调超参。
候选版本先过 XD 再原样重跑 UCF；XD 若明显不过线，不浪费一轮 UCF 正式训练。

按项目约定继续采用 VAD 常用的 test-primary-metric best checkpoint 协议，不新增 validation
split；结果文件必须如实保留 `test_used_for_selection=true`。本项目“无数据泄露”门聚焦于：
训练和结构拟合不读取帧级 GT；UCF 流程不读取 XD 数据/统计，XD 流程不读取 UCF 数据/统计。

#### 下一步先审计，不直接开第六次训练

第五条结果表明，给现有输出增加更贴近 AP 的排序目标仍不能改变 epoch-1 峰值。下一步只读
审计**视频内标准化是否删除跨视频校准**。注意当前 W6 `WitnessRouter` 已在 `b6badcc` 删除
`event_gap`，直接使用 `delta_anomaly = eta * masked_standardize(evidence)`；不得按旧 W2 结构
分析。复用当前 best 的缓存曲线，分别报告 raw `evidence`、`delta_anomaly`、host 与 corrected
的 pooled/cross/within 指标，并统计正常帧、异常帧、固定 `host<0.5` 漏检异常帧上的正校正
覆盖。测试 GT 只用于诊断分组，不拟合阈值或系数。只有 raw evidence 保留有用 cross-video
信息、而标准化 residual 明显丢失时，才允许设计“训练正常分布校准 + 视频内残差”的单一
通用 correction；否则停止改 router，回到 witness 表示本身。任何新 correction 必须同一
commit 依次通过 XD、UCF。

### DSANet-XD 当前 W6 跨视频校准审计（2026-09-07）

脚本 `vin_vad/audit_w6_calibration.py` 只读取 commit `440953e` best epoch 1 的正式缓存曲线；
GT 只用于报告分组和指标，没有拟合阈值、系数或结构。远端产物：

~~~text
/root/autodl-tmp/vadmy_data/witness_vad/dsanet/diagnostics/xd_w6_cross_video_calibration_audit.json
~~~

| 曲线 | pooled AUC | cross-AUC | within-AUC | pooled AP |
|---|---:|---:|---:|---:|
| host | 95.40110 | 95.40816 | 85.46846 | 86.95090 |
| corrected | 95.51465 | 95.52125 | 86.23118 | 87.31431 |
| raw witness evidence | 74.67940 | 74.67562 | 79.99238 | 47.73728 |
| `delta_anomaly`（视频内标准化后） | 59.99488 | 59.98066 | 79.99257 | 31.31636 |

raw evidence 经视频内标准化后，within-AUC 几乎不变，但 cross-AUC 从 `74.67562%` 降到
`59.98066%`，说明当前 router 明确丢掉了 witness 的跨视频绝对可信度。raw evidence 均值在
正常视频帧、真实异常帧、固定 `host<0.5` 的漏检异常帧上分别为 `0.32922/0.48189/0.42049`；
但漏检异常帧的 `delta_anomaly` 均值为 `-0.02293`，只有 `46.60028%` 得到正校正，反而低于
正常视频帧的正校正比例 `55.84265%`。这解释了“within 明显涨、cross 和 AP 只小涨”。

下一条允许的结构假设只修改同一 evidence 的 correction 表达：

\[
r_t=\operatorname{StdWithinVideo}(e_t)
+\tanh\!\left(\operatorname{logit}(e_t)-b\right).
\]

第一项保留已验证的局部排序；第二项保留 raw evidence 的跨视频可信度。`b` 是由现有袋级损失
学习的单一全局标量，固定初始化为 0；它不是数据集专属阈值，不增加 expert、context、q 或
后处理，也不增加需扫描的超参。正式实验前应删除已判无效的 final tail-ranking 项，避免把
失败 loss 与新 correction 捆绑。该候选先在 XD 运行；若接近或达到 `+1 pp`，必须在完全相同
commit 上重跑 UCF。若 XD 仍停留在零点几，否决当前三角色 evidence 本身，停止 router 微调。

### DSANet-UCF 单一固定 Top-32 干净参考（2026-09-08）

用户取消两组选点的 union 方案后，commit `bf3010d` 已由 `b5b89fc` 完整撤回。当前
commit `fc2b4da` 只用训练袋正常/异常的标准化反事实效应，在每层选择一组固定 Top-32；
Primary、Normality、Context 三种读出必须共享这组坐标，代码会拒绝第二套 role mask。
同时删除 XD 已判无效的 median/MAD 选点和 corrected-score tail-ranking loss；router 只保留
视频级正常抑制、一致位置保护和带符号的局部有界残差，不含 host-miss complement、event gap、
event completion 或 confidence gain。

seed 42、20 epoch、31 项远程测试通过。UCF best epoch 16：

| 指标 | DSANet host | clean W6 | 变化 |
|---|---:|---:|---:|
| pooled AUC | 89.44464 | 89.61455 | +0.16991 pp |
| cross-AUC | 89.50613 | 89.67459 | +0.16846 pp |
| within-AUC | 73.67507 | 74.21588 | +0.54082 pp |
| macro-within-AUC | 69.91278 | 70.66935 | +0.75657 pp |
| pooled AP | 37.41960 | 37.70695 | +0.28735 pp |
| normal-frame FPR @ 95% TPR | 10.19904 | 8.45063 | -1.74841 pp |

裁决：这是一套可复现、明显比 Universal 干净的统一代码参考，但 UCF 只提升 `+0.16991 pp`，
没有达到 `+1 pp`。历史双选点/复杂 router 的 `5709e6e` 为 `+1.07115 pp`，两者差
`0.90124 pp`；禁止把历史数字贴到当前方法上。当前 within 增益高于 cross，说明同一 witness
对视频内定位有效，主要缺口仍是跨视频校准。按用户要求，本轮到此停止，不跑 XD、不再启动
autoresearch。

正式复现命令：

~~~powershell
powershell -NoProfile -ExecutionPolicy Bypass -File run_instructions/verify_witness_vad_f3_2_remote.ps1 -Dataset ucf
~~~

当前产物：

~~~text
/root/autodl-tmp/vadmy_data/witness_vad/dsanet/f3_2_signed_support/ucf/w6/evaluation/metrics.json
/root/autodl-tmp/vadmy_data/witness_vad/dsanet/f3_2_signed_support/ucf/w6/selection/selection.json
/root/autodl-tmp/vadmy_data/witness_vad/dsanet/f3_2_signed_support/ucf/w6/training/checkpoints/best.pt
~~~

受保护归档：

~~~text
/root/autodl-tmp/vadmy_data/witness_vad/dsanet/diagnostics/formal_trials/20260908T004900Z-clean-ucf-fc2b4da/
~~~

查看命令：

~~~bash
cat /root/autodl-tmp/vadmy_data/witness_vad/dsanet/f3_2_signed_support/ucf/w6/evaluation/metrics.json
cat /root/autodl-tmp/vadmy_data/witness_vad/dsanet/f3_2_signed_support/ucf/w6/selection/selection.json
~~~
