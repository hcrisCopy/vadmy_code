# Intrinsic Anomaly Subspace

这个实验只回答一个问题：**CLIP CLS hidden states 中是否存在一小组能够直接做帧级异常定位的功能神经元？**

它不接 DSANet、DeSC 或 LaGoVAD 的网络，也不微调 CLIP。唯一训练参数是层探针和最终线性分类器。这样可以先判断 hidden states 本身有没有足够强的定位能力，再决定是否值得接回三个 baseline。

## 方法主线

### 1. 正负 snippet

完全沿用 `shift global768` 的构造：对每个异常训练视频，将 baseline 分数最高的 top 10% snippets 记为正样本，最低的 bottom 10% 记为负样本。两端来自同一视频、数量相等，因而控制了场景和视频身份差异。

这一步明确依赖 baseline 排序。测试帧标注不会参与样本构造、神经元发现或选模。异常视频按视频划分为互不重叠的 discovery/train/validation，避免同一视频同时负责“选神经元”和“证明神经元有效”。

### 2. V-FIND 式选层

每个 discovery 视频分别对正、负尾部 snippets 求均值，得到每层的正负视频表示。第 `l` 层的两个分数为：

```text
D_cos(l) = 1 - cos(mu_pos(l), mu_neg(l))

sigma_bar(l)^2 = 两类、全部神经元的类内方差均值

D_shift(l) = ||mu_pos(l)-mu_neg(l)||_2 / sqrt(M * sigma_bar(l)^2)
```

阈值采用 V-FIND：

```text
tau_cos   = mean(D_cos)   + std(D_cos)
tau_shift = mean(D_shift) + std(D_shift)

L_cos   = {D_cos > tau_cos}
L_shift = {D_shift > tau_shift}
```

V-FIND 在其32层伪造检测器上取交集。UCF实测显示，CLIP的幅值位移集中在早层，而方向分离集中在末层，二者交集为空。因此本任务默认显式采用 `L_cos ∪ L_shift` 作为**候选层**，随后仍由神经元效应量筛选；它不是手选层，也不会隐藏原始交集，`layer_metrics.csv` 会同时保存两种结果。需要核验原论文规则时可用 `--layer-rule intersection`，交集为空会直接停止。

### 3. V-FIND 式神经元探测

每个关键层单独训练一个线性 probe。神经元 `n` 对样本 `x` 的响应和正负效应量为：

```text
r_n(x) = |a_n(x)| * |w_n|

d_n = |mean_pos(r_n)-mean_neg(r_n)| / (pooled_std_n + epsilon)
```

默认保留 `d_n >= 1.5` 的神经元，与 V-FIND 一致。最终分类器读取的是这些神经元的**原始 activation**，不是人为构造的 `r_n` 或 baseline 分数。

### 4. 直接帧级分类

只在 train 视频的已选 activation 上训练一个线性分类器；按 validation snippet AP 选模型。测试时：

```text
test hidden [T, L, 768]
  -> 取选中 (layer, dimension)
  -> 线性分类器
  -> T 个 snippet 异常分数
  -> 每个分数重复 16 帧
  -> UCF frame AUC / XD frame AP
```

完整 CLIP、三个 baseline 都不会前向或更新。测试 hidden 会先对齐到官方 512D CLIP 特征的 snippet 数，因此评测顺序和长度仍采用原 benchmark 协议。

## 必须看的证据

正式结论不能只看 selected 的 AUC/AP，还要看两个同宽对照：

- `same_layer_random`：每个关键层随机取同样数量、且不与 selected 重叠的维度。
- `global_random`：从全部层随机取同样数量、且不与 selected 重叠的维度。

只有 selected 同时显著超过二者，才能说发现的是功能神经元；否则最多说明关键层整体有判别信息。

可视化只保留能回答关键问题的四张图：两种层证据与候选层、神经元效应量热力图、等宽随机对照、代表性时间曲线。

## 产物

所有新产物写入 `../vadmy_data/intrinsic_anomaly_subspace/{ucf,xd}`：

```text
pairs/                       # 可恢复的逐视频正负 snippet 与三折清单
discovery/                   # 层分数、probe、神经元效应量、selected_subspace.json
readout/{mode}/              # checkpoint_last.pth、model_best.pth、history.jsonl
evaluation/{mode}/           # 帧分数缓存和 metrics.json
diagnostics/                 # 四张证据图和 diagnostic_summary.json
```

XD 训练 hidden 已知缺 4 个视频，`build_shift_pairs` 会跳过并写入 `skipped_videos.csv`。测试视频不允许跳过。

完整、无需替换参数的 UCF/XD 指令见 [COMMANDS.md](COMMANDS.md)。研究价值和风险见 [IDEA_REVIEW.md](IDEA_REVIEW.md)。
