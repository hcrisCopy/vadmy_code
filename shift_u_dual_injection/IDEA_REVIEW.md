# 方案审查

## 核心假设

旧方案只有早期注入：

$$
h_t=E\left(f_t+\sigma(g_e)R(n_t)\right),
$$

冻结的时序模块 $E$ 可能平滑或抑制片段级神经元残差。新方案增加绕过 $E$ 的后期路径：

$$
e_t=T(n_t),
$$

$$
f'_t=f_t+\sigma(g_e)P_e(e_t),
$$

$$
h'_t=E(f'_t)+\sigma(g_l)P_l(e_t).
$$

$T$ 是共享主干，$P_e/P_l$ 是两个零初始化出口。两个出口使用独立门控，但不读取baseline分数；baseline分数只在训练前构建top/bottom样本。

## 为什么它是干净的变化

1. pseudo score、选中神经元、normal mean/std和aligned feature均与上一实验共用同一文件。
2. baseline checkpoint、loss、batch、lr、scheduler、epoch和选模规则不变。
3. baseline所有参数冻结，`parameter_report.json`必须显示可训练baseline参数为0。
4. 相比单注入只增加一个 `1024→512` 出口和一个标量门控，约增加52.5万参数。

## Idea质量

| 维度 | 评价 |
|---|---|
| 问题针对性 | 高。直接针对冻结时序主干可能稀释神经元证据的问题。 |
| 控制变量 | 高。输入证据和baseline训练状态完全不变。 |
| 通用性 | 高。三个baseline都存在时序前和时序后512D共同节点。 |
| 可解释性 | 中高。可以分别观察early/late门控和实际残差RMS。 |
| 计算开销 | 低。CLIP不运行，baseline不反传参数梯度，只多一个投影出口。 |
| 论文新颖性 | 中。比普通单点adapter更有结构动机，但仍需要位置消融和跨baseline结果支撑。 |
| 显著增益把握 | 中低。它解决信息衰减，不会修复Shift神经元本身不稳定的问题。 |

## 结果如何解释

- late RMS明显大于early RMS且指标提升：支持“冻结时序模块稀释证据”。
- 两个分支都有能量且双注入优于单注入：支持时间上下文与直接证据互补。
- residual能量增加但指标不升：问题在神经元信息，而不在注入位置。
- 两个分支都接近0：baseline的原始表示已经足够，或训练loss不给残差有效梯度。

本实验本身不能单独证明双注入优于late-only。若结果有效，下一步必须补 early-only、late-only、dual 三组位置消融；当前阶段不把额外消融混入主代码，保证本次请求只有一个变化。
