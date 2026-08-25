# 方案审查：责任引导的语义时间拼接

## 一句话结论

**值得完成DSANet/UCF正式实验，但目前只能评为“有依据的高风险增量方法”，不能提前保证高于强baseline 2～3点。**

方法只解决一个共同问题：UCF/XD只有视频级弱标签，三个baseline最终却需要输出snippet级分数。输入仍然是同一个CLIP，再拼一份hidden state不会产生新监督；因此本方法让可解释hidden states负责挖掘位置，再用KNN拼接产生边界监督。

## 统一逻辑

```text
公开事件Prompt定义“什么异常”
              ↓
神经元责任统计判断“哪一完整层负责”
              ↓
完整层文本边际判断“发生在哪里”
              ↓
KNN相似正常段拼接产生密集边界
              ↓
三个baseline原有交互模块和打分头学习边界
```

这不是Prompt、选层、KNN三个独立模块。文本方向用于分解神经元贡献；神经元数量决定层；同一层的文本边际决定被拼接的候选段。

## 1. 文本设置

- UCF和XD异常Prompt直接采用LAP论文附录Tables A1/A2公开的原子事件字典。
- 正常Prompt采用LaGoVAD公开的`DatasetSpecVerbalizer`。
- 不自行生成Prompt，不使用`a video of <class>`模板，不运行视频字幕模型。
- LAP做法是对一个类别的多个原子事件取最大相似度；本方法保持这一规则。

依据：LAP报告完整事件句子相对短语在XD提升3.1点、UCF提升0.7点。LaGoVAD人工描述相对类别名只提升0.68 UCF AUC，说明描述有用但不能单独承担主要增益。

## 2. 可解释选层

责任探测只使用视频标签、正常统计和冻结CLIP文本方向，不使用DSANet、DeSC或LaGoVAD分数。对12层分别统计类别相关神经元的充分数量、方向稳定性和正常特异性，自动选择责任最集中的少量层。后续使用整层768维CLS，不拼接零散Top-K维度。

层权重等于“责任神经元并集数量×方向稳定性”的归一化值，因此每个时间分数都能拆成层贡献。

## 3. 无训练的语义定位

选中层通过CLIP自身冻结的`ln_post`和`visual.proj`进入512维文本空间。每个snippet的类别分数为LAP原子事件最大相似度减去正常Prompt最大相似度。候选区域使用LAP的动态阈值：

```text
threshold = mean(segment_score) + tau * std(segment_score)
```

相邻片段合并为连续段。这里没有可学习投影器、CoOp token、额外打分头或baseline分数，避免定位器学习自己的伪标签。

## 4. KNN时间拼接

候选异常段作为anchor；用已有最终512维CLIP特征在纯正常训练集检索场景相近的正常片段，按LaGoVAD公开算法随机拼接1～5段并产生边界标签。拼接位置准确，但异常anchor仍是伪标签，所以仅保留高于动态阈值的连续段。

LaGoVAD消融中，KNN相对无KNN约提升1.14 UCF AUC和5.30 XD AP。该数字来自LaGoVAD的零样本设置，不能直接当作本方法预期增益。

## 5. baseline训练范围

CLIP视觉和文本backbone、baseline时序主干全部冻结。只训练作者已有的：

| baseline | 可训练部分 |
|---|---|
| DSANet | Text Adapter、`mlp1/mlp2`视觉文本交互、classifier |
| DeSC | learnable prompt、两个流的交互/打分头 |
| LaGoVAD | soft prompt、fusion、binary/similarity head |

原始正常与异常batch仍按作者方式成对训练；合成batch单独增加LaGoVAD式dense BCE和候选正段MIL。UCF/DSANet仍按作者每1280个原始样本评测并用frame AUC选模。

## Idea质量

| 维度 | 评分 | 原因 |
|---|---:|---|
| 可解释性 | 8/10 | 分数可拆到Prompt、层和神经元；需要热力图验证不是背景捷径 |
| 通用性 | 8/10 | 伪片段和合成数据不依赖baseline，同一份产物可供三模型使用 |
| 有效性依据 | 7/10 | LAP文本伪标、LaGoVAD KNN拼接、DSANet轻量文本适配均有消融；三者串联尚未被验证 |
| 新颖性 | 6/10 | 单个机制已有；新意主要是责任选层驱动的跨baseline监督生成 |
| 开销 | 9/10 | 不重跑视觉CLIP、不微调CLIP、无额外最终推理开销 |
| 达到+2～3点把握 | 5/10 | 方向对准定位瓶颈，但强baseline上的伪标签噪声可能抵消收益 |

## 决定性风险与停止条件

最大风险不是训练开销，而是CLIP中间层文本边际仍可能选中火焰、人群、车辆等显眼背景。必须检查：

1. `class_layer_neuron_heatmap.png`：是否只有少数类别/少数层支配全部责任；
2. `temporal_text_margin.png`：候选段是否连续且不同层在同一事件处共同升高；
3. `synthetic_boundaries.png`：是否确实形成正常—异常穿插，而不是只有单一异常段；
4. DSANet/UCF是否在作者相同选模规则下超过作者初始化。

如果最终AUC不超过作者权重，不能继续靠调tau、损失权重堆小技巧，应判定“冻结CLIP语义仍不足以提供更准的时间伪标签”，再决定是否放弃该主线。

## 最终评价

**Accept with major experimental validation。** 方案逻辑已经干净，文献依据也明确；真正的不确定性只剩一个：公开Prompt和责任层能否在UCF/XD长视频中定位出比原MIL更可靠的异常段。
