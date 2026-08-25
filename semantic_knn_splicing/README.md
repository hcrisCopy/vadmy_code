# Responsibility-Guided Semantic Splicing

这个目录实现一条完整流程：

```text
神经元责任选层 → 冻结文本边际定位 → KNN时间拼接 → 局部训练baseline
```

先看 [IDEA_REVIEW.md](IDEA_REVIEW.md) 理解方法和风险，再按 [COMMANDS.md](COMMANDS.md) 运行。公开代码与Prompt来源见 [SOURCE_PROVENANCE.md](SOURCE_PROVENANCE.md)。

## 主要文件

- `extract_full_layers.py`：提取责任探测选中的完整CLS层。
- `semantic_lens.py`：用冻结CLIP投影和LAP事件Prompt计算可分解文本边际，没有训练参数。
- `select_pseudo_segments.py`：用LAP动态阈值生成连续候选异常段，不读取baseline分数。
- `build_knn_cache.py`：按LaGoVAD方式检索场景相似的正常片段。
- `build_synthetic_features.py`：生成正常—候选异常穿插序列和密集边界标签。
- `train_baseline.py`：保留原始MIL数据与选模规则，只解冻原有视觉—文本交互模块和打分头。
- `evaluate_baseline.py`：使用baseline原二值分数计算UCF frame AUC或XD frame AP。
- `visualize_outputs.py`：输出层责任图、神经元数量热力图、时间边际曲线和拼接边界图。

## 开销

- 复用已有512维CLIP特征和CLS hidden states。
- 不提取光流、patch token、字幕或其他视频信息。
- 不微调CLIP，不训练额外定位网络。
- baseline推理时不读取hidden states，因此不增加最终推理开销。

所有新增产物写到同级目录 `../vadmy_data/semantic_knn_splicing/`。已有hidden states继续从 `../vad_data/` 读取，不复制原始12层文件。
