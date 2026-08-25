# Semantic KNN Splicing

这个目录实现“完整层语义定位 → KNN 正常/异常时间拼接 → 局部解冻 baseline”。

先看 [IDEA_REVIEW.md](IDEA_REVIEW.md) 理解为什么这样做，再按 [COMMANDS.md](COMMANDS.md) 运行。开源代码对应关系记录在 [SOURCE_PROVENANCE.md](SOURCE_PROVENANCE.md)。

## 文件作用

- `extract_full_layers.py`：读取已有无 baseline 分数的回路图，保存被选中的完整 CLS 层。
- `train_localizer.py`：训练冻结 CLIP 的 AnomalyCLIP 式文本定位器。
- `select_pseudo_segments.py`：按真实异常类别选 top-k 片段并合并相邻段。
- `build_knn_cache.py`：从纯正常训练视频建立 LaGoVAD 式 KNN 缓存。
- `build_synthetic_features.py`：离线生成正常—异常穿插序列和稠密段标签。
- `train_baseline.py`：保留作者损失，增加 LaGoVAD 的 dense BCE 与 pseudo-supervised MIL，渐进解冻交互/头和最后时序部分。
- `evaluate_baseline.py`：使用 baseline 原二值分数计算 UCF frame AUC 或 XD frame AP。

## 开销

- 不提取光流、patch token、字幕、姿态或其他视频信息。
- 不运行 CLIP 图像编码器；直接复用已有 512 维特征和 CLS hidden states。
- CLIP 图像和文本 backbone 参数始终冻结。
- 中间层只在独立定位器中使用，测试 baseline 不读取 hidden states，因此 baseline 推理开销不增加。

## 中断与清理

- `COMMANDS.md` 中的同一条命令可以原样重跑：数据阶段复用逐文件缓存，训练阶段自动读取 `checkpoint_last.pth`。
- 只有明确要废弃旧实验时才额外传入 `--clean`；显式 `--resume` 仍可用，但通常不需要。
- 不要同时使用 `--clean` 和 `--resume`。

所有新增产物写到同级目录 `../vadmy_data/semantic_knn_splicing/`。原始 hidden states 继续从 `../vad_data/` 读取，不复制整套 12 层产物。
