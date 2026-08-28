# 开源来源与改动边界

| 本目录内容 | 公开来源 | 保留的机制 | 本项目必要适配 |
|---|---|---|---|
| UCF/XD异常Prompt | LAP论文附录Tables A1/A2 | 每个数据集30条原子事件句子；每类取最大相似度 | 按UCF/XD标签映射到对应Prompt组 |
| 正常Prompt | LaGoVAD `DatasetSpecVerbalizer` | 发布的正常描述集合 | 与异常Prompt形成类别—正常边际 |
| 动态阈值 | LAP Eq. 13 | `mean + tau * std` | 在单视频32段的文本边际上计算 |
| 神经元责任选层 | 本项目已有`discover_definition_circuits.py` | 视频标签、正常统计、冻结CLIP文本方向；不读baseline分数 | 神经元只用于选完整层，不再拼接零散维度 |
| 中间层文本空间 | CLIP发布的`ln_post`和`visual.proj` | 冻结的768→512视觉投影 | 同一冻结投影作为中间层语义透镜；不训练Adapter |
| KNN时间拼接 | LaGoVAD Algorithm 1 | 正常KNN、1～5段、随机插入位置、密集标签 | 异常anchor改为文本边际候选段 |
| 伪监督损失 | LaGoVAD dynamic video synthesis loss | 有效片段dense BCE、候选正段MIL | 仅作用于离线合成batch |
| 三baseline训练评测 | DSANet、DeSC、LaGoVAD发布代码 | 模型结构、原二值输出、UCF AUC/XD AP、作者选模规则 | 冻结时序主干，只训练原有交互模块和打分头 |

参考仓库：

- LAP：<https://github.com/shiwoaz/lap>，完整参考副本位于`rely/LAP/`。
- LaGoVAD：<https://github.com/Kamino666/LaGoVAD-PreVAD>。
- AnomalyCLIP：<https://github.com/lucazanella/AnomalyCLIP>，作为早期方案审查参考，当前冻结语义透镜不使用其CoOp训练模块。
- DSANet、DeSC、LaGoVAD作者代码位于`baseline/`，未修改。

`rely/`和`baseline/`中的文件只作来源审查。正式代码不在运行时引用这些目录；所需作者代码副本位于`semantic_knn_splicing/vendor/`。
