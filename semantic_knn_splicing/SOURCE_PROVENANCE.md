# 开源代码来源

本目录只组合已有公开机制；没有把未经验证的自创模块伪装成文献方法。

| 本目录模块 | 公开来源 | 复用内容 | 必要改动 |
|---|---|---|---|
| `DescriptionPromptLearner` | AnomalyCLIP `coop.py` | shared CoOp context token | 一个类别允许多条 LaGoVAD 描述，按类别平均 |
| 文本 top-k 定位 | AnomalyCLIP `selector_model.py` | 正常中心重定位、无仿射 BatchNorm、按真实类别选 top-k | 支持 XD 多标签，合并相邻 top-k 段 |
| 文本损失 | AnomalyCLIP `loss.py` | abnormal top-k、normal suppression、bottom-k、smooth/sparse | 改为数值稳定的 `softplus`，支持可变长度 |
| 时间拼接 | LaGoVAD `PreVAD.py` | 1～5 段、随机异常位置、KNN/随机正常混合、稠密段标签 | 异常源改为文本 selector 的候选片段 |
| 稠密监督与伪监督 MIL | LaGoVAD `losses.py` | 全有效片段 BCE、候选正段内部 top-k | 只对离线合成样本启用 |
| `768→512` 投影 | DSANet `adapter_modules.py` | Linear + LeakyReLU 轻量 Adapter | 只在独立定位器中使用，不注入 baseline |
| 三 baseline 训练/评测 | DSANet、DeSC、LaGoVAD 发布代码 | 发布模型、二值分数、数据集指标；DSANet/LaGoVAD 的训练损失 | 增加伪监督损失和功能等价的局部解冻；DeSC 因官方未发布训练程序而使用标准 CLIPVAD MIL |

参考仓库：

- AnomalyCLIP: <https://github.com/lucazanella/AnomalyCLIP>
- LaGoVAD-PreVAD: <https://github.com/Kamino666/LaGoVAD-PreVAD>
- DSANet、DeSC：本项目 `baseline/` 中的作者发布版本

完整 AnomalyCLIP 仓库已按规范克隆到 `rely/AnomalyCLIP/`，其中没有保留 `.git`。`rely/` 和 `baseline/` 均未被本方法修改。
