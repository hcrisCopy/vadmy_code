# 开源来源

本目录不直接 import `baseline/` 或 `rely/`。所需 baseline 源码已经复制到
`responsibility_cross_expert/vendor/`，该目录只作为不可修改的工作副本。

| 本方案部分 | 开源来源 | 实际采用内容 |
|---|---|---|
| 三个 baseline 的模型、作者损失、评测 | `baseline/DSANet`、`baseline/DeSC`、`baseline/LaGoVAD-PreVAD` | 复制到 `vendor/`；不修改 vendor，只在外层写 adapter |
| 完整层事件文本 | `rely/LAP` | 使用 LAP 已发表的 UCF/XD 原子事件描述与 max-over-prompts 形式 |
| 正常文本与困难负样本思想 | LaGoVAD | 使用其公开 normal definitions；本版暂不启用 KNN/DVS |
| 小型残差 Adapter | DSANet | 按其 residual adapter 思路改成每个选中层一个零初始化 bottleneck |
| 两阶段稀疏自训练 | `rely/MIST_VAD` | 语义专家先独立训练，再离线生成监督；不在线学习自身当前输出 |
| 多尺度标签精炼 | `rely/CPLVAD/models/model.py::Generate_gt` | 复制并整理 threshold、grouping、filter、flat-Gaussian boundary 逻辑 |
| 双专家职责 | CPL-VAD、DeSC | semantic/binary 分支独立；一致片段训练，分歧片段 ignore |

## 重要限制

- CPL-VAD 的公开 README 截至本次开发明确写着训练代码尚未发布。因此本方案只复用仓库中已经公开的 `Generate_gt` 标签精炼逻辑，不声称复现其完整训练方法。
- DeSC 的公开仓库只发布模型、权重和推理脚本，没有训练脚本。本模块严格保留其双流模型、独立 classifier、发布的 UCF/XD 预处理和概率平均推理；head-only 阶段的 MIL 训练循环是依据论文目标重建，不能表述为逐行复现作者训练器。
- TPWNG 提供“正常性引导前必须做目标域适配”的论文依据，但没有在本方案中声称复制其代码。可执行实现来自已经克隆的 DSANet/LAP/MIST/CPL-VAD 代码。
- `rely/CPLVAD` 和 `rely/MIST_VAD` 均已移除 `.git`，按照项目规范保留为只读参考，不在其中做修改。
