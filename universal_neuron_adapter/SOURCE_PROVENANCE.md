# 开源实现依据

- Baseline 加载与作者推理协议来自仓库中只读的官方代码副本：`baseline/DSANet`、`baseline/DeSC`、`baseline/LaGoVAD-PreVAD`。
- 多实例 top-k、时间卷积和 normal-video 负约束参考仓库内 `rely/CPLVAD`、`rely/MIST_VAD` 的公开实现思路；本目录没有直接导入或修改这些代码。
- Baseline 适配接口和 DeSC 官方 UCF/XD 推理规则复用本项目已有的 `neuron_responsibility` 实现；新增方法代码全部位于本目录。
- 神经元严格定义为预提取 CLIP ViT-B/16 CLS hidden states 的原始维度，不把投影通道重新命名为神经元。

