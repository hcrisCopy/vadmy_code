# 旧版 head-only 对照说明

head-only 方案已经被渐进解冻方案替代，旧命令与新版 checkpoint 不兼容，不要继续运行。

正式的 UCF-Crime / XD-Violence、DSANet / DeSC / LaGoVAD 共 6 条完整训练命令和 6 条评测命令都在 `neuron_responsibility/README.md`。其中已经明确写入：

- 前 2 个 epoch 仅训练零初始化责任校正头；
- 第 3 个 epoch 起训练 baseline heads 和最后时序精炼块；
- CLIP 与 probe 全程冻结；
- UCF 用官方帧级 AUC 选模，XD 用官方帧级 AP 选模；
- DSANet/UCF 按作者代码每 1280 个训练样本验证一次。

若需要做公平消融，应在同一份新版代码上增加独立的消融开关和独立输出目录，不能拿旧 head-only 产物与新版主实验直接比较。
