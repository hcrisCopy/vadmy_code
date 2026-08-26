# 来源与代码边界

## V-FIND

- 论文：[V-FIND: Revealing the Intrinsic Forgery Knowledge Encoded in Video Forgery Detectors](https://arxiv.org/abs/2608.03008)
- 使用内容：层定位公式 (5)、(7)、(17)～(19)，probe response 公式 (12)～(15)、(20)～(21)，默认 `effect threshold=1.5`，以及 discovery/train/validation 三折隔离、同层随机固定预算对照。
- 明确差异：论文使用两种层阈值的交集；UCF的CLIP实测交集为空，因此正式VAD命令显式使用阈值并集作为候选层，再用论文原效应量筛选神经元。代码仍支持 `--layer-rule intersection` 复核原规则。
- 明确差异：UCF候选神经元没有达到论文的绝对效应阈值1.5。正式命令采用论文Figure 10已有的Top-200固定预算敏感性设置，并保留绝对阈值模式供复核；没有根据测试AUC调一个新的效应阈值。
- 截至 2026-08-26，没有检索到作者公开的官方代码仓库。因此没有声称逐行复现其实现；本目录逐式实现论文公开定义。

## 开源 neuron probing 参考

- 仓库：`https://github.com/fdalvi/neuron-comparative-analysis`
- 本地只读副本：`rely/neuron-comparative-analysis`（克隆后已移除 `.git`，未修改）
- 使用内容：按层训练线性 probe、根据 probe 权重排序/筛选神经元、在固定神经元预算下重新训练并比较分类性能的实验组织方式。
- 新代码不运行时导入 `rely`。

## 用户已有 shift global768

- 样本构造核对目录：`../vad_code/experiments/intravideo_paired_shift` 与 `../vadclip_code/experiments/vadclip_neuron_injection`。
- 复用规则：异常视频内 baseline pseudo-score top/bottom 10%，数量相等，稳定排序，分数长度线性对齐 hidden 长度。
- 本项目没有导入或修改其他项目源码，只通过命令读取已有 `../vad_data` 数据产物；所有新输出写入 `../vadmy_data`。
