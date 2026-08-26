# 统一稀疏神经元分数纠错

## 方法

方法只读取已经提取的 CLIP ViT-B/16 每层 CLS hidden states。每个数据集训练一个共享的稀疏神经元专家；每层固定保留 32 个原始 hidden-state 维度，并记录层号、维度、门值和权重。三个 baseline 完全冻结，各自只训练相同结构的零初始化分数纠错头。

纠错头使用 baseline 曲线、神经元曲线及其时间变化。训练同时保留视频级 MIL、纯正常视频密集负约束、异常包与 hard-normal 的排序约束，以及对作者分数的轻量锚定。UCF-Crime 和 XD-Violence 使用相同网络、损失权重、训练轮数和随机种子。

不提取光流、patch token 或新视觉特征，也不反向传播 CLIP 或 baseline。

## 正式运行

从 `vadmy_code` 根目录运行：

```bash
conda activate dsanet
bash universal_neuron_adapter/commands/run_all.sh
python -m universal_neuron_adapter.aggregate_metric \
  --results-root ../vadmy_data/universal_neuron_adapter
```

每个 Git 提交的完整产物写到 `../vadmy_data/universal_neuron_adapter/runs/<commit>/`。最终一行输出六个组合相对论文指标的最小提升百分点。

中断后用完全相同的命令重跑；训练脚本会从 `checkpoint_last.pth` 续训。不同提交使用隔离目录，不覆盖历史实验。

