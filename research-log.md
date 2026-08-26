# Research Log

## Autoresearch run `e8a06449ff0048d88091ee19adc46b14`

- 目标：六个 baseline/dataset 组合的最小论文基线增益达到 1.0 个百分点。
- 指标：`minimum_paper_gain_pp`，初始值 0.0。
- 硬约束：只用预提取 CLS hidden states；单卡 RTX 4090；不修改 `baseline/`、`rely/`、`vad_data`。

### Experiment 1: 统一稀疏神经元专家与分数纠错

假设：此前失败源于把约 75 AUC 的弱神经元探针当作强 baseline 的密集教师。改为每个数据集共享一个稀疏神经元专家，冻结所有 baseline，仅让零初始化纠错头利用纯正常视频和 hard-normal 排序学习小幅重排，可保留作者能力并利用互补神经元语义。

状态：远程正式验证由 autoresearch 审计日志和 `../vadmy_data/universal_neuron_adapter/runs/<commit>/summary.json` 记录；下一轮在本文件补充数值结论。

