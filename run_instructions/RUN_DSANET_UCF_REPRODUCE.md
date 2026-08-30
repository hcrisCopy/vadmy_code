# DSANet UCF 从头复现

运行：

```bash
conda activate dsanet
bash run_instructions/run_dsanet_ucf_reproduce.sh
```

脚本使用固定 seed 234，从训练集重新训练主神经元专家、常态专家、上下文学生和保守校正头。所有中间产物彼此隔离，不复用其他实验的模型缓存。

结果位于：

```text
../vadmy_data/universal_neuron_adapter/reproductions/<RUN_ID>/ucf/dsanet/evaluation/metrics.json
```

中断续跑时，使用日志第一行打印的 `RUN_ID`：

```bash
RUN_ID=<RUN_ID> bash run_instructions/run_dsanet_ucf_reproduce.sh
```

该脚本复现已经固定的 development-test 协议，不用测试标签训练或选择 checkpoint。90.50 是复现目标，不是代码强制保证值。
