# 当前方法累积消融

该目录只做一件事：对一个冻结 Baseline 按 M0 到 M5 逐步加入当前方法模块，并直接报告 UCF-Crime AUC 或 XD-Violence AP。

| 阶段 | 累积设置 |
|---|---|
| M0 | 冻结 Baseline |
| M1 | M0 + 主稀疏神经元分数校正 |
| M2 | M1 + 多源神经元一致性与冲突抑制 |
| M3 | M2 + 多尺度神经元事件门控 |
| M4 | M3 + 正常视频抑制 |
| M5 | M4 + 时间持续性和边界恢复 |

脚本使用当前 `evaluate.py`，不读取已经删除的 Top-64 `expert2`。它会检查源实验是否为 seed 234、测试视频数量是否为 UCF 290/XD 800，以及所需缓存是否完整。禁止把不同实验目录的文件拼到一起。

## 运行一个 Baseline

在远程服务器的 `vadmy_code/` 下运行：

```bash
source /etc/network_turbo
conda activate dsanet
bash ablation/run_cumulative.sh desc formal_seed234_final both
```

第一个参数只允许 `lagovad`、`desc`、`dsanet` 或 `vadclip`。第三个参数可使用 `ucf`、`xd` 或 `both`。默认断点复用已有阶段；需要重算该 Baseline 时在末尾传入 `--clean`：

```bash
bash ablation/run_cumulative.sh desc formal_seed234_final both --clean
```

目前服务器上的 `formal_seed234_final` 只有 DeSC 的 UCF/XD 产物完整，因此只能用于 `desc`。其余 Baseline 应先运行各自正式指令，并向本脚本传入对应的完整实验名。

输出位置：

```text
../vadmy_data/universal_neuron_adapter/ablations/<source-run-name>/
```

每个 Baseline 生成一份 CSV 和 JSON，字段包括绝对指标、相对 Baseline 的累计提升，以及相对上一步的模块增量。
